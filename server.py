"""
StyleGAN2 Latent Space Server
==============================
Serves Walk Explorer (/ui), Latent Browser (/browser), Infinite Map (/infinite).

Usage:
    python server.py --pkl path/to/your_model.pkl

Then open:
    http://localhost:5000           ← Landing page
    http://localhost:5000/browser   ← 2D Latent Browser
    http://localhost:5000/ui        ← Walk Explorer
    http://localhost:5000/infinite  ← Infinite Latent Map
"""

import argparse, os, sys, glob, io, base64, uuid, time
import warnings
warnings.filterwarnings('ignore', message='Failed to build CUDA kernels', category=UserWarning)
warnings.filterwarnings('ignore', message='TORCH_CUDA_ARCH_LIST', category=UserWarning)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stylegan2-ada-pytorch'))

# ── Suppress upfirdn2d_plugin spam ───────────────────────────────────────────
# StyleGAN2 tries to JIT-compile CUDA kernels on every G() call if the cache
# is missing. Monkey-patch the plugin loader to fail silently after first try.
import builtins as _builtins
_real_print = _builtins.print
_upfirdn_failed = False
def _filtered_print(*args, **kwargs):
    global _upfirdn_failed
    msg = ' '.join(str(a) for a in args)
    if 'upfirdn2d_plugin' in msg or 'bias_act_plugin' in msg:
        return  # suppress completely
    _real_print(*args, **kwargs)
_builtins.print = _filtered_print
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
from PIL import Image
from flask import Flask, jsonify, request, Response, send_from_directory, make_response
from flask_cors import CORS
import threading
import legacy

# ── Real-ESRGAN (optional — lazy-loaded on first /upscale call) ───────────────
try:
    import cv2 as _cv2_check  # noqa — just to detect availability at startup
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

_upsampler = None   # lazy-loaded

_recording     = False
_record_frames = []

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--pkl',  type=str, default='', help='Path to .pkl model file')
parser.add_argument('--port', type=int, default=5000)
parser.add_argument('--grid', type=int, default=6,  help='Grid size (grid x grid images)')
parser.add_argument('--res',  type=int, default=128, help='Thumbnail resolution for grid')
args = parser.parse_args()

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[server] Device: {device}')

# ── Model state ───────────────────────────────────────────────────────────────
G         = None
Z_DIM     = None
RES       = None
PKL_PATH  = None

def load_model(path):
    global G, Z_DIM, RES, PKL_PATH
    print(f'[server] Loading: {path}')
    with open(path, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    G.eval()
    Z_DIM    = G.z_dim
    RES      = G.img_resolution
    PKL_PATH = path
    _reset_walk()
    _grid_cache.clear()
    _tile_cache.clear()
    print(f'[server] ✅ z_dim={Z_DIM}  resolution={RES}')

# ── Image generation ──────────────────────────────────────────────────────────
def z_to_pil(z_vector, truncation_psi=0.7, size=None):
    with torch.no_grad():
        z     = torch.tensor(z_vector, dtype=torch.float32, device=device).unsqueeze(0)
        label = torch.zeros([1, G.c_dim], device=device)
        img   = G(z, label, truncation_psi=truncation_psi, noise_mode='const')
        img   = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    pil = Image.fromarray(img[0].cpu().numpy(), 'RGB')
    if size:
        pil = pil.resize((size, size), Image.LANCZOS)
    elif pil.width > 512:
        pil = pil.resize((512, 512), Image.LANCZOS)
    return pil

def pil_to_b64(pil, quality=85):
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

def z_to_b64(z_vector, truncation_psi=0.7, size=None):
    return pil_to_b64(z_to_pil(z_vector, truncation_psi, size))

def z_to_seed(z):
    """Deterministic seed number from a z vector."""
    return int(abs(hash(np.array(z).tobytes())) % 999999)

def slerp(z1, z2, t):
    z1n = z1 / (np.linalg.norm(z1) + 1e-8)
    z2n = z2 / (np.linalg.norm(z2) + 1e-8)
    dot = np.clip(np.dot(z1n, z2n), -1.0, 1.0)
    omega = np.arccos(dot)
    if np.abs(omega) < 1e-6:
        return (1 - t) * z1 + t * z2
    return (np.sin((1-t)*omega)/np.sin(omega))*z1 + (np.sin(t*omega)/np.sin(omega))*z2

# ── Walk state ────────────────────────────────────────────────────────────────
_walk = {}

# Server-side waypoint cache — keyed by session token.
# Stores the full z-vector arrays so the frontend never needs to round-trip them.
# Format: { token: { "zs": [np.array, ...], "b64s": [str, ...] } }
_waypoint_cache: dict = {}

def _reset_walk():
    global _walk
    if Z_DIM is None: return
    z_start = np.random.randn(Z_DIM)
    _walk = {
        'z':          z_start.copy(),
        'z_start':    z_start.copy(),
        'z_target':   np.random.randn(Z_DIM),
        't':          0.0,
        'truncation': 0.7,
        'step_size':  0.05,
        'pinned_z':   None,
    }

# ── Pending z — one-shot Browser → Explorer handoff ──────────────────────────
_pending_z = None

# ── PCA Grid ──────────────────────────────────────────────────────────────────
_grid_cache = {}

def build_pca_grid(n_samples=512, grid_size=None, thumb_res=None, truncation=0.7):
    """
    Sample n_samples z vectors, compute 2D PCA, build a grid_size x grid_size
    thumbnail grid covering the PCA space. Returns grid metadata + images.
    """
    if grid_size is None: grid_size = args.grid
    if thumb_res is None: thumb_res = args.res

    print(f'[PCA] Sampling {n_samples} vectors...')
    zs = np.random.randn(n_samples, Z_DIM).astype(np.float32)

    # PCA — manual 2-component via SVD (no sklearn needed)
    zs_centered = zs - zs.mean(axis=0)
    _, _, Vt = np.linalg.svd(zs_centered, full_matrices=False)
    pc1 = Vt[0]
    pc2 = Vt[1]

    # Project all samples onto PC1/PC2
    coords = zs_centered @ np.stack([pc1, pc2], axis=1)
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    # Build grid: for each cell, find the z closest to that PCA coordinate
    print(f'[PCA] Building {grid_size}x{grid_size} grid...')
    grid_zs   = []
    grid_imgs = []

    for row in range(grid_size):
        for col in range(grid_size):
            tx = x_min + (col + 0.5) / grid_size * (x_max - x_min)
            ty = y_min + (row + 0.5) / grid_size * (y_max - y_min)
            dists = np.sum((coords - np.array([tx, ty])) ** 2, axis=1)
            idx   = np.argmin(dists)
            z     = zs[idx]
            grid_zs.append(z.tolist())
            img_b64 = z_to_b64(z, truncation, size=thumb_res)
            grid_imgs.append(img_b64)
            print(f'[PCA] Generated {row*grid_size+col+1}/{grid_size*grid_size}', end='\r')

    print(f'\n[PCA] Grid ready.')

    result = {
        'grid_size':  grid_size,
        'thumb_res':  thumb_res,
        'n_samples':  n_samples,
        'x_min': float(x_min), 'x_max': float(x_max),
        'y_min': float(y_min), 'y_max': float(y_max),
        'pc1': pc1.tolist(),
        'pc2': pc2.tolist(),
        'z_mean': zs.mean(axis=0).tolist(),
        'grid_zs':   grid_zs,
        'grid_imgs': grid_imgs,
    }
    _grid_cache['data'] = result
    return result


def z_from_pca_coord(nx, ny, truncation=0.7):
    """
    Given normalized mouse position (nx, ny) in [0,1],
    reconstruct a z vector in PCA space.
    NOTE: This is an approximation — use grid/get_z for exact thumbnail z.
    """
    d = _grid_cache.get('data')
    if d is None:
        return None

    pc1    = np.array(d['pc1'])
    pc2    = np.array(d['pc2'])
    z_mean = np.array(d['z_mean'])
    x_min, x_max = d['x_min'], d['x_max']
    y_min, y_max = d['y_min'], d['y_max']

    px = x_min + nx * (x_max - x_min)
    py = y_min + ny * (y_max - y_min)

    z = z_mean + px * pc1 + py * pc2
    return z


# ── Infinite map tile cache ───────────────────────────────────────────────────
_tile_cache = {}

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='.')
CORS(app, origins='*')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route('/status')
def status():
    return jsonify({
        'status':     'ok' if G is not None else 'no_model',
        'z_dim':      int(Z_DIM) if Z_DIM else None,
        'resolution': int(RES)   if RES   else None,
        'pkl':        PKL_PATH,
    })


@app.route('/list_models')
def list_models():
    pkls = glob.glob(os.path.join(SCRIPT_DIR, '**', '*.pkl'), recursive=True)
    # Return paths relative to SCRIPT_DIR for portability
    rel = [os.path.relpath(p, SCRIPT_DIR).replace('\\', '/') for p in pkls]
    return jsonify({'models': sorted(set(rel))})


@app.route('/load_model', methods=['POST'])
def load_model_route():
    path = (request.json or {}).get('path', '')
    # Resolve relative paths against SCRIPT_DIR
    if not os.path.isabs(path):
        path = os.path.join(SCRIPT_DIR, path)
    if not os.path.isfile(path):
        return jsonify({'error': 'not found'}), 400
    try:
        load_model(path)
        return jsonify({'status': 'ok', 'z_dim': int(Z_DIM), 'resolution': int(RES)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Walk endpoints ────────────────────────────────────────────────────────────

@app.route('/walk', methods=['POST'])
def walk():
    if G is None: return jsonify({'error': 'no model'}), 400
    step = float((request.json or {}).get('step_size', _walk['step_size']))
    _walk['step_size'] = step
    _walk['t'] += step
    if _walk['t'] >= 1.0:
        _walk['z_start']  = _walk['z_target'].copy()
        _walk['z']        = _walk['z_target'].copy()
        _walk['z_target'] = np.random.randn(Z_DIM)
        _walk['t']        = 0.0
    else:
        _walk['z'] = slerp(_walk['z_start'], _walk['z_target'], _walk['t'])
    img = z_to_b64(_walk['z'], _walk['truncation'])
    if _recording:
        _record_frames.append(img)
    return jsonify({'image': img})


@app.route('/random', methods=['POST'])
def random_jump():
    if G is None: return jsonify({'error': 'no model'}), 400
    z = np.random.randn(Z_DIM)
    _walk['z']        = z.copy()
    _walk['z_start']  = z.copy()
    _walk['z_target'] = np.random.randn(Z_DIM)
    _walk['t']        = 0.0
    return jsonify({'image': z_to_b64(_walk['z'], _walk['truncation'])})


@app.route('/pin', methods=['POST'])
def pin():
    _walk['pinned_z'] = _walk['z'].copy()
    return jsonify({'status': 'pinned'})


@app.route('/recall', methods=['POST'])
def recall():
    if _walk.get('pinned_z') is None:
        return jsonify({'error': 'nothing pinned'}), 400
    _walk['z']        = _walk['pinned_z'].copy()
    _walk['z_start']  = _walk['pinned_z'].copy()
    _walk['z_target'] = np.random.randn(Z_DIM)
    _walk['t']        = 0.0
    return jsonify({'image': z_to_b64(_walk['z'], _walk['truncation'])})


@app.route('/set_seed', methods=['POST'])
def set_seed():
    if G is None: return jsonify({'error': 'no model'}), 400
    seed = int((request.json or {}).get('seed', 0))
    rng  = np.random.RandomState(seed)
    z    = rng.randn(Z_DIM)
    _walk['z']        = z.copy()
    _walk['z_start']  = z.copy()
    _walk['z_target'] = np.random.randn(Z_DIM)
    _walk['t']        = 0.0
    return jsonify({'image': z_to_b64(_walk['z'], _walk['truncation'])})


@app.route('/truncation', methods=['POST'])
def set_truncation():
    if G is None: return jsonify({'error': 'no model'}), 400
    _walk['truncation'] = float((request.json or {}).get('value', 0.7))
    return jsonify({'image': z_to_b64(_walk['z'], _walk['truncation'])})


# ── Pending z — one-shot Browser → Explorer handoff ──────────────────────────

@app.route('/pending_z')
def get_pending_z():
    """
    Returns the z vector sent from Browser to Explorer.
    Clears itself after the first read — one-shot delivery.
    """
    global _pending_z
    if _pending_z is None:
        return jsonify({'image': None})
    z          = _pending_z
    _pending_z = None   # clear after reading
    img  = z_to_b64(z, _walk.get('truncation', 0.7))
    seed = z_to_seed(z)
    return jsonify({'image': img, 'seed': seed})


# ── PCA Grid endpoints ────────────────────────────────────────────────────────

@app.route('/grid/build', methods=['POST'])
def grid_build():
    """Build (or rebuild) the PCA grid."""
    if G is None: return jsonify({'error': 'no model'}), 400
    data       = request.json or {}
    n_samples  = int(data.get('n_samples', 512))
    grid_size  = int(data.get('grid_size', args.grid))
    thumb_res  = int(data.get('thumb_res', args.res))
    truncation = float(data.get('truncation', 0.7))

    result = build_pca_grid(n_samples, grid_size, thumb_res, truncation)

    return jsonify({
        'status':    'ok',
        'grid_size': result['grid_size'],
        'thumb_res': result['thumb_res'],
        'grid_imgs': result['grid_imgs'],
    })


@app.route('/grid/probe', methods=['POST'])
def grid_probe():
    """
    Generate a preview image at a PCA coordinate (mouse position).
    Returns PCA-reconstructed z — for preview only, not for pinning.
    Use /grid/get_z for the exact thumbnail z vector.
    """
    if G is None: return jsonify({'error': 'no model'}), 400
    data       = request.json or {}
    nx         = float(data.get('nx', 0.5))
    ny         = float(data.get('ny', 0.5))
    truncation = float(data.get('truncation', 0.7))

    z = z_from_pca_coord(nx, ny, truncation)
    if z is None:
        return jsonify({'error': 'grid not built yet'}), 400

    img  = z_to_b64(z, truncation, size=512)
    seed = z_to_seed(z)
    return jsonify({'image': img, 'z': z.tolist(), 'seed': seed})


@app.route('/grid/get_z', methods=['POST'])
def grid_get_z():
    """Return the exact z vector of a grid thumbnail by row/col index."""
    data = request.json or {}
    row  = int(data.get('row', 0))
    col  = int(data.get('col', 0))
    d    = _grid_cache.get('data')
    if d is None:
        return jsonify({'error': 'grid not built'}), 400
    gs  = d['grid_size']
    idx = row * gs + col
    if idx >= len(d['grid_zs']):
        return jsonify({'error': 'out of range'}), 400
    z    = np.array(d['grid_zs'][idx])
    seed = z_to_seed(z)
    return jsonify({'z': d['grid_zs'][idx], 'seed': seed})


@app.route('/grid/pin_probe', methods=['POST'])
def grid_pin_probe():
    """
    Pin a z vector from the Browser.
    Sets _pending_z so Walk Explorer gets the exact same image on open.
    """
    global _pending_z
    data = request.json or {}
    z    = data.get('z')
    if z is None:
        return jsonify({'error': 'no z provided'}), 400
    za = np.array(z)
    _walk['z']        = za.copy()
    _walk['z_start']  = za.copy()
    _walk['z_target'] = np.random.randn(Z_DIM)
    _walk['t']        = 0.0
    _walk['pinned_z'] = za.copy()
    _pending_z        = za.copy()   # one-shot for Explorer init
    return jsonify({'status': 'ok'})


@app.route('/grid/pin_infinite', methods=['POST'])
def grid_pin_infinite():
    """
    Pin a z vector as the center point for the Infinite Map.
    Clears tile cache so new tiles are generated from this center.
    """
    global _tile_cache
    data = request.json or {}
    z    = data.get('z')
    if z is None:
        return jsonify({'error': 'no z provided'}), 400
    za = np.array(z)
    _walk['z']               = za.copy()
    _walk['z_start']         = za.copy()
    _walk['z_target']        = np.random.randn(Z_DIM)
    _walk['t']               = 0.0
    _walk['pinned_z']        = za.copy()
    _walk['infinite_center'] = za.copy()
    _tile_cache = {}   # clear so new tiles use this center
    return jsonify({'status': 'ok'})


@app.route('/infinite/center', methods=['GET'])
def infinite_center():
    """Return the pinned center z vector for the Infinite Map, if any."""
    center = _walk.get('infinite_center')
    if center is None:
        return jsonify({'center': None})
    return jsonify({'center': center.tolist()})


@app.route('/infinite/center/clear', methods=['POST'])
def infinite_center_clear():
    """Clear the infinite center after it has been read by the client."""
    _walk.pop('infinite_center', None)
    return jsonify({'status': 'ok'})


# ── Infinite map tiles ────────────────────────────────────────────────────────

def _z_hash(lx, ly):
    """Deterministic unit-normalised z for a grid position."""
    seed = int(abs(hash((round(lx, 3), round(ly, 3)))) % (2**31))
    rng  = np.random.RandomState(seed)
    z    = rng.randn(Z_DIM).astype(np.float32)
    return z / (np.linalg.norm(z) + 1e-8) * np.sqrt(Z_DIM)


def _z_for_tile(lx, ly, mode, center_z, step_size=1.0):
    """Compute z vector for one tile according to the chosen mode."""

    # ── Mode A: Hash — every tile fully independent ───────────────────────────
    if mode == 'hash':
        z_tile = _z_hash(lx, ly)
        if center_z is not None:
            cz    = np.array(center_z, dtype=np.float32)
            dist  = np.sqrt(lx**2 + ly**2)
            alpha = np.exp(-dist * 0.4)
            return slerp(z_tile, cz, alpha)
        return z_tile

    # ── Mode B: Slerp — smooth walk outward from center ───────────────────────
    elif mode == 'slerp':
        if center_z is not None:
            origin = np.array(center_z, dtype=np.float32)
        else:
            origin = _z_hash(0.0, 0.0)

        dist = np.sqrt(lx**2 + ly**2)
        if dist < 1e-6:
            return origin.copy()

        # Direction seeded by raw grid integers (not normalized ratios),
        # so every tile position always gets a unique direction regardless of stepSize
        grid_ix = int(round(lx / step_size)) if step_size > 1e-6 else int(round(lx * 1000))
        grid_iy = int(round(ly / step_size)) if step_size > 1e-6 else int(round(ly * 1000))
        dir_seed = int(abs(hash((grid_ix, grid_iy))) % (2**31))
        rng_dir   = np.random.RandomState(dir_seed)
        direction = rng_dir.randn(Z_DIM).astype(np.float32)
        origin_n  = origin / (np.linalg.norm(origin) + 1e-8)
        direction = direction - np.dot(direction, origin_n) * origin_n
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        # Angle based on grid distance (integer steps), not float distance
        grid_dist = np.sqrt(grid_ix**2 + grid_iy**2)
        angle = grid_dist * 0.12
        angle = np.clip(angle, 0, np.pi * 0.9)
        z = np.cos(angle) * origin + np.sin(angle) * direction * np.sqrt(Z_DIM)
        return z.astype(np.float32)

    # ── Mode C: Bilinear — 4 random corner anchors ────────────────────────────
    elif mode == 'bilinear':
        corners = {}
        for i, (cx, cy) in enumerate([(-1,-1),(1,-1),(-1,1),(1,1)]):
            rng_c = np.random.RandomState(100 + i)
            z_c   = rng_c.randn(Z_DIM).astype(np.float32)
            z_c   = z_c / (np.linalg.norm(z_c) + 1e-8) * np.sqrt(Z_DIM)
            corners[(cx,cy)] = z_c

        # Always work in grid-integer space so stepSize doesn't matter
        grid_ix = int(round(lx / step_size)) if step_size > 1e-6 else int(round(lx * 1000))
        grid_iy = int(round(ly / step_size)) if step_size > 1e-6 else int(round(ly * 1000))

        # Window of ±20 grid tiles — gives full 0..1 range across the visible map
        window = 20
        tx = np.clip((grid_ix + window) / (2 * window), 0, 1)
        ty = np.clip((grid_iy + window) / (2 * window), 0, 1)

        z00, z10 = corners[(-1,-1)], corners[(1,-1)]
        z01, z11 = corners[(-1, 1)], corners[(1, 1)]
        z_top    = slerp(z00, z10, tx)
        z_bot    = slerp(z01, z11, tx)
        z        = slerp(z_top, z_bot, ty)

        # Only blend toward center_z for tiles very close to origin (dist < 2 tiles)
        # — avoids overwriting bilinear variation across the whole map
        if center_z is not None:
            grid_dist = np.sqrt(grid_ix**2 + grid_iy**2)
            alpha = np.exp(-grid_dist * 1.5)   # sharp falloff: ~0.05 at dist=2
            if alpha > 0.01:
                cz = np.array(center_z, dtype=np.float32)
                z  = slerp(z, cz, alpha)
        return z.astype(np.float32)

    # Fallback
    return _z_hash(lx, ly)


@app.route('/infinite/tile', methods=['POST'])
def infinite_tile():
    if G is None: return jsonify({'error': 'no model'}), 400
    data      = request.json or {}
    lx        = float(data.get('lx', 0))
    ly        = float(data.get('ly', 0))
    trunc     = float(data.get('truncation', 0.7))
    center_z  = data.get('center_z', None)
    mode      = data.get('mode', 'hash')
    step_size = float(data.get('step_size', 1.0))

    center_key = hash(tuple(center_z[:8])) if center_z else 0
    cache_key  = f'{lx:.3f}_{ly:.3f}_{trunc:.2f}_{center_key}_{mode}_{step_size:.3f}'
    if cache_key in _tile_cache:
        return jsonify(_tile_cache[cache_key])

    z      = _z_for_tile(lx, ly, mode, center_z, step_size)
    img    = z_to_b64(z, trunc, size=128)
    result = {'image': img, 'z': z.tolist(), 'seed': z_to_seed(z)}
    _tile_cache[cache_key] = result
    return jsonify(result)


# ── Recording ─────────────────────────────────────────────────────────────────

@app.route('/record/start', methods=['POST'])
def record_start():
    global _recording, _record_frames
    _recording     = True
    _record_frames = []
    return jsonify({'status': 'recording'})


@app.route('/record/stop', methods=['POST'])
def record_stop():
    global _recording, _record_frames
    _recording = False
    if not _record_frames:
        return jsonify({'error': 'no frames'}), 400

    import subprocess, tempfile

    tmpdir = tempfile.mkdtemp()
    for i, frame_b64 in enumerate(_record_frames):
        img_data = base64.b64decode(frame_b64)
        img      = Image.open(io.BytesIO(img_data))
        img.save(os.path.join(tmpdir, f'frame_{i:05d}.png'))

    output_path = os.path.join(SCRIPT_DIR, f'output_{int(torch.randint(0,9999,(1,)).item())}.mp4')
    subprocess.run([
        r'C:\ffmpeg\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe', '-y',
        '-framerate', '12',
        '-i', os.path.join(tmpdir, 'frame_%05d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ], check=True)

    _record_frames = []
    return jsonify({'status': 'saved', 'path': output_path})


# ── Upscale with Real-ESRGAN ──────────────────────────────────────────────────

def _get_upsampler():
    global _upsampler
    if _upsampler is not None:
        return _upsampler
    import cv2
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    weights_path = os.path.join(SCRIPT_DIR, 'weights', 'RealESRGAN_x4plus.pth')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f'Weights not found: {weights_path}\n'
            'Download: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        )
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)
    _upsampler = RealESRGANer(
        scale=4, model_path=weights_path, model=model,
        tile=512, tile_pad=10, pre_pad=0, half=True,
    )
    print('[server] ✅ Real-ESRGAN loaded')
    return _upsampler


@app.route('/upscale_status')
def upscale_status():
    weights_path = os.path.join(SCRIPT_DIR, 'weights', 'RealESRGAN_x4plus.pth')
    try:
        import realesrgan  # noqa
        has_pkg = True
    except ImportError:
        has_pkg = False
    return jsonify({
        'available':   has_pkg and os.path.exists(weights_path),
        'has_package': has_pkg,
        'has_weights': os.path.exists(weights_path),
        'weights_path': weights_path,
    })


@app.route('/upscale', methods=['POST'])
def upscale():
    import cv2
    import numpy as _np
    data = request.get_json(force=True) or {}
    if 'image_b64' not in data:
        return jsonify({'error': 'missing image_b64'}), 400
    scale = int(data.get('scale', 4))
    if scale not in (2, 4):
        return jsonify({'error': 'scale must be 2 or 4'}), 400
    # Decode input
    try:
        raw     = base64.b64decode(data['image_b64'])
        arr     = _np.frombuffer(raw, dtype=_np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError('cv2 could not decode image')
    except Exception as e:
        return jsonify({'error': f'decode error: {e}'}), 400
    # Run upscale
    try:
        up = _get_upsampler()
        out_bgr, _ = up.enhance(img_bgr, outscale=scale)
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        print(f'[upscale] error: {e}')
        return jsonify({'error': f'upscale failed: {e}'}), 500
    # Encode as PNG (lossless)
    ok, buf = cv2.imencode('.png', out_bgr)
    if not ok:
        return jsonify({'error': 'png encode failed'}), 500
    return jsonify({
        'image_b64': base64.b64encode(buf.tobytes()).decode(),
        'mime': 'png',
    })


# ── Random z ─────────────────────────────────────────────────────────────────

@app.route('/random_z')
def random_z():
    """Return a fresh random z vector — used by Ausstellung on startup."""
    if G is None: return jsonify({'error': 'no model'}), 400
    z = np.random.randn(Z_DIM).astype(np.float32)
    return jsonify({'z': z.tolist()})


@app.route('/seed_to_z', methods=['POST'])
def seed_to_z():
    """Convert a user-supplied integer seed to a deterministic z vector.
    Uses np.random.RandomState(seed) — fully invertible, unlike z_to_seed()."""
    if G is None: return jsonify({'error': 'no model'}), 400
    data = request.json or {}
    try:
        seed = int(data.get('seed', 0))
    except (ValueError, TypeError):
        return jsonify({'error': 'invalid seed'}), 400
    rng = np.random.RandomState(seed)
    z   = rng.randn(Z_DIM).astype(np.float32)
    return jsonify({'z': z.tolist(), 'seed': seed})


@app.route('/render_z', methods=['POST'])
def render_z():
    """Render a single image from a z-vector. Used by waypoint seed inputs."""
    if G is None: return jsonify({'error': 'no model'}), 400
    data       = request.get_json(force=True) or {}
    z_raw      = data.get('z')
    truncation = float(data.get('truncation', 0.7))
    size       = int(data.get('size', 256))
    if not z_raw:
        return jsonify({'error': 'z required'}), 400
    z   = np.array(z_raw, dtype=np.float32)
    pil = z_to_pil(z, truncation, size=size)
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=80)
    return jsonify({'image': base64.b64encode(buf.getvalue()).decode()})


# ── Loop Walk — closed latent walk returning to start seed ────────────────────

@app.route('/walk/waypoints', methods=['POST'])
def walk_waypoints():
    """
    Generate N waypoint slots (slot 0 = seed, slots 1..N-1 = random intermediates).
    Stores zs server-side in _waypoint_cache; returns a token so the frontend
    never needs to round-trip the raw z-vector arrays.

    POST {
        "seed":        int,       # start seed
        "z":           [...],     # z vector (overrides seed)
        "n":           4,         # total slots incl. seed at [0]
        "truncation":  0.7,
        "wp_token":    str|null,  # existing token (for refresh)
        "refresh_idx": int|null   # slot to regenerate (never 0)
    }
    -> { "wp_token": str, "waypoint_b64s": [...] }
    """
    if G is None: return jsonify({"error": "no model"}), 400

    data        = request.get_json(force=True) or {}
    n           = max(2, int(data.get("n", 5)))  # min 2 = seed + 1 intermediate
    truncation  = float(data.get("truncation", 0.7))
    wp_token    = data.get("wp_token", None)
    refresh_idx = data.get("refresh_idx", None)

    # Resolve seed z (slot 0)
    if "z" in data and data["z"]:
        seed_z = np.array(data["z"], dtype=np.float32)
    else:
        raw_seed = data.get("seed", 0)
        try:
            seed = int(raw_seed)
        except (ValueError, TypeError):
            seed = 0
        rng    = np.random.RandomState(seed)
        seed_z = rng.randn(Z_DIM).astype(np.float32)

    # Retrieve cached zs or build fresh
    existing = _waypoint_cache.get(wp_token) if wp_token else None
    if existing and len(existing["zs"]) == n:
        waypoint_zs = [z.copy() for z in existing["zs"]]
        waypoint_zs[0] = seed_z  # always re-pin seed
        if refresh_idx is not None:
            idx = int(refresh_idx)
            if 1 <= idx < n:
                waypoint_zs[idx] = np.random.randn(Z_DIM).astype(np.float32)
    else:
        intermediates = [np.random.randn(Z_DIM).astype(np.float32) for _ in range(n - 1)]
        waypoint_zs   = [seed_z] + intermediates

    # Render thumbnails
    waypoint_b64s = []
    for wz in waypoint_zs:
        pil = z_to_pil(wz, truncation, size=256)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=80)
        waypoint_b64s.append(base64.b64encode(buf.getvalue()).decode())

    # Cache and mint/reuse token
    token = wp_token if (wp_token and existing and len(existing["zs"]) == n) else str(uuid.uuid4())
    _waypoint_cache[token] = {"zs": waypoint_zs, "b64s": waypoint_b64s}
    print(f"[walk_waypoints] token={token[:8]}  n={n}  seed_z[:3]={seed_z[:3].tolist()}")

    return jsonify({
        "wp_token":      token,
        "waypoint_zs":   [wz.tolist() for wz in waypoint_zs],
        "waypoint_b64s": waypoint_b64s,
    })


@app.route('/walk/waypoints/dump', methods=['POST'])
def walk_waypoints_dump():
    """
    Export cached waypoint z-vectors for a given token (for project persistence).
    Called by audio_server.py at save time so zs are written into project.json.

    POST { "wp_token": str }
    -> { "waypoint_zs": [[...], ...] }  or { "error": ... }
    """
    data  = request.get_json(force=True) or {}
    token = data.get("wp_token")
    cached = _waypoint_cache.get(token) if token else None
    if not cached:
        return jsonify({"error": f"no cache for token {token!r}"}), 404
    return jsonify({"waypoint_zs": [z.tolist() for z in cached["zs"]]})


@app.route('/walk/waypoints/restore', methods=['POST'])
def walk_waypoints_restore():
    """
    Re-ingest saved waypoint z-vectors (from project.json) into the server cache.
    Returns a fresh wp_token so the session can Preview/Final without re-generating.

    POST { "waypoint_zs": [[...], ...] }
    -> { "wp_token": str, "n": int }
    """
    if G is None: return jsonify({"error": "no model"}), 400
    data = request.get_json(force=True) or {}
    zs_raw = data.get("waypoint_zs", [])
    if not zs_raw:
        return jsonify({"error": "waypoint_zs required"}), 400
    waypoint_zs = [np.array(z, dtype=np.float32) for z in zs_raw]
    token = str(uuid.uuid4())
    _waypoint_cache[token] = {"zs": waypoint_zs, "b64s": []}
    print(f"[walk_waypoints/restore] token={token[:8]}  n={len(waypoint_zs)}")
    return jsonify({"wp_token": token, "n": len(waypoint_zs)})


@app.route('/walk/loop', methods=['POST'])
def walk_loop():
    """
    Generate a looping latent walk as MP4.
    Supports preview mode (fast, low-res) and final mode (full quality).

    POST {
        "seed":            int,           # start/end seed
        "z":               list[float],   # z vector (overrides seed)
        "duration":        30,
        "fps":             24,
        "waypoints":       4,             # number of intermediate waypoints
        "truncation":      0.7,
        "preview":         false,         # if true: fast low-res preview
        "waypoint_zs":     null,          # list of z vectors to reuse (for final pass)
        "output_name":     "walk_loop"
    }
    -> {
        "status": "ok", "path": "...", "frames": N, "video_b64": "...",
        "waypoint_zs": [[...], ...],      # always returned for reuse in final pass
        "waypoint_b64s": ["...", ...]     # preview thumbnails of each waypoint
    }
    """
    if G is None: return jsonify({"error": "no model"}), 400

    import subprocess, tempfile

    data           = request.get_json(force=True) or {}
    duration         = int(data.get("duration", 30))
    fps              = int(data.get("fps", 24))
    n_waypoints      = int(data.get("waypoints", 4))
    truncation       = float(data.get("truncation", 0.7))
    out_name         = data.get("output_name", "walk_loop")
    preview          = bool(data.get("preview", False))
    wp_token         = data.get("wp_token", None)
    project_dir_name = data.get("project_dir", None)

    # Resolve output directory
    if project_dir_name:
        out_dir = os.path.join(PROJECTS_DIR, project_dir_name)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = SCRIPT_DIR

    # Preview settings: low-res, low-fps
    target_size    = int(data.get("size", 512))
    do_upscale     = False
    ffmpeg_upscale = False   # post-render FFmpeg upscale (fast, replaces per-frame ESRGAN)
    if preview:
        res  = 256
        fps  = 8
        crf  = 28
    else:
        res  = 512
        crf  = 18
        if target_size > 512:
            ffmpeg_upscale = True   # upscale after render via FFmpeg — much faster than per-frame ESRGAN

    # Resolve waypoints from server-side cache (via wp_token) — no z-transfer needed
    cached = _waypoint_cache.get(wp_token) if wp_token else None
    waypoint_zs_direct = data.get("waypoint_zs", None)  # direct transfer from frontend

    if cached:
        waypoint_zs_list = cached["zs"]   # list of np.float32 arrays, slot 0 = seed
        z_start      = waypoint_zs_list[0].copy()
        intermediate = [z.copy() for z in waypoint_zs_list[1:]]
        print(f"[walk_loop] cache hit token={wp_token[:8]}  n={len(waypoint_zs_list)}  seed[:3]={z_start[:3].tolist()}")
    elif waypoint_zs_direct:
        # Frontend sent zs directly (after drag/drop reorder or cache miss)
        all_zs       = [np.array(z, dtype=np.float32) for z in waypoint_zs_direct]
        z_start      = all_zs[0]
        intermediate = all_zs[1:]
        # Drop trailing duplicate seed if present
        if len(intermediate) > 0 and np.allclose(intermediate[-1], z_start, atol=1e-5):
            intermediate = intermediate[:-1]
        print(f"[walk_loop] direct zs  n={len(all_zs)}  seed[:3]={z_start[:3].tolist()}")
    else:
        # Fallback: resolve seed from request body and generate random intermediates
        print(f"[walk_loop] WARNING: no cache and no zs for token={wp_token!r} — generating random walk")
        if "z" in data and data["z"]:
            z_start = np.array(data["z"], dtype=np.float32)
        else:
            raw_seed = data.get("seed", 0)
            try:
                seed = int(raw_seed)
            except (ValueError, TypeError):
                seed = 0
            rng     = np.random.RandomState(seed)
            z_start = rng.randn(Z_DIM).astype(np.float32)
        intermediate = [np.random.randn(Z_DIM).astype(np.float32) for _ in range(n_waypoints)]

    waypoints = [z_start] + intermediate + [z_start]  # close loop

    # Generate waypoint thumbnails (always, for UI preview)
    waypoint_b64s = []
    for wz in intermediate:
        pil = z_to_pil(wz, truncation, size=128)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=70)
        waypoint_b64s.append(base64.b64encode(buf.getvalue()).decode())

    total_frames   = duration * fps
    n_segments     = len(waypoints) - 1   # number of interpolation segments
    frames_per_seg = total_frames // n_segments

    if not preview:
        print(f"[walk_loop] size={target_size}, upscale={'ffmpeg→' + str(target_size) if ffmpeg_upscale else 'no'}, frames={total_frames}")

    # Optional source image override for frame 0 (when z-vector is unavailable)
    source_b64 = data.get("source_b64", None)
    source_pil = None
    if source_b64:
        try:
            import io as _io
            src_bytes  = base64.b64decode(source_b64)
            source_pil = Image.open(_io.BytesIO(src_bytes)).convert("RGB")
            source_pil = source_pil.resize((res, res), Image.LANCZOS)
        except Exception as e:
            print(f"[walk_loop] source_b64 decode failed: {e}")
            source_pil = None

    # Generate frames
    tmpdir    = tempfile.mkdtemp()
    frame_idx = 0

    # Crossfade from source_pil into latent walk over first N frames (fallback for missing z)
    crossfade_frames = min(fps, frames_per_seg // 3) if source_pil is not None else 0

    for seg in range(n_segments):
        z_a = waypoints[seg]
        z_b = waypoints[seg + 1]
        seg_frames = frames_per_seg if seg < n_segments - 1 else (total_frames - frame_idx)
        for f in range(seg_frames):
            # Half-open interval [0, 1): t never reaches 1.0 so waypoints are never
            # duplicated at segment boundaries, and the last frame approaches the seed
            # without equalling it — the video loop then transitions seamlessly back
            # to frame 0 (seed) without a freeze or jump.
            t   = f / seg_frames
            z_f = slerp(z_a, z_b, t)
            pil = z_to_pil(z_f, truncation, size=res)
            # Crossfade: blend source_pil → latent over first N frames (only when no z available)
            if source_pil is not None and frame_idx < crossfade_frames:
                alpha = frame_idx / crossfade_frames
                pil = Image.blend(source_pil, pil, alpha)
            # Per-frame ESRGAN removed — post-render FFmpeg upscale is used instead (see below)
            pil.save(os.path.join(tmpdir, f"frame_{frame_idx:05d}.png"))
            frame_idx += 1

    # ffmpeg -> MP4
    output_path = os.path.join(out_dir, f"{out_name}_{int(np.random.randint(0,9999))}.mp4")
    ffmpeg_exe  = r"C:\ffmpeg\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"

    try:
        subprocess.run([
            ffmpeg_exe, "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmpdir, "frame_%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(crf),
            output_path
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"ffmpeg failed: {e.stderr.decode()[:300]}"}), 500

    # Post-render upscale via FFmpeg lanczos (fast — single pass on finished video)
    if ffmpeg_upscale:
        upscaled_path = output_path.replace('.mp4', f'_up{target_size}.mp4')
        try:
            subprocess.run([
                ffmpeg_exe, "-y",
                "-i", output_path,
                "-vf", f"scale={target_size}:{target_size}:flags=lanczos",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", str(crf),
                "-preset", "slow", "-tune", "film",
                upscaled_path
            ], check=True, capture_output=True)
            print(f"[walk_loop] upscaled {target_size}px → {upscaled_path}")
            output_path = upscaled_path   # serve the upscaled version
        except subprocess.CalledProcessError as e:
            print(f"[walk_loop] FFmpeg upscale failed: {e.stderr.decode()[:300]} — using 512px")

    with open(output_path, "rb") as vf:
        video_b64 = base64.b64encode(vf.read()).decode()

    return jsonify({
        "status":        "ok",
        "path":          output_path,
        "frames":        frame_idx,
        "duration":      duration,
        "fps":           fps,
        "preview":       preview,
        "video_b64":     video_b64,
        "waypoint_zs":   [wz.tolist() for wz in intermediate],
        "waypoint_b64s": waypoint_b64s,
    })


@app.route('/walk/upscale_video', methods=['POST'])
def upscale_video():
    """
    Post-render upscale an existing MP4 to a target resolution using FFmpeg lanczos.
    Body: { "video_b64": "...", "target_size": 1024, "output_name": "walk_1024", "project_dir": "..." }
    Returns: { "video_b64": "...", "path": "..." }
    """
    import tempfile as _tmp
    data        = request.get_json(force=True) or {}
    video_b64   = data.get('video_b64', '')
    target_size = int(data.get('target_size', 1024))
    out_name    = data.get('output_name', 'walk_upscaled')
    proj_name   = data.get('project_dir', None)

    if not video_b64:
        return jsonify({'error': 'video_b64 required'}), 400

    if proj_name:
        out_dir = os.path.join(PROJECTS_DIR, proj_name)
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = SCRIPT_DIR

    ffmpeg_exe = r"C:\ffmpeg\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"

    # Write input video to temp file
    with _tmp.NamedTemporaryFile(suffix='.mp4', delete=False) as tf:
        tf.write(base64.b64decode(video_b64))
        in_path = tf.name

    out_path = os.path.join(out_dir, f"{out_name}_{target_size}.mp4")
    try:
        subprocess.run([
            ffmpeg_exe, "-y",
            "-i", in_path,
            "-vf", f"scale={target_size}:{target_size}:flags=lanczos",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "14",
            "-preset", "slow", "-tune", "film",
            out_path
        ], check=True, capture_output=True)
        print(f"[upscale_video] {target_size}px → {out_path}")
    except subprocess.CalledProcessError as e:
        os.unlink(in_path)
        return jsonify({'error': f'ffmpeg failed: {e.stderr.decode()[:300]}'}), 500
    finally:
        try: os.unlink(in_path)
        except: pass

    with open(out_path, 'rb') as f:
        out_b64 = base64.b64encode(f.read()).decode()

    return jsonify({'status': 'ok', 'path': out_path, 'video_b64': out_b64})


# ── Serve HTML files ──────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(SCRIPT_DIR, 'index.html')

@app.route('/ui')
def serve_explorer():
    return send_from_directory(SCRIPT_DIR, 'latent_explorer.html')

@app.route('/browser')
def serve_browser():
    return send_from_directory(SCRIPT_DIR, 'latent_browser.html')

@app.route('/infinite')
def serve_infinite():
    return send_from_directory(SCRIPT_DIR, 'latent_infinite.html')

@app.route('/ausstellung')
def serve_ausstellung():
    return send_from_directory(SCRIPT_DIR, 'latent_ausstellung.html')

@app.route('/merzpoet')
def serve_merzpoet():
    return send_from_directory(SCRIPT_DIR, 'latent_merzpoet.html')

@app.route('/view')
def serve_view():
    return send_from_directory(SCRIPT_DIR, 'latent_view.html')

@app.route('/audiolab')
def serve_audiolab():
    return send_from_directory(SCRIPT_DIR, 'audio_lab.html')

@app.route('/medialab')
def serve_medialab():
    return send_from_directory(SCRIPT_DIR, 'medialab.html')

@app.route('/ragui')
def serve_ragui():
    return send_from_directory(SCRIPT_DIR, 'rag_ui.html')


# ── Poem-Proxy: leitet /poem_proxy/* an Port 5001 weiter ─────────────────────
# Ermoeglicht Tablet/ngrok-Zugriff ohne direkte Port-5001-Verbindung
import requests as _proxy_requests

@app.route('/poem_proxy/<path:subpath>', methods=['GET', 'POST', 'OPTIONS'])
def poem_proxy(subpath):
    target = f'http://localhost:5001/{subpath}'
    try:
        if request.method == 'OPTIONS':
            resp = make_response()
            resp.headers['Access-Control-Allow-Origin'] = '*'
            resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            return resp
        r = _proxy_requests.request(
            method=request.method,
            url=target,
            json=request.get_json(silent=True),
            params=request.args,
            timeout=120,
        )
        resp = make_response(r.content, r.status_code)
        resp.headers['Content-Type'] = r.headers.get('Content-Type', 'application/json')
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    except Exception as e:
        return jsonify({'error': str(e)}), 502


# ── Mux video + audio into final MP4 ─────────────────────────────────────────

@app.route('/walk/preview_frame', methods=['POST'])
def walk_preview_frame():
    """
    Generate a side-by-side preview PNG:
    left = source image (base64 JPEG), right = black panel with text.
    No video needed — instant preview of final composition.

    POST {
        "image_b64":  "base64 JPEG of source image",
        "text":       "poem text",
        "font_size":  48
    }
    -> { "preview_b64": "base64 PNG" }
    """
    import textwrap
    from PIL import Image, ImageDraw, ImageFont
    import io as _io

    data       = request.get_json(force=True) or {}
    image_b64  = data.get('image_b64', '')
    text       = data.get('text', '')
    font_size  = int(data.get('font_size', 48))
    text_offset_x = int(data.get('text_offset_x', 60))
    text_offset_y = int(data.get('text_offset_y', 0))

    size = 512  # panel size

    # ── Left: source image ────────────────────────────────────────────────────
    if image_b64:
        try:
            img_bytes = base64.b64decode(image_b64)
            left_img  = Image.open(_io.BytesIO(img_bytes)).convert('RGB')
            left_img  = left_img.resize((size, size), Image.LANCZOS)
        except Exception:
            left_img = Image.new('RGB', (size, size), (30, 30, 30))
    else:
        left_img = Image.new('RGB', (size, size), (30, 30, 30))

    # ── Right: text panel ─────────────────────────────────────────────────────
    padding = 60
    font    = None
    font_candidates = [
        r'C:\Windows\Fonts\meiryo.ttc',
        r'C:\Windows\Fonts\msgothic.ttc',
        r'C:\Windows\Fonts\YuGothR.ttc',
        os.path.join(SCRIPT_DIR, 'fonts', 'Inter_28pt-SemiBold.ttf'),
        r'C:\Users\vladi\AppData\Local\Microsoft\Windows\Fonts\Inter_28pt-SemiBold.ttf',
        r'C:\Windows\Fonts\calibri.ttf',
        r'C:\Windows\Fonts\arial.ttf',
    ]
    for fp in font_candidates:
        if os.path.isfile(fp):
            try:
                idx = 0 if fp.lower().endswith('.ttc') else None
                font = ImageFont.truetype(fp, font_size, index=idx or 0)
                print(f"Font loaded (preview): {fp}")
                break
            except Exception as e:
                print(f"Font failed {fp}: {e}")
    if font is None:
        font = ImageFont.load_default()
        print("WARNING: Using PIL default font — no CJK font found")

    usable_w = size - padding * 2

    def char_width_p(ch):
        cp = ord(ch)
        if (0x3000 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF or
                0xFF00 <= cp <= 0xFFEF or 0x20000 <= cp <= 0x2FA1F):
            return font_size * 1.0
        return font_size * 0.52

    def wrap_text_p(txt, w):
        result = []
        for para in txt.split('\n'):
            if not para.strip():
                result.append('')
                continue
            line, lw = '', 0
            for ch in para:
                cw = char_width_p(ch)
                if lw + cw > w and line:
                    result.append(line)
                    line, lw = ch, cw
                else:
                    line += ch
                    lw += cw
            if line:
                result.append(line)
        return result

    lines  = wrap_text_p(text, size - text_offset_x - padding)
    line_h         = font_size + int(font_size * 0.45)
    total_h        = len(lines) * line_h
    y_start        = text_offset_y if text_offset_y > 0 else max(padding, (size - total_h) // 2)

    right_img  = Image.new('RGB', (size, size), (0, 0, 0))
    draw       = ImageDraw.Draw(right_img)
    for i, line in enumerate(lines):
        y = y_start + i * line_h
        if y + line_h > size - padding:
            draw.text((text_offset_x, y), '…', font=font, fill=(140, 140, 140))
            break
        draw.text((text_offset_x, y), line, font=font, fill=(240, 240, 240))

    # ── Composite: hstack ─────────────────────────────────────────────────────
    composite = Image.new('RGB', (size * 2, size), (0, 0, 0))
    composite.paste(left_img,  (0,    0))
    composite.paste(right_img, (size, 0))

    buf = _io.BytesIO()
    composite.save(buf, format='JPEG', quality=90)
    buf.seek(0)
    preview_b64 = base64.b64encode(buf.read()).decode()

    return jsonify({'preview_b64': preview_b64})


@app.route('/walk/mux', methods=['POST'])
def walk_mux():
    """
    Combine walk MP4 + audio WAV into side-by-side composite:
    left = video walk, right = black panel with poem text (PIL rendered).

    POST {
        "video_path":  "C:/path/to/walk.mp4",
        "audio_b64":   "base64-encoded WAV",
        "text":        "poem text for right panel",
        "font_size":   48,
        "volume":      1.0,
        "fade_in":     2,
        "fade_out":    3,
        "output_name": "medialab_export_1234"
    }
    """
    import subprocess, tempfile, textwrap, wave, contextlib, re
    from PIL import Image, ImageDraw, ImageFont

    data        = request.get_json(force=True) or {}
    video_path  = data.get('video_path', '')
    audio_b64   = data.get('audio_b64', '')
    text        = data.get('text', '')
    font_size       = int(data.get('font_size', 48))
    out_name        = data.get('output_name', 'medialab_export')
    volume          = max(0.0, min(2.0, float(data.get('volume', 1.0))))
    fade_in         = int(data.get('fade_in', 2))
    fade_out        = int(data.get('fade_out', 3))
    export_duration = data.get('export_duration', None)
    export_crf      = max(8, min(28, int(data.get('crf', 14))))
    layout_format   = data.get('format', 'landscape')
    text_offset_x   = int(data.get('text_offset_x', 60))
    text_offset_y   = int(data.get('text_offset_y', 0))
    project_dir_name = data.get('project_dir', None)
    if export_duration:
        export_duration = float(export_duration)

    mux_out_dir = SCRIPT_DIR
    if project_dir_name:
        _pd = os.path.join(PROJECTS_DIR, project_dir_name)
        if os.path.isdir(_pd):
            mux_out_dir = _pd

    # Accept video_b64 as alternative to video_path (free upload from browser)
    video_b64_in = data.get('video_b64', '')
    tmpdir = tempfile.mkdtemp()
    if (not video_path or not os.path.isfile(video_path)) and video_b64_in:
        _vf = os.path.join(tmpdir, 'uploaded_walk.mp4')
        with open(_vf, 'wb') as f:
            f.write(base64.b64decode(video_b64_in))
        video_path = _vf
    if not video_path or not os.path.isfile(video_path):
        return jsonify({'error': f'video not found: {video_path}'}), 400
    if not audio_b64:
        return jsonify({'error': 'no audio provided'}), 400

    audio_path = os.path.join(tmpdir, 'audio.wav')
    with open(audio_path, 'wb') as f:
        f.write(base64.b64decode(audio_b64))

    ffmpeg_exe = r'C:\ffmpeg\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe'

    # ── Get video dimensions ──────────────────────────────────────────────────
    probe  = subprocess.run([ffmpeg_exe, '-i', video_path], capture_output=True, text=True)
    vid_w, vid_h = 512, 512
    for line in probe.stderr.split('\n'):
        if 'Video:' in line:
            m = re.search(r'(\d{3,4})x(\d{3,4})', line)
            if m:
                vid_w, vid_h = int(m.group(1)), int(m.group(2))
                break

    # ── Shared font loader ────────────────────────────────────────────────────
    padding = 60
    font    = None
    font_candidates = [
        r'C:\Windows\Fonts\meiryo.ttc',
        r'C:\Windows\Fonts\msgothic.ttc',
        r'C:\Windows\Fonts\YuGothR.ttc',
        r'C:\Users\vladi\AppData\Local\Microsoft\Windows\Fonts\Inter_28pt-SemiBold.ttf',
        r'C:\Users\vladi\AppData\Local\Microsoft\Windows\Fonts\Inter_24pt-SemiBold.ttf',
        r'C:\Windows\Fonts\calibri.ttf',
        r'C:\Windows\Fonts\arial.ttf',
    ]
    for fp in font_candidates:
        if os.path.isfile(fp):
            try:
                idx = 0 if fp.lower().endswith('.ttc') else None
                font = ImageFont.truetype(fp, font_size, index=idx or 0)
                print(f"Font loaded (mux): {fp}")
                break
            except Exception as e:
                print(f"Font failed {fp}: {e}")
    if font is None:
        font = ImageFont.load_default()
        print("WARNING: Using PIL default font — no CJK font found")

    def char_width(ch):
        cp = ord(ch)
        if (0x3000 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF or
                0xFF00 <= cp <= 0xFFEF or 0x20000 <= cp <= 0x2FA1F):
            return font_size * 1.0
        return font_size * 0.52

    def wrap_text(txt, usable_w):
        lines = []
        for para in txt.split('\n'):
            if not para.strip():
                lines.append(''); continue
            line, w = '', 0
            for ch in para:
                cw = char_width(ch)
                if w + cw > usable_w and line:
                    lines.append(line); line, w = ch, cw
                else:
                    line += ch; w += cw
            if line: lines.append(line)
        return lines

    # ── Audio filter ──────────────────────────────────────────────────────────
    # Use soundfile to read duration — supports float32 WAV (format 3) from Stable Audio
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        audio_dur = info.frames / info.samplerate
    except Exception:
        import contextlib, wave as _wave
        with contextlib.closing(_wave.open(audio_path, 'r')) as wf:
            audio_dur = wf.getnframes() / float(wf.getframerate())
    master_dur     = export_duration if export_duration else audio_dur
    fade_out_start = max(0.0, master_dur - fade_out)
    af_parts = [f'volume={volume:.2f}']
    if fade_in  > 0: af_parts.append(f'afade=t=in:st=0:d={fade_in}')
    if fade_out > 0: af_parts.append(f'afade=t=out:st={fade_out_start:.2f}:d={fade_out}')
    af_filter = ','.join(af_parts)

    output_path = os.path.join(mux_out_dir, f'{out_name}.mp4')

    if layout_format == 'vertical':
        # ── Vertical 9:16: image top, text panel bottom ───────────────────────
        out_w  = vid_w
        out_h  = vid_w * 16 // 9          # 512 → 910, 1024 → 1820
        text_h = out_h - vid_h
        pad    = max(20, text_h // 8)

        line_h   = font_size + int(font_size * 0.45)
        lines    = wrap_text(text, out_w - pad * 2)
        total_h  = len(lines) * line_h
        y_start  = text_offset_y if text_offset_y > 0 else (pad + max(0, (text_h - pad * 2 - total_h) // 2))

        bot = Image.new('RGB', (out_w, text_h), (0, 0, 0))
        drw = ImageDraw.Draw(bot)
        for i, line in enumerate(lines):
            y = y_start + i * line_h
            if y + line_h > text_h: break
            lw  = sum(char_width(ch) for ch in line)
            # text_offset_x: 0 = centered, >0 = shift right from center
            x   = max(pad, int((out_w - lw) // 2) + text_offset_x)
            drw.text((x, y), line, font=font, fill=(240, 240, 240))

        bot_frame = os.path.join(tmpdir, 'bottom_panel.png')
        bot.save(bot_frame)

        try:
            subprocess.run([
                ffmpeg_exe, '-y',
                '-i', video_path,
                '-loop', '1', '-i', bot_frame,
                '-i', audio_path,
                '-filter_complex',
                f'[0:v]scale={out_w}:{vid_h}:flags=lanczos,trim=duration={master_dur:.3f},setpts=PTS-STARTPTS[top];'
                f'[1:v]scale={out_w}:{text_h}[bot];'
                f'[top][bot]vstack=inputs=2[out]',
                '-map', '[out]', '-map', '2:a',
                '-af', af_filter,
                '-c:v', 'libx264', '-c:a', 'aac',
                '-pix_fmt', 'yuv420p', '-crf', str(export_crf), '-preset', 'slow', '-tune', 'film',
                '-t', f'{master_dur:.3f}', output_path
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            return jsonify({'error': f'ffmpeg failed: {e.stderr.decode()[:400]}'}), 500

    else:
        # ── Landscape: video left, text panel right ───────────────────────────
        usable_w = vid_w - text_offset_x - padding
        lines    = wrap_text(text, usable_w)
        line_h   = font_size + int(font_size * 0.45)
        total_h  = len(lines) * line_h
        y_start  = text_offset_y if text_offset_y > 0 else max(padding, (vid_h - total_h) // 2)

        img  = Image.new('RGB', (vid_w, vid_h), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        for i, line in enumerate(lines):
            y = y_start + i * line_h
            if y + line_h > vid_h - padding:
                draw.text((text_offset_x, y), '…', font=font, fill=(140, 140, 140))
                break
            draw.text((text_offset_x, y), line, font=font, fill=(240, 240, 240))

        text_frame = os.path.join(tmpdir, 'text_panel.png')
        img.save(text_frame)

        try:
            subprocess.run([
                ffmpeg_exe, '-y',
                '-i', video_path,
                '-loop', '1', '-i', text_frame,
                '-i', audio_path,
                '-filter_complex',
                f'[0:v]scale={vid_w}:{vid_h}:flags=lanczos,trim=duration={master_dur:.3f},setpts=PTS-STARTPTS[left];'
                f'[1:v]scale={vid_w}:{vid_h}[right];'
                f'[left][right]hstack=inputs=2[out]',
                '-map', '[out]', '-map', '2:a',
                '-af', af_filter,
                '-c:v', 'libx264', '-c:a', 'aac',
                '-pix_fmt', 'yuv420p', '-crf', str(export_crf), '-preset', 'slow', '-tune', 'film',
                '-t', f'{master_dur:.3f}', output_path
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            return jsonify({'error': f'ffmpeg failed: {e.stderr.decode()[:400]}'}), 500

    with open(output_path, 'rb') as f:
        video_b64 = base64.b64encode(f.read()).decode()

    return jsonify({
        'video_b64': video_b64,
        'filename':  os.path.basename(output_path),
        'path':      output_path,
    })


@app.route('/gallery3d')
def serve_gallery3d():
    return send_from_directory(SCRIPT_DIR, 'latent_gallery3d.html')


# ── Start ─────────────────────────────────────────────────────────────────────

if args.pkl:
    load_model(args.pkl)



# ── Project filesystem endpoints ──────────────────────────────────────────────

PROJECTS_DIR = os.path.join(SCRIPT_DIR, 'projects')
os.makedirs(PROJECTS_DIR, exist_ok=True)


# ── GAN Projection (Z-finder) ─────────────────────────────────────────────────
_proj_jobs = {}
_proj_lock = threading.Lock()

def _run_projection(job_id, target_pil, num_steps, truncation_psi):
    import torch.nn.functional as F
    try:
        with _proj_lock:
            _proj_jobs[job_id].update({'status':'running','message':'Preparing...'})
        res = G.img_resolution
        target = target_pil.convert('RGB').resize((res, res), Image.LANCZOS)
        target_np = np.array(target, dtype=np.float32) / 127.5 - 1.0
        target_t  = torch.tensor(target_np, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
        label = torch.zeros([1, G.c_dim], device=device)
        z = torch.randn([1, G.z_dim], device=device, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=0.01)
        def lr_lambda(s):
            w = num_steps // 10
            return (s/w) if s < w else max(0.01, 1.0 - (s-w)/(num_steps-w))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        best_z, best_loss = z.detach().clone(), float('inf')
        for step in range(num_steps):
            optimizer.zero_grad()
            img = G(z, label, truncation_psi=truncation_psi, noise_mode='const')
            loss = F.mse_loss(img, target_t) + 0.5 * F.mse_loss(F.avg_pool2d(img,4), F.avg_pool2d(target_t,4))
            loss.backward(); optimizer.step(); scheduler.step()
            with torch.no_grad(): z.clamp_(-4.0, 4.0)
            lv = loss.item()
            if lv < best_loss: best_loss, best_z = lv, z.detach().clone()
            if step % max(1, num_steps//50) == 0:
                with _proj_lock:
                    _proj_jobs[job_id].update({'progress':int(step/num_steps*100),'message':f'Step {step}/{num_steps} — loss {lv:.4f}'})
        z_np = best_z.squeeze(0).cpu().numpy().tolist()
        proj_b64 = pil_to_b64(z_to_pil(best_z.squeeze(0).cpu().numpy(), truncation_psi))
        with _proj_lock:
            _proj_jobs[job_id].update({'status':'done','progress':100,'message':f'Done — loss {best_loss:.4f}','z':z_np,'image_b64':proj_b64})
        print(f"[project_z] {job_id[:8]} done steps={num_steps} loss={best_loss:.4f}")
    except Exception as e:
        import traceback; tb = traceback.format_exc()
        print(f"[project_z] ERROR: {e}\n{tb}")
        with _proj_lock:
            _proj_jobs[job_id].update({'status':'error','message':str(e)})

@app.route('/project_z/start', methods=['POST'])
def project_z_start():
    if G is None: return jsonify({'error':'Model not loaded'}), 400
    data = request.get_json(force=True) or {}
    img_b64 = data.get('image_b64','')
    num_steps = max(50, min(1000, int(data.get('num_steps', 300))))
    truncation = float(data.get('truncation', 0.7))
    if not img_b64: return jsonify({'error':'image_b64 required'}), 400
    try:
        target_pil = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert('RGB')
    except Exception as e:
        return jsonify({'error':f'Image decode failed: {e}'}), 400
    job_id = str(uuid.uuid4())
    with _proj_lock:
        _proj_jobs[job_id] = {'status':'queued','progress':0,'message':'Queued...','z':None,'image_b64':None,'error':None}
    threading.Thread(target=_run_projection, args=(job_id,target_pil,num_steps,truncation), daemon=True).start()
    print(f"[project_z] started job={job_id[:8]} steps={num_steps}")
    return jsonify({'job_id':job_id})

@app.route('/project_z/status/<job_id>', methods=['GET'])
def project_z_status(job_id):
    with _proj_lock: job = _proj_jobs.get(job_id)
    if not job: return jsonify({'error':'Job not found'}), 404
    return jsonify(job)

@app.route('/project_z/cancel/<job_id>', methods=['POST'])
def project_z_cancel(job_id):
    with _proj_lock:
        if job_id in _proj_jobs: _proj_jobs[job_id]['status'] = 'cancelled'
    return jsonify({'ok':True})

@app.route('/project/init', methods=['POST'])
def project_init():
    """Create an empty project directory and return its name."""
    data     = request.get_json(force=True) or {}
    dir_name = data.get('dir_name', f"project_{int(time.time())}")
    # Sanitize
    import re as _re
    dir_name = _re.sub(r'[^\w\-äöüÄÖÜß]', '_', dir_name).strip('_')
    project_dir = os.path.join(PROJECTS_DIR, dir_name)
    suffix = 0
    while os.path.exists(project_dir):
        suffix += 1
        project_dir = os.path.join(PROJECTS_DIR, f"{dir_name}_{suffix}")
    os.makedirs(project_dir)
    print(f"Project initialized: {project_dir}")
    return jsonify({'project_dir': project_dir,
                    'project_dir_name': os.path.basename(project_dir)})


@app.route('/project/save', methods=['POST'])
def project_save():
    """
    Save a full project to disk.
    POST {
        "seed":         int or str,
        "date":         "2026-05-03",
        "poem":         "...",
        "audio_prompt": "...",
        "genre":        "...",
        "image_b64":    "base64 JPEG",
        "walk_path":    "C:/path/to/walk.mp4",   # existing file to copy
        "sounds": [
            { "model": "musicgen-small", "sampler": null, "audio_b64": "...", "elapsed": 42 },
            ...
        ],
        "selected_sound_idx": 0,
        "volume": 100,
        "fade_in": 2,
        "fade_out": 3,
    }
    -> { "project_dir": "...", "name": "2026-05-03_seed-1234" }
    """
    import shutil

    data      = request.get_json(force=True) or {}
    seed      = str(data.get('seed', 'unknown')).replace('—', 'unknown')
    date      = data.get('date', '2026-01-01')
    poem      = data.get('poem', '')
    audio_prompt = data.get('audio_prompt', '')
    genre     = data.get('genre', '')
    image_b64 = data.get('image_b64', '')
    walk_path = data.get('walk_path', '')
    sounds    = data.get('sounds', [])
    sel_idx   = data.get('selected_sound_idx', -1)
    volume    = data.get('volume', 100)
    fade_in   = data.get('fade_in', 2)
    fade_out  = data.get('fade_out', 3)

    # Create project directory
    dir_name    = f"{date}_seed-{seed}"
    project_dir = os.path.join(PROJECTS_DIR, dir_name)
    # Avoid collisions
    suffix = 0
    while os.path.exists(project_dir):
        suffix += 1
        project_dir = os.path.join(PROJECTS_DIR, f"{dir_name}_{suffix}")
    os.makedirs(project_dir)

    saved_files = {}

    # Save source image
    if image_b64:
        try:
            img_path = os.path.join(project_dir, 'source.jpg')
            with open(img_path, 'wb') as f:
                f.write(base64.b64decode(image_b64))
            saved_files['image'] = 'source.jpg'
        except Exception as e:
            print("WARNING:", f'Image save failed: {e}')

    # Copy walk video
    if walk_path and os.path.isfile(walk_path):
        try:
            dst = os.path.join(project_dir, 'walk.mp4')
            shutil.copy2(walk_path, dst)
            saved_files['walk'] = 'walk.mp4'
        except Exception as e:
            print("WARNING:", f'Walk copy failed: {e}')

    # Save poem text
    if poem:
        txt_path = os.path.join(project_dir, 'text.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(poem)
        saved_files['text'] = 'text.txt'

    # Save audio prompt
    if audio_prompt:
        ap_path = os.path.join(project_dir, 'audio_prompt.txt')
        with open(ap_path, 'w', encoding='utf-8') as f:
            f.write(audio_prompt)

    # Save sound files
    saved_sounds = []
    for i, s in enumerate(sounds):
        ab64 = s.get('audio_b64', '')
        if not ab64:
            continue
        model   = s.get('model', f'sound_{i}').replace('/', '-')
        sampler = s.get('sampler', '')
        if sampler:
            safe_sampler = sampler.replace('/', '-').replace(':', '-')
            fname = f"{model}_{safe_sampler}.wav"
        else:
            fname = f"{model}.wav"
        # Avoid name collisions
        fpath = os.path.join(project_dir, fname)
        cnt   = 1
        while os.path.exists(fpath):
            fpath = os.path.join(project_dir, fname.replace('.wav', f'_{cnt}.wav'))
            cnt  += 1
        try:
            with open(fpath, 'wb') as f:
                f.write(base64.b64decode(ab64))
            saved_sounds.append({
                'file':    os.path.basename(fpath),
                'model':   s.get('model'),
                'sampler': sampler,
                'elapsed': s.get('elapsed'),
                'rating':  s.get('rating', 0),
                'selected': i == sel_idx,
            })
        except Exception as e:
            print("WARNING:", f'Sound {i} save failed: {e}')

    # Save project.json metadata
    meta = {
        'name':         os.path.basename(project_dir),
        'seed':         seed,
        'date':         date,
        'genre':        genre,
        'poem':         poem,
        'audio_prompt': audio_prompt,
        'sounds':       saved_sounds,
        'settings':     {'volume': volume, 'fade_in': fade_in, 'fade_out': fade_out},
        'files':        saved_files,
    }
    with open(os.path.join(project_dir, 'project.json'), 'w', encoding='utf-8') as f:
        import json as _json
        _json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Project saved: {project_dir}")
    return jsonify({
        'project_dir': project_dir,
        'name':        os.path.basename(project_dir),
        'sounds_saved': len(saved_sounds),
    })


@app.route('/project/list', methods=['GET'])
def project_list():
    """List all saved projects."""
    import json as _json
    projects = []
    if not os.path.isdir(PROJECTS_DIR):
        return jsonify({'projects': []})
    for name in sorted(os.listdir(PROJECTS_DIR), reverse=True):
        proj_dir  = os.path.join(PROJECTS_DIR, name)
        meta_path = os.path.join(proj_dir, 'project.json')
        if not os.path.isdir(proj_dir) or not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = _json.load(f)
            # Add thumbnail if source.jpg exists
            img_path = os.path.join(proj_dir, 'source.jpg')
            if os.path.isfile(img_path):
                with open(img_path, 'rb') as f:
                    meta['image_b64'] = base64.b64encode(f.read()).decode()
            meta['dir'] = proj_dir
            projects.append(meta)
        except Exception as e:
            print("WARNING:", f'Project list error for {name}: {e}')
    return jsonify({'projects': projects})


@app.route('/project/load', methods=['POST'])
@app.route('/project/load/<project_id>', methods=['GET'])
def project_load(project_id=None):
    """
    Load a project — return all files as base64.
    POST { "dir": "C:/path/to/project_dir" }
    """
    import json as _json

    data        = request.get_json(force=True) if request.method == 'POST' else {}
    # GET: resolve by project_id; POST: use 'dir' field
    if project_id:
        project_dir = os.path.join(PROJECTS_DIR, project_id)
    else:
        project_dir = (data or {}).get('dir', '')

    if not project_dir or not os.path.isdir(project_dir):
        return jsonify({'error': f'Project dir not found: {project_dir}'}), 400

    meta_path = os.path.join(project_dir, 'project.json')
    if not os.path.isfile(meta_path):
        return jsonify({'error': 'project.json not found'}), 400

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = _json.load(f)

    # Load image
    img_path = os.path.join(project_dir, 'source.jpg')
    if os.path.isfile(img_path):
        with open(img_path, 'rb') as f:
            meta['image_b64'] = base64.b64encode(f.read()).decode()

    # Load walk — check project dir first, then migrate from root if needed
    import shutil as _shutil
    walk_in_project = os.path.join(project_dir, 'walk.mp4')

    if not os.path.isfile(walk_in_project):
        # Look for walk in root SCRIPT_DIR by old naming patterns
        candidates = []
        # Check meta for a stored walk_path
        old_walk = (meta.get('settings') or {}).get('walk_path') or meta.get('walk_path') or ''
        if old_walk and os.path.isfile(old_walk):
            candidates.append(old_walk)
        # Also scan SCRIPT_DIR for medialab_walk*.mp4 and walk_loop*.mp4
        for fname in os.listdir(SCRIPT_DIR):
            if fname.endswith('.mp4') and any(fname.startswith(p) for p in
                    ('medialab_walk', 'medialab_preview', 'walk_loop', 'walk_512', 'walk_1024', 'walk_2048')):
                candidates.append(os.path.join(SCRIPT_DIR, fname))

        if candidates:
            # Pick the newest file
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            src = candidates[0]
            try:
                _shutil.copy2(src, walk_in_project)
                print(f"Migrated walk to project: {src} → {walk_in_project}")
                # Remove from root to keep it clean
                if os.path.dirname(src) == SCRIPT_DIR:
                    os.remove(src)
                    print(f"Removed old walk from root: {src}")
            except Exception as e:
                print("WARNING:", f"Walk migration failed: {e}")

    meta['walk_path'] = walk_in_project if os.path.isfile(walk_in_project) else None

    # Load walk as base64 so frontend can display it on project load
    if os.path.isfile(walk_in_project):
        with open(walk_in_project, 'rb') as f:
            meta['walk_b64'] = base64.b64encode(f.read()).decode()

    # Load sounds as base64
    for s in meta.get('sounds', []):
        fpath = os.path.join(project_dir, s.get('file', ''))
        if os.path.isfile(fpath):
            with open(fpath, 'rb') as f:
                s['audio_b64'] = base64.b64encode(f.read()).decode()

    meta['dir'] = project_dir
    return jsonify(meta)


@app.route('/project/rename', methods=['POST'])
def project_rename():
    """Rename a project directory."""
    import re as _re, shutil as _shutil
    data   = request.get_json(force=True) or {}
    old_id = data.get('old_id', '')
    new_id = data.get('new_id', '')
    if not old_id or not new_id:
        return jsonify({'error': 'old_id and new_id required'}), 400
    # Sanitize new_id
    new_id = _re.sub(r'[^\w\-äöüÄÖÜß]', '_', new_id).strip('_')
    old_path = os.path.join(PROJECTS_DIR, old_id)
    new_path = os.path.join(PROJECTS_DIR, new_id)
    if not os.path.isdir(old_path):
        return jsonify({'error': f'Source not found: {old_id}'}), 400
    if os.path.exists(new_path):
        # Add suffix to avoid collision
        new_path = new_path + '_2'
        new_id   = new_id + '_2'
    try:
        os.rename(old_path, new_path)
        print(f"Project renamed: {old_id} → {new_id}")
        return jsonify({'new_id': new_id, 'new_path': new_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/project/delete', methods=['POST'])
def project_delete():
    """Delete a project directory."""
    import shutil
    data        = request.get_json(force=True) or {}
    project_dir = data.get('dir', '')
    # Safety: only delete inside projects dir
    if not project_dir.startswith(PROJECTS_DIR):
        return jsonify({'error': 'Invalid path'}), 400
    if os.path.isdir(project_dir):
        shutil.rmtree(project_dir)
        return jsonify({'deleted': project_dir})
    return jsonify({'error': 'Not found'}), 404

if __name__ == '__main__':
    print(f'\n✅  Open http://localhost:{args.port} in your browser')
    print(f'✅  Ausstellungs-App:  http://localhost:{args.port}/ausstellung')
    print(f'✅  3D Gallery:        http://localhost:{args.port}/gallery3d\n')
    app.run(port=args.port, debug=False, threaded=True)
