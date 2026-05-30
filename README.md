# Latent Ausstellung

**An interactive installation for exploring the latent space of a StyleGAN2 neural network.**

The visitor navigates a seemingly infinite map of AI-generated images — each tile a unique point in the model's learned visual space. The map is explorable by mouse drag or trackball. After 30 seconds of inactivity, the installation enters **Autopilot mode** and drifts through latent space on its own.

> **Model:** `mem.pkl` — a StyleGAN2 model trained on memory-related imagery.  
> Developed by [Merzmensch](https://github.com/merzmensch) (Vladimir Alexeev) as part of the *Latent MERZpoet* project.

---

## Demo

| Infinite tile map | Selected tile preview |
|---|---|
| Pan by dragging or trackball | Click any tile to see full preview |

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 64-bit | Windows 10/11 64-bit |
| GPU | NVIDIA GTX 1060 6 GB | NVIDIA RTX 3060 or better |
| VRAM | 6 GB | 12 GB |
| RAM | 16 GB | 32 GB |
| CUDA | 11.8 | 12.1 or 12.4 |
| Python | 3.9 | 3.9 (required — StyleGAN2 is not compatible with 3.10+) |
| Storage | 5 GB free | 10 GB free |

> **CPU-only mode** is possible but very slow (several seconds per tile). Not recommended for live exhibition.

---

## Step-by-Step Installation

### Step 1 — Install Miniconda

Download and install **Miniconda** (Python package manager):  
👉 https://docs.conda.io/en/latest/miniconda.html

During installation:
- Choose **"Just Me"** (no admin required)
- Check **"Add Miniconda3 to PATH"** *(or use "Anaconda Prompt" later)*

---

### Step 2 — Install CUDA Toolkit

Check which CUDA version your GPU supports:

1. Open **NVIDIA Control Panel** → Help → System Information → Components  
   Look for the line `NVCUDA.DLL` — it shows your max supported CUDA version.

2. Download the matching toolkit:
   - CUDA 12.1: https://developer.nvidia.com/cuda-12-1-0-download-archive
   - CUDA 12.4: https://developer.nvidia.com/cuda-12-4-0-download-archive

Install with default settings.

---

### Step 3 — Create the Conda Environment

Open **Anaconda Prompt** (from Start Menu) and run these commands one by one:

```bash
conda create -n stylegan python=3.9 -y
conda activate stylegan
```

---

### Step 4 — Install PyTorch with CUDA

Still in the Anaconda Prompt (environment `stylegan` must be active):

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify the installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
This should print `True`. If it prints `False`, your CUDA installation may have an issue — see Troubleshooting below.

---

### Step 5 — Install Python Dependencies

```bash
pip install flask flask-cors numpy Pillow requests
```

---

### Step 6 — Get the StyleGAN2 Code

The server requires the official **StyleGAN2-ADA-PyTorch** repository from NVIDIA.

```bash
cd C:\Users\YourName\
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
```

> If you do not have **Git** installed: https://git-scm.com/download/win  
> During installation, accept all defaults.

The cloned folder must be placed **inside** the `stylegan2-explorer` project folder (see Step 7).

---

### Step 7 — Set Up the Project Folder

Create a folder for the installation, for example:
```
C:\Users\YourName\stylegan2-explorer\
```

Place the following files inside it:

```
stylegan2-explorer/
├── server.py                        ← from this repository
├── latent_ausstellung.html          ← from this repository
├── start_ausstellung.bat            ← from this repository
├── stylegan2-ada-pytorch/           ← cloned in Step 6
│   ├── legacy.py
│   ├── dnnlib/
│   └── ... (other NVIDIA files)
└── models/
    └── mem.pkl                      ← model file (provided separately)
```

> **Important:** `mem.pkl` is distributed separately by the artist. Place it in the `models/` subfolder.

---

### Step 8 — Adapt `start_ausstellung.bat`

Open `start_ausstellung.bat` in a text editor (right-click → Edit) and update the paths to match your system:

```bat
@echo off
title Latent Ausstellung

call C:\Users\YourName\miniconda3\Scripts\activate.bat stylegan

cd /d C:\Users\YourName\stylegan2-explorer

start "" http://localhost:5000/ausstellung

python server.py --pkl models\mem.pkl
```

Replace `YourName` with your actual Windows username.

---

### Step 9 — Launch the Installation

Double-click `start_ausstellung.bat`.

A terminal window will open. Wait until you see:
```
[server] ✅ z_dim=512  resolution=1024
✅  Open http://localhost:5000 in your browser
✅  Ausstellungs-App:  http://localhost:5000/ausstellung
```

The browser will open automatically. If not, navigate to:
```
http://localhost:5000/ausstellung
```

> **First launch** may take 1–2 minutes while PyTorch compiles CUDA kernels. This is normal.  
> Subsequent launches are faster.

---

## Controls

| Action | Input |
|---|---|
| Pan the map | Mouse drag *or* trackball |
| Toggle trackball / drag mode | Key `T` |
| Zoom | Mouse wheel |
| Select a tile | Click |
| Random jump | Key `R` or button |
| Reset view | Key `0` or button |
| Toggle Autopilot | Key `P` |
| Save selected image | "Save PNG" button |
| Adjust Truncation ψ | Slider (right panel) |
| Adjust Step Size | Slider (right panel) |

### Autopilot Mode

If no interaction occurs for **30 seconds**, a countdown ring appears. After 10 more seconds, Autopilot activates and the map drifts slowly through latent space — ideal for unmanned exhibition display.

Any mouse movement, scroll, or keypress cancels Autopilot immediately.

### Interpolation Modes

Three modes for how neighboring tiles relate to each other:

- **Hash** — every tile fully independent, maximum variety
- **Slerp** — smooth spherical interpolation outward from center
- **Bilinear** — four corner anchors, gradual transitions across the whole map

### Truncation ψ (psi)

Controls how "typical" the generated images are:
- `ψ = 1.0` — maximum diversity, may include unusual/noisy images
- `ψ = 0.7` *(default)* — balanced quality and variety
- `ψ = 0.4` — images closer to the training average, cleaner but more similar

---

## Troubleshooting

**`torch.cuda.is_available()` returns `False`**  
→ Check that your GPU driver is up to date (https://www.nvidia.com/drivers).  
→ Make sure the CUDA version of the PyTorch install matches your driver.

**Server starts but tiles don't load**  
→ Check the terminal for error messages.  
→ Try a smaller `--res` value: `python server.py --pkl models\mem.pkl --res 64`

**CUDA kernel compilation warnings on startup**  
→ These are suppressed in the server. If you still see them, they are harmless — StyleGAN2 falls back to CPU-compiled kernels automatically.

**`ModuleNotFoundError: No module named 'legacy'`**  
→ The `stylegan2-ada-pytorch/` folder is not in the right place. Check Step 7.

**Very slow tile generation (several seconds per tile)**  
→ CUDA is not active — running on CPU. Check PyTorch CUDA installation (Step 4).

**Browser does not open automatically**  
→ Manually navigate to `http://localhost:5000/ausstellung`

---

## Architecture

```
latent_ausstellung.html   ← Single-file frontend (HTML/CSS/JS, no build step)
server.py                 ← Flask backend, StyleGAN2 inference, tile cache
start_ausstellung.bat     ← Windows launcher (activates conda env, starts server)
models/
  mem.pkl                 ← StyleGAN2 model weights (provided separately)
stylegan2-ada-pytorch/    ← NVIDIA's official StyleGAN2-ADA-PyTorch code
```

The frontend requests tiles via `POST /infinite/tile` — the server computes the latent vector for each grid position, runs it through the Generator, and returns a JPEG thumbnail as base64. Tiles are cached server-side to avoid redundant GPU computation.

---

## API Endpoints Used by This Installation

| Endpoint | Method | Purpose |
|---|---|---|
| `/ausstellung` | GET | Serves the HTML frontend |
| `/status` | GET | Model status check (z_dim, resolution) |
| `/infinite/tile` | POST | Generate one tile image |
| `/render_z` | POST | Render a full-resolution image from a z vector |
| `/list_models` | GET | List available `.pkl` files |
| `/load_model` | POST | Load a different model at runtime |

---

## Credits

- **Artist / Developer:** [Merzmensch](https://github.com/merzmensch) (Vladimir Alexeev)
- **StyleGAN2-ADA-PyTorch:** [NVIDIA Research](https://github.com/NVlabs/stylegan2-ada-pytorch)
- **Model:** `mem.pkl` — trained by Merzmensch

---

## License

The code in this repository is released under the **MIT License**.

`mem.pkl` and any other model weights are **not** included and are subject to separate terms — contact the artist for exhibition licensing.

StyleGAN2-ADA-PyTorch is licensed under the [NVIDIA Source Code License](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/LICENSE.txt).
