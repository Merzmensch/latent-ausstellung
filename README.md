# Latent Ausstellung

> ### 🧳 Bring your own checkpoint. / Bringen Sie Ihren eigenen Checkpoint mit.
> This installation is **model-agnostic** — it works with any StyleGAN2-ADA-PyTorch compatible `.pkl` file.  
> Diese Installation ist **modell-agnostisch** — sie funktioniert mit jeder StyleGAN2-ADA-PyTorch-kompatiblen `.pkl`-Datei.  
> The software is open. The vision is yours. / Die Software ist offen. Die Vision gehört Ihnen.

---

*[English version below / Deutsche Version unten](#german)*

---

## English

**An interactive exhibition installation for exploring the latent space of a StyleGAN2 neural network.**

The visitor navigates a seemingly infinite map of AI-generated images — each tile a unique point in the model's learned visual space. The map is explorable by mouse drag or trackball. After 30 seconds of inactivity, the installation enters **Autopilot mode** and drifts through latent space on its own — ideal for unmanned display.

The installation was designed to be **portable across models and exhibitions**: swap the `.pkl` checkpoint and the entire visual world changes. A model trained on faces produces a portrait landscape. A model trained on architecture produces an infinite city. A model trained on your own dataset produces your own latent universe.

> Developed by [Merzmensch](https://github.com/merzmensch) (Vladimir Alexeev) as part of the *Latent MERZpoet* project.

---

### What You Need to Bring

| What | Details |
|---|---|
| **Your `.pkl` checkpoint** | Any StyleGAN2-ADA-PyTorch model, any resolution, any subject |
| **A Windows PC with NVIDIA GPU** | See system requirements below |
| **~30 minutes for setup** | First-time installation only |

Place your `.pkl` file in the `models/` subfolder and adjust the path in `start_ausstellung.bat`. That's it.

> **Don't have a checkpoint yet?**  
> Pre-trained models are available at [NVIDIA's model repository](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan2) and [community collections on Hugging Face](https://huggingface.co/models?search=stylegan2).

---

### System Requirements

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

### Step-by-Step Installation

#### Step 1 — Install Miniconda

Download and install **Miniconda** (Python package manager):  
👉 https://docs.conda.io/en/latest/miniconda.html

During installation:
- Choose **"Just Me"** (no admin required)
- Check **"Add Miniconda3 to PATH"** *(or use "Anaconda Prompt" later)*

---

#### Step 2 — Install CUDA Toolkit

Check which CUDA version your GPU supports:

1. Open **NVIDIA Control Panel** → Help → System Information → Components  
   Look for the line `NVCUDA.DLL` — it shows your max supported CUDA version.

2. Download the matching toolkit:
   - CUDA 12.1: https://developer.nvidia.com/cuda-12-1-0-download-archive
   - CUDA 12.4: https://developer.nvidia.com/cuda-12-4-0-download-archive

Install with default settings.

---

#### Step 3 — Create the Conda Environment

Open **Anaconda Prompt** (from Start Menu) and run these commands one by one:

```bash
conda create -n stylegan python=3.9 -y
conda activate stylegan
```

---

#### Step 4 — Install PyTorch with CUDA

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

#### Step 5 — Install Python Dependencies

```bash
pip install flask flask-cors numpy Pillow requests
```

---

#### Step 6 — Get the StyleGAN2 Code

The server requires the official **StyleGAN2-ADA-PyTorch** repository from NVIDIA.

```bash
cd C:\Users\YourName\
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
```

> If you do not have **Git** installed: https://git-scm.com/download/win  
> During installation, accept all defaults.

The cloned folder must be placed **inside** the project folder (see Step 7).

---

#### Step 7 — Set Up the Project Folder

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
    └── your_model.pkl               ← your checkpoint file
```

---

#### Step 8 — Adapt `start_ausstellung.bat`

Open `start_ausstellung.bat` in a text editor (right-click → Edit) and update the paths:

```bat
@echo off
title Latent Ausstellung

call C:\Users\YourName\miniconda3\Scripts\activate.bat stylegan

cd /d C:\Users\YourName\stylegan2-explorer

start "" http://localhost:5000/ausstellung

python server.py --pkl models\your_model.pkl
```

Replace `YourName` and `your_model.pkl` with your actual values.

---

#### Step 9 — Launch the Installation

Double-click `start_ausstellung.bat`.

Wait until you see:
```
[server] ✅ z_dim=512  resolution=1024
✅  Ausstellungs-App:  http://localhost:5000/ausstellung
```

The browser opens automatically. If not: `http://localhost:5000/ausstellung`

> **First launch** may take 1–2 minutes while PyTorch compiles CUDA kernels. This is normal.

---

### Controls

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

After **30 seconds** without interaction, a countdown ring appears. After 10 more seconds, Autopilot activates and the map drifts slowly through latent space. Any mouse movement, scroll, or keypress cancels it immediately.

### Interpolation Modes

- **Hash** — every tile fully independent, maximum variety
- **Slerp** — smooth spherical interpolation outward from center
- **Bilinear** — four corner anchors, gradual transitions across the map

### Truncation ψ (psi)

- `ψ = 1.0` — maximum diversity
- `ψ = 0.7` *(default)* — balanced quality and variety
- `ψ = 0.4` — closer to training average, cleaner but more uniform

---

### Troubleshooting

**`torch.cuda.is_available()` returns `False`**  
→ Update GPU driver: https://www.nvidia.com/drivers  
→ Make sure PyTorch CUDA version matches your driver.

**Server starts but tiles don't load**  
→ Check terminal for errors.  
→ Try: `python server.py --pkl models\your_model.pkl --res 64`

**`ModuleNotFoundError: No module named 'legacy'`**  
→ The `stylegan2-ada-pytorch/` folder is missing or in the wrong place. Check Step 7.

**Very slow tile generation**  
→ CUDA is not active — running on CPU. Re-check Step 4.

**Browser does not open automatically**  
→ Navigate manually to `http://localhost:5000/ausstellung`

---

### Credits

- **Artist / Developer:** [Merzmensch](https://github.com/merzmensch) (Vladimir Alexeev)
- **StyleGAN2-ADA-PyTorch:** [NVIDIA Research](https://github.com/NVlabs/stylegan2-ada-pytorch)

---

### License

Code: **MIT License**.  
Model weights (`.pkl`) are not included and subject to separate terms.  
StyleGAN2-ADA-PyTorch: [NVIDIA Source Code License](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/LICENSE.txt).

---
---

<a name="german"></a>

## Deutsch

**Eine interaktive Ausstellungsinstallation zur Erkundung des latenten Raums eines StyleGAN2-Neuronalen Netzes.**

Der Besucher navigiert durch eine scheinbar unendliche Karte KI-generierter Bilder — jedes Kachel ein einzigartiger Punkt im gelernten visuellen Raum des Modells. Die Karte lässt sich per Maus-Drag oder Trackball erkunden. Nach 30 Sekunden Inaktivität aktiviert sich der **Autopilot-Modus** und die Karte driftet selbstständig durch den latenten Raum — ideal für unbemannte Ausstellungsdisplays.

Die Installation wurde so konzipiert, dass sie **modell- und ausstellungsübergreifend portierbar** ist: Den `.pkl`-Checkpoint austauschen, und die gesamte visuelle Welt der Installation verändert sich. Ein Modell, das auf Gesichter trainiert wurde, erzeugt eine Porträtlandschaft. Ein Modell, das auf Architektur trainiert wurde, erzeugt eine unendliche Stadt. Ein Modell, das auf Ihrem eigenen Datensatz trainiert wurde, erzeugt Ihr eigenes latentes Universum.

> Entwickelt von [Merzmensch](https://github.com/merzmensch) (Vladimir Alexeev) als Teil des *Latent MERZpoet*-Projekts.

---

### Was Sie mitbringen müssen

| Was | Details |
|---|---|
| **Ihren `.pkl`-Checkpoint** | Beliebiges StyleGAN2-ADA-PyTorch-Modell, beliebige Auflösung, beliebiges Sujet |
| **Einen Windows-PC mit NVIDIA-GPU** | Siehe Systemanforderungen unten |
| **~30 Minuten für die Einrichtung** | Nur beim ersten Mal |

Legen Sie Ihre `.pkl`-Datei in den Unterordner `models/` und passen Sie den Pfad in `start_ausstellung.bat` an. Das war's.

> **Noch keinen Checkpoint?**  
> Vortrainierte Modelle finden Sie im [NVIDIA-Modell-Repository](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan2) und in [Community-Sammlungen auf Hugging Face](https://huggingface.co/models?search=stylegan2).

---

### Systemanforderungen

| Komponente | Minimum | Empfohlen |
|---|---|---|
| Betriebssystem | Windows 10 64-bit | Windows 10/11 64-bit |
| GPU | NVIDIA GTX 1060 6 GB | NVIDIA RTX 3060 oder besser |
| VRAM | 6 GB | 12 GB |
| RAM | 16 GB | 32 GB |
| CUDA | 11.8 | 12.1 oder 12.4 |
| Python | 3.9 | 3.9 (erforderlich — StyleGAN2 ist nicht kompatibel mit 3.10+) |
| Speicherplatz | 5 GB frei | 10 GB frei |

> **Nur-CPU-Modus** ist möglich, aber sehr langsam (mehrere Sekunden pro Kachel). Für Live-Ausstellungen nicht empfohlen.

---

### Schritt-für-Schritt-Installation

#### Schritt 1 — Miniconda installieren

**Miniconda** (Python-Paketverwaltung) herunterladen und installieren:  
👉 https://docs.conda.io/en/latest/miniconda.html

Während der Installation:
- **„Just Me"** wählen (keine Adminrechte erforderlich)
- **„Add Miniconda3 to PATH"** aktivieren *(oder später „Anaconda Prompt" verwenden)*

---

#### Schritt 2 — CUDA Toolkit installieren

Prüfen, welche CUDA-Version Ihre GPU unterstützt:

1. **NVIDIA-Systemsteuerung** öffnen → Hilfe → Systeminformationen → Komponenten  
   Zeile `NVCUDA.DLL` suchen — sie zeigt die maximal unterstützte CUDA-Version.

2. Passendes Toolkit herunterladen:
   - CUDA 12.1: https://developer.nvidia.com/cuda-12-1-0-download-archive
   - CUDA 12.4: https://developer.nvidia.com/cuda-12-4-0-download-archive

Mit Standardeinstellungen installieren.

---

#### Schritt 3 — Conda-Umgebung erstellen

**Anaconda Prompt** öffnen (im Startmenü) und folgende Befehle nacheinander ausführen:

```bash
conda create -n stylegan python=3.9 -y
conda activate stylegan
```

---

#### Schritt 4 — PyTorch mit CUDA installieren

Weiterhin im Anaconda Prompt (Umgebung `stylegan` muss aktiv sein):

**Für CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Für CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Installation überprüfen:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Ausgabe sollte `True` sein. Bei `False` → siehe Fehlerbehebung weiter unten.

---

#### Schritt 5 — Python-Abhängigkeiten installieren

```bash
pip install flask flask-cors numpy Pillow requests
```

---

#### Schritt 6 — StyleGAN2-Code herunterladen

Der Server benötigt das offizielle **StyleGAN2-ADA-PyTorch**-Repository von NVIDIA:

```bash
cd C:\Users\IhrName\
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
```

> Falls **Git** nicht installiert ist: https://git-scm.com/download/win  
> Alle Standardeinstellungen bestätigen.

Der geklonte Ordner muss **innerhalb** des Projektordners liegen (siehe Schritt 7).

---

#### Schritt 7 — Projektordner einrichten

Einen Ordner für die Installation erstellen, zum Beispiel:
```
C:\Users\IhrName\stylegan2-explorer\
```

Folgende Dateien darin ablegen:

```
stylegan2-explorer/
├── server.py                        ← aus diesem Repository
├── latent_ausstellung.html          ← aus diesem Repository
├── start_ausstellung.bat            ← aus diesem Repository
├── stylegan2-ada-pytorch/           ← in Schritt 6 geklont
│   ├── legacy.py
│   ├── dnnlib/
│   └── ... (weitere NVIDIA-Dateien)
└── models/
    └── ihr_modell.pkl               ← Ihr Checkpoint
```

---

#### Schritt 8 — `start_ausstellung.bat` anpassen

`start_ausstellung.bat` im Texteditor öffnen (Rechtsklick → Bearbeiten) und Pfade anpassen:

```bat
@echo off
title Latent Ausstellung

call C:\Users\IhrName\miniconda3\Scripts\activate.bat stylegan

cd /d C:\Users\IhrName\stylegan2-explorer

start "" http://localhost:5000/ausstellung

python server.py --pkl models\ihr_modell.pkl
```

`IhrName` und `ihr_modell.pkl` durch Ihre tatsächlichen Werte ersetzen.

---

#### Schritt 9 — Installation starten

`start_ausstellung.bat` doppelklicken.

Ein Terminalfenster öffnet sich. Warten bis folgendes erscheint:
```
[server] ✅ z_dim=512  resolution=1024
✅  Ausstellungs-App:  http://localhost:5000/ausstellung
```

Der Browser öffnet sich automatisch. Falls nicht: `http://localhost:5000/ausstellung`

> **Beim ersten Start** kann es 1–2 Minuten dauern, während PyTorch CUDA-Kernels kompiliert. Das ist normal.  
> Folgestarts sind deutlich schneller.

---

### Bedienung

| Aktion | Eingabe |
|---|---|
| Karte verschieben | Maus-Drag *oder* Trackball |
| Trackball / Drag-Modus wechseln | Taste `T` |
| Zoom | Mausrad |
| Kachel auswählen | Klick |
| Zufälliger Sprung | Taste `R` oder Button |
| Ansicht zurücksetzen | Taste `0` oder Button |
| Autopilot ein/aus | Taste `P` |
| Bild speichern | Button „Save PNG" |
| Truncation ψ einstellen | Schieberegler (rechtes Panel) |
| Schrittweite einstellen | Schieberegler (rechtes Panel) |

### Autopilot-Modus

Nach **30 Sekunden** ohne Interaktion erscheint ein Countdown-Ring. Nach weiteren 10 Sekunden aktiviert sich der Autopilot und die Karte driftet langsam durch den latenten Raum. Jede Mausbewegung, jedes Scrollen oder Tastendruck deaktiviert ihn sofort.

### Interpolationsmodi

- **Hash** — jede Kachel vollständig unabhängig, maximale Vielfalt
- **Slerp** — sphärische Interpolation vom Zentrum nach außen
- **Bilinear** — vier Eck-Anker, sanfte Übergänge über die gesamte Karte

### Truncation ψ (Psi)

- `ψ = 1,0` — maximale Diversität
- `ψ = 0,7` *(Standard)* — ausgewogene Qualität und Vielfalt
- `ψ = 0,4` — näher am Trainings-Durchschnitt, einheitlicher aber cleaner

---

### Fehlerbehebung

**`torch.cuda.is_available()` gibt `False` zurück**  
→ GPU-Treiber aktualisieren: https://www.nvidia.com/drivers  
→ Sicherstellen, dass PyTorch-CUDA-Version zum Treiber passt.

**Server startet, aber Kacheln laden nicht**  
→ Terminal auf Fehlermeldungen prüfen.  
→ Versuch mit kleinerer Auflösung: `python server.py --pkl models\ihr_modell.pkl --res 64`

**`ModuleNotFoundError: No module named 'legacy'`**  
→ Der Ordner `stylegan2-ada-pytorch/` fehlt oder liegt am falschen Ort. Schritt 7 prüfen.

**Kacheln werden sehr langsam generiert**  
→ CUDA ist nicht aktiv — läuft auf CPU. Schritt 4 wiederholen.

**Browser öffnet sich nicht automatisch**  
→ Manuell aufrufen: `http://localhost:5000/ausstellung`

---

### Credits

- **Künstler / Entwickler:** [Merzmensch](https://github.com/merzmensch) (Vladimir Alexeev)
- **StyleGAN2-ADA-PyTorch:** [NVIDIA Research](https://github.com/NVlabs/stylegan2-ada-pytorch)

---

### Lizenz

Code: **MIT-Lizenz**.  
Modellgewichte (`.pkl`) sind nicht enthalten und unterliegen gesonderten Bedingungen.  
StyleGAN2-ADA-PyTorch: [NVIDIA Source Code License](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/LICENSE.txt).
