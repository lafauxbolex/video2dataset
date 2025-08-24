# 🎬 Video2Dataset

**Turn raw videos into AI‑ready image datasets in one command.**

This repo takes you from videos ➜ evenly spaced frames ➜ subject tracking ➜ smart square crops ➜ auto captions ➜ de‑dup & packaging. The output is a flat `images/` folder + a single `metadata.jsonl` suitable for AI Toolkit, Diffusers, and Kohya training.

---

## ✨ Features
- **One‑liner pipeline** (`00_run_pipeline.py`) – sensible defaults, GPU‑optional
- **Evenly spaced frame sampling** – avoid near‑duplicates from long clips
- **Main‑subject tracking** (YOLO) – picks the most persistent subject (prefers `person`)
- **Smart square crops** – padding, upward bias, and headroom to keep faces/hair
- **Auto‑captioning** (BLIP) – caption sidecars/metadata for training
- **CLIP de‑dup** – remove near‑identical images; optional blur filter
- **Flat dataset layout** – **no filename collisions**, single `metadata.jsonl`

---

## 🧰 Requirements
- **Python** 3.9+
- **ffmpeg/ffprobe** on PATH
- **Pip packages** listed below (install via `requirements.txt` or `pip install ...`)

### Install ffmpeg
- **macOS (Homebrew)**
  ```bash
  brew install ffmpeg
  ```
- **Ubuntu/Debian**
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```
- **Windows (PowerShell)**
  ```powershell
  choco install ffmpeg   # or: scoop install ffmpeg
  ```
Verify:
```bash
ffmpeg -version
ffprobe -version
```

### Create & activate a virtual environment
> Do this in the repo folder.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Python deps
You can use either `requirements.txt` or the direct pip command.

**Option A — requirements.txt**
```txt
ultralytics
transformers
torch
torchvision
torchaudio
pillow
tqdm
opencv-python
```
Install:
```bash
pip install -r requirements.txt
```

**Option B — direct**
```bash
pip install ultralytics transformers torch torchvision torchaudio pillow tqdm opencv-python
```

> **GPU note:** Torch will install a CPU build by default. For NVIDIA CUDA builds, follow PyTorch’s official selector.

---

## 🚀 One‑line usage (recommended)
Run the full pipeline end‑to‑end with defaults tuned for portraits/people:

```bash
python 00_run_pipeline.py \
  --src /path/to/video_or_folder \
  --out outputs
```

**What happens:**
1) `01_extract_frames.py` – evenly spaced **3** frames per video
2) `02_track_subject.py`  – YOLO detect/track, prefer `person`
3) `03_square_crop.py`    – `size=1024`, `pad=0.15`, `y_bias=0.15`, `headroom=0.05`
4) `04_caption.py`        – BLIP captions (lowercased, no trailing period)
5) `05_dedup_package.py`  – CLIP de‑dup (`sim‑thresh=0.985`, `blur‑thresh=0`), **flat** dataset

**Final output:**
```
outputs/dataset/
  images/
    <unique_filenames>.jpg
  metadata.jsonl
```
Each line in `metadata.jsonl` looks like:
```json
{"file_name": "wan_00103_00000.jpg", "text": "portrait, soft light, high detail"}
```

### Common quick tweaks
```bash
# more/less frames per video
python 00_run_pipeline.py --src /path/to/src --out outputs --count 5

# smaller crops
python 00_run_pipeline.py --src /path/to/src --out outputs --crop-size 768

# tighter or looser framing
python 00_run_pipeline.py --src /path/to/src --out outputs --pad 0.12 --y-bias 0.20 --headroom 0.08

# stronger detector
python 00_run_pipeline.py --src /path/to/src --out outputs --yolo-model yolov8s.pt

# different final naming (still flat, collision‑safe)
python 00_run_pipeline.py --src /path/to/src --out outputs --name-template "{seq}_{stem}_{idx}{ext}"
```

---

## 🧪 Power‑user: run steps manually
If you want full control or to troubleshoot, each step is callable.

**1) Extract frames** (even spacing)
```bash
python 01_extract_frames.py <video_or_folder> --count 3 --out outputs/frames_raw
```

**2) Detect + track**
```bash
python 02_track_subject.py --in-root outputs/frames_raw --prefer-person --model yolov8n.pt --out outputs/tracks
```

**3) Square crops** (defaults tuned for portraits)
```bash
python 03_square_crop.py \
  --in-root outputs/frames_raw \
  --boxes-root outputs/tracks \
  --out outputs/crops \
  --size 1024 --pad 0.15 --y-bias 0.15 --headroom 0.05
```

**4) Caption** (BLIP, safe defaults baked in)
```bash
python 04_caption.py --in-root outputs/crops --model blip --lower --strip-period
```

**5) De‑dup + package** (flat dataset w/ single metadata.jsonl)
```bash
python 05_dedup_package.py \
  --in-root outputs/crops \
  --out outputs/dataset \
  --flat \
  --name-template "{seq}_{idx}{ext}" \
  --sim-thresh 0.985 --blur-thresh 0
```

---

## 🛠️ Troubleshooting
**ffmpeg / ffprobe not found**  
Install via Homebrew/apt/Chocolatey/Scoop (see above) and restart the shell.

**BLIP captioning error about min/max length**  
We use safe defaults (`minlen=1`, `maxlen=50`, beams=1, repetition=1.0). If you changed flags and error returns, revert to defaults.

**“All images filtered out by blur threshold; skipping”**  
Set `--blur-thresh 0` (disables). If you want some filtering, calibrate by printing Laplacian variance and pick a lower cutoff (e.g., 5–10).

**Cropping cuts off hairline**  
Increase `--y-bias` to shift crop upward and/or `--headroom` to add extra space above. For very tight verticals, reduce `--pad`.

**Near‑duplicate flood**  
Raise `--sim-thresh` (e.g., 0.99) to be more aggressive.

---

## 📄 License
MIT (or your preference). Add a LICENSE file before publishing.

## 🙌 Credits
- Ultralytics YOLO for detection
- Salesforce BLIP for captions
- OpenAI CLIP for embeddings

---

## 📦 Repo layout
```
00_run_pipeline.py        # one‑command runner
01_extract_frames.py      # even‑spacing frame extraction (ffmpeg)
02_track_subject.py       # YOLO detection + greedy IOU tracker
03_square_crop.py         # square crops with pad + y‑bias + headroom
04_caption.py             # BLIP captioning
05_dedup_package.py       # CLIP de‑dup + flat dataset packaging
```

---

## 💡 Tagline
**Video2Dataset — turn raw videos into AI‑ready datasets in one command.**

