#!/usr/bin/env python3
"""
00_run_pipeline.py

All-in-one runner: given a source video *file or folder*, execute Steps 1–5 and
produce a flat AI-Toolkit-friendly dataset (images/ + metadata.jsonl).

Minimal usage (defaults chosen from our earlier tuning):
  python 00_run_pipeline.py --src /path/to/video_or_folder --out outputs

This will run:
  01_extract_frames.py  (evenly spaced count = 3)
  02_track_subject.py   (YOLO, prefer person)
  03_square_crop.py     (size=1024, pad=0.20, y_bias=0.35, headroom=0.10)
  04_caption.py         (BLIP, safe decoding defaults)
  05_dedup_package.py   (flat layout, sim-thresh=0.985, blur-thresh=0)

Advanced knobs if you want to tweak without going step-by-step:
  --count N         # frames per video (evenly spaced) for Step 1 (default 3)
  --crop-size PX    # final square size (default 1024)
  --pad FRAC        # padding around subject per side (default 0.20)
  --y-bias FRAC     # 0..1, shift crop center upward (default 0.35)
  --headroom FRAC   # extra space above bbox top as fraction of square (default 0.10)
  --sim-thresh V    # CLIP duplicate threshold (default 0.985)
  --blur-thresh V   # Laplacian variance cutoff; 0 disables (default 0)
  --name-template   # e.g. "{seq}_{idx}{ext}" (default)

Assumptions:
  - You are running this from the repo root where 01..05 scripts live.
  - Required Python deps installed (see README/previous steps).
  - ffmpeg/ffprobe available on PATH.
"""

from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def which_or_die(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise SystemExit(f"Required binary not found on PATH: {name}")
    return exe


def run(cmd: list[str]):
    print("\n$", " ".join(str(c) for c in cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Command failed with exit code {e.returncode}: {' '.join(str(c) for c in cmd)}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the full video2lora pipeline end-to-end.")
    ap.add_argument("--src", required=True, help="Source video file or folder of videos")
    ap.add_argument("--out", default="outputs", help="Root output folder (frames, tracks, crops, dataset live under here)")

    # Common quick knobs
    ap.add_argument("--count", type=int, default=3, help="Evenly spaced frames per video (Step 1)")
    ap.add_argument("--crop-size", type=int, default=1024, help="Final square size (Step 3)")
    ap.add_argument("--pad", type=float, default=0.20, help="Padding per side as fraction of subject size (Step 3)")
    ap.add_argument("--y-bias", type=float, default=0.35, help="0..1 upward bias toward head (Step 3)")
    ap.add_argument("--headroom", type=float, default=0.10, help="Extra space above bbox top (Step 3)")
    ap.add_argument("--sim-thresh", type=float, default=0.985, help="CLIP cosine similarity threshold for dedup (Step 5)")
    ap.add_argument("--blur-thresh", type=float, default=0.0, help="Laplacian variance threshold; 0 disables (Step 5)")
    ap.add_argument("--name-template", type=str, default="{seq}_{idx}{ext}", help="Filename template for final images (Step 5)")

    # Model/options shortcuts
    ap.add_argument("--yolo-model", default="yolov8n.pt", help="Ultralytics model for detection (Step 2)")

    return ap.parse_args()


def main():
    args = parse_args()

    # sanity checks
    which_or_die("ffmpeg"); which_or_die("ffprobe")

    repo = Path(__file__).resolve().parent
    src = Path(args.src)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    frames_root = out_root / "frames_raw"
    tracks_root = out_root / "tracks"
    crops_root  = out_root / "crops"
    dataset_root = out_root / "dataset"

    # Step 1: extract frames (evenly spaced count)
    run([
        sys.executable, str(repo / "01_extract_frames.py"), str(src),
        "--out", str(frames_root),
        "--count", str(args.count),
    ])

    # Step 2: detect + track (prefer person)
    # If user passed a single sequence, --in-root still works (it will find child manifests).
    run([
        sys.executable, str(repo / "02_track_subject.py"),
        "--in-root", str(frames_root),
        "--prefer-person",
        "--model", args.yolo_model,
        "--out", str(tracks_root),
    ])

    # Step 3: square crops with head bias
    run([
        sys.executable, str(repo / "03_square_crop.py"),
        "--in-root", str(frames_root),
        "--boxes-root", str(tracks_root),
        "--out", str(crops_root),
        "--size", str(args.crop_size),
        "--pad", str(args.pad),
        "--y-bias", str(args.y_bias),
        "--headroom", str(args.headroom),
    ])

    # Step 4: captions (BLIP defaults baked into the script)
    run([
        sys.executable, str(repo / "04_caption.py"),
        "--in-root", str(crops_root),
        "--model", "blip2",
        "--lower", "--strip-period",
    ])

    # Step 5: dedup + package (flat, single metadata.jsonl)
    run([
        sys.executable, str(repo / "05_dedup_package.py"),
        "--in-root", str(crops_root),
        "--out", str(dataset_root),
        "--flat",
        "--name-template", args.name_template,
        "--sim-thresh", str(args.sim_thresh),
        "--blur-thresh", str(args.blur_thresh),
    ])

    print("\n✅ All done! Final dataset:")
    print("  ", dataset_root)
    print("   ├─ images/")
    print("   └─ metadata.jsonl")


if __name__ == "__main__":
    main()
