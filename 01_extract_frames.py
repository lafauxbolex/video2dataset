#!/usr/bin/env python3
"""
01_extract_frames.py

Step 1 of the video2lora pipeline: extract frames from one or more videos.
Now supports TWO strategies:
- Fixed FPS (previous behavior)
- Evenly spaced frame COUNT across a clip (new: great to avoid near-duplicates)

Examples:
  python 01_extract_frames.py video.mp4 --count 3 --out outputs/frames_raw   # begin/middle/end
  python 01_extract_frames.py /folder --count 12                              # 12 evenly spaced frames each
  python 01_extract_frames.py video.mp4 --fps 2                               # fixed-FPS (legacy)
  python 01_extract_frames.py video.mp4 --count 5 --start 00:00:10 --duration 30

Notes:
- Requires ffmpeg + ffprobe on PATH.
- Cropping/resizing happens later in Step 3.
"""

from __future__ import annotations
import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mts", ".m2ts", ".mpg", ".mpeg", ".wmv"}

@dataclass
class VideoJob:
    input_path: Path
    out_dir: Path
    # strategy A: fixed fps
    fps: Optional[float] = None
    max_frames: Optional[int] = None
    # strategy B: evenly spaced count
    count: Optional[int] = None
    # time window control
    start: Optional[str] = None
    duration: Optional[str] = None
    # output format
    quality: int = 2
    ext: str = "jpg"

def find_exe(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise FileNotFoundError(
            f"{name} binary not found on PATH.\n"
            "macOS:  brew install ffmpeg\n"
            "Ubuntu: sudo apt update && sudo apt install -y ffmpeg\n"
            "Windows (PowerShell): choco install ffmpeg   or   scoop install ffmpeg"
        )
    return exe

def list_videos(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return [p for p in sorted(input_path.rglob("*")) if p.is_file() and p.suffix.lower() in VIDEO_EXTS]

def slugify(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "-", name)
    name = re.sub(r"-+", "-", name)
    return name.strip("-_") or "video"

def uniquify_dir(base: Path, name: str) -> Path:
    candidate = base / name
    i = 2
    while candidate.exists():
        candidate = base / f"{name}_{i}"
        i += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate

def ffprobe_duration(ffprobe_exe: str, video_path: Path) -> float:
    """Return duration in seconds (float)."""
    cmd = [ffprobe_exe, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)

def to_seconds(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    # Accept seconds or hh:mm:ss(.ms)
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return float(s)
    parts = s.split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        h, m, sec = parts
        return h * 3600 + m * 60 + sec
    if len(parts) == 2:
        m, sec = parts
        return m * 60 + sec
    raise ValueError(f"Invalid time format: {s}")

def build_ffmpeg_fps_cmd(ffmpeg_exe: str, job: VideoJob, pattern: str) -> list:
    cmd = [ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-stats", "-y", "-i", str(job.input_path)]
    if job.start:
        cmd += ["-ss", str(job.start)]
    if job.duration:
        cmd += ["-t", str(job.duration)]
    vf = f"fps=fps={job.fps}"
    cmd += ["-vf", vf]
    out_args = []
    if job.ext == "jpg":
        out_args += ["-q:v", str(job.quality)]
    if job.max_frames is not None:
        out_args += ["-frames:v", str(job.max_frames)]
    cmd += out_args + [pattern]
    return cmd

def extract_fps(ffmpeg_exe: str, job: VideoJob, out_dir: Path) -> int:
    pattern = str(out_dir / ("%06d." + job.ext))
    cmd = build_ffmpeg_fps_cmd(ffmpeg_exe, job, pattern)
    print("  → Command:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)
    return len(list(out_dir.glob(f"*.{job.ext}")))

def extract_count(ffmpeg_exe: str, ffprobe_exe: str, job: VideoJob, out_dir: Path) -> int:
    # Compute absolute time window [t0, t1]
    full_dur = ffprobe_duration(ffprobe_exe, job.input_path)
    t0 = to_seconds(job.start) or 0.0
    if t0 < 0: t0 = 0.0
    if job.duration is not None:
        span = to_seconds(job.duration) or 0.0
        t1 = min(t0 + span, full_dur)
    else:
        t1 = full_dur
    if t1 <= t0:
        raise SystemExit(f"Empty time window for {job.input_path} (t0={t0}, t1={t1})")

    n = int(job.count or 0)
    if n <= 0:
        raise SystemExit("--count must be a positive integer")

    # Generate evenly-spaced timestamps. Include endpoints when n>=2; if n==1, pick midpoint.
    # safety margin so we never request the very last instant
    SAFETY = 0.05  # seconds

    # generate midpoints of N equal segments in [t0, t1 - SAFETY]
    span = max(0.0, (t1 - SAFETY) - t0)
    if n == 1:
        ts = [t0 + span / 2.0]
    else:
        ts = [t0 + ((i + 0.5) / n) * span for i in range(n)]

    # final clamp just in case
    ts = [min(t, t1 - SAFETY) for t in ts]

    # Extract one frame per timestamp (accurate seek, single frame)
    count = 0
    for i, t in enumerate(ts, start=1):
        hh = int(t // 3600)
        mm = int((t % 3600) // 60)
        ss = t % 60
        t_str = f"{hh:02d}:{mm:02d}:{ss:06.3f}"
        out_file = out_dir / f"{i:06d}.{job.ext}"
        cmd = [
            ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", t_str, "-i", str(job.input_path),
            "-frames:v", "1",
        ]
        if job.ext == "jpg":
            cmd += ["-q:v", str(job.quality)]
        cmd += [str(out_file)]
        print("  → Frame", i, "@", t_str)
        subprocess.run(cmd, check=True)
        count += 1
    return count

def run_job(ffmpeg_exe: str, ffprobe_exe: str, job: VideoJob) -> dict:
    slug = slugify(job.input_path.stem)
    out_parent = job.out_dir
    out_parent.mkdir(parents=True, exist_ok=True)
    out_dir = uniquify_dir(out_parent, slug)

    print("\n— Extracting:", job.input_path)
    print("  → Output:", out_dir)

    if job.count is not None:
        n = extract_count(ffmpeg_exe, ffprobe_exe, job, out_dir)
        strategy = {"mode": "count", "count": job.count}
    else:
        n = extract_fps(ffmpeg_exe, job, out_dir)
        strategy = {"mode": "fps", "fps": job.fps, "max_frames": job.max_frames}

    frames = sorted(out_dir.glob(f"*.{job.ext}"))
    manifest = {
        "input": str(job.input_path.resolve()),
        "output_dir": str(out_dir.resolve()),
        "strategy": strategy,
        "start": job.start,
        "duration": job.duration,
        "ext": job.ext,
        "quality": job.quality if job.ext == "jpg" else None,
        "num_frames": n,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "frames": [f.name for f in frames],
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"  ✓ Extracted {n} frame(s). Manifest written to {out_dir/'manifest.json'}")
    return manifest

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract frames from video(s) using either fixed FPS or evenly spaced COUNT.")
    p.add_argument("input", type=str, help="Path to a video file or a directory of videos")
    p.add_argument("--out", type=str, default="outputs/frames_raw", help="Root output directory")

    # Mutually exclusive strategy
    group = p.add_mutually_exclusive_group()
    group.add_argument("--fps", type=float, help="Frames per second to extract (e.g., 0.5, 1, 2)")
    group.add_argument("--count", type=int, help="Evenly spaced number of frames across the time window")

    p.add_argument("--start", type=str, default=None, help="Optional start (e.g., 12 or 00:00:12.5)")
    p.add_argument("--duration", type=str, default=None, help="Optional duration (e.g., 30 or 00:00:30)")
    p.add_argument("--max-frames", type=int, default=None, help="Optional hard cap (FPS mode only)")
    p.add_argument("--quality", type=int, default=2, help="JPEG quality (2–31, lower is better; ignored for PNG)")
    p.add_argument("--ext", type=str, choices=["jpg", "png"], default="jpg", help="Output image format")
    return p.parse_args()

def main():
    args = parse_args()
    ffmpeg_exe = find_exe("ffmpeg")
    ffprobe_exe = find_exe("ffprobe")

    in_path = Path(args.input)
    videos = list_videos(in_path)
    if not videos:
        raise SystemExit(f"No video files found in: {in_path}")

    out_root = Path(args.out)

    for v in videos:
        job = VideoJob(
            input_path=v,
            out_dir=out_root,
            fps=args.fps,
            max_frames=args.max_frames,
            count=args.count,
            start=args.start,
            duration=args.duration,
            quality=args.quality,
            ext=args.ext,
        )
        run_job(ffmpeg_exe, ffprobe_exe, job)

if __name__ == "__main__":
    main()
