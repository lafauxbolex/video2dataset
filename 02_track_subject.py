#!/usr/bin/env python3
"""
02_track_subject.py

Step 2 of the video2lora pipeline: detect and track the main subject across frames.

Input:
  - A single frames folder produced by step 01 (e.g., outputs/frames_raw/<video_slug>)
    or the parent root (e.g., outputs/frames_raw). If a root is passed, the script
    processes each immediate subfolder that contains a manifest.json and images.

What it does:
  - Runs YOLO (Ultralytics) detection on each frame in order.
  - Performs a lightweight IOU-based tracker to keep a stable track id per subject.
  - Picks the "main" subject track by default as:
      1) the longest-lasting PERSON track (class==0 in COCO), else
      2) the track with the most frames overall.
  - Writes outputs per sequence under outputs/tracks/<video_slug>/ :
      - tracks.json: all tracks with per-frame boxes
      - main_track.json: the chosen track summary
      - main_subject_boxes.json: { frame_name: [x1,y1,x2,y2] } for easy use in step 03

Requirements:
  - ultralytics (YOLOv8/YOLOv10)
  - pillow, numpy, tqdm

Examples:
  python 02_track_subject.py --in-seq outputs/frames_raw/my_video_slug
  python 02_track_subject.py --in-root outputs/frames_raw  # processes all sequences

Notes:
  - This script uses a simple greedy IOU tracker (no external deps). Works well when the main
    subject stays on screen. You can swap it later for ByteTrack if desired.
  - Boxes are [x1, y1, x2, y2] in pixel coordinates relative to the frame size.
"""

from __future__ import annotations
import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    raise SystemExit("Ultralytics not installed. `pip install ultralytics` in your venv.")


COCO_PERSON_CLASS_ID = 0  # Ultralytics default models trained on COCO


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = max(1e-9, area_a + area_b - inter)
    return float(inter / union)


@dataclass
class Det:
    frame_idx: int
    cls: int
    conf: float
    box: List[float]  # [x1,y1,x2,y2]
    name: str

@dataclass
class Track:
    id: int
    cls_mode: Optional[int]  # most frequent class in this track
    frames: List[int]
    boxes: List[List[float]]
    confs: List[float]

    def add(self, frame_idx: int, det: Det):
        self.frames.append(frame_idx)
        self.boxes.append(det.box)
        self.confs.append(det.conf)


class GreedyTracker:
    """A tiny IOU-based tracker.
    - Assigns detections to existing tracks greedily by IoU.
    - Creates new tracks for unmatched detections.
    - Drops tracks that were not matched for `max_missed` frames.
    """
    def __init__(self, iou_thresh: float = 0.3, max_missed: int = 15):
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.last_seen: Dict[int, int] = {}
        self.cls_counts: Dict[int, Dict[int, int]] = {}  # track_id -> {cls: count}

    def _update_cls(self, tid: int, cls: int):
        d = self.cls_counts.setdefault(tid, {})
        d[cls] = d.get(cls, 0) + 1

    def _finalize_classes(self):
        for tid, t in self.tracks.items():
            counts = self.cls_counts.get(tid, {})
            if counts:
                t.cls_mode = max(counts.items(), key=lambda kv: kv[1])[0]
            else:
                t.cls_mode = None

    def step(self, frame_idx: int, detections: List[Det]):
        # Build IoU matrix between active tracks and current detections
        active_ids = list(self.tracks.keys())
        iou_mat = np.zeros((len(active_ids), len(detections)), dtype=np.float32)
        for i, tid in enumerate(active_ids):
            last_box = np.array(self.tracks[tid].boxes[-1]) if self.tracks[tid].boxes else None
            if last_box is None:
                continue
            for j, det in enumerate(detections):
                iou_mat[i, j] = iou_xyxy(last_box, np.array(det.box))

        # Greedy matching
        matched_tracks = set()
        matched_dets = set()
        while True:
            if iou_mat.size == 0:
                break
            i_idx, j_idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            if iou_mat[i_idx, j_idx] < self.iou_thresh:
                break
            tid = active_ids[i_idx]
            det = detections[j_idx]
            if tid in matched_tracks or j_idx in matched_dets:
                iou_mat[i_idx, :] = -1
                iou_mat[:, j_idx] = -1
                continue
            # assign
            self.tracks[tid].add(frame_idx, det)
            self.last_seen[tid] = frame_idx
            self._update_cls(tid, det.cls)
            matched_tracks.add(tid)
            matched_dets.add(j_idx)
            # invalidate row/col
            iou_mat[i_idx, :] = -1
            iou_mat[:, j_idx] = -1

        # New tracks for unmatched detections
        for j, det in enumerate(detections):
            if j in matched_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            t = Track(id=tid, cls_mode=None, frames=[], boxes=[], confs=[])
            t.add(frame_idx, det)
            self.tracks[tid] = t
            self.last_seen[tid] = frame_idx
            self._update_cls(tid, det.cls)

        # Drop stale tracks
        to_drop = []
        for tid in active_ids:
            if tid in matched_tracks:
                continue
            if frame_idx - self.last_seen.get(tid, frame_idx) > self.max_missed:
                to_drop.append(tid)
        for tid in to_drop:
            self.tracks.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.cls_counts.pop(tid, None)

    def finalize(self):
        self._finalize_classes()


def load_sequence(seq_dir: Path) -> List[Path]:
    frames = sorted([p for p in seq_dir.glob("*.jpg")] + [p for p in seq_dir.glob("*.png")])
    if not frames:
        raise SystemExit(f"No frames found in {seq_dir}")
    return frames


def run_sequence(seq_dir: Path, model: YOLO, conf: float, prefer_person: bool, out_root: Path):
    frames = load_sequence(seq_dir)
    out_dir = out_root / seq_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = GreedyTracker(iou_thresh=0.3, max_missed=15)

    for idx, img_path in enumerate(tqdm(frames, desc=f"Detect+Track [{seq_dir.name}]")):
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        res = model.predict(source=np.array(im), conf=conf, verbose=False)[0]
        dets: List[Det] = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            clses = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            for b, c, cf in zip(xyxy, clses, confs):
                x1, y1, x2, y2 = [float(max(0, v)) for v in b]
                x1 = min(x1, w - 1); x2 = min(x2, w - 1)
                y1 = min(y1, h - 1); y2 = min(y2, h - 1)
                if x2 <= x1 or y2 <= y1:
                    continue
                dets.append(Det(frame_idx=idx, cls=int(c), conf=float(cf), box=[x1, y1, x2, y2], name=img_path.name))
        tracker.step(idx, dets)

    tracker.finalize()

    # Choose main track
    tracks = list(tracker.tracks.values())
    if not tracks:
        raise SystemExit(f"No detections found in {seq_dir} — consider lowering --conf or using a larger YOLO model.")

    def track_length(t: Track) -> int:
        return len(t.frames)

    chosen: Optional[Track] = None
    if prefer_person:
        person_tracks = [t for t in tracks if t.cls_mode == COCO_PERSON_CLASS_ID]
        if person_tracks:
            chosen = max(person_tracks, key=track_length)
    if chosen is None:
        chosen = max(tracks, key=track_length)

    # Write outputs
    tracks_out = {
        str(t.id): {
            "id": t.id,
            "cls_mode": t.cls_mode,
            "num_frames": len(t.frames),
            "frames": t.frames,
            "boxes": t.boxes,
            "confs": t.confs,
        } for t in tracks
    }
    with open(out_dir / "tracks.json", "w", encoding="utf-8") as f:
        json.dump(tracks_out, f, indent=2)

    main_out = {
        "id": chosen.id,
        "cls_mode": chosen.cls_mode,
        "num_frames": len(chosen.frames),
        "frames": chosen.frames,
        "boxes": chosen.boxes,
        "confs": chosen.confs,
    }
    with open(out_dir / "main_track.json", "w", encoding="utf-8") as f:
        json.dump(main_out, f, indent=2)

    # Map filename -> box for chosen track
    name_to_box: Dict[str, List[float]] = {}
    for frame_idx, box in zip(chosen.frames, chosen.boxes):
        name_to_box[frames[frame_idx].name] = box
    with open(out_dir / "main_subject_boxes.json", "w", encoding="utf-8") as f:
        json.dump(name_to_box, f, indent=2)

    print(f"✓ {seq_dir.name}: wrote tracks to {out_dir}")


def iter_sequences(root_or_seq: Path) -> List[Path]:
    if (root_or_seq / "manifest.json").exists():
        return [root_or_seq]
    # else treat as root containing multiple sequences
    seqs = []
    for p in sorted(root_or_seq.iterdir()):
        if not p.is_dir():
            continue
        if (p / "manifest.json").exists():
            seqs.append(p)
    if not seqs:
        raise SystemExit(f"No sequences found in {root_or_seq}. Pass a frames folder with manifest.json or the parent root.")
    return seqs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Detect and track the main subject across frames (YOLO + greedy tracker).")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--in-seq", type=str, help="Path to a single sequence folder (with manifest.json)")
    group.add_argument("--in-root", type=str, help="Path to the frames root; processes all child sequences")
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics YOLO model (e.g., yolov8n.pt, yolov8s.pt, yolov8x.pt)")
    ap.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    ap.add_argument("--prefer-person", action="store_true", help="Prefer the longest person track if present")
    ap.add_argument("--out", type=str, default="outputs/tracks", help="Output root directory for tracks")
    return ap.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.in_seq:
        seqs = iter_sequences(Path(args.in_seq))
    else:
        seqs = iter_sequences(Path(args.in_root))

    for seq_dir in seqs:
        run_sequence(seq_dir, model, conf=args.conf, prefer_person=args.prefer_person, out_root=out_root)


if __name__ == "__main__":
    main()