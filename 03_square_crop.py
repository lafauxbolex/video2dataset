#!/usr/bin/env python3
"""
03_square_crop.py

Step 3 of the video2lora pipeline: create smart **square crops** centered on the main subject.

Inputs per sequence (video slug):
  - Frames folder:           outputs/frames_raw/<seq>/
  - Main subject boxes JSON: outputs/tracks/<seq>/main_subject_boxes.json

What it does:
  - For each frame listed in main_subject_boxes.json, takes the bbox [x1,y1,x2,y2]
  - Inflates it by a padding percentage around the center
  - Converts to a square by expanding the short side (not by cutting the long side)
  - Clamps to image bounds (shifts the box inside; if the requested square can't fit, reduces it)
  - Crops and resizes to target size (e.g., 1024x1024)
  - Writes to outputs/crops/<seq>/*.jpg and a manifest

Options:
  --size            Output square size in pixels (default: 1024)
  --pad             Margin around subject as a fraction of the subject's max side (default: 0.25 means 25% on each side)
  --fallback        What to do if a frame has no bbox entry: 'skip' (default) or 'center'
  --draw-debug      Also save a debug image with the final box overlay

Examples:
  python 03_square_crop.py --in-seq outputs/frames_raw/my_seq --boxes-root outputs/tracks --out outputs/crops --size 1024 --pad 0.25
  python 03_square_crop.py --in-root outputs/frames_raw --boxes-root outputs/tracks --out outputs/crops --size 768 --pad 0.2
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw


@dataclass
class CropConfig:
    size: int = 1024
    pad: float = 0.25
    fallback: str = "skip"  # or "center"
    draw_debug: bool = False
    y_bias: float = 0.0       # 0.0=centered; 0.3 shifts crop up toward head (top-biased)
    headroom: float = 0.08    # fraction of final square side to keep above bbox top



def load_frames(seq_dir: Path) -> List[Path]:
    frames = sorted(list(seq_dir.glob("*.jpg")) + list(seq_dir.glob("*.png")))
    if not frames:
        raise SystemExit(f"No frames found in {seq_dir}")
    return frames


def load_boxes(boxes_root: Path, seq_name: str) -> Dict[str, List[float]]:
    p = boxes_root / seq_name / "main_subject_boxes.json"
    if not p.exists():
        raise SystemExit(f"Missing boxes file: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp_square_inside(img_w: int, img_h: int, cx: float, cy: float, side: float) -> Tuple[float, float, float]:
    """Shift (cx, cy) so that [cx - s/2, cy - s/2, cx + s/2, cy + s/2] fits inside the image.
    If side is larger than either dimension, reduce it to fit.
    Returns (cx, cy, side)."""
    side = min(side, img_w, img_h)
    half = side / 2.0
    x1 = cx - half; y1 = cy - half
    x2 = cx + half; y2 = cy + half

    # Shift to fit horizontally
    if x1 < 0:
        cx += -x1
    if x2 > img_w:
        cx -= (x2 - img_w)

    # Recompute after horizontal shift
    x1 = cx - half; x2 = cx + half

    # Shift to fit vertically
    if y1 < 0:
        cy += -y1
    if y2 > img_h:
        cy -= (y2 - img_h)

    # Final clamp (in case of rounding)
    half = side / 2.0
    cx = max(half, min(img_w - half, cx))
    cy = max(half, min(img_h - half, cy))
    return cx, cy, side


def expand_to_square_with_pad(box: List[float], pad_frac: float, img_w: int, img_h: int, *, y_bias: float = 0.0, headroom: float = 0.0) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    # base square side is max of bbox sides
    base = max(bw, bh)
    # add symmetric padding on each side (pad_frac of base per side)
    side = base * (1.0 + 2.0 * pad_frac)

    # center (optionally biased toward the top of the bbox)
    cx = (x1 + x2) / 2.0
    cy_center = (y1 + y2) / 2.0
    top_anchor = y1 + 0.35 * bh  # rough location of face within a person box
    y_bias = max(0.0, min(1.0, y_bias))
    cy = (1.0 - y_bias) * cy_center + y_bias * top_anchor

    # ensure inside image (or reduced to fit)
    cx, cy, side = clamp_square_inside(img_w, img_h, cx, cy, side)

    # enforce headroom: keep some space above bbox top
    if headroom > 0.0:
        desired_top = max(0.0, y1 - headroom * side)
        current_top = cy - side / 2.0
        if current_top > desired_top:
            # shift up to achieve desired headroom
            cy -= (current_top - desired_top)
            cx, cy, side = clamp_square_inside(img_w, img_h, cx, cy, side)

    half = side / 2.0
    x1n = int(round(cx - half))
    y1n = int(round(cy - half))
    x2n = int(round(cx + half))
    yn2 = int(round(cy + half))

    # Final safety clamps
    x1n = max(0, min(img_w - 1, x1n))
    y1n = max(0, min(img_h - 1, y1n))
    x2n = max(1, min(img_w, x2n))
    yn2 = max(1, min(img_h, yn2))

    return x1n, y1n, x2n, yn2


def process_sequence(seq_dir: Path, boxes_root: Path, out_root: Path, cfg: CropConfig):
    frames = load_frames(seq_dir)
    boxes = load_boxes(boxes_root, seq_dir.name)

    seq_out = out_root / seq_dir.name
    seq_out.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0

    for img_path in frames:
        im = Image.open(img_path).convert("RGB")
        W, H = im.size

        if img_path.name in boxes:
            box = boxes[img_path.name]
            crop = expand_to_square_with_pad(box, cfg.pad, W, H, y_bias=cfg.y_bias, headroom=cfg.headroom)
        else:
            if cfg.fallback == "center":
                # center square box using min(W,H)
                side = min(W, H)
                cx, cy = W / 2.0, H / 2.0
                half = side / 2.0
                crop = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
            else:
                skipped += 1
                continue

        x1, y1, x2, y2 = crop
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)
        if x2 <= x1 or y2 <= y1:
            skipped += 1
            continue

        sub = im.crop((x1, y1, x2, y2)).resize((cfg.size, cfg.size), Image.LANCZOS)
        out_img = seq_out / img_path.name
        sub.save(out_img, quality=95)
        kept += 1

        if cfg.draw_debug:
            dbg = im.copy()
            drw = ImageDraw.Draw(dbg)
            drw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            dbg.save(seq_out / f"_debug_{img_path.name}")

    manifest = {
        "sequence": seq_dir.name,
        "input_frames_dir": str(seq_dir.resolve()),
        "boxes_from": str((boxes_root / seq_dir.name / 'main_subject_boxes.json').resolve()),
        "output_dir": str(seq_out.resolve()),
        "size": cfg.size,
        "pad": cfg.pad,
        "kept": kept,
        "skipped": skipped,
    }
    with open(seq_out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ {seq_dir.name}: wrote {kept} crop(s), skipped {skipped}")


def iter_sequences(root_or_seq: Path) -> List[Path]:
    # If a manifest exists inside, assume it's a sequence folder
    if (root_or_seq / "manifest.json").exists():
        return [root_or_seq]
    seqs = []
    for p in sorted(root_or_seq.iterdir()):
        if p.is_dir() and (p / "manifest.json").exists():
            seqs.append(p)
    if not seqs:
        raise SystemExit(f"No sequences found in {root_or_seq}. Pass a frames folder with manifest.json or the parent root.")
    return seqs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Square-crop frames around the main subject using boxes from Step 2.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--in-seq", type=str, help="Path to a single frames sequence (with manifest.json)")
    group.add_argument("--in-root", type=str, help="Path to the frames root; processes all child sequences")
    ap.add_argument("--boxes-root", type=str, default="outputs/tracks", help="Root where Step 2 wrote tracks")
    ap.add_argument("--out", type=str, default="outputs/crops", help="Output root for cropped squares")
    ap.add_argument("--size", type=int, default=1024, help="Output square size in px")
    ap.add_argument("--pad", type=float, default=0.25, help="Padding as fraction of subject max side (per side)")
    ap.add_argument("--y-bias", type=float, default=0.0, help="0=center; 0.3 biases upward toward head")
    ap.add_argument("--headroom", type=float, default=0.08, help="Keep this fraction of square side above bbox top")
    ap.add_argument("--fallback", choices=["skip", "center"], default="skip", help="If a frame lacks a box: skip or center-crop")
    ap.add_argument("--draw-debug", action="store_true", help="Save debug overlays of final crop boxes")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = CropConfig(size=args.size, pad=args.pad, fallback=args.fallback, draw_debug=args.draw_debug, y_bias=args["y_bias"] if isinstance(args, dict) else args.y_bias, headroom=args["headroom"] if isinstance(args, dict) else args.headroom)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    boxes_root = Path(args.boxes_root)

    if args.in_seq:
        seqs = iter_sequences(Path(args.in_seq))
    else:
        seqs = iter_sequences(Path(args.in_root))

    for seq_dir in seqs:
        process_sequence(seq_dir, boxes_root, out_root, cfg)


if __name__ == "__main__":
    main()
