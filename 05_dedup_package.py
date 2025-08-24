#!/usr/bin/env python3
"""
05_dedup_package.py

Step 5 of the video2lora pipeline: clean up and package dataset for training.

Inputs:
  - Cropped + captioned sequences (outputs/crops/<seq>) with captions/*.txt and metadata.jsonl

What it does:
  - Loads all images + captions
  - Computes CLIP embeddings and prunes near-duplicates (cosine similarity above threshold)
  - Optionally drops blurry/low-quality images (Laplacian variance threshold)
  - Writes cleaned dataset to outputs/dataset/images + metadata.jsonl (LoRA/AI Toolkit-ready)

Outputs:
  - outputs/dataset/images/*.jpg
  - outputs/dataset/metadata.jsonl

Usage:
  python 05_dedup_package.py --in-root outputs/crops --out outputs/dataset --sim-thresh 0.985 --blur-thresh 0 --flat

Options:
  --sim-thresh   Cosine similarity above which images are considered duplicates (default 0.985)
  --blur-thresh  If >0, drop images with Laplacian variance < threshold (default 0.0)
  --max-per-seq  Optional cap on images per sequence

Dependencies:
  pip install torch torchvision pillow tqdm transformers opencv-python
"""

from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".PNG"}


def list_sequences(root: Path) -> List[Path]:
    seqs = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "metadata.jsonl").exists():
            seqs.append(p)
    if not seqs:
        raise SystemExit(f"No captioned sequences found in {root}")
    return seqs


def list_images(seq: Path) -> List[Path]:
    files = [p for p in sorted(seq.iterdir()) if p.is_file()]
    imgs = [p for p in files if p.suffix.lower() in IMG_EXTS and not p.name.startswith("_debug_")]
    return imgs


def is_blurry(img: Image.Image, thresh: float) -> bool:
    if thresh <= 0:
        return False
    gray = np.array(img.convert("L"))
    val = cv2.Laplacian(gray, cv2.CV_64F).var()
    return val < thresh


def embed_images(imgs: List[Image.Image], model: CLIPModel, processor: CLIPProcessor, device: str) -> torch.Tensor:
    if not imgs:
        return torch.empty((0, model.config.projection_dim))
    embs = []
    for i in tqdm(range(0, len(imgs), 8), desc="Embedding"):
        batch = imgs[i:i+8]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            e = model.get_image_features(**inputs)
            e = F.normalize(e, dim=-1)
        embs.append(e.cpu())
    return torch.cat(embs, dim=0)


def deduplicate(imgs: List[Path], embs: torch.Tensor, thresh: float) -> List[int]:
    N = len(imgs)
    if N == 0 or embs.numel() == 0:
        return []
    keep = []
    dropped = set()
    for i in range(N):
        if i in dropped:
            continue
        keep.append(i)
        vi = embs[i].unsqueeze(0)
        sims = F.cosine_similarity(vi, embs, dim=-1)
        dup_idx = (sims > thresh).nonzero(as_tuple=True)[0].tolist()
        for j in dup_idx:
            if j > i:
                dropped.add(j)
    return keep


def _render_name(template: str, seq: str, stem: str, idx: int, ext: str) -> str:
    return template.format(seq=seq, stem=stem, idx=f"{idx:06d}", ext=ext)


def process_sequence(seq: Path, out_root: Path, model: CLIPModel, processor: CLIPProcessor, device: str,
                     sim_thresh: float, blur_thresh: float, max_per_seq: int,
                     flat: bool, name_template: str, sidecar: bool,
                     global_seen: Dict[str, int]) -> List[Tuple[str, str]]:
    images = list_images(seq)
    if not images:
        print(f"! {seq.name}: no images found; skipping")
        return []
    captions_dir = seq / "captions"
    captions = {p.stem: p.read_text(encoding="utf-8").strip() for p in captions_dir.glob("*.txt")} if captions_dir.exists() else {}

    imgs_pil = [Image.open(p).convert("RGB") for p in images]

    keep_idxs = [i for i, im in enumerate(imgs_pil) if not is_blurry(im, blur_thresh)]
    imgs_pil = [imgs_pil[i] for i in keep_idxs]
    images = [images[i] for i in keep_idxs]

    if not images:
        print(f"! {seq.name}: all images filtered out by blur threshold; skipping")
        return []

    embs = embed_images(imgs_pil, model, processor, device)
    keep = deduplicate(images, embs, sim_thresh)

    if max_per_seq > 0 and len(keep) > max_per_seq:
        keep = keep[:max_per_seq]

    records: List[Tuple[str, str]] = []

    if flat:
        img_out = out_root / "images"
        img_out.mkdir(parents=True, exist_ok=True)
        running_idx = 1
        for i in keep:
            src = images[i]
            stem, ext = src.stem, src.suffix
            name = _render_name(name_template, seq.name, stem, running_idx, ext)
            while name in global_seen:
                running_idx = global_seen[name] + 1
                name = _render_name(name_template, seq.name, stem, running_idx, ext)
            global_seen[name] = running_idx
            running_idx += 1

            dst = img_out / name
            shutil.copy2(src, dst)
            text = captions.get(src.stem, "")
            if sidecar:
                (img_out / (Path(name).stem + ".txt")).write_text(text + "\n", encoding="utf-8")
            records.append((f"images/{name}", text))
        print(f"✓ {seq.name}: kept {len(keep)} images (flat)")
        return records

    else:
        seq_out = out_root / seq.name
        img_out = seq_out / "images"
        img_out.mkdir(parents=True, exist_ok=True)

        for i in keep:
            src = images[i]
            dst = img_out / src.name
            shutil.copy2(src, dst)
            text = captions.get(src.stem, "")
            if sidecar:
                (img_out / (src.stem + ".txt")).write_text(text + "\n", encoding="utf-8")
            records.append((f"images/{dst.name}", text))

        with open(seq_out / "metadata.jsonl", "w", encoding="utf-8") as f:
            for rel, text in records:
                f.write(json.dumps({"file_name": Path(rel).name, "text": text}, ensure_ascii=False) + "\n")
        print(f"✓ {seq.name}: kept {len(keep)} images (from {len(images)})")
        return records


def parse_args():
    ap = argparse.ArgumentParser(description="Deduplicate and package dataset")
    ap.add_argument("--in-root", type=str, required=True, help="Root of captioned crops")
    ap.add_argument("--out", type=str, default="outputs/dataset", help="Output root")
    ap.add_argument("--sim-thresh", type=float, default=0.985)
    ap.add_argument("--blur-thresh", type=float, default=0.0)
    ap.add_argument("--max-per-seq", type=int, default=0, help="Cap images per sequence (0=no cap)")
    ap.add_argument("--flat", action="store_true", help="Put all images in a single <out>/images folder and write one metadata.jsonl")
    ap.add_argument("--name-template", type=str, default="{seq}_{idx}{ext}", help="Template for filenames (supports {seq}, {stem}, {idx}, {ext})")
    ap.add_argument("--sidecar", action="store_true", help="Write <image>.txt captions next to images")
    # Defaults for cropping stage (Step 3)
    ap.add_argument("--pad", type=float, default=0.15, help="Default padding per side for crop step")
    ap.add_argument("--y-bias", type=float, default=0.15, help="Default upward bias for crop step")
    ap.add_argument("--headroom", type=float, default=0.05, help="Default headroom fraction for crop step")
    return ap.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    in_root = Path(args.in_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    seqs = list_sequences(in_root)

    merged: List[Tuple[str, str]] = []
    global_seen: Dict[str, int] = {}

    for seq in seqs:
        recs = process_sequence(
            seq, out_root, model, processor, device,
            args.sim_thresh, args.blur_thresh, args.max_per_seq,
            args.flat, args.name_template, args.sidecar,
            global_seen,
        )
        if args.flat:
            merged.extend(recs)

    if args.flat:
        meta_path = out_root / "metadata.jsonl"
        with open(meta_path, "w", encoding="utf-8") as f:
            for rel, text in merged:
                f.write(json.dumps({"file_name": Path(rel).name, "text": text}, ensure_ascii=False) + "\n")
        print(f"✓ Wrote merged metadata.jsonl with {len(merged)} records at {meta_path}")

    print("✓ Packaging complete.")


if __name__ == "__main__":
    main()
