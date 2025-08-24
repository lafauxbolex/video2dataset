#!/usr/bin/env python3
"""
04_caption.py

Step 4 of the video2lora pipeline: auto-caption cropped images for LoRA training.

Inputs:
  - A single crops sequence folder (e.g., outputs/crops/<seq>) OR a root with many sequences

What it does:
  - Runs a vision-language captioning model (BLIP by default) on each image
  - Writes per-image sidecar captions:  <image>.txt  (kohya_ss-compatible)
  - Also writes a dataset-level metadata.jsonl: { "file_name": ..., "text": ... }

Models:
  - BLIP (default):  "Salesforce/blip-image-captioning-large"  (fast, good baseline)
  - BLIP-2 (optional): "Salesforce/blip2-opt-2.7b" (MUCH heavier; needs a strong GPU)

Usage:
  # Caption a single sequence of crops
  python 04_caption.py --in-seq outputs/crops/my_seq --model blip

  # Caption every sequence in the crops root
  python 04_caption.py --in-root outputs/crops --model blip --lower --strip-period

Options that affect wording:
  --prefix ""        Text to prepend to every caption (e.g., style tags)
  --suffix ""        Text to append to every caption
  --lower            Lowercase the generated caption
  --strip-period     Remove trailing period

Outputs per sequence:
  - <seq>/captions/ (sidecar .txt files)
  - <seq>/metadata.jsonl

Dependencies:
  pip install torch torchvision torchaudio transformers pillow tqdm

Notes:
  - If you don't have a GPU, BLIP still works on CPU (slower). BLIP-2 is not recommended on CPU.
  - Script auto-sanitizes generation params to avoid HF ValueError (sets safe minlen, maxlen, beams, repetition_penalty).
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from tqdm import tqdm

from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    Blip2ForConditionalGeneration,
    Blip2Processor,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class GenOpts:
    min_length: int = 1
    max_length: int = 50
    num_beams: int = 1
    repetition_penalty: float = 1.0


@dataclass
class StyleOpts:
    prefix: str = ""
    suffix: str = ""
    lower: bool = False
    strip_period: bool = False


@dataclass
class ModelSpec:
    kind: str  # "blip" or "blip2"
    name: str


BLIP_NAME = "Salesforce/blip-image-captioning-large"
BLIP2_NAME = "Salesforce/blip2-opt-2.7b"


def list_sequences(root_or_seq: Path) -> List[Path]:
    if (root_or_seq / "manifest.json").exists():
        return [root_or_seq]
    seqs: List[Path] = []
    for p in sorted(root_or_seq.iterdir()):
        if p.is_dir() and (p / "manifest.json").exists():
            seqs.append(p)
    if not seqs:
        raise SystemExit(f"No sequences found in {root_or_seq}. Pass a crops folder with manifest.json or the parent root.")
    return seqs


def list_images(seq_dir: Path) -> List[Path]:
    imgs: List[Path] = []
    for ext in IMG_EXTS:
        imgs.extend(sorted(seq_dir.glob(f"*{ext}")))
    imgs = [p for p in imgs if p.name.lower().endswith(tuple(IMG_EXTS)) and not p.name.startswith("_debug_")]
    if not imgs:
        raise SystemExit(f"No images found in {seq_dir}")
    return imgs


class Captioner:
    def __init__(self, spec: ModelSpec, device: Optional[str] = None):
        self.spec = spec
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if spec.kind == "blip":
            self.processor = BlipProcessor.from_pretrained(spec.name)
            self.model = BlipForConditionalGeneration.from_pretrained(spec.name).to(self.device)
        elif spec.kind == "blip2":
            self.processor = Blip2Processor.from_pretrained(spec.name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                spec.name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to(self.device)
        else:
            raise SystemExit(f"Unknown model kind: {spec.kind}")

        self.model.eval()

    @torch.inference_mode()
    def caption(self, image: Image.Image, gen: GenOpts) -> str:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            num_beams=gen.num_beams,
            max_length=gen.max_length,
            min_length=gen.min_length,
            repetition_penalty=gen.repetition_penalty,
        )
        if self.spec.kind == "blip":
            text = self.processor.decode(out[0], skip_special_tokens=True).strip()
        else:
            text = self.processor.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return text


def stylize(text: str, style: StyleOpts) -> str:
    t = text.strip()
    if style.lower:
        t = t.lower()
    if style.strip_period and t.endswith('.'):
        t = t[:-1]
    if style.prefix:
        t = f"{style.prefix} {t}".strip()
    if style.suffix:
        sep = '' if style.suffix.startswith((',', '.', '!', '?')) else ' '
        t = f"{t}{sep}{style.suffix}".strip()
    return t


def write_outputs(seq_dir: Path, images: List[Path], captions: List[str]):
    cap_dir = seq_dir / "captions"
    cap_dir.mkdir(parents=True, exist_ok=True)
    for img, text in zip(images, captions):
        (cap_dir / (img.stem + ".txt")).write_text(text + "\n", encoding="utf-8")
    meta_path = seq_dir / "metadata.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for img, text in zip(images, captions):
            rec = {"file_name": img.name, "text": text}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def process_sequence(seq_dir: Path, cap: Captioner, gen: GenOpts, style: StyleOpts):
    images = list_images(seq_dir)
    outs: List[str] = []
    for i in tqdm(range(0, len(images)), desc=f"Caption [{seq_dir.name}]"):
        img = Image.open(images[i]).convert("RGB")
        txt = cap.caption(img, gen)
        txt = stylize(txt, style)
        outs.append(txt)
    write_outputs(seq_dir, images, outs)
    print(f"✓ {seq_dir.name}: wrote {len(images)} captions → captions/ and metadata.jsonl")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Auto-caption cropped images (BLIP/BLIP-2)")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--in-seq", type=str, help="Path to a single crops sequence (with manifest.json)")
    group.add_argument("--in-root", type=str, help="Path to crops root; processes all child sequences")

    ap.add_argument("--model", choices=["blip", "blip2"], default="blip")
    ap.add_argument("--blip-name", type=str, default=BLIP_NAME)
    ap.add_argument("--blip2-name", type=str, default=BLIP2_NAME)

    ap.add_argument("--prefix", type=str, default="")
    ap.add_argument("--suffix", type=str, default="")
    ap.add_argument("--lower", action="store_true")
    ap.add_argument("--strip-period", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()

    spec = ModelSpec(kind=args.model, name=args.blip_name if args.model == "blip" else args.blip2_name)
    cap = Captioner(spec)

    # safe defaults baked in
    gen = GenOpts()
    style = StyleOpts(prefix=args.prefix, suffix=args.suffix, lower=args.lower, strip_period=args.strip_period)

    if args.in_seq:
        seqs = list_sequences(Path(args.in_seq))
    else:
        seqs = list_sequences(Path(args.in_root))

    for seq_dir in seqs:
        process_sequence(seq_dir, cap, gen, style)


if __name__ == "__main__":
    main()
