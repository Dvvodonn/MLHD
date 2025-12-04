#!/usr/bin/env python
"""
Create a mini dataset in the same layout as training_5 by copying a subset of
images/labels into a new directory.

Example:
python -m utils.make_mini_dataset \\
    --source data/processed/training_5 \\
    --dest data/processed/training_5_mini \\
    --train-count 100 \\
    --val-count 40
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _collect_pairs(image_dir: Path, label_dir: Path, count: int) -> List[Tuple[Path, Path]]:
    """Return up to `count` (image, label) pairs where labels exist."""
    images = sorted(
        p for p in image_dir.glob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    pairs: List[Tuple[Path, Path]] = []
    for img_path in images:
        lbl_path = label_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        pairs.append((img_path, lbl_path))
        if len(pairs) >= count:
            break
    return pairs


def _copy_pairs(pairs: Iterable[Tuple[Path, Path]], out_img_dir: Path, out_lbl_dir: Path) -> int:
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for img_path, lbl_path in pairs:
        shutil.copy2(img_path, out_img_dir / img_path.name)
        shutil.copy2(lbl_path, out_lbl_dir / lbl_path.name)
        copied += 1
    return copied


def build_mini_dataset(source_root: Path, dest_root: Path, train_count: int, val_count: int) -> None:
    src_img_root = source_root / "images"
    src_lbl_root = source_root / "labels"

    for sub in ("train", "val"):
        if not (src_img_root / sub).exists() or not (src_lbl_root / sub).exists():
            raise FileNotFoundError(f"Missing split '{sub}' under {source_root}")

    dest_img_root = dest_root / "images"
    dest_lbl_root = dest_root / "labels"

    train_pairs = _collect_pairs(src_img_root / "train", src_lbl_root / "train", train_count)
    val_pairs = _collect_pairs(src_img_root / "val", src_lbl_root / "val", val_count)

    copied_train = _copy_pairs(train_pairs, dest_img_root / "train", dest_lbl_root / "train")
    copied_val = _copy_pairs(val_pairs, dest_img_root / "val", dest_lbl_root / "val")

    print(f"Source: {source_root}")
    print(f"Destination: {dest_root}")
    print(f"Copied train: {copied_train} images/labels")
    print(f"Copied val:   {copied_val} images/labels")
    if copied_train < train_count:
        print(f"Warning: requested {train_count} train, only found {copied_train} with labels.")
    if copied_val < val_count:
        print(f"Warning: requested {val_count} val, only found {copied_val} with labels.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create a mini dataset from training_5-style layout.")
    ap.add_argument("--source", type=Path, default=Path("data/processed/training_5"), help="Root of source dataset.")
    ap.add_argument("--dest", type=Path, required=True, help="Output root for the mini dataset.")
    ap.add_argument("--train-count", type=int, required=True, help="Number of training images to copy.")
    ap.add_argument("--val-count", type=int, default=0, help="Number of validation images to copy.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    build_mini_dataset(args.source, args.dest, args.train_count, args.val_count)


if __name__ == "__main__":
    main()
