#!/usr/bin/env python
"""
Count YOLO label files and box annotations for training_5 train/val splits.

Hardcoded paths:
- Labels root: data/processed/training_5/labels/{train,val}
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent
LABEL_ROOT = ROOT / "data/processed/training_5/labels"
SPLITS = ("train", "val")


def _count_split(split: str) -> Tuple[int, int]:
    """Return (#label_files, #boxes) for a given split."""
    label_dir = LABEL_ROOT / split
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing labels directory: {label_dir}")

    file_count = 0
    box_count = 0
    for path in sorted(label_dir.glob("*.txt")):
        file_count += 1
        with path.open() as f:
            box_count += sum(1 for line in f if line.strip())
    return file_count, box_count


def main():
    totals_files = 0
    totals_boxes = 0

    print(f"Label counts under {LABEL_ROOT}:")
    for split in SPLITS:
        files, boxes = _count_split(split)
        totals_files += files
        totals_boxes += boxes
        print(f"  {split}: {files} files | {boxes} boxes")

    print(f"\nTotal: {totals_files} files | {totals_boxes} boxes")


if __name__ == "__main__":
    main()
