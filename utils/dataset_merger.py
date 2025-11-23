#!/usr/bin/env python3
"""
Utility for merging two YOLO-style datasets (images/labels train & val).

The script expects each dataset root to follow this structure:
    dataset_root/
        images/
            train/*.jpg
            val/*.jpg
        labels/
            train/*.txt
            val/*.txt

Example usage (from the project root):
    python utils/dataset_merger.py \
        --dataset-a data/raw/training_3 \
        --dataset-b data/raw/final_4 \
        --output data/raw/merged_training3_final4
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml

IMAGE_EXT = ".jpg"
LABEL_EXT = ".txt"
DEFAULT_SPLITS = ("train", "val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two YOLO-format datasets (train/val splits)."
    )
    parser.add_argument("--dataset-a", required=True, help="Path to the first dataset root.")
    parser.add_argument("--dataset-b", required=True, help="Path to the second dataset root.")
    parser.add_argument("--output", required=True, help="Path for the merged dataset.")
    parser.add_argument(
        "--prefix-a",
        help="Filename prefix for dataset A (defaults to dataset directory name)."
    )
    parser.add_argument(
        "--prefix-b",
        help="Filename prefix for dataset B (defaults to dataset directory name)."
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks (labels always copy fallback)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove the output directory if it already exists."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to merge (default: train val)."
    )
    return parser.parse_args()


def ensure_dataset_dirs(dataset_root: Path, splits: Sequence[str]) -> None:
    """Verify the dataset contains expected split folders."""
    missing: List[str] = []
    for split in splits:
        for subtree in ("images", "labels"):
            path = dataset_root / subtree / split
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            f"Dataset at {dataset_root} is missing required folders:\n  "
            + "\n  ".join(missing)
        )


def collect_pairs(dataset_root: Path, split: str) -> List[Tuple[Path, Path]]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    pairs: List[Tuple[Path, Path]] = []

    for img_path in sorted(image_dir.glob(f"*{IMAGE_EXT}")):
        label_path = label_dir / f"{img_path.stem}{LABEL_EXT}"
        if not label_path.exists():
            print(f"[WARN] Missing label for {img_path} (expected {label_path}), skipping.")
            continue
        pairs.append((img_path, label_path))
    return pairs


def make_unique_name(base: str, used: set[str]) -> str:
    """Ensure filenames remain unique after merging by appending counters."""
    candidate = base
    suffix = 1
    while candidate in used:
        candidate = f"{base}_{suffix:04d}"
        suffix += 1

    used.add(candidate)
    return candidate


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if copy:
        shutil.copy2(src, dst)
    else:
        target = src.resolve()
        try:
            dst.symlink_to(target)
        except OSError:
            # Some environments (e.g., Windows without privileges) disallow symlinks.
            shutil.copy2(src, dst)


def load_dataset_metadata(datasets: Sequence[Path]) -> Dict[str, object]:
    for root in datasets:
        yaml_path = root / "data.yaml"
        if not yaml_path.exists():
            continue
        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, dict) and ("names" in data or "nc" in data):
                return {
                    "names": data.get("names"),
                    "nc": data.get("nc"),
                }
        except Exception as exc:
            print(f"[WARN] Failed to parse {yaml_path}: {exc}")
    # Fallback for person-only datasets
    return {"names": ["person"], "nc": 1}


def write_data_yaml(output_root: Path, metadata: Dict[str, object]) -> None:
    names = metadata.get("names") or ["person"]
    if isinstance(names, str):
        names = [names]
    nc = metadata.get("nc")
    if not isinstance(nc, int) or nc <= 0:
        nc = len(names)
    if nc <= 0:
        nc = 1
    if len(names) != nc:
        # Harmonize in case yaml had mismatched counts
        if len(names) > nc:
            names = names[:nc]
        else:
            names = list(names) + [f"class_{i}" for i in range(len(names), nc)]

    yaml_data = {
        "train": str(output_root / "images" / "train"),
        "val": str(output_root / "images" / "val"),
        "nc": nc,
        "names": names,
    }
    yaml_path = output_root / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False)


def merge_datasets(
    datasets: Sequence[Path],
    prefixes: Sequence[str],
    output_root: Path,
    splits: Sequence[str],
    copy: bool,
) -> Dict[str, int]:
    used_names = {split: set() for split in splits}
    counts = {split: 0 for split in splits}

    for dataset_root, prefix in zip(datasets, prefixes):
        for split in splits:
            pairs = collect_pairs(dataset_root, split)
            if not pairs:
                print(f"[INFO] No samples found under {dataset_root}/images/{split}")
                continue

            for img_path, label_path in pairs:
                base_stem = f"{prefix}_{img_path.stem}"
                unique_stem = make_unique_name(base_stem, used_names[split])

                dst_img = output_root / "images" / split / f"{unique_stem}{IMAGE_EXT}"
                dst_label = output_root / "labels" / split / f"{unique_stem}{LABEL_EXT}"

                link_or_copy(img_path, dst_img, copy=copy)
                # Ensure labels always end up as real files for easy editing
                link_or_copy(label_path, dst_label, copy=True)
                counts[split] += 1

    return counts


def main() -> None:
    args = parse_args()

    dataset_a = Path(args.dataset_a)
    dataset_b = Path(args.dataset_b)
    output_dir = Path(args.output)

    for ds in (dataset_a, dataset_b):
        ensure_dataset_dirs(ds, args.splits)

    if output_dir.exists():
        if not args.force:
            print(f"[ERROR] Output directory {output_dir} already exists. Use --force to overwrite.")
            sys.exit(1)
        shutil.rmtree(output_dir)

    for split in args.splits:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    prefixes = [
        args.prefix_a or dataset_a.name,
        args.prefix_b or dataset_b.name,
    ]

    metadata = load_dataset_metadata([dataset_a, dataset_b])
    counts = merge_datasets(
        datasets=[dataset_a, dataset_b],
        prefixes=prefixes,
        output_root=output_dir,
        splits=args.splits,
        copy=args.copy,
    )
    write_data_yaml(output_dir, metadata)

    print("\nMerge complete:")
    for split in args.splits:
        print(f"  {split:>5}: {counts.get(split, 0)} images")
    print(f"\nMerged dataset root: {output_dir}")
    print(f"Data config written to: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
