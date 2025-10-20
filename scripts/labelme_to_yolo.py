#!/usr/bin/env python3
"""
LabelMe -> YOLO converter

Features:
- Reads LabelMe JSON files (rectangle or polygon) and outputs YOLOv5/v8 .txt labels.
- Preserves subfolder structure under output/images and output/labels.
- Allows limiting to a class list (default: ["person"]). Non-matching labels are skipped.
- Can create a train/val split (by folder-level random split or by file-level split).
- Generates a minimal data.yaml suitable for Ultralytics YOLO (v5/v8).

Usage:
  python scripts/labelme_to_yolo.py \
      --src "/path/to/source_root" \
      --dst "/path/to/output_root" \
      --classes person \
      --split 0.9 \
      --copy

Notes:
- Source tree is expected to contain pairs like 000001.jpg and 000001.json (LabelMe format).
- Destination tree will look like:
    dst/
      images/
        train/...subfolders and jpg...
        val/...subfolders and jpg...
      labels/
        train/...matching .txt...
        val/...matching .txt...
- By default, the script symlinks images into train/val. Use --copy to copy files instead.
- The generated data.yaml is placed at dst/data.yaml (you can edit paths afterward if needed).
"""

import argparse
import json
from pathlib import Path
import random
import shutil
import sys

# -----------------------------
# Helpers
# -----------------------------

def clamp01(x):
    return max(0.0, min(1.0, x))

def bbox_norm(xmin, ymin, xmax, ymax, img_w, img_h):
    x_c = ((xmin + xmax) / 2.0) / img_w
    y_c = ((ymin + ymax) / 2.0) / img_h
    w   = (xmax - xmin) / img_w
    h   = (ymax - ymin) / img_h
    return clamp01(x_c), clamp01(y_c), clamp01(w), clamp01(h)

def rect_from_points(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, ymin = min(xs), min(ys)
    xmax, ymax = max(xs), max(ys)
    return xmin, ymin, xmax, ymax

def poly_to_bbox(points):
    return rect_from_points(points)

def load_labelme(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def write_yolo_txt(txt_path, lines):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

def relpath_under(root, path):
    return str(Path(path).resolve().relative_to(Path(root).resolve()))

def find_pairs(src_root):
    """Find (image, json) pairs under src_root where both exist."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for p in Path(src_root).rglob("*"):
        if p.suffix.lower() in exts:
            j = p.with_suffix(".json")
            if j.exists():
                yield p, j

def make_split(items, split_ratio):
    """Split a list of items into train and val (by file)."""
    items = list(items)
    random.shuffle(items)
    n_train = int(len(items) * split_ratio)
    return items[:n_train], items[n_train:]

def symlink_or_copy(src, dst, copy=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        # If symlink exists pointing to the same target, skip; otherwise recreate
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        # Use relative symlink for portability
        rel = Path(src).resolve()
        try:
            dst.symlink_to(rel)
        except OSError:
            # On systems that don't allow symlinks, fallback to copy
            shutil.copy2(src, dst)

def build_class_map(class_names):
    """Return {label: idx} map."""
    return {name: i for i, name in enumerate(class_names)}

def gen_data_yaml(dst_root, nc, names):
    content = f"""train: {str(Path(dst_root)/'images'/'train')}
val: {str(Path(dst_root)/'images'/'val')}

nc: {nc}
names: {names}
"""
    with open(Path(dst_root)/"data.yaml", "w") as f:
        f.write(content)

# -----------------------------
# Main
# -----------------------------

def convert(src_root, dst_root, classes, split=0.9, copy=False, verbose=False):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    cls_map = build_class_map(classes)

    # Collect pairs
    pairs = list(find_pairs(src_root))
    if not pairs:
        print("No image+json pairs found under", src_root, file=sys.stderr)
        sys.exit(1)

    # Split into train/val by file
    train_pairs, val_pairs = make_split(pairs, split)

    # Process a split
    def process_split(pairs, split_name):
        for img_path, json_path in pairs:
            data = load_labelme(json_path)
            img_w = data.get("imageWidth")
            img_h = data.get("imageHeight")
            if not img_w or not img_h:
                # Fallback: skip if dimensions missing
                if verbose:
                    print(f"[WARN] Missing image dims for {json_path}, skipping")
                continue

            yolo_lines = []
            for shape in data.get("shapes", []):
                label = shape.get("label", "")

                # Manual remap option: force '0' -> 'person'
                if str(label).strip() == "0":
                    label = "person"

                # --- Normalize label -----------------------------------------------------
                # If LabelMe stored class indices as strings (e.g., "0"), map them to names.
                # Accept slight variations like whitespace or actual ints.
                if label not in cls_map:
                    try:
                        # Allow "0", " 0 ", 0, etc.
                        lbl_str = str(label).strip()
                        if lbl_str.isdigit():
                            idx = int(lbl_str)
                            if 0 <= idx < len(classes):
                                label = classes[idx]
                    except Exception:
                        pass  # fall through; we'll filter via cls_map below

                # Skip if (still) not a class we want
                if label not in cls_map:
                    continue
                # ------------------------------------------------------------------------

                stype = shape.get("shape_type", "rectangle")
                pts = shape.get("points", [])
                if not pts:
                    continue

                # Rectangle => two points; Polygon => many points; both converted to bbox
                if stype == "rectangle":
                    xmin, ymin, xmax, ymax = rect_from_points(pts)
                else:
                    xmin, ymin, xmax, ymax = poly_to_bbox(pts)

                # YOLO normalized
                x_c, y_c, w, h = bbox_norm(xmin, ymin, xmax, ymax, img_w, img_h)

                # Guard tiny or invalid boxes
                if w <= 0 or h <= 0:
                    if verbose:
                        print(f"[WARN] Non-positive box at {json_path}")
                    continue

                yolo_lines.append(f"{cls_map[label]} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

            # Destination paths (preserve subfolders)
            rel = img_path.relative_to(src_root)
            out_img = dst_root / "images" / split_name / rel
            out_lbl = dst_root / "labels" / split_name / rel.with_suffix(".txt")

            # Ensure image present (symlink/copy)
            symlink_or_copy(img_path, out_img, copy=copy)

            # Write label file (even if empty — YOLO is OK with empty txt meaning no objects)
            write_yolo_txt(out_lbl, yolo_lines)

    process_split(train_pairs, "train")
    process_split(val_pairs, "val")

    # data.yaml
    gen_data_yaml(dst_root, nc=len(classes), names=classes)

    print("Done.")
    print("Wrote:", dst_root)
    print("  - images/train, labels/train")
    print("  - images/val, labels/val")
    print("Also wrote:", dst_root/"data.yaml")
    print("Classes:", classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Root directory with LabelMe .json + .jpg")
    ap.add_argument("--dst", required=True, help="Output root")
    ap.add_argument("--classes", nargs="+", default=["person"], help="Class names to keep (in order)")
    ap.add_argument("--split", type=float, default=0.9, help="Train split ratio (0-1)")
    ap.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    convert(args.src, args.dst, args.classes, split=args.split, copy=args.copy, verbose=args.verbose)

if __name__ == "__main__":
    main()
