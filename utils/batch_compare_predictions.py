#!/usr/bin/env python
"""
Generate side-by-side bounding-box visualizations for a folder of images using:
- MLHD checkpoint (checkpoints/best.pt)
- YOLOv5s weights (yolov5s.pt)

For every input image, three outputs are written:
  1) YOLOv5s predictions
  2) MLHD predictions
  3) Both overlaid

Outputs are stored under --output-dir (default: <images_dir>/predictions/{yolo,mlhd,both}).

Example:
python -m utils.batch_compare_predictions \\
    --images-dir data/processed/training_5/images/val \\
    --mlhd-checkpoint checkpoints/best.pt \\
    --yolo-weights yolov5s.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.transforms import letterbox_image  # noqa: E402
from models.backbone import backbone_output_grid, backbone_input_resolution  # noqa: E402
from models.yolo_like import Model  # noqa: E402
from utils.device import get_best_device, describe_device  # noqa: E402


# -----------------
# Decoding helpers
# -----------------
def _letterbox_to_orig(box, params, img_w, img_h):
    """Map letterboxed normalized box (cx, cy, w, h in [0,1]) back to original image pixels."""
    cx, cy, w, h = box
    target_w, target_h = params["target_wh"]
    scale = params["scale"]
    pad_w = params["pad_w"]
    pad_h = params["pad_h"]

    cx_l = cx * target_w
    cy_l = cy * target_h
    w_l = w * target_w
    h_l = h * target_h

    x1_l = cx_l - w_l / 2
    y1_l = cy_l - h_l / 2
    x2_l = cx_l + w_l / 2
    y2_l = cy_l + h_l / 2

    x1 = int(np.clip((x1_l - pad_w) / scale, 0, img_w))
    y1 = int(np.clip((y1_l - pad_h) / scale, 0, img_h))
    x2 = int(np.clip((x2_l - pad_w) / scale, 0, img_w))
    y2 = int(np.clip((y2_l - pad_h) / scale, 0, img_h))
    return x1, y1, x2, y2


def _decode_mlhd(pred: torch.Tensor, conf_thresh: float, S: int, params, img_w: int, img_h: int):
    """Decode MLHD grid predictions to pixel boxes."""
    obj = pred[..., 4]
    keep = obj >= conf_thresh
    boxes = []
    for j, i in keep.nonzero(as_tuple=False):
        score = obj[j, i].item()
        tx, ty, tw, th = pred[j, i, 0:4].tolist()
        cx = (i + tx) / S
        cy = (j + ty) / S
        w = tw
        h = th
        x1, y1, x2, y2 = _letterbox_to_orig((cx, cy, w, h), params, img_w, img_h)
        boxes.append((x1, y1, x2, y2, score))
    return boxes


def _load_yolov5(yolo_weights: str, device: torch.device):
    """
    Load YOLOv5 model via torch.hub using provided weights.
    Note: torch.hub will fetch ultralytics/yolov5 if not already cached.
    """
    # Avoid clashes with our local modules when YOLOv5 repo is added to sys.path
    for name in list(sys.modules):
        if name == "models" or name.startswith("models.") or name == "utils" or name.startswith("utils."):
            sys.modules.pop(name, None)

    try:
        if yolo_weights and Path(yolo_weights).exists():
            # Local weights: YOLOv5 custom() does not accept a pretrained kwarg.
            model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=yolo_weights,
            )
        else:
            # Fallback to vanilla yolov5s pretrained
            model = torch.hub.load(
                "ultralytics/yolov5",
                "yolov5s",
                pretrained=True,
            )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load YOLOv5 model. Ensure ultralytics/yolov5 is available or internet access is enabled."
        ) from exc

    model.to(device)
    model.eval()
    # Restrict detections to person class (COCO class 0)
    try:
        model.classes = [0]
    except Exception:
        pass
    return model


# -----------------
# Drawing helpers
# -----------------
def _draw_boxes(img, boxes, color, label_prefix: str, thickness: int = 2):
    for idx, (x1, y1, x2, y2, score) in enumerate(boxes, start=1):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        cv2.putText(
            img,
            f"{label_prefix}{idx}:{score:.2f}",
            (int(x1) + 2, max(14, int(y1) + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            max(2, thickness - 1),
            cv2.LINE_AA,
        )


def _ensure_outdirs(base: Path):
    yolo_dir = base / "yolo"
    mlhd_dir = base / "mlhd"
    both_dir = base / "both"
    for d in (yolo_dir, mlhd_dir, both_dir):
        d.mkdir(parents=True, exist_ok=True)
    return yolo_dir, mlhd_dir, both_dir


# -----------------
# Main pipeline
# -----------------
def process_folder(
    images_dir: Path,
    mlhd_ckpt: Path,
    yolo_weights: Path,
    backbone: str,
    mlhd_conf: float,
    yolo_conf: float,
    output_dir: Optional[Path],
):
    img_paths = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not img_paths:
        raise ValueError(f"No images found in {images_dir}")

    device = get_best_device()
    print(f"Using device: {describe_device(device)}")

    # MLHD model
    grid = backbone_output_grid(backbone)
    img_size = backbone_input_resolution(backbone)
    mlhd = Model(S=grid, backbone_name=backbone)
    ckpt = torch.load(str(mlhd_ckpt), map_location=device)
    mlhd.load_state_dict(ckpt["model"])
    mlhd.to(device).eval()

    # YOLOv5
    yolo_model = _load_yolov5(str(yolo_weights), device)
    yolo_model.conf = yolo_conf

    out_root = output_dir if output_dir is not None else images_dir / "predictions"
    yolo_dir, mlhd_dir, both_dir = _ensure_outdirs(out_root)
    print(f"Saving outputs to: {out_root}")

    for idx, img_path in enumerate(img_paths, start=1):
        print(f"[{idx}/{len(img_paths)}] {img_path.name}")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Skipping (failed to load): {img_path}")
            continue
        img_h, img_w = img.shape[:2]

        # MLHD inference
        img_tensor, params = letterbox_image(str(img_path), target_size=(img_size, img_size))
        with torch.no_grad():
            mlhd_pred = mlhd(img_tensor.unsqueeze(0).to(device))[0].cpu()
        mlhd_boxes = _decode_mlhd(mlhd_pred, mlhd_conf, grid, params, img_w, img_h)

        # YOLOv5 inference (expects RGB)
        with torch.no_grad():
            yolo_results = yolo_model(img[..., ::-1])
        yolo_boxes = []
        for *xyxy, conf, cls_idx in yolo_results.xyxy[0].cpu().tolist():
            if int(cls_idx) != 0:
                continue  # keep only person class
            x1, y1, x2, y2 = map(int, xyxy)
            yolo_boxes.append((x1, y1, x2, y2, conf))

        # Compose images
        yolo_color = (0, 0, 255)  # Red for YOLO
        mlhd_color = (255, 0, 0)  # Blue for MLHD

        yolo_img = img.copy()
        _draw_boxes(yolo_img, yolo_boxes, yolo_color, "Y", thickness=3)

        mlhd_img = img.copy()
        _draw_boxes(mlhd_img, mlhd_boxes, mlhd_color, "M", thickness=2)

        both_img = img.copy()
        _draw_boxes(both_img, yolo_boxes, yolo_color, "Y", thickness=3)
        _draw_boxes(both_img, mlhd_boxes, mlhd_color, "M", thickness=2)

        # Save
        cv2.imwrite(str(yolo_dir / img_path.name), yolo_img)
        cv2.imwrite(str(mlhd_dir / img_path.name), mlhd_img)
        cv2.imwrite(str(both_dir / img_path.name), both_img)


def parse_args():
    ap = argparse.ArgumentParser(description="Batch-generate YOLOv5 vs MLHD prediction visualizations.")
    ap.add_argument("--images-dir", required=True, help="Directory of images.")
    ap.add_argument("--mlhd-checkpoint", default=str(ROOT / "checkpoints" / "best.pt"), help="Path to MLHD checkpoint.")
    ap.add_argument("--yolo-weights", default=str(ROOT / "yolov5s.pt"), help="Path to YOLOv5s weights.")
    ap.add_argument("--backbone", default="13x13", choices=["13x13", "8x8"], help="Backbone used for MLHD checkpoint.")
    ap.add_argument("--mlhd-conf", type=float, default=0.3, help="Confidence threshold for MLHD.")
    ap.add_argument("--yolo-conf", type=float, default=0.25, help="Confidence threshold for YOLOv5s.")
    ap.add_argument("--output-dir", default=None, help="Optional output root directory.")
    return ap.parse_args()


def main():
    args = parse_args()
    process_folder(
        images_dir=Path(args.images_dir),
        mlhd_ckpt=Path(args.mlhd_checkpoint),
        yolo_weights=Path(args.yolo_weights),
        backbone=args.backbone,
        mlhd_conf=args.mlhd_conf,
        yolo_conf=args.yolo_conf,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
