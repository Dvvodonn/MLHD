#!/usr/bin/env python
"""
Compare YOLOv5s and MLHD predictions on a single image with GT overlay.

Usage:
python -m eval.compare_models \
  --image data/processed/training_5/images/val/xxx.jpg \
  --mlhd-checkpoint checkpoints/best.pt \
  --mlhd-backbone 13x13 \
  --output outputs/compare.jpg
"""
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import sys
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.target_encoding import load_yolo_labels  # noqa: E402
from datasets.transforms import letterbox_image  # noqa: E402
from models.yolo_like import Model  # noqa: E402
from models.backbone import backbone_output_grid, backbone_input_resolution  # noqa: E402
from utils.device import get_best_device, describe_device  # noqa: E402


def _load_gt_pixels(label_path: Path, img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
    boxes = load_yolo_labels(str(label_path))
    pixels = []
    for cx, cy, w, h in boxes:
        cx_pix = cx * img_w
        cy_pix = cy * img_h
        w_pix = w * img_w
        h_pix = h * img_h
        x1 = int(cx_pix - w_pix / 2)
        y1 = int(cy_pix - h_pix / 2)
        x2 = int(cx_pix + w_pix / 2)
        y2 = int(cy_pix + h_pix / 2)
        pixels.append((x1, y1, x2, y2))
    return pixels


def _letterbox_to_orig(box, params, img_w, img_h):
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


def _load_yolov5(yolo_weights: str, device: str):
    """
    Load YOLOv5 model via torch.hub, avoiding name clashes between our local
    `models`/`utils` packages and YOLOv5's own `models`/`utils`.
    """

    # 1) Remove any already-imported local `models`/`utils` modules so that
    #    when torch.hub adds the YOLOv5 repo to sys.path, imports like
    #    `from models.common import ...` and `from utils import TryExcept`
    #    will resolve to YOLOv5's code instead of ours.
    for name in list(sys.modules):
        if (
            name == "models"
            or name.startswith("models.")
            or name == "utils"
            or name.startswith("utils.")
        ):
            sys.modules.pop(name, None)

    try:
        if yolo_weights is None:
            # Standard small YOLOv5 model
            model = torch.hub.load(
                "ultralytics/yolov5",
                "yolov5s",
                pretrained=True,
            )
        else:
            # Custom weights if you ever pass them
            model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=yolo_weights,
                pretrained=True,
            )
    except Exception as e:
        raise RuntimeError(
            "Failed to load YOLOv5 model. Ensure ultralytics/yolov5 is available or provide --yolo-weights."
        ) from e

    model.to(device)
    model.eval()
    return model


def draw_boxes(img, boxes, color, label_prefix: str, thickness: int = 2):
    for idx, (x1, y1, x2, y2, score) in enumerate(boxes, start=1):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        cv2.putText(
            img,
            f"{label_prefix} {score:.2f}",
            (int(x1) + 2, max(14, int(y1) + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            max(2, thickness - 1),
            cv2.LINE_AA,
        )


def draw_gt(img, boxes):
    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            img,
            f"GT {idx}",
            (x1 + 2, max(14, y1 + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )


def _draw_legend(img):
    entries = [
        ((0, 200, 0), "Ground truth"),
        ((255, 0, 0), "YOLOv5s"),
        ((0, 0, 255), "MLHD"),
    ]
    pad = 10
    box_w, box_h = 20, 16
    x0, y0 = 12, 14
    total_h = len(entries) * 28 + pad
    bg_end = (x0 + 230, y0 + total_h)
    cv2.rectangle(img, (x0 - 8, y0 - 10), bg_end, (0, 0, 0), thickness=-1)
    cv2.rectangle(img, (x0 - 8, y0 - 10), bg_end, (255, 255, 255), thickness=2)

    for idx, (color, text) in enumerate(entries):
        y = y0 + idx * 28
        cv2.rectangle(img, (x0, y), (x0 + box_w, y + box_h), color, thickness=-1)
        cv2.putText(
            img,
            text,
            (x0 + box_w + 12, y + box_h + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def visualize(image_path: str, mlhd_ckpt: str, backbone: str, mlhd_conf: float, yolo_conf: float, yolo_weights: Optional[str], output: Optional[str]):
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    label_path = None
    if "images" in img_path.parts:
        idx = img_path.parts.index("images")
        parts = list(img_path.parts)
        parts[idx] = "labels"
        label_path = Path(*parts).with_suffix(".txt")
    else:
        label_path = img_path.with_suffix(".txt")

    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {img_path}")
    img_h, img_w = img.shape[:2]

    device = get_best_device()
    print(f"Using device: {describe_device(device)}")

    # MLHD inference
    grid = backbone_output_grid(backbone)
    img_size = backbone_input_resolution(backbone)
    mlhd = Model(S=grid, backbone_name=backbone)
    ckpt = torch.load(mlhd_ckpt, map_location=device)
    mlhd.load_state_dict(ckpt["model"])
    mlhd.to(device).eval()

    img_tensor, params = letterbox_image(str(img_path), target_size=(img_size, img_size))
    with torch.no_grad():
        mlhd_pred = mlhd(img_tensor.unsqueeze(0).to(device))[0].cpu()
    mlhd_boxes = _decode_mlhd(mlhd_pred, mlhd_conf, grid, params, img_w, img_h)
    print(f"MLHD boxes: {len(mlhd_boxes)} (conf>={mlhd_conf})")

    # YOLOv5 inference
    yolo_model = _load_yolov5(yolo_weights, device)
    yolo_model.conf = yolo_conf
    with torch.no_grad():
        results = yolo_model(img[..., ::-1])  # BGR -> RGB for yolov5
    yolo_boxes = []
    for *xyxy, conf, _cls in results.xyxy[0].cpu().tolist():
        x1, y1, x2, y2 = map(int, xyxy)
        yolo_boxes.append((x1, y1, x2, y2, conf))
    print(f"YOLOv5 boxes: {len(yolo_boxes)} (conf>={yolo_conf})")

    # Draw (order so YOLOv5 boxes remain visible)
    draw_gt(img, _load_gt_pixels(label_path, img_w, img_h))
    draw_boxes(img, mlhd_boxes, (0, 0, 255), "M", thickness=2)
    draw_boxes(img, yolo_boxes, (255, 0, 0), "Y", thickness=4)
    _draw_legend(img)

    if output is None:
        out_path = ROOT / "outputs" / f"{img_path.stem}_compare.jpg"
    else:
        out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"Saved: {out_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="Compare YOLOv5s and MLHD predictions on one image.")
    ap.add_argument("--image", required=True, help="Path to image file.")
    ap.add_argument("--mlhd-checkpoint", required=True, help="Path to MLHD checkpoint (state dict).")
    ap.add_argument("--mlhd-backbone", default="13x13", choices=["13x13", "8x8"], help="Backbone used for MLHD checkpoint.")
    ap.add_argument("--mlhd-conf", type=float, default=0.3, help="Confidence threshold for MLHD predictions.")
    ap.add_argument("--yolo-conf", type=float, default=0.25, help="Confidence threshold for YOLOv5s predictions.")
    ap.add_argument("--yolo-weights", default=None, help="Optional path to YOLOv5 weights. If omitted, loads yolov5s pretrained.")
    ap.add_argument("--output", default=None, help="Optional output path.")
    return ap.parse_args()


def main():
    args = parse_args()
    visualize(
        image_path=args.image,
        mlhd_ckpt=args.mlhd_checkpoint,
        backbone=args.mlhd_backbone,
        mlhd_conf=args.mlhd_conf,
        yolo_conf=args.yolo_conf,
        yolo_weights=args.yolo_weights,
        output=args.output,
    )


if __name__ == "__main__":
    main()
