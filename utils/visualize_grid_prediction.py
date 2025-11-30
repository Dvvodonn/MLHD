#!/usr/bin/env python
"""
Visualize a single model prediction against ground truth with grid overlay.

Example:
python -m utils.visualize_grid_prediction \
    --image data/processed/training_5/images/val/xxx.jpg \
    --checkpoint checkpoints/best.pt \
    --backbone 13x13 \
    --conf 0.3
"""
import argparse
import sys
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.target_encoding import load_yolo_labels  # noqa: E402
from datasets.transforms import letterbox_image, transform_boxes  # noqa: E402
from models.yolo_like import Model  # noqa: E402
from models.backbone import backbone_output_grid, backbone_input_resolution  # noqa: E402
from utils.device import get_best_device, describe_device  # noqa: E402


def _default_label_path(image_path: Path) -> Path:
    if "images" in image_path.parts:
        idx = image_path.parts.index("images")
        label_parts = list(image_path.parts)
        label_parts[idx] = "labels"
        return Path(*label_parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _draw_grid(img, cells: int, highlight: Tuple[int, int]) -> None:
    h, w = img.shape[:2]
    step_x = w / cells
    step_y = h / cells
    for i in range(1, cells):
        x = int(i * step_x)
        y = int(i * step_y)
        cv2.line(img, (x, 0), (x, h), (200, 200, 200), 1, cv2.LINE_AA)
        cv2.line(img, (0, y), (w, y), (200, 200, 200), 1, cv2.LINE_AA)
    j, i = highlight
    x1, y1 = int(i * step_x), int(j * step_y)
    x2, y2 = int((i + 1) * step_x), int((j + 1) * step_y)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 3)
    cv2.putText(
        img,
        f"cell ({i},{j})",
        (x1 + 4, y1 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 200, 0),
        2,
        cv2.LINE_AA,
    )


def _draw_legend(img) -> None:
    entries = [
        ((0, 200, 0), "Ground truth"),
        ((0, 0, 255), "Prediction"),
        ((255, 200, 0), "Cell Containing Centre"),
        ((200, 200, 200), "Grid lines"),
    ]
    pad = 8
    box_w, box_h = 16, 10
    x0, y0 = 10, 10

    # Background for readability
    total_h = len(entries) * 18 + pad
    bg_end = (x0 + 200, y0 + total_h)
    cv2.rectangle(img, (x0 - 6, y0 - 6), bg_end, (0, 0, 0), thickness=-1)
    cv2.rectangle(img, (x0 - 6, y0 - 6), bg_end, (255, 255, 255), thickness=1)

    for idx, (color, label) in enumerate(entries):
        y = y0 + idx * 18
        cv2.rectangle(img, (x0, y), (x0 + box_w, y + box_h), color, thickness=-1)
        cv2.putText(
            img,
            label,
            (x0 + box_w + 8, y + box_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _load_ground_truth(label_path: Path, params, img_size: int) -> List[Tuple[int, int, int, int]]:
    boxes = load_yolo_labels(str(label_path))
    boxes = transform_boxes(boxes, params)
    px = []
    for cx, cy, w, h in boxes:
        cx_pix = cx * img_size
        cy_pix = cy * img_size
        w_pix = w * img_size
        h_pix = h * img_size
        x1 = int(cx_pix - w_pix / 2)
        y1 = int(cy_pix - h_pix / 2)
        x2 = int(cx_pix + w_pix / 2)
        y2 = int(cy_pix + h_pix / 2)
        px.append((x1, y1, x2, y2))
    return px


def _decode_top_cell(pred: torch.Tensor, conf_thresh: float, S: int):
    obj = pred[..., 4]
    flat_idx = torch.argmax(obj).item()
    j, i = divmod(flat_idx, S)
    score = obj[j, i].item()
    tx, ty, tw, th = pred[j, i, 0:4].tolist()
    cx = (i + tx) / S
    cy = (j + ty) / S
    w = tw
    h = th
    return (i, j), score, (cx, cy, w, h)


def visualize(image_path: str, checkpoint_path: str, backbone: str, conf_thresh: float, output: Optional[str]):
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    label_path = _default_label_path(img_path)
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    grid = backbone_output_grid(backbone)
    img_size = backbone_input_resolution(backbone)

    img_tensor, params = letterbox_image(str(img_path), target_size=(img_size, img_size))
    vis_img = (
        img_tensor.permute(1, 2, 0).numpy() * 255.0
    ).clip(0, 255).astype(np.uint8)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    device = get_best_device()
    print(f"Using device: {describe_device(device)}")

    model = Model(S=grid, backbone_name=backbone)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        preds = model(img_tensor.unsqueeze(0).to(device))[0].cpu()

    cell, score, box = _decode_top_cell(preds, conf_thresh, grid)
    cx, cy, w, h = box
    if score < conf_thresh:
        print(f"Highest objectness {score:.3f} is below conf threshold {conf_thresh}")
    print(f"Top cell: {cell} with obj {score:.3f}")

    px = _load_ground_truth(label_path, params, img_size)
    for (x1, y1, x2, y2) in px:
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(vis_img, "GT", (x1 + 2, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2, cv2.LINE_AA)

    # predicted box
    x1 = int(max(0.0, (cx - w / 2) * img_size))
    y1 = int(max(0.0, (cy - h / 2) * img_size))
    x2 = int(min(img_size, (cx + w / 2) * img_size))
    y2 = int(min(img_size, (cy + h / 2) * img_size))
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(
        vis_img,
        f"Pred {score:.2f}",
        (x1 + 2, y2 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    _draw_grid(vis_img, grid, (cell[1], cell[0]))
    _draw_legend(vis_img)

    if output is None:
        out_path = ROOT / "outputs" / f"{img_path.stem}_grid.jpg"
    else:
        out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis_img)
    print(f"Saved visualization to: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize prediction grid cell and ground truth.")
    parser.add_argument("--image", required=True, help="Path to dataset image.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--backbone", default="13x13", choices=["13x13", "8x8"], help="Backbone used for the checkpoint.")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold for highlighting prediction.")
    parser.add_argument("--output", default=None, help="Optional output path for the rendered image.")
    return parser.parse_args()


def main():
    args = parse_args()
    visualize(args.image, args.checkpoint, args.backbone, args.conf, args.output)


if __name__ == "__main__":
    main()
