#!/usr/bin/env python
"""
Inference script for MLHD YOLO-like detector.
Usage: python scripts/infer.py --checkpoint checkpoints/best.pt --image path/to/image.jpg
"""
import sys
import os
from pathlib import Path
import argparse

import torch
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.yolo_like import Model
from datasets.transforms import letterbox_image


def decode_predictions(pred_grid, conf_thresh=0.5, grid_size=26, img_size=416):
    """
    Decode grid predictions [S, S, 5] to bounding boxes.

    Args:
        pred_grid: [S, S, 5] tensor with [tx, ty, tw, th, obj]
        conf_thresh: Confidence threshold for filtering
        grid_size: Grid size S
        img_size: Image size (416x416)

    Returns:
        List of boxes: [(x1, y1, x2, y2, conf), ...]
    """
    S = grid_size
    boxes = []

    for j in range(S):
        for i in range(S):
            # Model outputs [tx, ty, tw, th, obj] (obj is last!)
            tx = pred_grid[j, i, 0].item()
            ty = pred_grid[j, i, 1].item()
            tw = pred_grid[j, i, 2].item()
            th = pred_grid[j, i, 3].item()
            obj_conf = pred_grid[j, i, 4].item()

            if obj_conf < conf_thresh:
                continue

            # Convert to normalized center coordinates
            cx = (i + tx) / S
            cy = (j + ty) / S
            w = tw
            h = th

            # Convert to pixel coordinates
            cx_pix = cx * img_size
            cy_pix = cy * img_size
            w_pix = w * img_size
            h_pix = h * img_size

            # Convert to corner format
            x1 = cx_pix - w_pix / 2
            y1 = cy_pix - h_pix / 2
            x2 = cx_pix + w_pix / 2
            y2 = cy_pix + h_pix / 2

            boxes.append((x1, y1, x2, y2, obj_conf))

    return boxes


def nms(boxes, iou_thresh=0.5):
    """
    Non-maximum suppression to remove duplicate detections.

    Args:
        boxes: List of (x1, y1, x2, y2, conf)
        iou_thresh: IoU threshold for suppression

    Returns:
        Filtered list of boxes
    """
    if len(boxes) == 0:
        return []

    # Sort by confidence descending
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []

    while len(boxes) > 0:
        # Keep highest confidence box
        best = boxes[0]
        keep.append(best)
        boxes = boxes[1:]

        # Remove boxes with high IoU with best box
        filtered = []
        for box in boxes:
            iou = compute_iou(best, box)
            if iou < iou_thresh:
                filtered.append(box)
        boxes = filtered

    return keep


def compute_iou(box1, box2):
    """Compute IoU between two boxes (x1, y1, x2, y2, conf)."""
    x1_1, y1_1, x2_1, y2_1, _ = box1
    x1_2, y1_2, x2_2, y2_2, _ = box2

    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    inter_area = (x2_i - x1_i) * (y2_i - y1_i)

    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def draw_boxes(image, boxes, output_path=None):
    """
    Draw bounding boxes on image.

    Args:
        image: OpenCV image (BGR)
        boxes: List of (x1, y1, x2, y2, conf)
        output_path: Optional path to save output

    Returns:
        Image with boxes drawn
    """
    img_drawn = image.copy()

    for (x1, y1, x2, y2, conf) in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw rectangle
        cv2.rectangle(img_drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw confidence
        label = f"{conf:.2f}"
        cv2.putText(img_drawn, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if output_path:
        cv2.imwrite(output_path, img_drawn)
        print(f"Saved to: {output_path}")

    return img_drawn


def infer_single_image(model, image_path, conf_thresh=0.5, iou_thresh=0.5,
                       grid_size=26, img_size=416, device='cpu'):
    """
    Run inference on a single image.

    Args:
        model: Trained Model instance
        image_path: Path to input image
        conf_thresh: Confidence threshold
        iou_thresh: NMS IoU threshold
        grid_size: Grid size S
        img_size: Image size
        device: torch device

    Returns:
        boxes: List of (x1, y1, x2, y2, conf)
        orig_image: Original image (for visualization)
        params: Letterbox parameters (for coordinate transformation)
    """
    model.eval()

    # Load and preprocess image
    img_tensor, params = letterbox_image(str(image_path), target_size=(img_size, img_size))
    img_tensor = img_tensor.unsqueeze(0).to(device)  # [1, 3, 416, 416]

    # Run inference
    with torch.no_grad():
        pred = model(img_tensor)  # [1, S, S, 5]

    # Decode predictions
    pred_grid = pred[0]  # [S, S, 5]
    boxes = decode_predictions(pred_grid, conf_thresh, grid_size, img_size)

    # Apply NMS
    boxes = nms(boxes, iou_thresh)

    # Load original image for visualization
    orig_image = cv2.imread(str(image_path))

    # Transform boxes back to original image coordinates
    boxes_orig = transform_boxes_to_original(boxes, params, img_size)

    return boxes_orig, orig_image, params


def transform_boxes_to_original(boxes, params, img_size):
    """Transform boxes from letterbox coordinates back to original image."""
    scale = params['scale']
    pad_w = params['pad_w']
    pad_h = params['pad_h']
    orig_w, orig_h = params['orig_wh']

    transformed = []
    for (x1, y1, x2, y2, conf) in boxes:
        # Remove padding and scale back
        x1_orig = (x1 - pad_w) / scale
        y1_orig = (y1 - pad_h) / scale
        x2_orig = (x2 - pad_w) / scale
        y2_orig = (y2 - pad_h) / scale

        # Clip to original image bounds
        x1_orig = max(0, min(x1_orig, orig_w))
        y1_orig = max(0, min(y1_orig, orig_h))
        x2_orig = max(0, min(x2_orig, orig_w))
        y2_orig = max(0, min(y2_orig, orig_h))

        transformed.append((x1_orig, y1_orig, x2_orig, y2_orig, conf))

    return transformed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output image (default: input_pred.jpg)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='NMS IoU threshold')
    parser.add_argument('--grid-size', type=int, default=26,
                       help='Grid size S')
    parser.add_argument('--img-size', type=int, default=416,
                       help='Image size')
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = Model(S=args.grid_size)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    print("Model loaded successfully")

    # Run inference
    print(f"\nRunning inference on: {args.image}")
    boxes, orig_image, params = infer_single_image(
        model, args.image,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        grid_size=args.grid_size,
        img_size=args.img_size,
        device=device
    )

    print(f"Detected {len(boxes)} objects")
    for i, (x1, y1, x2, y2, conf) in enumerate(boxes):
        print(f"  Box {i+1}: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), conf={conf:.3f}")

    # Draw and save
    if args.output is None:
        img_path = Path(args.image)
        args.output = str(img_path.parent / f"{img_path.stem}_pred{img_path.suffix}")

    draw_boxes(orig_image, boxes, args.output)
    print(f"\nInference complete!")


if __name__ == '__main__':
    main()
