#!/usr/bin/env python
"""
Video inference script for MLHD YOLO-like detector.
Usage: python scripts/infer_video.py --video path/to/video.mp4
"""
import sys
import os
from pathlib import Path
import argparse
import time

import torch
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.yolo_like import Model
from datasets.transforms import letterbox_image


def decode_predictions(pred_grid, conf_thresh=0.5, grid_size=13, img_size=416):
    """Decode grid predictions [S, S, 5] to bounding boxes."""
    S = grid_size
    boxes = []

    for j in range(S):
        for i in range(S):
            tx = pred_grid[j, i, 0].item()
            ty = pred_grid[j, i, 1].item()
            tw = pred_grid[j, i, 2].item()
            th = pred_grid[j, i, 3].item()
            obj_conf = pred_grid[j, i, 4].item()

            if obj_conf < conf_thresh:
                continue

            cx = (i + tx) / S
            cy = (j + ty) / S
            w = tw
            h = th

            cx_pix = cx * img_size
            cy_pix = cy * img_size
            w_pix = w * img_size
            h_pix = h * img_size

            x1 = cx_pix - w_pix / 2
            y1 = cy_pix - h_pix / 2
            x2 = cx_pix + w_pix / 2
            y2 = cy_pix + h_pix / 2

            boxes.append((x1, y1, x2, y2, obj_conf))

    return boxes


def nms(boxes, iou_thresh=0.5):
    """Non-maximum suppression to remove duplicate detections."""
    if len(boxes) == 0:
        return []

    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []

    while len(boxes) > 0:
        best = boxes[0]
        keep.append(best)
        boxes = boxes[1:]

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

    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    inter_area = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def letterbox_frame(frame, target_size=(416, 416)):
    """Apply letterbox preprocessing to a video frame."""
    h, w = frame.shape[:2]
    target_h, target_w = target_size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

    frame_tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0

    params = {
        'scale': scale,
        'pad_w': pad_w,
        'pad_h': pad_h,
        'orig_wh': (w, h)
    }

    return frame_tensor, params


def transform_boxes_to_original(boxes, params, img_size):
    """Transform boxes from letterbox coordinates back to original image."""
    scale = params['scale']
    pad_w = params['pad_w']
    pad_h = params['pad_h']
    orig_w, orig_h = params['orig_wh']

    transformed = []
    for (x1, y1, x2, y2, conf) in boxes:
        x1_orig = (x1 - pad_w) / scale
        y1_orig = (y1 - pad_h) / scale
        x2_orig = (x2 - pad_w) / scale
        y2_orig = (y2 - pad_h) / scale

        x1_orig = max(0, min(x1_orig, orig_w))
        y1_orig = max(0, min(y1_orig, orig_h))
        x2_orig = max(0, min(x2_orig, orig_w))
        y2_orig = max(0, min(y2_orig, orig_h))

        transformed.append((x1_orig, y1_orig, x2_orig, y2_orig, conf))

    return transformed


def draw_boxes(frame, boxes):
    """Draw bounding boxes on frame."""
    for (x1, y1, x2, y2, conf) in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


def process_video(model, video_path, output_path, conf_thresh=0.5, iou_thresh=0.5,
                  grid_size=13, img_size=416, device='cpu', show_fps=True):
    """Process video and save output with detections."""

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    model.eval()

    frame_count = 0
    start_time = time.time()

    print("\nProcessing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.time()

        frame_tensor, params = letterbox_frame(frame, target_size=(img_size, img_size))
        frame_tensor = frame_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(frame_tensor)

        pred_grid = pred[0]
        boxes = decode_predictions(pred_grid, conf_thresh, grid_size, img_size)
        boxes = nms(boxes, iou_thresh)
        boxes_orig = transform_boxes_to_original(boxes, params, img_size)

        frame_with_boxes = draw_boxes(frame.copy(), boxes_orig)

        if show_fps:
            frame_time = time.time() - frame_start
            fps_text = f"FPS: {1/frame_time:.1f} | Objects: {len(boxes_orig)}"
            cv2.putText(frame_with_boxes, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(frame_with_boxes)

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_avg = frame_count / elapsed
            progress = (frame_count / total_frames) * 100
            print(f"  Frame {frame_count}/{total_frames} ({progress:.1f}%) - {fps_avg:.1f} FPS")

    cap.release()
    out.release()

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time

    print(f"\nProcessing complete!")
    print(f"  Processed {frame_count} frames in {total_time:.1f}s")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run MLHD detector on video")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output video (default: input_output.mp4)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='NMS IoU threshold')
    parser.add_argument('--grid-size', type=int, default=13,
                       help='Grid size S')
    parser.add_argument('--img-size', type=int, default=416,
                       help='Image size')
    parser.add_argument('--no-fps', action='store_true',
                       help='Disable FPS display on video')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = Model(S=args.grid_size)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    print("Model loaded successfully")

    if args.output is None:
        video_path = Path(args.video)
        args.output = str(video_path.parent / f"{video_path.stem}_output.mp4")

    process_video(
        model, args.video, args.output,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        grid_size=args.grid_size,
        img_size=args.img_size,
        device=device,
        show_fps=not args.no_fps
    )


if __name__ == '__main__':
    main()
