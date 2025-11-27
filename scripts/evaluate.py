#!/usr/bin/env python
"""
Comprehensive evaluation script for MLHD YOLO-like detector.
Evaluates model on validation set and generates detailed metrics.
Usage: python scripts/evaluate.py --checkpoint checkpoints/best.pt
"""
import sys
import os
from pathlib import Path
import argparse
from glob import glob

import torch
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.yolo_like import Model
from datasets.target_encoding import load_yolo_labels
from datasets.transforms import letterbox_image
from eval.evaluator import ObjectDetectionEvaluator, evaluate_predictions


def decode_predictions(pred_grid, conf_thresh=0.5, grid_size=13, img_size=416):
    """
    Decode grid predictions [S, S, 5] to bounding boxes.
    Returns: List of (x1, y1, x2, y2, conf)
    """
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
    """Apply Non-Maximum Suppression to remove duplicate detections."""
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


def infer_image(model, image_path, device, img_size=416, grid_size=13,
                conf_thresh=0.5, iou_thresh=0.5):
    """
    Run inference on a single image.
    Returns: (predictions, params) where predictions are in image coordinates
    """
    # Load and preprocess
    img_tensor, params = letterbox_image(image_path, (img_size, img_size))
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        pred = model(img_tensor)

    # Decode predictions
    boxes = decode_predictions(pred[0], conf_thresh, grid_size, img_size)

    # Apply NMS
    boxes = nms(boxes, iou_thresh)

    return boxes, params


def transform_boxes_to_original(boxes, params, img_size=416):
    """Transform boxes from letterbox coordinates back to original image coordinates."""
    if len(boxes) == 0:
        return []

    scale = params['scale']
    pad_w = params['pad_w']
    pad_h = params['pad_h']
    orig_w, orig_h = params['orig_wh']

    transformed = []
    for (x1, y1, x2, y2, conf) in boxes:
        # Remove padding
        x1_no_pad = x1 - pad_w
        y1_no_pad = y1 - pad_h
        x2_no_pad = x2 - pad_w
        y2_no_pad = y2 - pad_h

        # Unscale
        x1_orig = x1_no_pad / scale
        y1_orig = y1_no_pad / scale
        x2_orig = x2_no_pad / scale
        y2_orig = y2_no_pad / scale

        # Clip to image bounds
        x1_orig = max(0, min(x1_orig, orig_w))
        y1_orig = max(0, min(y1_orig, orig_h))
        x2_orig = max(0, min(x2_orig, orig_w))
        y2_orig = max(0, min(y2_orig, orig_h))

        transformed.append((x1_orig, y1_orig, x2_orig, y2_orig, conf))

    return transformed


def evaluate_dataset(model, image_dir, label_dir, device, img_size=416,
                     grid_size=13, conf_thresh=0.5, iou_thresh=0.5,
                     max_images=None):
    """
    Evaluate model on all images in a directory.

    Returns:
        (all_predictions, all_ground_truth, image_ids, stats)
    """
    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))

    if max_images:
        image_paths = image_paths[:max_images]

    all_predictions = []
    all_ground_truth = []
    image_ids = []
    stats = {
        'total': 0,
        'processed': 0,
        'errors': 0,
        'images_with_objects': 0,
        'total_pred_objects': 0,
        'total_gt_objects': 0,
        'processing_time': []
    }

    print(f"\nEvaluating {len(image_paths)} images...")
    print("-" * 70)

    for idx, img_path in enumerate(image_paths):
        stats['total'] += 1

        try:
            # Get label path
            basename = os.path.basename(img_path)
            label_name = os.path.splitext(basename)[0] + '.txt'
            label_path = os.path.join(label_dir, label_name)

            # Load ground truth
            gt_boxes = load_yolo_labels(label_path)
            if len(gt_boxes) > 0:
                stats['images_with_objects'] += 1
            stats['total_gt_objects'] += len(gt_boxes)

            # Convert GT boxes to pixel coordinates (from normalized)
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            gt_pixel = []
            for (cx, cy, bw, bh) in gt_boxes:
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                gt_pixel.append((x1, y1, x2, y2))

            # Run inference
            preds, params = infer_image(
                model, img_path, device, img_size, grid_size,
                conf_thresh, iou_thresh
            )

            # Transform predictions back to original image coordinates
            preds_orig = transform_boxes_to_original(preds, params, img_size)

            all_predictions.append(preds_orig)
            all_ground_truth.append(gt_pixel)
            image_ids.append(basename)

            stats['processed'] += 1
            stats['total_pred_objects'] += len(preds_orig)

            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(image_paths)} images...")

        except Exception as e:
            stats['errors'] += 1
            print(f"Error processing {img_path}: {e}")

    print("-" * 70)
    print(f"Evaluation complete: {stats['processed']} images processed, "
          f"{stats['errors']} errors")

    return all_predictions, all_ground_truth, image_ids, stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate MLHD model")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU threshold for NMS')
    parser.add_argument('--iou-eval', type=float, nargs='+',
                       default=[0.5, 0.75],
                       help='IoU thresholds for evaluation metrics')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Limit evaluation to N images (for testing)')
    parser.add_argument('--output', type=str, default='outputs/evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--grid-size', type=int, default=13,
                        help='Detection grid size S')
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
    print("\nLoading model...")
    model = Model(S=args.grid_size)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Evaluate on validation set
    val_images = ROOT / 'data/processed/training_5/images/val'
    val_labels = ROOT / 'data/processed/training_5/labels/val'

    all_preds, all_gt, image_ids, stats = evaluate_dataset(
        model, str(val_images), str(val_labels), device,
        grid_size=args.grid_size,
        conf_thresh=args.conf, iou_thresh=args.iou,
        max_images=args.max_images
    )

    # Compute metrics
    print("\n" + "="*70)
    print("COMPUTING EVALUATION METRICS")
    print("="*70)

    results = evaluate_predictions(
        all_preds, all_gt,
        iou_thresholds=args.iou_eval,
        verbose=True
    )

    # Compute mAP
    print("\n" + "="*70)
    print("COMPUTING mAP (Mean Average Precision)")
    print("="*70)

    evaluator = ObjectDetectionEvaluator(args.iou_eval)
    map_scores = evaluator.compute_map(all_preds, all_gt)

    print("\nmean Average Precision (mAP):")
    for iou_thresh, ap in map_scores.items():
        print(f"  IoU {iou_thresh}: {ap:.4f}")

    # Detailed statistics
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Total images evaluated: {stats['processed']}")
    print(f"Images with objects: {stats['images_with_objects']}")
    print(f"Total ground truth objects: {stats['total_gt_objects']}")
    print(f"Total predicted objects: {stats['total_pred_objects']}")

    if stats['total_gt_objects'] > 0:
        recall_ceiling = stats['total_pred_objects'] / stats['total_gt_objects']
        print(f"Recall ceiling (pred/gt): {recall_ceiling:.2%}")

    print(f"Average objects per image (GT): {stats['total_gt_objects'] / max(1, stats['processed']):.2f}")
    print(f"Average objects per image (Pred): {stats['total_pred_objects'] / max(1, stats['processed']):.2f}")

    # Per-image statistics
    print("\n" + "="*70)
    print("PER-IMAGE PERFORMANCE")
    print("="*70)

    per_image_stats = []
    iou_05_threshold = 0.5

    for i, (preds, gts, img_id) in enumerate(zip(all_preds, all_gt, image_ids)):
        tp, fp, fn, _ = evaluator.match_predictions_to_ground_truth(
            preds, gts, iou_05_threshold
        )
        precision, recall, f1 = evaluator.compute_precision_recall_f1(tp, fp, fn)
        per_image_stats.append({
            'image_id': img_id,
            'pred': len(preds),
            'gt': len(gts),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    # Sort by F1 score (descending)
    per_image_stats.sort(key=lambda x: x['f1'], reverse=True)

    print("\nTop 10 best performing images (by F1 score):")
    print(f"{'Image':<30} {'Pred':<5} {'GT':<5} {'TP':<4} {'FP':<4} {'FN':<4} {'F1':<8}")
    print("-" * 70)
    for stat in per_image_stats[:10]:
        print(f"{stat['image_id']:<30} {stat['pred']:<5} {stat['gt']:<5} "
              f"{stat['tp']:<4} {stat['fp']:<4} {stat['fn']:<4} "
              f"{stat['f1']:<8.4f}")

    print("\nBottom 10 worst performing images (by F1 score):")
    print(f"{'Image':<30} {'Pred':<5} {'GT':<5} {'TP':<4} {'FP':<4} {'FN':<4} {'F1':<8}")
    print("-" * 70)
    for stat in per_image_stats[-10:]:
        print(f"{stat['image_id']:<30} {stat['pred']:<5} {stat['gt']:<5} "
              f"{stat['tp']:<4} {stat['fp']:<4} {stat['fn']:<4} "
              f"{stat['f1']:<8.4f}")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary to text file
    with open(output_dir / 'evaluation_summary.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("MLHD MODEL EVALUATION SUMMARY\n")
        f.write("="*70 + "\n\n")

        f.write("Configuration:\n")
        f.write(f"  Checkpoint: {args.checkpoint}\n")
        f.write(f"  Confidence threshold: {args.conf}\n")
        f.write(f"  NMS IoU threshold: {args.iou}\n")
        f.write(f"  Evaluation IoU thresholds: {args.iou_eval}\n\n")

        f.write("Results:\n")
        for iou_thresh in sorted(results.keys()):
            metrics = results[iou_thresh]
            f.write(f"\nIoU {iou_thresh}:\n")
            f.write(f"  Precision: {metrics.precision:.4f}\n")
            f.write(f"  Recall:    {metrics.recall:.4f}\n")
            f.write(f"  F1 Score:  {metrics.f1:.4f}\n")
            f.write(f"  AP:        {metrics.ap:.4f}\n")
            f.write(f"  TP: {metrics.tp}, FP: {metrics.fp}, FN: {metrics.fn}\n")

        f.write(f"\nmean Average Precision (mAP):\n")
        for iou_thresh, ap in map_scores.items():
            f.write(f"  IoU {iou_thresh}: {ap:.4f}\n")

        f.write(f"\nDataset Statistics:\n")
        f.write(f"  Total images: {stats['processed']}\n")
        f.write(f"  Images with objects: {stats['images_with_objects']}\n")
        f.write(f"  Total GT objects: {stats['total_gt_objects']}\n")
        f.write(f"  Total predicted objects: {stats['total_pred_objects']}\n")

    print(f"\nResults saved to {output_dir / 'evaluation_summary.txt'}")

    # Save per-image results
    with open(output_dir / 'per_image_results.txt', 'w') as f:
        f.write(f"{'Image':<30} {'Pred':<5} {'GT':<5} {'TP':<4} {'FP':<4} "
                f"{'FN':<4} {'Precision':<12} {'Recall':<12} {'F1':<8}\n")
        f.write("-" * 100 + "\n")
        for stat in per_image_stats:
            f.write(f"{stat['image_id']:<30} {stat['pred']:<5} {stat['gt']:<5} "
                    f"{stat['tp']:<4} {stat['fp']:<4} {stat['fn']:<4} "
                    f"{stat['precision']:<12.4f} {stat['recall']:<12.4f} "
                    f"{stat['f1']:<8.4f}\n")

    print(f"Per-image results saved to {output_dir / 'per_image_results.txt'}")

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
