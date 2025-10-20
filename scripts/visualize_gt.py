#!/usr/bin/env python
"""
Visualize ground truth boxes and predictions on images.
Usage: python scripts/visualize_gt.py --image path/to/image.jpg --checkpoint checkpoints/best.pt --conf 0.5
"""
import argparse
import sys
from pathlib import Path
import cv2
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.yolo_like import Model
from datasets.transforms import letterbox_image


def visualize_ground_truth(image_path, checkpoint_path=None, conf_thresh=0.5, output_path=None):
    """Draw ground truth bounding boxes and predictions on image."""

    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    img_h, img_w = img.shape[:2]
    print(f"Image size: {img_w}x{img_h}")

    # Find corresponding label file
    image_path = Path(image_path)

    # Try to find label file
    # Assume structure: images/[train|val]/xxx.jpg -> labels/[train|val]/xxx.txt
    label_path = None
    if 'images' in str(image_path):
        # Replace 'images' with 'labels' and .jpg with .txt
        label_path = Path(str(image_path).replace('/images/', '/labels/')).with_suffix('.txt')

    if label_path is None or not label_path.exists():
        print(f"Error: Could not find label file at {label_path}")
        return

    print(f"Label file: {label_path}")

    # Read YOLO format labels
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append((cx, cy, w, h))

    print(f"Ground truth boxes: {len(boxes)}")

    # Get predictions if checkpoint provided
    pred_boxes = []
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"\nLoading model from: {checkpoint_path}")

        # Device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

        # Load model
        model = Model(S=26)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()

        # Run inference
        img_tensor, params = letterbox_image(str(image_path), target_size=(416, 416))
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_tensor)

        # Decode predictions
        pred_grid = pred[0]
        S = 26
        for j in range(S):
            for i in range(S):
                tx = pred_grid[j, i, 0].item()
                ty = pred_grid[j, i, 1].item()
                tw = pred_grid[j, i, 2].item()
                th = pred_grid[j, i, 3].item()
                obj_conf = pred_grid[j, i, 4].item()

                if obj_conf < conf_thresh:
                    continue

                # Convert to normalized coords
                cx = (i + tx) / S
                cy = (j + ty) / S
                w = tw
                h = th

                # Convert to letterbox pixel coords
                cx_pix = cx * 416
                cy_pix = cy * 416
                w_pix = w * 416
                h_pix = h * 416

                x1 = cx_pix - w_pix / 2
                y1 = cy_pix - h_pix / 2
                x2 = cx_pix + w_pix / 2
                y2 = cy_pix + h_pix / 2

                # Transform to original image coords
                scale = params['scale']
                pad_w = params['pad_w']
                pad_h = params['pad_h']

                x1_orig = (x1 - pad_w) / scale
                y1_orig = (y1 - pad_h) / scale
                x2_orig = (x2 - pad_w) / scale
                y2_orig = (y2 - pad_h) / scale

                x1_orig = max(0, min(x1_orig, img_w))
                y1_orig = max(0, min(y1_orig, img_h))
                x2_orig = max(0, min(x2_orig, img_w))
                y2_orig = max(0, min(y2_orig, img_h))

                pred_boxes.append((x1_orig, y1_orig, x2_orig, y2_orig, obj_conf))

        print(f"Predicted boxes: {len(pred_boxes)}")

    # Draw ground truth boxes (GREEN)
    for i, (cx, cy, w, h) in enumerate(boxes):
        # Convert normalized to pixel coordinates
        cx_pix = cx * img_w
        cy_pix = cy * img_h
        w_pix = w * img_w
        h_pix = h * img_h

        # Convert to corner format
        x1 = int(cx_pix - w_pix/2)
        y1 = int(cy_pix - h_pix/2)
        x2 = int(cx_pix + w_pix/2)
        y2 = int(cy_pix + h_pix/2)

        # Draw rectangle (green for ground truth)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label = f"GT {i+1}"
        cv2.putText(img, label, (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        print(f"  GT Box {i+1}: ({x1}, {y1}, {x2}, {y2}) - size: {w_pix:.0f}x{h_pix:.0f}")

    # Draw prediction boxes (RED)
    for i, (x1, y1, x2, y2, conf) in enumerate(pred_boxes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw rectangle (red for predictions)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw label
        label = f"P {conf:.2f}"
        cv2.putText(img, label, (x1, y2+15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        print(f"  Pred Box {i+1}: ({x1}, {y1}, {x2}, {y2}) - conf: {conf:.3f}")

    # Save output
    if output_path is None:
        img_name = image_path.stem
        suffix = "_compare" if checkpoint_path else "_gt"
        output_path = ROOT / 'outputs' / f"{img_name}{suffix}.jpg"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), img)
    print(f"\nSaved to: {output_path}")

    if checkpoint_path:
        print("\nLegend:")
        print("  GREEN boxes = Ground Truth (correct labels)")
        print("  RED boxes = Predictions (model output)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (optional, for predictions)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold for predictions (default: 0.5)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output image (default: outputs/{name}_compare.jpg)')
    args = parser.parse_args()

    visualize_ground_truth(args.image, args.checkpoint, args.conf, args.output)


if __name__ == '__main__':
    main()
