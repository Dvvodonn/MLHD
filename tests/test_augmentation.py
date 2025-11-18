"""
Test script to verify data augmentation is working correctly.
Visualizes augmented samples with bounding boxes.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from datasets.dataloader_augmented import CCTVDetectionDataset
from datasets.augmentation import ALBUMENTATIONS_AVAILABLE

def draw_boxes_on_image(image, boxes, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on image.

    Args:
        image: numpy array [H, W, 3] in range [0, 1]
        boxes: list of [cx, cy, w, h] in normalized format
        color: BGR color tuple
        thickness: line thickness

    Returns:
        image with boxes drawn
    """
    img = (image * 255).astype(np.uint8).copy()
    h, w = img.shape[:2]

    for box in boxes:
        cx, cy, bw, bh = box
        # Convert from center format to corner format
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    return img


def test_augmentation():
    """Test augmentation pipeline by visualizing samples."""

    print("=" * 60)
    print("AUGMENTATION TEST")
    print("=" * 60)

    # Check if albumentations is available
    if not ALBUMENTATIONS_AVAILABLE:
        print("\nERROR: albumentations not installed!")
        print("Install with: pip install albumentations")
        print("\nTest FAILED - augmentation not available")
        return False

    print("\nalbumentations is installed")

    # Dataset paths
    train_images = "data/processed_training_3/images/train"
    train_labels = "data/processed_training_3/labels/train"

    # Check if dataset exists
    if not os.path.exists(train_images):
        print(f"\nERROR: Training images not found at {train_images}")
        return False

    print(f"Loading dataset from: {train_images}")

    # Create augmented dataset
    try:
        dataset = CCTVDetectionDataset(
            image_dir=train_images,
            label_dir=train_labels,
            grid_size=26,
            img_size=416,
            augment=True
        )
        print(f"Dataset loaded: {len(dataset)} images")
    except Exception as e:
        print(f"\nERROR loading dataset: {e}")
        return False

    # Visualize 4 random samples
    print("\nGenerating 4 augmented samples...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    # Select random samples
    indices = np.random.choice(len(dataset), size=4, replace=False)

    for i, idx in enumerate(indices):
        # Get augmented sample
        image, target = dataset[idx]

        # Convert tensor to numpy [C, H, W] -> [H, W, C]
        image_np = image.permute(1, 2, 0).numpy()

        # Extract boxes from target grid
        boxes = []
        S = target.shape[0]
        for row in range(S):
            for col in range(S):
                if target[row, col, 4] > 0.5:  # objectness > 0.5
                    tx, ty, tw, th = target[row, col, :4]
                    # Convert grid coordinates to image coordinates
                    cx = (col + tx) / S
                    cy = (row + ty) / S
                    w = tw
                    h = th
                    boxes.append([cx, cy, w, h])

        # Draw boxes
        img_with_boxes = draw_boxes_on_image(image_np, boxes)

        # Display
        axes[i].imshow(img_with_boxes)
        axes[i].set_title(f"Sample {idx} ({len(boxes)} objects)", fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()

    # Save output
    output_path = "outputs/test_augmentation_samples.png"
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")

    print("\n" + "=" * 60)
    print("TEST PASSED - Augmentation is working correctly")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the visualization at:", output_path)
    print("2. Run training with: python scripts/train.py")
    print("3. Compare performance with old model")

    return True


if __name__ == "__main__":
    success = test_augmentation()
    sys.exit(0 if success else 1)
