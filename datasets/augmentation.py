"""
Data augmentation for CCTV person detection.
Uses Albumentations for bbox-aware augmentations.
"""
try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not installed. Run: pip install albumentations")


def get_training_augmentation():
    """
    Create augmentation pipeline for training.
    Designed specifically for CCTV person detection scenarios.

    Returns:
        Albumentations Compose object or None if not available
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return None

    return A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),

        # Color/lighting augmentations (common in CCTV)
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5
        ),

        # Simulate different lighting conditions
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        # CCTV-specific augmentations
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # Camera noise
        A.MotionBlur(blur_limit=5, p=0.3),  # Motion/camera shake

        # Contrast enhancement (helps with low-quality CCTV)
        A.CLAHE(clip_limit=2.0, p=0.2),

        # Simulate compression artifacts
        A.ImageCompression(quality_lower=75, quality_upper=100, p=0.2),

    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_area=0,
        min_visibility=0.3
    ))


def get_validation_augmentation():
    """
    No augmentation for validation (identity transform).
    """
    return None


def apply_augmentation(image, boxes, augmentation):
    """
    Apply augmentation to image and bounding boxes.

    Args:
        image: numpy array [H, W, 3] in range [0, 255]
        boxes: list of [cx, cy, w, h] in normalized YOLO format
        augmentation: Albumentations Compose object or None

    Returns:
        augmented_image: numpy array [H, W, 3]
        augmented_boxes: list of [cx, cy, w, h]
    """
    if augmentation is None or not ALBUMENTATIONS_AVAILABLE:
        return image, boxes

    if len(boxes) == 0:
        # No boxes, apply image-only augmentation
        augmented = augmentation(image=image, bboxes=[], class_labels=[])
        return augmented['image'], []

    # Clamp boxes to valid [0, 1] range and convert to lists
    # Add small epsilon to avoid edge cases during YOLO->corner conversion
    eps = 1e-6
    clamped_boxes = []
    for box in boxes:
        cx, cy, w, h = box

        # Clamp with epsilon margin
        cx = max(w/2 + eps, min(1.0 - w/2 - eps, cx))
        cy = max(h/2 + eps, min(1.0 - h/2 - eps, cy))
        w = max(eps, min(1.0, w))
        h = max(eps, min(1.0, h))

        clamped_boxes.append([cx, cy, w, h])

    # Create dummy class labels (all 0 for person class)
    class_labels = [0] * len(clamped_boxes)

    # Apply augmentation
    augmented = augmentation(
        image=image,
        bboxes=clamped_boxes,
        class_labels=class_labels
    )

    return augmented['image'], augmented['bboxes']
