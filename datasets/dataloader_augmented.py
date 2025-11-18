import os
from glob import glob
from typing import Tuple
import torch
from torch.utils.data import Dataset
import cv2

from .target_encoding import load_yolo_labels, encode_targets
from .transforms import letterbox_image, transform_boxes
from .augmentation import get_training_augmentation, apply_augmentation


class CCTVDetectionDataset(Dataset):
    """
    CCTV object detection dataset for YOLO-like training with augmentation support.
    Loads images and YOLO format labels, applies optional augmentation, letterbox resize, encodes to grid targets.
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        grid_size: int = 26,
        img_size: int = 416,
        augment: bool = False
    ):
        """
        Args:
            image_dir: Directory containing .jpg images
            label_dir: Directory containing .txt label files (YOLO format)
            grid_size: Grid dimension S (default 26 for 416x416 images)
            img_size: Target image size (default 416)
            augment: Whether to apply data augmentation (training only)
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.grid_size = grid_size
        self.img_size = img_size
        self.augment = augment

        # Get augmentation pipeline if requested
        self.augmentation = get_training_augmentation() if augment else None
        if augment and self.augmentation is not None:
            print(f"  Data augmentation enabled for {image_dir}")
        elif augment and self.augmentation is None:
            print(f"  Warning: Augmentation requested but albumentations not available")

        self.image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))

        if len(self.image_paths) == 0:
            raise ValueError(f"No .jpg images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _get_label_path(self, image_path: str) -> str:
        """Get corresponding label file path for an image."""
        basename = os.path.basename(image_path)
        label_name = os.path.splitext(basename)[0] + '.txt'
        return os.path.join(self.label_dir, label_name)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and process a single sample.

        Returns:
            image: Tensor [3, img_size, img_size] normalized to [0, 1]
            target: Tensor [S, S, 5] where last dim is [tx, ty, tw, th, obj]
        """
        img_path = self.image_paths[idx]
        label_path = self._get_label_path(img_path)

        # Load boxes in YOLO format (normalized)
        boxes = load_yolo_labels(label_path)

        # Apply augmentation BEFORE letterbox resize (if enabled)
        if self.augment and self.augmentation is not None and len(boxes) > 0:
            # Load raw image for augmentation
            img_raw = cv2.imread(img_path)
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

            # Apply augmentation
            img_aug, boxes_aug = apply_augmentation(img_raw, boxes, self.augmentation)

            # Convert augmented image to letterbox format
            # Save temporarily and use letterbox_image
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR))

            image, params = letterbox_image(tmp_path, target_size=(self.img_size, self.img_size))
            os.remove(tmp_path)

            # Transform augmented boxes
            transformed_boxes = transform_boxes(boxes_aug, params)
        else:
            # No augmentation - standard pipeline
            image, params = letterbox_image(img_path, target_size=(self.img_size, self.img_size))
            transformed_boxes = transform_boxes(boxes, params)

        # Encode to grid targets
        target = encode_targets(transformed_boxes, grid_size=self.grid_size)

        return image, target
