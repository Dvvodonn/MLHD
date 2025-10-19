import os
from glob import glob
from typing import Tuple
import torch
from torch.utils.data import Dataset

from .target_encoding import load_yolo_labels, encode_targets
from .transforms import letterbox_image, transform_boxes


class CCTVDetectionDataset(Dataset):
    """
    CCTV object detection dataset for YOLO-like training.
    Loads images and YOLO format labels, applies letterbox resize, encodes to grid targets.
    """

    def __init__(self, image_dir: str, label_dir: str, grid_size: int = 13, img_size: int = 416):
        """
        Args:
            image_dir: Directory containing .jpg images
            label_dir: Directory containing .txt label files (YOLO format)
            grid_size: Grid dimension S (default 13 for 416x416 images)
            img_size: Target image size (default 416)
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.grid_size = grid_size
        self.img_size = img_size

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
            target: Tensor [S, S, 5] where last dim is [obj, tx, ty, tw, th]
        """
        img_path = self.image_paths[idx]
        label_path = self._get_label_path(img_path)

        boxes = load_yolo_labels(label_path)

        image, params = letterbox_image(img_path, target_size=(self.img_size, self.img_size))

        transformed_boxes = transform_boxes(boxes, params)

        target = encode_targets(transformed_boxes, grid_size=self.grid_size)

        return image, target
