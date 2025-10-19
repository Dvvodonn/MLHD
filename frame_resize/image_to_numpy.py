

"""
Utility to convert images (e.g., JPEG, PNG) into NumPy arrays suitable for MLHD preprocessing.
"""

import cv2
import numpy as np
from typing import Union, List

def image_to_numpy(path: str, normalize: bool = True, color_mode: str = "rgb") -> np.ndarray:
    """
    Read an image from disk and return it as a NumPy array.

    Args:
        path (str): Path to the image file.
        normalize (bool): If True, divide by 255 to map values to [0,1].
        color_mode (str): 'rgb' (default) or 'bgr' depending on model convention.

    Returns:
        np.ndarray: Image array of shape (H, W, 3), dtype=float32 if normalized else uint8.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")

    if color_mode.lower() == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if normalize:
        img = img.astype(np.float32) / 255.0

    return img


def batch_images_to_numpy(paths: List[str], normalize: bool = True, color_mode: str = "rgb") -> np.ndarray:
    """
    Load multiple images into a single NumPy batch.

    Args:
        paths (List[str]): List of image file paths.
        normalize (bool): Whether to normalize pixel values to [0,1].
        color_mode (str): 'rgb' or 'bgr'.

    Returns:
        np.ndarray: Batch of shape (B, H, W, 3), dtype=float32 if normalized.
    """
    images = [image_to_numpy(p, normalize=normalize, color_mode=color_mode) for p in paths]

    # Ensure all shapes match
    first_shape = images[0].shape
    for i, img in enumerate(images):
        if img.shape != first_shape:
            raise ValueError(f"Image at index {i} has different shape {img.shape}, expected {first_shape}")

    return np.stack(images, axis=0)