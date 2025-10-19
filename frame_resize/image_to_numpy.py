


import cv2
import numpy as np
from typing import List

def frames_to_numpy_batch(
    frame_paths: List[str],
    normalize: bool = True,
    color_mode: str = "rgb"
) -> np.ndarray:
    """
    Load a list of frame file paths and stack them into a NumPy batch.

    Args:
        frame_paths (List[str]): List of paths to frame image files.
        normalize (bool): If True, normalize pixel values to [0,1].
        color_mode (str): 'rgb' (default) or 'bgr'.

    Returns:
        np.ndarray: Batch array of shape (B, H, W, 3).
    """
    frames = []
    for path in frame_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot load frame: {path}")
        if color_mode.lower() == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if normalize:
            img = img.astype(np.float32) / 255.0
        frames.append(img)

    # Ensure all frames have the same shape
    first_shape = frames[0].shape
    for i, frame in enumerate(frames):
        if frame.shape != first_shape:
            raise ValueError(f"Frame at index {i} has shape {frame.shape}, expected {first_shape}")

    # Stack frames into batch
    batch = np.stack(frames, axis=0)
    return batch