

import cv2
import numpy as np
from typing import List, Tuple

def frames_to_resized_batch(
    frame_paths: List[str],
    target_size: Tuple[int, int] = (416, 416),
    normalize: bool = True,
    color_mode: str = "rgb"
) -> np.ndarray:
    """
    Load a list of video frame paths, resize each to a fixed target size, and
    return them stacked as a NumPy batch.

    Args:
        frame_paths (List[str]): Paths to frame image files.
        target_size (Tuple[int, int]): Desired output size (width, height).
        normalize (bool): If True, normalize pixel values to [0,1].
        color_mode (str): 'rgb' (default) or 'bgr'.

    Returns:
        np.ndarray: Array of shape (B, H, W, 3) with resized frames.
    """
    resized_frames = []
    width, height = target_size

    for path in frame_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot load frame: {path}")
        if color_mode.lower() == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        if normalize:
            img = img.astype(np.float32) / 255.0
        resized_frames.append(img)

    batch = np.stack(resized_frames, axis=0)
    return batch