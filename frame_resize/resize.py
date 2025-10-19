


import numpy as np
import cv2
from typing import Tuple, List, Dict, Union

def letterbox_resize_batch_from_numpy(
    batch: np.ndarray,
    target_size: Tuple[int, int] = (416, 416),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, List[Dict[str, Union[int, float, Tuple[int, int]]]]]:
    """
    Apply letterbox resizing to a batch of NumPy images.

    Args:
        batch (np.ndarray): Array of shape (B, H, W, 3) with float or uint8 values.
        target_size (Tuple[int, int]): (width, height) target for all outputs.
        color (Tuple[int, int, int]): Padding color for letterbox borders.

    Returns:
        Tuple[np.ndarray, List[Dict]]: 
            - Resized batch of shape (B, target_h, target_w, 3).
            - List of parameter dicts for each image ('scale', 'pad_w', 'pad_h', etc.).
    """
    if batch.ndim != 4:
        raise ValueError(f"Expected batch shape (B, H, W, 3), got {batch.shape}")

    out_images = []
    params_list = []
    tw, th = target_size

    for i in range(batch.shape[0]):
        img = batch[i]
        h, w = img.shape[:2]

        # Compute scale factor to fit within target
        scale = min(tw / w, th / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))

        # Resize with preserved aspect ratio
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Compute symmetric padding
        pad_w_total = tw - nw
        pad_h_total = th - nh
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left
        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top

        # Add padding
        new_image = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=color
        )

        params = {
            "scale": float(scale),
            "pad_w": int(pad_left),
            "pad_h": int(pad_top),
            "new_wh": (nw, nh),
            "target_wh": (tw, th),
            "orig_wh": (w, h),
        }

        out_images.append(new_image)
        params_list.append(params)

    resized_batch = np.stack(out_images, axis=0)
    return resized_batch, params_list