import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import cv2
import numpy as np
from typing import Tuple, List, Dict
from frame_resize.image_to_numpy import frames_to_numpy_batch
from frame_resize.resize import letterbox_resize_batch_from_numpy


def letterbox_image(image_path: str, target_size: Tuple[int, int] = (416, 416)) -> Tuple[torch.Tensor, Dict]:
    """
    Load and apply letterbox resize to single image using existing batch utilities.
    Returns normalized tensor [3, H, W] and resize parameters.
    """
    batch = frames_to_numpy_batch([image_path], normalize=True, color_mode='rgb')
    resized_batch, params_list = letterbox_resize_batch_from_numpy(batch, target_size)

    img_tensor = torch.from_numpy(resized_batch[0]).permute(2, 0, 1).float()

    return img_tensor, params_list[0]


def transform_boxes(boxes: List[Tuple[float, float, float, float]],
                    params: Dict) -> List[Tuple[float, float, float, float]]:
    """
    Transform normalized box coordinates to account for letterbox padding and scaling.

    Args:
        boxes: List of (cx, cy, w, h) normalized to original image [0, 1]
        params: Letterbox parameters from letterbox_resize_batch_from_numpy

    Returns:
        Transformed boxes normalized to letterbox image [0, 1]
    """
    scale = params['scale']
    pad_w = params['pad_w']
    pad_h = params['pad_h']
    orig_w, orig_h = params['orig_wh']
    target_w, target_h = params['target_wh']

    transformed = []
    for (cx, cy, w, h) in boxes:
        new_cx = (cx * orig_w * scale + pad_w) / target_w
        new_cy = (cy * orig_h * scale + pad_h) / target_h
        new_w = (w * orig_w * scale) / target_w
        new_h = (h * orig_h * scale) / target_h

        transformed.append((new_cx, new_cy, new_w, new_h))

    return transformed
