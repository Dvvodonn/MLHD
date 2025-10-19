"""
Utilities to resize frames for detection while preserving aspect ratio (YOLO-style letterboxing),
and to transform YOLO-format boxes accordingly.

Intended usage in the pipeline:
    from frame_resize.resize import letterbox_resize, remap_yolo_boxes

    img_resized, params = letterbox_resize(img, target_size=(416, 416))
    boxes_resized = remap_yolo_boxes(boxes, orig_shape=img.shape[:2], params=params, to='resized')

    # If you later need to map detections back to the original image:
    boxes_orig = remap_yolo_boxes(boxes_resized, orig_shape=img.shape[:2], params=params, to='original')

`boxes` can be either:
    - shape (N, 5): [class_id, x, y, w, h] with x,y,w,h normalized in [0,1]
    - shape (N, 4): [x, y, w, h] normalized in [0,1]
"""

from __future__ import annotations

from typing import Dict, Tuple, Union
import numpy as np

try:
    import cv2  # OpenCV is used for efficient resizing and padding
except ImportError as e:
    raise ImportError("OpenCV (cv2) is required for letterbox resizing. `pip install opencv-python`.") from e


def letterbox_resize(
    image: np.ndarray,
    target_size: Tuple[int, int] = (416, 416),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, Dict[str, Union[int, float, Tuple[int, int]]]]:
    """
    Resize an image while preserving aspect ratio by scaling to fit within `target_size`,
    then pad the remaining area (letterbox) with `color`.

    Args:
        image: HxWxC (BGR or RGB) uint8/float32 numpy array.
        target_size: (width, height) of the network input.
        color: padding color (in the same channel order as `image`).

    Returns:
        new_image: (target_h, target_w, C) numpy array.
        params: dict with information needed to remap boxes:
            - 'scale': float
            - 'pad_w': int (left padding in pixels)
            - 'pad_h': int (top padding in pixels)
            - 'new_wh': (nw, nh) scaled image size before padding
            - 'target_wh': (tw, th) == target_size
    """
    th, tw = int(target_size[1]), int(target_size[0])
    h, w = image.shape[:2]

    # Compute uniform scale to fit inside target
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    # Resize with preserved aspect ratio
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Compute symmetric padding to center the image
    pad_w_total = tw - nw
    pad_h_total = th - nh
    pad_left = pad_w_total // 2
    pad_right = pad_w_total - pad_left
    pad_top = pad_h_total // 2
    pad_bottom = pad_h_total - pad_top

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
    return new_image, params


def _to_pixels_yolo(
    boxes: np.ndarray,
    wh: Tuple[int, int],
) -> np.ndarray:
    """
    Convert YOLO normalized boxes to pixel xywh.

    Args:
        boxes: shape (N,4) normalized xywh in [0,1].
        wh: (width, height) of the image.

    Returns:
        (N,4) pixel xywh
    """
    W, H = wh
    px = boxes.copy().astype(np.float32)
    px[:, 0] *= W  # x
    px[:, 1] *= H  # y
    px[:, 2] *= W  # w
    px[:, 3] *= H  # h
    return px


def _to_normalized_yolo(
    boxes_px: np.ndarray,
    wh: Tuple[int, int],
) -> np.ndarray:
    """
    Convert pixel xywh to YOLO normalized xywh.
    """
    W, H = wh
    out = boxes_px.copy().astype(np.float32)
    out[:, 0] /= W
    out[:, 1] /= H
    out[:, 2] /= W
    out[:, 3] /= H
    return out


def remap_yolo_boxes(
    boxes: np.ndarray,
    orig_shape: Tuple[int, int],
    params: Dict[str, Union[int, float, Tuple[int, int]]],
    to: str = "resized",
) -> np.ndarray:
    """
    Remap YOLO-format boxes between the original image coordinates and the letterboxed (resized+pad) image.

    Args:
        boxes: shape (N,5) [cls, x, y, w, h] or (N,4) [x, y, w, h] with normalized values in [0,1]
               relative to the *original* image when to='resized', or relative to the *resized* image when to='original'.
        orig_shape: (H, W) of the original image.
        params: dict returned by `letterbox_resize` (must include 'scale', 'pad_w', 'pad_h', 'target_wh').
        to: 'resized'  -> map from original -> letterboxed target
            'original' -> map from letterboxed target -> original

    Returns:
        boxes_out: same shape as input `boxes`, with x,y,w,h normalized to the destination image.
    """
    if boxes.size == 0:
        return boxes

    # Separate class id if present
    has_class = boxes.shape[1] == 5
    if has_class:
        cls = boxes[:, 0:1]
        xywh = boxes[:, 1:].astype(np.float32)
    else:
        xywh = boxes.astype(np.float32)

    W_orig, H_orig = int(orig_shape[1]), int(orig_shape[0])

    scale = float(params["scale"])
    pad_w = int(params["pad_w"])
    pad_h = int(params["pad_h"])
    tw, th = params["target_wh"]

    if to == "resized":
        # 1) original normalized -> original pixels
        xywh_px = _to_pixels_yolo(xywh, (W_orig, H_orig))
        # 2) scale
        xywh_px[:, 0] *= scale
        xywh_px[:, 1] *= scale
        xywh_px[:, 2] *= scale
        xywh_px[:, 3] *= scale
        # 3) add padding offsets to centers
        xywh_px[:, 0] += pad_w
        xywh_px[:, 1] += pad_h
        # 4) normalize to target (tw, th)
        xywh_out = _to_normalized_yolo(xywh_px, (tw, th))

    elif to == "original":
        # 1) letterboxed normalized -> letterboxed pixels
        xywh_px = _to_pixels_yolo(xywh, (tw, th))
        # 2) remove padding offsets from centers
        xywh_px[:, 0] -= pad_w
        xywh_px[:, 1] -= pad_h
        # 3) inverse scale
        inv = 1.0 / scale
        xywh_px[:, 0] *= inv
        xywh_px[:, 1] *= inv
        xywh_px[:, 2] *= inv
        xywh_px[:, 3] *= inv
        # 4) normalize back to original (W_orig, H_orig)
        xywh_out = _to_normalized_yolo(xywh_px, (W_orig, H_orig))
    else:
        raise ValueError("`to` must be either 'resized' or 'original'.")

    if has_class:
        return np.concatenate([cls, xywh_out], axis=1)
    return xywh_out


# Backwards-compatible simple wrapper (kept name `resize` per existing import)
def resize(image: np.ndarray, target_size: Tuple[int, int] = (416, 416)):
    """
    Backwards-compatible wrapper that performs letterbox resizing.

    Returns:
        new_image, params
    """
    return letterbox_resize(image, target_size=target_size)