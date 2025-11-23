"""
Detection metrics helpers for YOLO-like outputs.

Provides utilities to decode grid predictions into absolute boxes, compute IoU,
and accumulate precision/recall at multiple IoU thresholds during training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch


def _decode_boxes(x: torch.Tensor) -> torch.Tensor:
    """
    Convert YOLO grid predictions/targets to normalized [x1, y1, x2, y2].

    Args:
        x: Tensor [B, S, S, 4] with [tx, ty, tw, th].
    """
    _, S, _, _ = x.shape
    device = x.device
    dtype = x.dtype

    tx = x[..., 0]
    ty = x[..., 1]
    tw = torch.clamp(x[..., 2], min=1e-6, max=1.0)
    th = torch.clamp(x[..., 3], min=1e-6, max=1.0)

    grid_y = torch.arange(S, device=device, dtype=dtype).view(1, S, 1)
    grid_x = torch.arange(S, device=device, dtype=dtype).view(1, 1, S)

    cx = torch.clamp((grid_x + tx) / S, 0.0, 1.0)
    cy = torch.clamp((grid_y + ty) / S, 0.0, 1.0)

    x1 = torch.clamp(cx - tw / 2.0, 0.0, 1.0)
    y1 = torch.clamp(cy - th / 2.0, 0.0, 1.0)
    x2 = torch.clamp(cx + tw / 2.0, 0.0, 1.0)
    y2 = torch.clamp(cy + th / 2.0, 0.0, 1.0)

    return torch.stack([x1, y1, x2, y2], dim=-1)


def _box_iou_xyxy(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between predicted and target boxes (already in xyxy format).
    """
    x1 = torch.maximum(pred[..., 0], tgt[..., 0])
    y1 = torch.maximum(pred[..., 1], tgt[..., 1])
    x2 = torch.minimum(pred[..., 2], tgt[..., 2])
    y2 = torch.minimum(pred[..., 3], tgt[..., 3])

    inter_w = torch.clamp(x2 - x1, min=0.0)
    inter_h = torch.clamp(y2 - y1, min=0.0)
    intersection = inter_w * inter_h

    area_pred = torch.clamp(pred[..., 2] - pred[..., 0], min=0.0) * torch.clamp(
        pred[..., 3] - pred[..., 1], min=0.0
    )
    area_tgt = torch.clamp(tgt[..., 2] - tgt[..., 0], min=0.0) * torch.clamp(
        tgt[..., 3] - tgt[..., 1], min=0.0
    )
    union = area_pred + area_tgt - intersection
    union = torch.clamp(union, min=1e-6)
    return intersection / union


def _format_threshold(thr: float) -> str:
    return f"{thr:.2f}".rstrip("0").rstrip(".")


@dataclass
class DetectionMetricTracker:
    """
    Tracks IoU, precision, and recall at configurable IoU thresholds during training.
    """

    iou_thresholds: Tuple[float, ...] = (0.5, 0.85)
    obj_threshold: float = 0.5
    sum_iou: float = 0.0
    count_iou: int = 0
    stats: Dict[float, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stats:
            self.stats = {
                thr: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for thr in self.iou_thresholds
            }

    def reset(self) -> None:
        self.sum_iou = 0.0
        self.count_iou = 0
        for thr in self.stats:
            self.stats[thr] = {"tp": 0.0, "fp": 0.0, "fn": 0.0}

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds_boxes = _decode_boxes(preds[..., :4])
        target_boxes = _decode_boxes(targets[..., :4])
        iou_map = _box_iou_xyxy(preds_boxes, target_boxes)

        gt_mask = targets[..., 4] == 1.0
        obj_scores = preds[..., 4]
        pred_mask = obj_scores >= self.obj_threshold

        gt_iou_sum = (iou_map * gt_mask).sum()
        self.sum_iou += float(gt_iou_sum.item())
        self.count_iou += int(gt_mask.sum().item())

        for thr in self.iou_thresholds:
            thr_mask = iou_map >= thr
            tp_mask = gt_mask & pred_mask & thr_mask

            tp = float(tp_mask.sum().item())
            fn = float((gt_mask & ~tp_mask).sum().item())
            fp = float((pred_mask & ~tp_mask).sum().item())

            self.stats[thr]["tp"] += tp
            self.stats[thr]["fp"] += fp
            self.stats[thr]["fn"] += fn

    def summary(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        mean_iou = self.sum_iou / self.count_iou if self.count_iou > 0 else 0.0
        result["mean_iou"] = mean_iou

        for thr in self.iou_thresholds:
            stats = self.stats[thr]
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            tag = _format_threshold(thr)
            result[f"precision@{tag}"] = precision
            result[f"recall@{tag}"] = recall
            # Alias for "correct if IoU >= threshold"
            result[f"correct@{tag}"] = recall

        return result
