

# train/trainer.py
"""
Generic training utilities for MLHD.

This module provides:
  - train_one_epoch: runs one epoch over a DataLoader
  - evaluate: runs evaluation over a DataLoader
  - fit: multi-epoch training loop with checkpointing

Expected DataLoader output per batch:
  imgs_t:   torch.FloatTensor of shape (B, 3, H, W)
  targets_t:torch.FloatTensor of shape (B, S, S, 5) where last dim is [tx, ty, tw, th, obj]
  paths:    list[str] (optional, for debugging)

Model expected output per forward:
  preds: torch.FloatTensor of shape (B, S, S, 5) where last dim is [tx, ty, tw, th, obj]

Loss function signature:
  detection_loss(preds, targets, lambda_coord=5.0, lambda_noobj=0.5, obj_from_logits=False)
"""
from __future__ import annotations

import math
import os
from typing import Optional, Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader

try:
    from losses.detection_loss import detection_loss
except Exception as e:
    raise ImportError("Could not import detection_loss from losses.detection_loss. Ensure the file exists.") from e

try:
    from datasets.dataloader import CCTVDetectionDataset
except Exception as e:
    raise ImportError("Could not import CCTVDetectionDataset from datasets.dataloader. Ensure the file exists.") from e

from .metrics import DetectionMetricTracker


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    lambda_coord: float = 5.0,
    lambda_noobj: float = 0.5,
    obj_from_logits: bool = False,
    max_batches: Optional[int] = None,
    metric_thresholds: Tuple[float, ...] = (0.5, 0.85),
    obj_threshold: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    """Run one training epoch.

    Args:
        model: nn.Module producing (B,S,S,5)
        loader: DataLoader yielding (imgs_t, targets_t, paths)
        optimizer: torch optimizer
        device: cpu/cuda/mps device
        lambda_coord: coord loss weight
        lambda_noobj: no-object BCE weight
        obj_from_logits: whether objectness is provided as logits
        max_batches: if set, limit number of batches (useful for smoke tests)
        metric_thresholds: IoU thresholds for metric tracking
        obj_threshold: Objectness cutoff for considering predictions positive

    Returns:
        Tuple (average loss, metric summary dict)
    """
    model.train()
    running = 0.0
    seen = 0
    metric_tracker = DetectionMetricTracker(
        iou_thresholds=metric_thresholds,
        obj_threshold=obj_threshold,
    )

    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break

        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            imgs_t, targets_t = batch[0], batch[1]
        else:
            raise ValueError("Expected batch to be a tuple/list: (imgs_t, targets_t, *_) ")

        non_blocking = device.type != "cpu"
        imgs_t = imgs_t.to(device, non_blocking=non_blocking)
        targets_t = targets_t.to(device, non_blocking=non_blocking)

        # Forward
        preds = model(imgs_t)

        # Loss
        loss = detection_loss(
            preds,
            targets_t,
            lambda_coord=lambda_coord,
            lambda_noobj=lambda_noobj,
            obj_from_logits=obj_from_logits,
        )

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Stats
        bs = imgs_t.size(0)
        running += loss.item() * bs
        seen += bs
        metric_tracker.update(preds.detach(), targets_t)

    return running / max(1, seen), metric_tracker.summary()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    lambda_coord: float = 5.0,
    lambda_noobj: float = 0.5,
    obj_from_logits: bool = False,
    max_batches: Optional[int] = None,
    metric_thresholds: Tuple[float, ...] = (0.5, 0.85),
    obj_threshold: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate average loss over a dataset along with IoU metrics.

    Args mirror train_one_epoch; metric_thresholds and obj_threshold control metric computation.
    """
    model.eval()
    running = 0.0
    seen = 0
    metric_tracker = DetectionMetricTracker(
        iou_thresholds=metric_thresholds,
        obj_threshold=obj_threshold,
    )

    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break

        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            imgs_t, targets_t = batch[0], batch[1]
        else:
            raise ValueError("Expected batch to be a tuple/list: (imgs_t, targets_t, *_) ")

        non_blocking = device.type != "cpu"
        imgs_t = imgs_t.to(device, non_blocking=non_blocking)
        targets_t = targets_t.to(device, non_blocking=non_blocking)

        preds = model(imgs_t)
        loss = detection_loss(
            preds,
            targets_t,
            lambda_coord=lambda_coord,
            lambda_noobj=lambda_noobj,
            obj_from_logits=obj_from_logits,
        )

        bs = imgs_t.size(0)
        running += loss.item() * bs
        seen += bs
        metric_tracker.update(preds, targets_t)

    return running / max(1, seen), metric_tracker.summary()


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epochs: int = 20,
    lambda_coord: float = 5.0,
    lambda_noobj: float = 0.5,
    obj_from_logits: bool = False,
    scheduler: Optional[Any] = None,
    ckpt_dir: str = "checkpoints",
    ckpt_name: str = "best.pt",
    early_stopping_patience: int = 10,
    print_fn=print,
    iou_thresholds: Tuple[float, ...] = (0.5, 0.85),
    obj_threshold: float = 0.5,
) -> Dict[str, float]:
    """Full training loop with optional validation and checkpointing.

    Args:
        early_stopping_patience: Stop training if val loss doesn't improve for this many epochs (default: 10)
        iou_thresholds: IoU thresholds (e.g., 0.5, 0.85) for precision/recall reporting
        obj_threshold: Objectness cutoff for counting detections in metrics

    Returns a dict with final metrics: {"best_val": float, "best_miou": float}
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val = math.inf
    best_miou = -math.inf
    epochs_without_improvement = 0

    # Track loss history for visualization
    train_losses = []
    val_losses = []

    model.to(device)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            lambda_coord=lambda_coord,
            lambda_noobj=lambda_noobj,
            obj_from_logits=obj_from_logits,
            metric_thresholds=iou_thresholds,
            obj_threshold=obj_threshold,
        )

        if val_loader is not None:
            va_loss, va_metrics = evaluate(
                model, val_loader, device,
                lambda_coord=lambda_coord,
                lambda_noobj=lambda_noobj,
                obj_from_logits=obj_from_logits,
                metric_thresholds=iou_thresholds,
                obj_threshold=obj_threshold,
            )
        else:
            va_loss = float('nan')
            va_metrics = {}

        if scheduler is not None:
            try:
                scheduler.step(va_loss if not math.isnan(va_loss) else tr_loss)
            except TypeError:
                scheduler.step()

        # Track loss history
        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        metrics_str = _format_metrics(tr_metrics, iou_thresholds)
        val_metrics_str = _format_metrics(va_metrics, iou_thresholds) if val_loader is not None else ""
        if val_metrics_str:
            print_fn(f"[Epoch {epoch:02d}] train {tr_loss:.4f} ({metrics_str}) | val {va_loss:.4f} ({val_metrics_str})")
        else:
            print_fn(f"[Epoch {epoch:02d}] train {tr_loss:.4f} ({metrics_str}) | val {va_loss:.4f}")

        # Save best checkpoint on validation
        current_miou = va_metrics.get("mean_iou", float("-inf")) if val_loader is not None else float("-inf")
        improved = val_loader is not None and current_miou > best_miou

        if improved:
            best_miou = current_miou
            best_val = va_loss
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": va_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_mean_iou": current_miou,
            }, os.path.join(ckpt_dir, ckpt_name))
            print_fn(f"  ↳ saved {os.path.join(ckpt_dir, ckpt_name)}")
        elif val_loader is not None:
            epochs_without_improvement += 1

        # Early stopping check
        if (
            val_loader is not None
            and early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            print_fn(f"\n[Early Stopping] No improvement for {early_stopping_patience} epochs. Stopping at epoch {epoch}.")
            print_fn(f"Best validation mIoU: {best_miou:.4f}")
            break

    return {"best_val": best_val, "best_miou": best_miou}


def _format_metrics(metrics: Dict[str, float], thresholds: Tuple[float, ...]) -> str:
    if not metrics:
        return ""

    parts = [f"mIoU {metrics.get('mean_iou', 0.0):.3f}"]
    for thr in thresholds:
        tag = f"{thr:.2f}".rstrip("0").rstrip(".")
        prec = metrics.get(f"precision@{tag}", 0.0)
        rec = metrics.get(f"recall@{tag}", 0.0)
        parts.append(f"P@{tag} {prec:.3f}")
        parts.append(f"R@{tag} {rec:.3f}")
    return " ".join(parts)
