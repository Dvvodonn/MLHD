

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
from typing import Optional, Dict, Any

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
) -> float:
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

    Returns:
        Average loss over the dataset (mean per sample)
    """
    model.train()
    running = 0.0
    seen = 0

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

    return running / max(1, seen)


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
) -> float:
    """Evaluate average loss over a dataset."""
    model.eval()
    running = 0.0
    seen = 0

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

    return running / max(1, seen)


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
) -> Dict[str, float]:
    """Full training loop with optional validation and checkpointing.

    Args:
        early_stopping_patience: Stop training if val loss doesn't improve for this many epochs (default: 10)

    Returns a dict with final metrics: {"best_val": float}
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val = math.inf
    epochs_without_improvement = 0

    # Track loss history for visualization
    train_losses = []
    val_losses = []

    model.to(device)

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            lambda_coord=lambda_coord,
            lambda_noobj=lambda_noobj,
            obj_from_logits=obj_from_logits,
        )

        if val_loader is not None:
            va_loss = evaluate(
                model, val_loader, device,
                lambda_coord=lambda_coord,
                lambda_noobj=lambda_noobj,
                obj_from_logits=obj_from_logits,
            )
        else:
            va_loss = float('nan')

        if scheduler is not None:
            try:
                scheduler.step(va_loss if not math.isnan(va_loss) else tr_loss)
            except TypeError:
                scheduler.step()

        # Track loss history
        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        print_fn(f"[Epoch {epoch:02d}] train {tr_loss:.4f} | val {va_loss:.4f}")

        # Save best checkpoint on validation
        if val_loader is not None and not math.isnan(va_loss) and va_loss < best_val:
            best_val = va_loss
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": va_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
            }, os.path.join(ckpt_dir, ckpt_name))
            print_fn(f"  ↳ saved {os.path.join(ckpt_dir, ckpt_name)}")
        else:
            epochs_without_improvement += 1

        # Early stopping check
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print_fn(f"\n[Early Stopping] No improvement for {early_stopping_patience} epochs. Stopping at epoch {epoch}.")
            print_fn(f"Best validation loss: {best_val:.4f}")
            break

    return {"best_val": best_val}
