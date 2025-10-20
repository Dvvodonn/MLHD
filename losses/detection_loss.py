

"""
Detection loss for the MLHD YOLO-like single-class detector.

Preds/targets shape: (B, S, S, 5)
Channel order: [tx, ty, tw, th, obj]

- Coordinate loss (xy + wh terms) is only applied to positive cells (obj* == 1).
- Objectness loss is BCE on all cells, with negatives down-weighted by lambda_noobj.
- Optionally, the objectness term can be passed as logits (pre-sigmoid) using obj_from_logits=True.

This mirrors the training dynamics we've described in the proposal.
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn.functional as F


def detection_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    lambda_coord: float = 5.0,
    lambda_noobj: float = 0.5,
    obj_from_logits: bool = False,
) -> torch.Tensor:
    """
    Compute the composite detection loss.

    Args:
        preds:   (B, S, S, 5) model predictions. If obj_from_logits=False, all channels are expected in (0,1)
                 (i.e., post-sigmoid). If obj_from_logits=True, only the last channel (objectness) is treated as
                 logits; t_x, t_y, t_w, t_h are still expected in (0,1).
        targets: (B, S, S, 5) ground-truth tensor with channels [tx*, ty*, tw*, th*, tobj*].
        lambda_coord: Weight for coordinate regression terms.
        lambda_noobj: Down-weighting factor for negative objectness BCE.
        obj_from_logits: If True, use BCEWithLogits for objectness; otherwise, use BCE on probabilities.

    Returns:
        Scalar loss tensor.
    """
    assert preds.shape == targets.shape, f"Shape mismatch: preds {preds.shape}, targets {targets.shape}"
    B = preds.shape[0]

    # Split channels
    px = preds[..., 0]
    py = preds[..., 1]
    pw = preds[..., 2]
    ph = preds[..., 3]
    pobj = preds[..., 4]

    tx = targets[..., 0]
    ty = targets[..., 1]
    tw = targets[..., 2]
    th = targets[..., 3]
    tobj = targets[..., 4]

    # Masks (convert to float for element-wise ops, MPS-compatible)
    pos_mask = (tobj == 1.0).float()
    neg_mask = 1.0 - pos_mask

    # -------------------------
    # 1) Coordinate loss (positives only)
    # -------------------------
    # Position (x, y) - use element-wise multiplication instead of indexing
    xy_err = (px - tx) ** 2 + (py - ty) ** 2
    xy_loss = (xy_err * pos_mask).sum()

    # Size (w, h): compare squared widths/heights to stabilize gradients near small boxes
    w2_pred = pw ** 2
    h2_pred = ph ** 2
    w2_tgt = tw ** 2
    h2_tgt = th ** 2

    wh_err = (w2_pred - w2_tgt) ** 2 + (h2_pred - h2_tgt) ** 2
    wh_loss = (wh_err * pos_mask).sum()

    coord_loss = lambda_coord * (xy_loss + wh_loss)

    # -------------------------
    # 2) Objectness loss on all cells
    # -------------------------
    if obj_from_logits:
        # pobj are logits - compute BCE per element, then mask
        pos_bce_all = F.binary_cross_entropy_with_logits(pobj, torch.ones_like(pobj), reduction="none")
        neg_bce_all = F.binary_cross_entropy_with_logits(pobj, torch.zeros_like(pobj), reduction="none")
        pos_bce = (pos_bce_all * pos_mask).sum()
        neg_bce = (neg_bce_all * neg_mask).sum()
    else:
        # pobj are probabilities in (0,1) - compute BCE per element, then mask
        pos_bce_all = F.binary_cross_entropy(pobj, torch.ones_like(pobj), reduction="none")
        neg_bce_all = F.binary_cross_entropy(pobj, torch.zeros_like(pobj), reduction="none")
        pos_bce = (pos_bce_all * pos_mask).sum()
        neg_bce = (neg_bce_all * neg_mask).sum()

    obj_loss = pos_bce + lambda_noobj * neg_bce

    # -------------------------
    # 3) Combine and normalize by batch size
    # -------------------------
    total = (coord_loss + obj_loss) / max(B, 1)

    return total