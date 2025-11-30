

"""
Detection loss for the MLHD YOLO-like single-class detector.

Preds/targets shape: (B, S, S, 5)
Channel order: [tx, ty, tw, th, obj]

- Coordinate loss uses CIoU and is only applied to positive cells (obj* == 1).
- Objectness loss is BCE on all cells, with negatives down-weighted by lambda_noobj.
- Optionally, the objectness term can be passed as logits (pre-sigmoid) using obj_from_logits=True.

This mirrors the training dynamics we've described in the proposal.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Helpers: safe BCE on possibly-empty tensors (fixes MPS crash on empty masks)
import math

def _safe_bce_probs(input_t: torch.Tensor, target_t: torch.Tensor) -> torch.Tensor:
    """Binary cross entropy on probabilities that returns 0 if the tensors are empty."""
    if input_t.numel() == 0:
        # scalar 0 with same device/dtype
        return input_t.new_zeros((), dtype=input_t.dtype)
    return F.binary_cross_entropy(input_t, target_t, reduction="sum")


def _safe_bce_logits(input_t: torch.Tensor, target_t: torch.Tensor) -> torch.Tensor:
    """BCE with logits that returns 0 if the tensors are empty."""
    if input_t.numel() == 0:
        return input_t.new_zeros((), dtype=input_t.dtype)
    return F.binary_cross_entropy_with_logits(input_t, target_t, reduction="sum")


def _ciou_loss(
    px: torch.Tensor,
    py: torch.Tensor,
    pw: torch.Tensor,
    ph: torch.Tensor,
    tx: torch.Tensor,
    ty: torch.Tensor,
    tw: torch.Tensor,
    th: torch.Tensor,
    pos_mask: torch.Tensor,
) -> torch.Tensor:
    """Complete IoU loss summed over positive cells."""
    if not torch.any(pos_mask):
        return px.new_zeros((), dtype=px.dtype)

    eps = 1e-7
    S = px.shape[1]
    dtype = px.dtype
    device = px.device

    grid_y, grid_x = torch.meshgrid(
        torch.arange(S, device=device, dtype=dtype),
        torch.arange(S, device=device, dtype=dtype),
        indexing="ij",
    )
    grid_x = grid_x.unsqueeze(0)
    grid_y = grid_y.unsqueeze(0)

    # Normalized centers
    cx_pred = (grid_x + px) / S
    cy_pred = (grid_y + py) / S
    cx_tgt = (grid_x + tx) / S
    cy_tgt = (grid_y + ty) / S

    # Clamp widths/heights to avoid degenerate boxes
    w_pred = pw.clamp(min=eps)
    h_pred = ph.clamp(min=eps)
    w_tgt = tw.clamp(min=eps)
    h_tgt = th.clamp(min=eps)

    x1_pred = cx_pred - w_pred / 2
    y1_pred = cy_pred - h_pred / 2
    x2_pred = cx_pred + w_pred / 2
    y2_pred = cy_pred + h_pred / 2

    x1_tgt = cx_tgt - w_tgt / 2
    y1_tgt = cy_tgt - h_tgt / 2
    x2_tgt = cx_tgt + w_tgt / 2
    y2_tgt = cy_tgt + h_tgt / 2

    inter_x1 = torch.max(x1_pred, x1_tgt)
    inter_y1 = torch.max(y1_pred, y1_tgt)
    inter_x2 = torch.min(x2_pred, x2_tgt)
    inter_y2 = torch.min(y2_pred, y2_tgt)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area_pred = w_pred * h_pred
    area_tgt = w_tgt * h_tgt
    union = area_pred + area_tgt - inter_area + eps
    iou = inter_area / union

    center_dist = (cx_pred - cx_tgt) ** 2 + (cy_pred - cy_tgt) ** 2
    enc_x1 = torch.min(x1_pred, x1_tgt)
    enc_y1 = torch.min(y1_pred, y1_tgt)
    enc_x2 = torch.max(x2_pred, x2_tgt)
    enc_y2 = torch.max(y2_pred, y2_tgt)
    enc_diag = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2
    enc_diag = enc_diag.clamp(min=eps)

    v = (4 / (math.pi ** 2)) * (torch.atan(w_tgt / (h_tgt + eps)) - torch.atan(w_pred / (h_pred + eps))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - center_dist / enc_diag - alpha * v
    ciou_loss = (1.0 - ciou)[pos_mask].sum()

    return ciou_loss


def detection_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    lambda_coord: float = 5.0,
    lambda_noobj: float = 0.5,
    obj_from_logits: bool = False,
) -> torch.Tensor:
    """
    Compute the composite detection loss with a CIoU localization term.

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

    # Masks
    pos_mask = (tobj == 1.0)
    neg_mask = ~pos_mask

    # -------------------------
    # 1) Coordinate loss (positives only) via CIoU
    # -------------------------
    ciou_loss = _ciou_loss(px, py, pw, ph, tx, ty, tw, th, pos_mask)
    coord_loss = lambda_coord * ciou_loss

    # -------------------------
    # 2) Objectness loss on all cells (safe for empty masks / MPS backend)
    # -------------------------
    if obj_from_logits:
        # pobj are logits
        pos_bce = _safe_bce_logits(pobj[pos_mask], torch.ones_like(pobj[pos_mask]))
        neg_bce = _safe_bce_logits(pobj[neg_mask], torch.zeros_like(pobj[neg_mask]))
    else:
        # pobj are probabilities in (0,1)
        pos_bce = _safe_bce_probs(pobj[pos_mask], torch.ones_like(pobj[pos_mask]))
        neg_bce = _safe_bce_probs(pobj[neg_mask], torch.zeros_like(pobj[neg_mask]))

    obj_loss = pos_bce + lambda_noobj * neg_bce

    # -------------------------
    # 3) Combine and normalize by batch size
    # -------------------------
    total = (coord_loss + obj_loss) / max(B, 1)

    return total
