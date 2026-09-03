"""Reconstruction losses that preserve sharp frame detail."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match spatial gradients to reduce blurry VQ reconstructions."""
    def _diff(x: torch.Tensor, dim: int) -> torch.Tensor:
        return x.diff(dim=dim)

    pred_dx = _diff(pred, -1)
    pred_dy = _diff(pred, -2)
    tgt_dx = _diff(target, -1)
    tgt_dy = _diff(target, -2)
    return F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    l1_weight: float = 1.0,
    grad_weight: float = 0.15,
    fg_weight: float = 0.0,
    fg_threshold: float = -0.8,
) -> torch.Tensor:
    """Smooth-L1 + gradient term for crisp frame reconstruction.

    fg_weight > 0 adds an extra foreground-weighted L1 term, useful for
    datasets with sparse bright objects on dark backgrounds (e.g. Pong) where
    the model can trivially collapse to predicting all-black.
    """
    loss = l1_weight * F.smooth_l1_loss(pred, target)
    if grad_weight > 0:
        loss = loss + grad_weight * gradient_loss(pred, target)
    if fg_weight > 0:
        # per-pixel weight: foreground pixels get boosted weight
        fg_mask = (target > fg_threshold).float()
        # normalise so total weight sums to batch size * spatial dims
        n = fg_mask.numel()
        n_fg = fg_mask.sum().clamp_min(1.0)
        weight = 1.0 + fg_weight * fg_mask * (n / n_fg)
        per_pixel = F.l1_loss(pred, target, reduction='none')
        loss = loss + fg_weight * (weight * per_pixel).mean()
    return loss
