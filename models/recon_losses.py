"""Reconstruction losses that preserve sharp 128×128 surgical detail."""

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
) -> torch.Tensor:
    """Smooth-L1 + gradient term for crisp frame reconstruction."""
    loss = l1_weight * F.smooth_l1_loss(pred, target)
    if grad_weight > 0:
        loss = loss + grad_weight * gradient_loss(pred, target)
    return loss
