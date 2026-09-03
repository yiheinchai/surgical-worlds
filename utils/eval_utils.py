"""Lightweight held-out eval helpers for W&B train/val curves."""

from __future__ import annotations

from typing import Callable, Optional

import torch


@torch.no_grad()
def mean_loader_loss(
    step_fn: Callable[[torch.Tensor], torch.Tensor],
    loader,
    device: str,
    n_batches: int = 4,
) -> Optional[float]:
    """Average `step_fn(batch)` over up to `n_batches` from `loader`.

    `step_fn` should return a scalar loss tensor (already reduced).
    Returns None if the loader is empty.
    """
    if loader is None or n_batches <= 0:
        return None

    total = 0.0
    count = 0
    it = iter(loader)
    for _ in range(n_batches):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        x = x.to(device, non_blocking=True)
        loss = step_fn(x)
        total += float(loss.detach().item())
        count += 1

    if count == 0:
        return None
    return total / count
