"""Small training-loop helpers shared by all accumulated stages."""

from __future__ import annotations

import torch


class MicrobatchLossMean:
    """Track the undivided mean while callers backpropagate a scaled loss."""

    def __init__(self, accumulation_steps: int):
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be at least 1")
        self.accumulation_steps = accumulation_steps
        self._total = 0.0
        self._count = 0

    def backward_loss(self, loss: torch.Tensor) -> torch.Tensor:
        self._total += float(loss.detach())
        self._count += 1
        return loss / self.accumulation_steps

    @property
    def mean(self) -> float:
        if self._count != self.accumulation_steps:
            raise RuntimeError(
                f"expected {self.accumulation_steps} microbatch losses, observed {self._count}"
            )
        return self._total / self._count
