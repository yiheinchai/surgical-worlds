"""Frame conversion utilities for the surgery simulator UI."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


DISPLAY_SIZE = 512
# CRCD / da Vinci feeds are 848x480; cached HDF5 was resized to square without
# center-crop, squashing circles into tall ovals. Stretch width when displaying.
ROBOTIC_WIDESCREEN_ASPECT = 848 / 480


def tensor_frame_to_pil(
    frame: torch.Tensor,
    upscale: bool = False,
    display_aspect_width_scale: float = 1.0,
    display_size: int = DISPLAY_SIZE,
) -> Image.Image:
    """Convert a single normalized frame [C,H,W] in [-1,1] to PIL RGB."""
    if frame.dim() == 4:
        frame = frame[0]
    img = frame.detach().cpu().float()
    img = ((img + 1) / 2).clamp(0, 1)
    arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    if upscale:
        if display_aspect_width_scale != 1.0:
            target_h = display_size
            target_w = max(1, int(round(target_h * display_aspect_width_scale)))
            pil = pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        elif max(pil.size) < display_size:
            pil = pil.resize((display_size, display_size), Image.Resampling.LANCZOS)
    return pil


def tensor_sequence_to_pil_list(
    frames: torch.Tensor,
    upscale: bool = False,
    display_aspect_width_scale: float = 1.0,
    display_size: int = DISPLAY_SIZE,
) -> list[Image.Image]:
    """Convert [T,C,H,W] or [1,T,C,H,W] batch to list of PIL images."""
    if frames.dim() == 5:
        frames = frames[0]
    return [
        tensor_frame_to_pil(
            frames[t],
            upscale=upscale,
            display_aspect_width_scale=display_aspect_width_scale,
            display_size=display_size,
        )
        for t in range(frames.shape[0])
    ]


def pil_to_model_tensor(image: Image.Image, size: int = 128) -> torch.Tensor:
    """Resize PIL image to model input tensor [1,1,C,H,W] normalized to [-1,1]."""
    image = image.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    t = t * 2.0 - 1.0
    return t.unsqueeze(0).unsqueeze(0)
