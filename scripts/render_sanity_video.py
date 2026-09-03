#!/usr/bin/env python3
"""3-panel sanity video: GT | tokenizer recon | dynamics prediction."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import cv2
import numpy as np
import torch
from einops import repeat
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.data_utils import load_data_and_data_loaders
from simulator.frame_utils import ROBOTIC_WIDESCREEN_ASPECT, tensor_frame_to_pil
from utils.inference_utils import load_models

ActionMode = Literal["gt", "fixed"]


def _font(size: int = 15) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _pil_to_bgr(img: Image.Image, size: Tuple[int, int]) -> np.ndarray:
    rgb = np.array(img.convert("RGB").resize(size, Image.Resampling.LANCZOS))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _annotate(frame_bgr: np.ndarray, title: str) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    bar = np.zeros((40, w, 3), dtype=np.uint8)
    bar[:] = (24, 24, 24)
    out = np.vstack([bar, frame_bgr])
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text((8, 10), title, fill=(255, 220, 180), font=_font())
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _panel_size(height: int, aspect: float) -> Tuple[int, int]:
    width = max(2, int(round(height * aspect)))
    if width % 2:
        width += 1
    if height % 2:
        height += 1
    return width, height


def _to_pil(tensor_frame: torch.Tensor, panel_size: Tuple[int, int], aspect: float) -> Image.Image:
    return tensor_frame_to_pil(
        tensor_frame,
        upscale=True,
        display_aspect_width_scale=aspect,
        display_size=panel_size[1],
    )


def _encode_h264(src: Path) -> None:
    """Re-encode OpenCV mp4v output to H.264 for browser/GitHub playback."""
    tmp = src.with_suffix(".h264.mp4")
    last_err = b""
    for codec in ("libx264", "libopenh264"):
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(src),
                "-c:v", codec, "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-crf", "20", "-an",
                str(tmp),
            ],
            capture_output=True,
        )
        if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(src)
            return
        last_err = result.stderr or b""
        tmp.unlink(missing_ok=True)
    raise RuntimeError(
        f"H.264 encode failed for {src} (need ffmpeg libx264). "
        f"stderr tail: {last_err[-500:]!r}"
    )


def _build_action_latent(
    latent_action_model,
    context_frames: torch.Tensor,
    action_id: int,
    context_window: int,
) -> torch.Tensor:
    recent_tensor = repeat(
        torch.tensor([action_id], device=context_frames.device, dtype=torch.long),
        "i -> 1 i",
    )
    action_latent = latent_action_model.quantizer.get_latents_from_indices(recent_tensor)
    pad_count = context_window
    gt_pad = latent_action_model.encode(context_frames[:, :pad_count])
    return torch.cat([latent_action_model.quantizer(gt_pad), action_latent], dim=1)


def _predict_frame(
    video_tokenizer,
    dynamics_model,
    context_frames: torch.Tensor,
    action_latent: torch.Tensor,
    maskgit_steps: int,
    temperature: float,
) -> torch.Tensor:
    video_latents = video_tokenizer.quantizer.get_latents_from_indices(
        video_tokenizer.tokenize(context_frames), dim=-1
    )

    def idx_to_latents(idx: torch.Tensor) -> torch.Tensor:
        return video_tokenizer.quantizer.get_latents_from_indices(idx, dim=-1)

    next_latents = dynamics_model.forward_inference(
        context_latents=video_latents,
        prediction_horizon=1,
        num_steps=maskgit_steps,
        index_to_latents_fn=idx_to_latents,
        conditioning=action_latent,
        temperature=temperature,
    )
    return video_tokenizer.detokenize(next_latents)[:, -1]


def render_sanity_video(
    output_path: Path,
    *,
    video_tokenizer_path: str,
    latent_actions_path: str,
    dynamics_path: str,
    device: str,
    dataset: str,
    seed_index: int,
    context_window: int,
    inference_steps: int,
    action_mode: ActionMode,
    fixed_action: int,
    temperature: float,
    maskgit_steps: int,
    panel_height: int,
    fps: int,
    preload_ratio: float,
) -> Path:
    aspect = ROBOTIC_WIDESCREEN_ASPECT if dataset == "ROBOTIC_LAPAROSCOPIC" else 1.0
    panel_size = _panel_size(panel_height, aspect)

    video_tokenizer, latent_action_model, dynamics_model = load_models(
        video_tokenizer_path, latent_actions_path, dynamics_path, device, use_actions=True
    )

    frames_to_load = context_window + inference_steps
    _, _, data_loader, _, _ = load_data_and_data_loaders(
        dataset=dataset,
        batch_size=1,
        num_frames=frames_to_load,
        preload_ratio=preload_ratio,
    )
    full_sequence = data_loader.dataset[seed_index][0].unsqueeze(0).to(device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (panel_size[0] * 3, panel_size[1] + 40),
    )

    with torch.inference_mode():
        # Context frames — no dynamics prediction yet; right panel = tokenizer recon.
        for ctx_i in range(context_window):
            gt_tensor = full_sequence[:, ctx_i]
            recon = video_tokenizer.detokenize(
                video_tokenizer.quantizer.get_latents_from_indices(
                    video_tokenizer.tokenize(gt_tensor.unsqueeze(1)), dim=-1
                )
            )[:, 0]

            panels = [
                _annotate(_pil_to_bgr(_to_pil(gt_tensor, panel_size, aspect), panel_size), f"GT  |  context {ctx_i + 1}/{context_window}"),
                _annotate(_pil_to_bgr(_to_pil(recon, panel_size, aspect), panel_size), f"Tokenizer recon  |  context {ctx_i + 1}"),
                _annotate(_pil_to_bgr(_to_pil(recon, panel_size, aspect), panel_size), "Dynamics  |  (waiting for context)"),
            ]
            writer.write(np.hstack(panels))

        # Inference steps — teacher-forced GT context window.
        for step in range(1, inference_steps + 1):
            ctx_start = step - 1
            context = full_sequence[:, ctx_start : ctx_start + context_window]
            gt_next = full_sequence[:, ctx_start + context_window]

            recon = video_tokenizer.detokenize(
                video_tokenizer.quantizer.get_latents_from_indices(
                    video_tokenizer.tokenize(gt_next.unsqueeze(1)), dim=-1
                )
            )[:, 0]

            gt_actions = latent_action_model.encode(context)
            gt_action_id = int(
                latent_action_model.quantizer.get_indices_from_latents(gt_actions)[0, -1].item()
            )
            action_id = gt_action_id if action_mode == "gt" else fixed_action
            action_latent = _build_action_latent(
                latent_action_model, context, action_id, context_window
            )
            pred = _predict_frame(
                video_tokenizer,
                dynamics_model,
                context,
                action_latent,
                maskgit_steps,
                temperature,
            )

            action_note = f"action {action_id}" + (" (GT)" if action_mode == "gt" else " (forced)")
            panels = [
                _annotate(_pil_to_bgr(_to_pil(gt_next, panel_size, aspect), panel_size), f"GT  |  step {step}"),
                _annotate(_pil_to_bgr(_to_pil(recon, panel_size, aspect), panel_size), f"Tokenizer recon  |  step {step}"),
                _annotate(_pil_to_bgr(_to_pil(pred[0], panel_size, aspect), panel_size), f"Dynamics  |  step {step}  |  {action_note}"),
            ]
            writer.write(np.hstack(panels))

    writer.release()
    _encode_h264(output_path)
    print(f"Wrote {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render 3-panel GT | tokenizer | dynamics sanity video.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-tokenizer-path", required=True)
    parser.add_argument("--latent-actions-path", required=True)
    parser.add_argument("--dynamics-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", default="ROBOTIC_LAPAROSCOPIC")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--context-window", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--action-mode", choices=["gt", "fixed"], default="gt")
    parser.add_argument("--fixed-action", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--maskgit-steps", type=int, default=8)
    parser.add_argument("--panel-height", type=int, default=384)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--preload-ratio", type=float, default=1.0)
    args = parser.parse_args()

    os.chdir(ROOT)
    render_sanity_video(
        args.output,
        video_tokenizer_path=args.video_tokenizer_path,
        latent_actions_path=args.latent_actions_path,
        dynamics_path=args.dynamics_path,
        device=args.device,
        dataset=args.dataset,
        seed_index=args.seed,
        context_window=args.context_window,
        inference_steps=args.steps,
        action_mode=args.action_mode,
        fixed_action=args.fixed_action,
        temperature=args.temperature,
        maskgit_steps=args.maskgit_steps,
        panel_height=args.panel_height,
        fps=args.fps,
        preload_ratio=args.preload_ratio,
    )


if __name__ == "__main__":
    main()
