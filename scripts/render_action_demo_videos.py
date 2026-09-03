#!/usr/bin/env python3
"""Render MP4 demos of world-model inference under scripted instrument actions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulator.engine import EngineConfig, SurgeryWorldEngine

IDLE, L_GRASP, L_RELEASE, R_GRASP, R_RELEASE, RETRACT, CAMERA, CAUTERY = range(8)
# Laparoscopic screen-plane motion (indices 3=left, 4=right in latent action codebook)
L_LEFT, L_RIGHT = 3, 4

ACTION_SEQUENCES: Dict[str, List[int]] = {
    "circle_instruments": [L_GRASP, RETRACT, R_GRASP, RETRACT, CAMERA, RETRACT] * 4,
    "dual_grasp_sweep": [L_GRASP, L_GRASP, R_GRASP, R_GRASP, RETRACT, RETRACT] * 3,
    "cautery_pass": [CAUTERY, CAUTERY, RETRACT, CAMERA, CAUTERY, RETRACT] * 3,
    "camera_orbit": [CAMERA, RETRACT, CAMERA, L_GRASP, CAMERA, R_GRASP] * 3,
    "left_then_right": [L_LEFT] * 8 + [L_RIGHT] * 8,
}


def _pil_to_bgr(img: Image.Image, size: Tuple[int, int]) -> np.ndarray:
    rgb = np.array(img.convert("RGB").resize(size, Image.Resampling.LANCZOS))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _annotate(frame_bgr: np.ndarray, title: str, action_id: int, step: int) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    bar = np.zeros((36, w, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)
    out = np.vstack([bar, out])
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 8), f"{title}  |  step {step}  |  action {action_id}{_action_label(action_id)}", fill=(255, 220, 180), font=font)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


_ACTION_NAMES = {
    0: "hold", 1: "grasp/L-grasp", 2: "release/L-rel", 3: "left", 4: "right",
    5: "retract/down", 6: "camera", 7: "cautery/up",
}


def _action_label(action_id: int) -> str:
    name = _ACTION_NAMES.get(action_id)
    return f" ({name})" if name else ""


def _panel_size(height: int = 384, aspect_width_scale: float = 1.0) -> Tuple[int, int]:
    width = max(2, int(round(height * aspect_width_scale)))
    if width % 2:
        width += 1
    if height % 2:
        height += 1
    return width, height


def render_sequence(
    engine: SurgeryWorldEngine,
    name: str,
    actions: List[int],
    output_path: Path,
    panel_size: Optional[Tuple[int, int]] = None,
    fps: int = 4,
    seed_index: int = 0,
) -> Path:
    aspect = engine._display_aspect()
    if panel_size is None:
        panel_size = _panel_size(384, aspect)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engine.reset(seed_index=seed_index)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (panel_size[0] * 2, panel_size[1] + 36))

    context_frames = engine.get_session_gif_frames()
    for i, frame in enumerate(context_frames):
        gt_bgr = _annotate(_pil_to_bgr(frame, panel_size), f"{name} — context (GT)", IDLE, i)
        pred_bgr = _annotate(_pil_to_bgr(frame, panel_size), f"{name} — context", IDLE, i)
        writer.write(np.hstack([gt_bgr, pred_bgr]))

    for step, action_id in enumerate(actions, start=1):
        result = engine.step(action_id)
        gt_img = getattr(result, "ground_truth_frame", None) or result.frame
        gt_bgr = _annotate(_pil_to_bgr(gt_img, panel_size), "Ground truth", action_id, step)
        pred_bgr = _annotate(_pil_to_bgr(result.frame, panel_size), "World model", action_id, step)
        writer.write(np.hstack([gt_bgr, pred_bgr]))

    writer.release()

    # Re-encode to H.264 — OpenCV mp4v is not playable in GitHub/browser players.
    h264_path = output_path.with_suffix(".h264.mp4")
    import subprocess
    for codec in ("libx264", "libopenh264"):
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(output_path),
                "-c:v", codec, "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-crf", "20",
                str(h264_path),
            ],
            capture_output=True,
        )
        if result.returncode == 0 and h264_path.exists() and h264_path.stat().st_size > 0:
            h264_path.replace(output_path)
            break

    print(f"Wrote {output_path} ({len(actions)} inference steps)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("inference_results/action_demos"))
    parser.add_argument("--dataset", default="ROBOTIC_LAPAROSCOPIC")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--sequences", nargs="*", default=list(ACTION_SEQUENCES.keys()))
    args = parser.parse_args()

    os.chdir(ROOT)
    cfg = EngineConfig(device=args.device, dataset=args.dataset, preload_ratio=0.08)
    engine = SurgeryWorldEngine(cfg)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.sequences:
        if name not in ACTION_SEQUENCES:
            continue
        render_sequence(engine, name, ACTION_SEQUENCES[name], out_dir / f"crcd_{name}.mp4", seed_index=args.seed)

    print(f"Done — videos in {out_dir}")


if __name__ == "__main__":
    main()
