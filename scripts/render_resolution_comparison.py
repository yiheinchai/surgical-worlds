#!/usr/bin/env python3
"""Render left-then-right inference at multiple display resolutions + timing report."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.render_action_demo_videos import (  # noqa: E402
    ACTION_SEQUENCES,
    L_LEFT,
    L_RIGHT,
    _action_label,
    _panel_size,
)
from simulator.engine import EngineConfig, SurgeryWorldEngine  # noqa: E402
from simulator.frame_utils import ROBOTIC_WIDESCREEN_ASPECT, tensor_frame_to_pil  # noqa: E402

DEFAULT_HEIGHTS = [128, 192, 256, 384, 512]
SEQUENCE = "left_then_right"


def _font(size: int = 16):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _upscale(img: Image.Image, panel_h: int, aspect: float) -> Image.Image:
    panel_w, panel_h = _panel_size(panel_h, aspect)
    return img.convert("RGB").resize((panel_w, panel_h), Image.Resampling.LANCZOS)


def _annotate_native(img: Image.Image, title: str, action_id: int, step: int) -> Image.Image:
    panel_w, panel_h = img.size
    bar_h = 36
    canvas = Image.new("RGB", (panel_w, panel_h + bar_h), (30, 30, 30))
    canvas.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 8),
        f"{title}  |  step {step}  |  action {action_id}{_action_label(action_id)}",
        fill=(255, 220, 180),
        font=_font(),
    )
    return canvas


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _encode_h264(src: Path) -> None:
    dst = src.with_suffix(".h264.mp4")
    for codec in ("libx264", "libopenh264"):
        r = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(src),
                "-c:v", codec, "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-crf", "20", str(dst),
            ],
            capture_output=True,
        )
        if r.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
            dst.replace(src)
            return


def collect_inference_frames_fixed(
    engine: SurgeryWorldEngine,
    actions: List[int],
    seed_index: int,
) -> Tuple[List[Tuple[Image.Image, Image.Image, int, int]], float]:
    aspect = engine._display_aspect()
    engine.reset(seed_index=seed_index)

    frames: List[Tuple[Image.Image, Image.Image, int, int]] = []
    ctx = engine.session.context_frames
    for i in range(ctx.shape[1]):
        pil = tensor_frame_to_pil(ctx[0, i], upscale=False)
        up = _upscale(pil, 128, aspect)
        frames.append((up, up, 0, i))

    t0 = time.perf_counter()
    for step, action_id in enumerate(actions, start=1):
        result = engine.step(action_id)
        gt_idx = engine.config.context_window + step - 1
        if engine.session.ground_truth_frames is not None and gt_idx < engine.session.ground_truth_frames.shape[1]:
            gt_pil = tensor_frame_to_pil(engine.session.ground_truth_frames[0, gt_idx], upscale=False)
        else:
            gt_pil = tensor_frame_to_pil(engine.session.context_frames[0, -1], upscale=False)
        # predicted frame tensor: need last generated — stored in session all_frames as upscaled PIL
        # Re-run detokenize path is heavy; use context last frame after step
        pred_pil = tensor_frame_to_pil(engine.session.context_frames[0, -1], upscale=False)
        frames.append((gt_pil, pred_pil, action_id, step))
    elapsed = time.perf_counter() - t0
    return frames, elapsed


def write_video(
    native_frames: List[Tuple[Image.Image, Image.Image, int, int]],
    panel_h: int,
    aspect: float,
    output_path: Path,
    fps: int = 4,
) -> None:
    sample = _annotate_native(_upscale(native_frames[0][0], panel_h, aspect), "GT", 0, 0)
    w, h = sample.size
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h)
    )
    for gt, pred, action_id, step in native_frames:
        gt_bgr = _pil_to_bgr(_annotate_native(_upscale(gt, panel_h, aspect), "Ground truth", action_id, step))
        pred_bgr = _pil_to_bgr(_annotate_native(_upscale(pred, panel_h, aspect), "World model", action_id, step))
        writer.write(np.hstack([gt_bgr, pred_bgr]))
    writer.release()
    _encode_h264(output_path)


def write_montage_still(
    native_frames: List[Tuple[Image.Image, Image.Image, int, int]],
    heights: List[int],
    aspect: float,
    output_path: Path,
    frame_index: int = 10,
) -> None:
    gt, pred, action_id, step = native_frames[min(frame_index, len(native_frames) - 1)]
    rows = []
    max_w = 0
    for h in heights:
        label = Image.new("RGB", (200, h + 36), (20, 20, 20))
        d = ImageDraw.Draw(label)
        d.text((10, (h + 36) // 2 - 10), f"{h}px tall\n(native 128²)", fill=(200, 220, 255), font=_font(14))
        row = np.hstack([
            cv2.cvtColor(np.array(label), cv2.COLOR_RGB2BGR),
            _pil_to_bgr(_annotate_native(_upscale(gt, h, aspect), f"GT @ {h}px", action_id, step)),
            _pil_to_bgr(_annotate_native(_upscale(pred, h, aspect), f"Model @ {h}px", action_id, step)),
        ])
        max_w = max(max_w, row.shape[1])
        rows.append(row)
    padded = []
    for row in rows:
        if row.shape[1] < max_w:
            pad = np.zeros((row.shape[0], max_w - row.shape[1], 3), dtype=row.dtype)
            row = np.hstack([row, pad])
        padded.append(row)
    montage = np.vstack(padded)
    cv2.imwrite(str(output_path), montage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("inference_results/resolution_sweep"))
    parser.add_argument("--dataset", default="ROBOTIC_LAPAROSCOPIC")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--heights", nargs="*", type=int, default=DEFAULT_HEIGHTS)
    args = parser.parse_args()

    os.chdir(ROOT)
    actions = ACTION_SEQUENCES[SEQUENCE]
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = EngineConfig(device=args.device, dataset=args.dataset, preload_ratio=0.08)
    engine = SurgeryWorldEngine(cfg)
    aspect = engine._display_aspect()

    native_frames, infer_sec = collect_inference_frames_fixed(engine, actions, args.seed)
    per_step_ms = 1000 * infer_sec / len(actions)

    report = {
        "model_native_resolution": 128,
        "note": "Display heights upscale the same 128x128 model output. True sharpness needs retraining at higher frame_size.",
        "sequence": SEQUENCE,
        "inference_steps": len(actions),
        "total_inference_sec": round(infer_sec, 2),
        "per_step_ms": round(per_step_ms, 1),
        "display_heights": [],
    }

    for h in args.heights:
        t0 = time.perf_counter()
        mp4 = out_dir / f"crcd_left_then_right_{h}p.mp4"
        write_video(native_frames, h, aspect, mp4)
        encode_sec = time.perf_counter() - t0
        pw, ph = _panel_size(h, aspect)
        report["display_heights"].append({
            "height_px": h,
            "panel_size": [pw, ph],
            "video_size": [pw * 2, ph + 36],
            "encode_sec": round(encode_sec, 2),
            "file": mp4.name,
        })
        print(f"Wrote {mp4} ({pw}x{ph} per panel)")

    montage = out_dir / "resolution_comparison_montage.png"
    write_montage_still(native_frames, args.heights, aspect, montage)
    report["montage"] = montage.name

    report_path = out_dir / "resolution_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Done — {out_dir}")


if __name__ == "__main__":
    main()
