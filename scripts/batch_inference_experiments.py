#!/usr/bin/env python3
"""Run a grid of inference experiments and write H.264 side-by-side MP4s + manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulator.engine import EngineConfig, SurgeryWorldEngine

IDLE, L_GRASP, L_RELEASE, R_GRASP, R_RELEASE, RETRACT, CAMERA, CAUTERY = range(8)
L_LEFT, L_RIGHT = 3, 4

ACTION_SEQUENCES: Dict[str, List[int]] = {
    "circle_instruments": [L_GRASP, RETRACT, R_GRASP, RETRACT, CAMERA, RETRACT] * 4,
    "dual_grasp_sweep": [L_GRASP, L_GRASP, R_GRASP, R_GRASP, RETRACT, RETRACT] * 3,
    "cautery_pass": [CAUTERY, CAUTERY, RETRACT, CAMERA, CAUTERY, RETRACT] * 3,
    "camera_orbit": [CAMERA, RETRACT, CAMERA, L_GRASP, CAMERA, R_GRASP] * 3,
    "left_then_right": [L_LEFT] * 8 + [L_RIGHT] * 8,
    "right_x16": [L_RIGHT] * 16,
    "left_x16": [L_LEFT] * 16,
    "idle_x12": [IDLE] * 12,
    "grasp_x12": [L_GRASP] * 12,
    "alt_lr_x12": [L_LEFT, L_RIGHT] * 6,
    "alt_34_x12": [3, 4] * 6,
}


@dataclass
class Experiment:
    name: str
    category: str
    seed: int = 3
    actions: Optional[List[int]] = None
    action_sequence: Optional[str] = None
    context_window: int = 4
    prediction_horizon: int = 1
    generation_steps: int = 16
    temperature: float = 0.0
    maskgit_steps: int = 8
    fps: int = 2
    dynamics_step: Optional[int] = None
    include_context: bool = True


def _font(size: int = 14) -> ImageFont.ImageFont:
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
    draw.text((6, 10), title, fill=(255, 220, 180), font=_font())
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _panel_size(height: int, aspect: float) -> Tuple[int, int]:
    width = max(2, int(round(height * aspect)))
    if width % 2:
        width += 1
    if height % 2:
        height += 1
    return width, height


def _encode_h264(path: Path) -> None:
    """Re-encode OpenCV mp4v output to H.264 for browser/GitHub playback.

    Without this step, browsers often show a solid green frame for MPEG-4 Part 2.
    """
    tmp = path.with_suffix(".h264.mp4")
    last_err = b""
    for codec in ("libx264", "libopenh264"):
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(path),
                "-c:v", codec, "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-crf", "20", "-an",
                str(tmp),
            ],
            capture_output=True,
        )
        if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(path)
            return
        last_err = result.stderr or b""
        tmp.unlink(missing_ok=True)
    raise RuntimeError(
        f"H.264 encode failed for {path} (need ffmpeg libx264). "
        f"stderr tail: {last_err[-500:]!r}"
    )


def _resolve_actions(exp: Experiment) -> List[int]:
    if exp.actions is not None:
        return list(exp.actions)
    if exp.action_sequence:
        return list(ACTION_SEQUENCES[exp.action_sequence])
    raise ValueError(f"Experiment {exp.name} needs actions or action_sequence")


def render_experiment(
    exp: Experiment,
    *,
    run_root: Path,
    output_path: Path,
    device: str,
    dataset: str,
    panel_height: int,
    default_dynamics_step: int,
) -> Dict[str, Any]:
    actions = _resolve_actions(exp)
    generation_steps = max(exp.generation_steps, len(actions))

    vt = run_root / "video_tokenizer" / "checkpoints" / "video_tokenizer_step_14000"
    lam = run_root / "latent_actions" / "checkpoints" / "latent_actions_step_7000"
    dyn_step = exp.dynamics_step or default_dynamics_step
    dyn = run_root / "dynamics" / "checkpoints" / f"dynamics_step_{dyn_step}"

    cfg = EngineConfig(
        device=device,
        dataset=dataset,
        preload_ratio=1.0,
        context_window=exp.context_window,
        generation_steps=generation_steps,
        prediction_horizon=exp.prediction_horizon,
        maskgit_steps=exp.maskgit_steps,
        temperature=exp.temperature,
        video_tokenizer_path=str(vt),
        latent_actions_path=str(lam),
        dynamics_path=str(dyn),
        use_latest_checkpoints=False,
        amp=device.startswith("cuda"),
        tf32=device.startswith("cuda"),
    )
    engine = SurgeryWorldEngine(cfg)
    aspect = engine._display_aspect()
    panel_size = _panel_size(panel_height, aspect)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        exp.fps,
        (panel_size[0] * 2, panel_size[1] + 40),
    )

    t0 = time.perf_counter()
    engine.reset(seed_index=exp.seed)

    if exp.include_context:
        for i, frame in enumerate(engine.get_session_gif_frames()):
            gt_bgr = _annotate(_pil_to_bgr(frame, panel_size), f"GT  |  ctx {i + 1}/{exp.context_window}")
            wm_bgr = _annotate(_pil_to_bgr(frame, panel_size), f"WM  |  ctx {i + 1} (GT copy)")
            writer.write(np.hstack([gt_bgr, wm_bgr]))

    for step, action_id in enumerate(actions, start=1):
        result = engine.step(action_id)
        gt_img = result.ground_truth_frame or result.frame
        meta = (
            f"seed={exp.seed} ctx={exp.context_window} ph={exp.prediction_horizon} "
            f"T={exp.temperature} dyn={dyn_step}"
        )
        gt_bgr = _annotate(_pil_to_bgr(gt_img, panel_size), f"GT  |  step {step}  |  a{action_id}  |  {meta}")
        wm_bgr = _annotate(_pil_to_bgr(result.frame, panel_size), f"WM  |  step {step}  |  a{action_id}")
        writer.write(np.hstack([gt_bgr, wm_bgr]))

    writer.release()
    _encode_h264(output_path)
    elapsed = time.perf_counter() - t0

    record = {
        **asdict(exp),
        "actions": actions,
        "output": str(output_path),
        "dynamics_step": dyn_step,
        "elapsed_sec": round(elapsed, 1),
        "size_bytes": output_path.stat().st_size if output_path.exists() else 0,
    }
    print(f"OK {exp.category}/{exp.name} ({elapsed:.1f}s) -> {output_path.name}")
    return record


def build_experiment_grid(default_dyn: int) -> List[Experiment]:
    exps: List[Experiment] = []

    # Different data sections (seeds)
    for seed in [0, 3, 10, 25, 50, 100, 250, 500]:
        for seq in ("left_then_right", "right_x16"):
            exps.append(Experiment(
                name=f"seed{seed:04d}_{seq}",
                category="by_seed",
                seed=seed,
                action_sequence=seq,
                generation_steps=20,
            ))

    # Action repertoire on seed 3
    for seq in ACTION_SEQUENCES:
        exps.append(Experiment(
            name=f"seed003_{seq}",
            category="by_action",
            seed=3,
            action_sequence=seq,
            generation_steps=max(20, len(ACTION_SEQUENCES[seq])),
        ))

    # Context length — trained at 4; architecture has no fixed T max (sinusoidal PE).
    # Practical inference cap ~16–32 on RTX 3090 @ 128² before VRAM blows up.
    for ctx in (2, 3, 4, 6, 8, 12, 16):
        exps.append(Experiment(
            name=f"ctx{ctx}_left_then_right",
            category="by_context",
            seed=3,
            action_sequence="left_then_right",
            context_window=ctx,
            generation_steps=20,
        ))
        exps.append(Experiment(
            name=f"ctx{ctx}_right_x12",
            category="by_context",
            seed=3,
            actions=[L_RIGHT] * 12,
            context_window=ctx,
            generation_steps=16,
        ))

    # Prediction horizon — decode H future frames per dynamics call
    for ph in (1, 2, 3, 4, 6, 8):
        exps.append(Experiment(
            name=f"ph{ph}_left_then_right",
            category="by_horizon",
            seed=3,
            action_sequence="left_then_right",
            prediction_horizon=ph,
            generation_steps=24,
        ))
        exps.append(Experiment(
            name=f"ph{ph}_right_x12",
            category="by_horizon",
            seed=3,
            actions=[L_RIGHT] * 12,
            prediction_horizon=ph,
            generation_steps=20,
        ))

    # Context × horizon interaction (small grid on seed 3)
    for ctx, ph in ((4, 2), (4, 4), (8, 2), (8, 4), (16, 2)):
        exps.append(Experiment(
            name=f"ctx{ctx}_ph{ph}_left_then_right",
            category="by_ctx_horizon",
            seed=3,
            action_sequence="left_then_right",
            context_window=ctx,
            prediction_horizon=ph,
            generation_steps=24,
        ))

    # Temperature
    for temp in (0.0, 0.4):
        exps.append(Experiment(
            name=f"temp{str(temp).replace('.', '')}_right_x16",
            category="by_temperature",
            seed=3,
            action_sequence="right_x16",
            temperature=temp,
            generation_steps=20,
        ))

    # Rollout length
    for n in (8, 16, 24):
        exps.append(Experiment(
            name=f"steps{n}_right_x{n}",
            category="by_length",
            seed=3,
            actions=[L_RIGHT] * n,
            generation_steps=n + 4,
            include_context=True,
        ))

    # MaskGIT decode steps
    for ms in (4, 16):
        exps.append(Experiment(
            name=f"maskgit{ms}_left_then_right",
            category="by_maskgit",
            seed=3,
            action_sequence="left_then_right",
            maskgit_steps=ms,
            generation_steps=20,
        ))

    # Dynamics checkpoint comparison (same clip, different training stages)
    for dyn in (20000, 25000, 27000, default_dyn):
        if dyn <= default_dyn:
            exps.append(Experiment(
                name=f"dyn{dyn}_left_then_right",
                category="by_checkpoint",
                seed=3,
                action_sequence="left_then_right",
                dynamics_step=dyn,
                generation_steps=20,
            ))

    return exps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True, help="e.g. results/2026_08_30_05_22_26")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/inference_sweep"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", default="ROBOTIC_LAPAROSCOPIC")
    parser.add_argument("--panel-height", type=int, default=384)
    parser.add_argument("--dynamics-step", type=int, default=30000)
    parser.add_argument("--categories", nargs="*", default=None, help="Filter categories")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.chdir(ROOT)
    run_root = args.run_root if args.run_root.is_absolute() else ROOT / args.run_root
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir

    experiments = build_experiment_grid(args.dynamics_step)
    if args.categories:
        experiments = [e for e in experiments if e.category in args.categories]
    if args.limit > 0:
        experiments = experiments[: args.limit]

    manifest: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_root": str(run_root),
        "dynamics_step_default": args.dynamics_step,
        "device": args.device,
        "dataset": args.dataset,
        "experiment_count": len(experiments),
        "experiments": [],
    }

    for i, exp in enumerate(experiments, 1):
        rel = out / exp.category / f"{exp.name}.mp4"
        try:
            record = render_experiment(
                exp,
                run_root=run_root,
                output_path=rel,
                device=args.device,
                dataset=args.dataset,
                panel_height=args.panel_height,
                default_dynamics_step=args.dynamics_step,
            )
            manifest["experiments"].append(record)
        except Exception as exc:
            print(f"FAIL {exp.category}/{exp.name}: {exc}", file=sys.stderr)
            manifest["experiments"].append({**asdict(exp), "error": str(exc)})

        if i % 5 == 0:
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ok = sum(1 for e in manifest["experiments"] if "error" not in e)
    print(f"\nDone: {ok}/{len(experiments)} succeeded -> {out}")


if __name__ == "__main__":
    main()
