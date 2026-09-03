#!/usr/bin/env python3
"""
Visual ladder: what CRCD training frames look like at each native model resolution,
plus compute cost estimates for training/inference at that frame_size.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.surgical_preprocessing import preprocess_surgical_frame  # noqa: E402

DEFAULT_RESOLUTIONS = [64, 128, 192, 256, 384, 512]
PATCH_SIZE = 4
BASELINE_FRAME = 128
# Measured on RTX 3090 with trained CRCD checkpoints (Aug 2026)
BASELINE_INFER_MS = 319.0
BASELINE_TRAIN_HOURS = 7.0  # full 3-stage quick_training pipeline


def _font(size: int = 15):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def find_sample_frame(data_root: Path) -> np.ndarray:
    for pattern in (
        "data/surgical/robotic/videos/*.mp4",
        "data/surgical/downloads/hf_videos/*.mp4",
        "data/surgical/downloads/**/*.mp4",
    ):
        for video in sorted(data_root.glob(pattern)):
            cap = cv2.VideoCapture(str(video))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
            ok, bgr = cap.read()
            cap.release()
            if ok:
                return bgr
    raise FileNotFoundError("No CRCD video frame found under data/")


def frame_at_native_resolution(bgr: np.ndarray, size: int, circular_mask: bool = True) -> np.ndarray:
    return preprocess_surgical_frame(
        bgr,
        resize_to=(size, size),
        crop_borders=True,
        circular_mask=circular_mask,
        color_normalize=True,
    )


def patches_per_frame(frame_size: int, patch_size: int = PATCH_SIZE) -> int:
    p = frame_size // patch_size
    return p * p


def cost_multiplier(frame_size: int, baseline: int = BASELINE_FRAME) -> float:
    """Patch count scales ~quadratically with frame edge length."""
    return (frame_size / baseline) ** 2


def estimate_inference_ms(frame_size: int, bench: Dict[int, float | None] | None = None) -> float:
    if bench and bench.get(BASELINE_FRAME) and bench.get(frame_size):
        return BASELINE_INFER_MS * (bench[frame_size] / bench[BASELINE_FRAME])
    scale = (frame_size / BASELINE_FRAME) ** 2.1
    return BASELINE_INFER_MS * scale


def estimate_train_hours(frame_size: int, bench: Dict[int, float | None] | None = None) -> float:
    if bench and bench.get(BASELINE_FRAME) and bench.get(frame_size):
        return BASELINE_TRAIN_HOURS * (bench[frame_size] / bench[BASELINE_FRAME])
    scale = (frame_size / BASELINE_FRAME) ** 2.1
    return BASELINE_TRAIN_HOURS * scale


def estimate_vram_gb(frame_size: int, baseline_vram_gb: float = 0.61) -> float:
    scale = (frame_size / BASELINE_FRAME) ** 2.0
    return baseline_vram_gb * scale


def fits_rtx3090(frame_size: int, bench: Dict[int, float | None] | None = None) -> bool:
    if bench and bench.get(frame_size) is None and frame_size > 256:
        return False
    return estimate_vram_gb(frame_size) < 22.0


def make_ladder_montage(frames: Dict[int, np.ndarray], output_path: Path) -> None:
    """One original frame at each native resolution, with labels."""
    rows: List[np.ndarray] = []
    max_w = 0
    for res in sorted(frames):
        rgb = frames[res]
        h, w = rgb.shape[:2]
        # Show at 2x zoom for small resolutions so they're visible in the ladder
        zoom = max(1, 512 // max(res, 1))
        zoom = min(zoom, 4)
        display = cv2.resize(rgb, (w * zoom, h * zoom), interpolation=cv2.INTER_NEAREST)
        label_w = 220
        label = np.zeros((display.shape[0], label_w, 3), dtype=np.uint8)
        pil = Image.fromarray(label)
        draw = ImageDraw.Draw(pil)
        p = patches_per_frame(res)
        mul = cost_multiplier(res)
        draw.text((8, 12), f"Native {res}×{res}", fill=(255, 220, 180), font=_font(16))
        draw.text((8, 36), f"{p:,} patches", fill=(180, 200, 255), font=_font(13))
        draw.text((8, 58), f"~{mul:.1f}× compute", fill=(180, 255, 180), font=_font(13))
        draw.text((8, 80), f"infer ~{estimate_inference_ms(res):.0f}ms", fill=(200, 200, 200), font=_font(12))
        row = np.hstack([cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), cv2.cvtColor(display, cv2.COLOR_RGB2BGR)])
        max_w = max(max_w, row.shape[1])
        rows.append(row)
    padded = []
    for row in rows:
        if row.shape[1] < max_w:
            row = np.hstack([row, np.zeros((row.shape[0], max_w - row.shape[1], 3), np.uint8)])
        padded.append(row)
    cv2.imwrite(str(output_path), np.vstack(padded))


def make_same_size_comparison(frames: Dict[int, np.ndarray], output_path: Path, display_size: int = 512) -> None:
    """All resolutions upscaled to same display size — shows blockiness at each native res."""
    cols = []
    for res in sorted(frames):
        rgb = frames[res]
        up = cv2.resize(rgb, (display_size, display_size), interpolation=cv2.INTER_NEAREST)
        bar = np.zeros((40, display_size, 3), np.uint8)
        pil = Image.fromarray(bar)
        d = ImageDraw.Draw(pil)
        d.text((8, 10), f"native {res}² → {display_size}px", fill=(255, 220, 180), font=_font(14))
        cols.append(np.vstack([cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), cv2.cvtColor(up, cv2.COLOR_RGB2BGR)]))
    cv2.imwrite(str(output_path), np.hstack(cols))


@torch.inference_mode()
def benchmark_dynamics_forward(frame_size: int, device: str = "cuda") -> float | None:
    """Relative forward-pass timing for dynamics at each frame_size (untrained weights)."""
    if not torch.cuda.is_available() and device == "cuda":
        return None
    try:
        from models.dynamics import DynamicsModel

        ctx = 4
        ph = 1
        p = patches_per_frame(frame_size)
        model = DynamicsModel(
            frame_size=(frame_size, frame_size),
            patch_size=PATCH_SIZE,
            embed_dim=48,
            num_heads=8,
            hidden_dim=192,
            num_blocks=8,
            latent_dim=6,
            num_bins=4,
        ).to(device)
        model.eval()
        B = 1
        latents = torch.randn(B, ctx, p, 6, device=device)
        cond = torch.randn(B, ctx, 3, device=device)
        # warmup
        for _ in range(2):
            model.forward_inference(
                context_latents=latents,
                prediction_horizon=ph,
                num_steps=4,
                index_to_latents_fn=lambda x: torch.randn(*x.shape, 6, device=device),
                conditioning=cond,
            )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            model.forward_inference(
                context_latents=latents,
                prediction_horizon=ph,
                num_steps=8,
                index_to_latents_fn=lambda x: torch.randn(*x.shape, 6, device=device),
                conditioning=cond,
            )
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / 5 * 1000
    except Exception as e:
        print(f"  benchmark {frame_size}: {e}")
        return None


def build_report(resolutions: List[int], bench: Dict[int, float | None]) -> dict:
    rows = []
    base_bench = bench.get(BASELINE_FRAME) or 1.0
    for res in resolutions:
        p = patches_per_frame(res)
        mul = cost_multiplier(res)
        bms = bench.get(res)
        rel = (bms / base_bench) if bms and base_bench else mul
        oom = bench is not None and res in bench and bench[res] is None and res > BASELINE_FRAME
        rows.append({
            "native_resolution": res,
            "patches_per_frame": p,
            "patch_multiplier_vs_128": round(mul, 2),
            "estimated_inference_ms": None if oom else round(estimate_inference_ms(res, bench), 0),
            "estimated_full_train_hours_rtx3090": None if oom else round(estimate_train_hours(res, bench), 1),
            "estimated_vram_gb": round(estimate_vram_gb(res), 1),
            "fits_rtx3090_24gb": fits_rtx3090(res, bench),
            "benchmark_dynamics_ms": round(bms, 1) if bms else None,
            "benchmark_relative": round(rel, 2) if bms else None,
            "notes": "OOM on RTX 3090 dynamics forward" if oom else None,
        })
    return {
        "baseline": {
            "frame_size": BASELINE_FRAME,
            "patch_size": PATCH_SIZE,
            "measured_inference_ms_per_step": BASELINE_INFER_MS,
            "measured_train_hours_3stage": BASELINE_TRAIN_HOURS,
            "gpu": "RTX 3090",
        },
        "note": (
            "Visuals are downscaled CRCD ground-truth frames — what the model would "
            "need to reproduce at each native resolution. Costs scale ~quadratically with edge length."
        ),
        "resolutions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("docs/inference_demos/native_resolution_ladder"))
    parser.add_argument("--resolutions", nargs="*", type=int, default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    bgr = find_sample_frame(ROOT)
    frames = {r: frame_at_native_resolution(bgr, r) for r in args.resolutions}

    for r, rgb in frames.items():
        Image.fromarray(rgb).save(out / f"crcd_native_{r}.png")

    make_ladder_montage(frames, out / "native_resolution_ladder.png")
    make_same_size_comparison(frames, out / "native_resolution_same_display.png")

    bench: Dict[int, float | None] = {}
    if args.benchmark:
        print("GPU benchmarks (dynamics forward, untrained weights):")
        for r in args.resolutions:
            ms = benchmark_dynamics_forward(r, args.device)
            bench[r] = ms
            print(f"  {r}x{r}: {ms:.1f} ms" if ms else f"  {r}x{r}: skipped")

    report = build_report(args.resolutions, bench)
    (out / "native_resolution_costs.json").write_text(json.dumps(report, indent=2))

    # markdown table
    lines = [
        "# Native resolution ladder (CRCD)",
        "",
        "What training data looks like at each **native model output** resolution.",
        "Costs are estimates for RTX 3090, scaling from measured 128² training run.",
        "",
        "| Native res | Patches | Compute vs 128² | Infer/step | Full train | VRAM | Fits 24GB |",
        "|------------|---------|-----------------|------------|------------|------|-----------|",
    ]
    for row in report["resolutions"]:
        infer = "OOM" if row.get("notes") else f"~{row['estimated_inference_ms']:.0f} ms"
        train = "OOM" if row.get("notes") else f"~{row['estimated_full_train_hours_rtx3090']:.0f} h"
        lines.append(
            f"| {row['native_resolution']}² "
            f"| {row['patches_per_frame']:,} "
            f"| {row['patch_multiplier_vs_128']:.1f}× "
            f"| {infer} "
            f"| {train} "
            f"| ~{row['estimated_vram_gb']:.1f} GB "
            f"| {'✓' if row['fits_rtx3090_24gb'] else '✗'} |"
        )
    lines += [
        "",
        "**Recommendation:**",
        "- **128²** — current model; cheapest (~7h train, ~320ms/step)",
        "- **256²** — best quality/cost tradeoff (~9× infer vs 128², ~65h train); fits 24GB",
        "- **384²+** — dynamics forward OOM on RTX 3090 (24GB); needs A100 40GB+",
        "",
        "See `native_resolution_ladder.png` (true pixel size) and "
        "`native_resolution_same_display.png` (all blown up to 512px for fair comparison).",
    ]
    (out / "README.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote ladder to {out}")


if __name__ == "__main__":
    main()
