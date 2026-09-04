#!/usr/bin/env python3
"""Interactive TinyWorlds PicoDoom demo — action-conditioned world-model play."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont

from simulator.engine import EngineConfig, SurgeryWorldEngine

_engine = None


def _blank(size: int = 256) -> Image.Image:
    return Image.new("RGB", (size, size), (10, 10, 30))


def _resolve_run_root() -> Optional[str]:
    env = os.environ.get("PICODOOM_RUN_ROOT") or os.environ.get("NG_RUN_ROOT_DIR")
    if env and os.path.isdir(env):
        return env
    results = os.path.join(ROOT, "results")
    if not os.path.isdir(results):
        return None
    candidates = sorted(
        (os.path.join(results, d) for d in os.listdir(results)),
        key=os.path.getmtime,
        reverse=True,
    )
    for root in candidates:
        dyn = os.path.join(root, "dynamics", "checkpoints")
        if os.path.isdir(dyn) and os.listdir(dyn):
            return root
    return None


def _latest_step(ckpt_dir: str, prefix: str) -> str:
    steps = []
    for name in os.listdir(ckpt_dir):
        if name.startswith(prefix):
            try:
                steps.append((int(name.split("_")[-1]), name))
            except ValueError:
                continue
    if not steps:
        raise FileNotFoundError(f"No {prefix}* in {ckpt_dir}")
    steps.sort()
    return os.path.join(ckpt_dir, steps[-1][1])


def _label_frame(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, 28), fill=(0, 0, 0))
    draw.text((8, 6), text, fill=(240, 240, 240))
    return out


def _montage(frames: List[Image.Image], labels: List[str]) -> Image.Image:
    labeled = [_label_frame(f, lab) for f, lab in zip(frames, labels)]
    w, h = labeled[0].size
    grid = Image.new("RGB", (w * 2, h * 2), (0, 0, 0))
    for i, im in enumerate(labeled):
        grid.paste(im, ((i % 2) * w, (i // 2) * h))
    return grid


def _init(device: str, temperature: float, maskgit_steps: int) -> Tuple[str, Image.Image, Optional[Image.Image], str]:
    global _engine
    run_root = _resolve_run_root()
    if not run_root:
        b = _blank()
        return "⚠️ No PicoDoom run found. Set PICODOOM_RUN_ROOT.", b, None, "Load failed."

    vt = _latest_step(os.path.join(run_root, "video_tokenizer", "checkpoints"), "video_tokenizer_step_")
    lam = _latest_step(os.path.join(run_root, "latent_actions", "checkpoints"), "latent_actions_step_")
    dyn = _latest_step(os.path.join(run_root, "dynamics", "checkpoints"), "dynamics_step_")

    cfg = EngineConfig(
        device=device,
        temperature=temperature,
        dataset="PICODOOM",
        preload_ratio=float(os.environ.get("DEMO_PRELOAD_RATIO", "0.05")),
        context_window=4,
        maskgit_steps=int(maskgit_steps),
        display_size=384,
        video_tokenizer_path=vt,
        latent_actions_path=lam,
        dynamics_path=dyn,
        use_latest_checkpoints=False,
    )
    _engine = SurgeryWorldEngine(cfg)
    frame = _engine.reset()
    dyn_step = os.path.basename(dyn).rsplit("_", 1)[-1]
    status = (
        f"✅ **PicoDoom world model loaded** (mid-training)\n\n"
        f"VT `{os.path.basename(vt)}` · LAM `{os.path.basename(lam)}` · Dyn `{os.path.basename(dyn)}` "
        f"(~{int(dyn_step)/3000:.0f}% of planned Dyn steps)\n\n"
        "**Actions are unsupervised latent codes** — they are *not* mapped to Doom keys "
        "(forward/strafe/shoot). The LAM discovered 4 discrete codes from video alone; "
        "use **Probe all 4** to see what each code does from the same context.\n\n"
        "**Collapse after a few steps** is expected this early: free rollout feeds predictions "
        "back into context. Use **Resync from GT** to reset context to real frames and compare "
        "single-step quality vs multi-step drift."
    )
    stats = "Game started. Prefer **Probe all 4** first, then step with a latent id."
    return status, frame, None, stats


def _step(action_id: int) -> Tuple[Image.Image, Optional[Image.Image], str]:
    if _engine is None:
        return _blank(), None, "Press **New Game** first."
    result = _engine.step(int(action_id))
    warn = ""
    if result.step_index >= 4:
        warn = " · ⚠️ free-rollout drift likely — try **Resync from GT**"
    stats = (
        f"Step **{result.step_index}** · latent action **{result.action_id}** · "
        f"{result.latency_ms:.0f} ms{warn}"
    )
    return result.frame, result.ground_truth_frame, stats


def _probe() -> Tuple[Image.Image, str]:
    if _engine is None:
        return _blank(768), "Press **New Game** first."
    frames = []
    for a in range(4):
        frames.append(_engine.preview_next(a))
    grid = _montage(frames, [f"latent {i}" for i in range(4)])
    # crude diversity hint
    arrs = [torch.tensor(list(f.getdata()), dtype=torch.float32) for f in frames]
    diffs = []
    for i in range(4):
        for j in range(i + 1, 4):
            diffs.append(float(torch.mean(torch.abs(arrs[i] - arrs[j]))))
    mean_diff = sum(diffs) / max(len(diffs), 1)
    note = (
        f"Same context → next frame for each latent id. "
        f"Mean pairwise pixel Δ ≈ **{mean_diff:.1f}** "
        f"(near 0 ⇒ actions look identical / weak conditioning)."
    )
    return grid, note


def _resync() -> Tuple[Image.Image, Optional[Image.Image], str]:
    if _engine is None:
        return _blank(), None, "Press **New Game** first."
    frame = _engine.resync_from_ground_truth()
    if frame is None:
        return _blank(), None, "No GT available to resync."
    return frame, None, "Context resynced from ground truth (teacher-force). Action history cleared."


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="TinyWorlds PicoDoom — Live Demo") as demo:
        gr.Markdown(
            """
            # TinyWorlds PicoDoom
            Mid-training interactive check. Dynamics is only ~10% done — expect blurry / collapsing rollouts.

            - **Left**: ground-truth video continuation  
            - **Right**: world-model prediction  
            - **Probe**: one-step previews for latent ids 0–3 (discover what actions mean)
            """
        )
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    gt = gr.Image(label="Ground truth continuation", type="pil", height=384)
                    pred = gr.Image(label="World model prediction", type="pil", height=384)
                probe_img = gr.Image(label="Action probe (same context → 4 latents)", type="pil", height=512)
                stats = gr.Markdown("Press **New Game** to start.")
            with gr.Column(scale=2):
                status = gr.Markdown("")
                device = gr.Radio(
                    choices=["cuda", "cpu"],
                    value="cuda" if torch.cuda.is_available() else "cpu",
                    label="Device",
                )
                temperature = gr.Slider(0.0, 1.5, value=0.0, step=0.1, label="Temperature (0 = greedy)")
                maskgit_steps = gr.Slider(1, 16, value=8, step=1, label="MaskGIT steps")
                new_game = gr.Button("🔄 New Game", variant="primary")
                probe_btn = gr.Button("🔎 Probe all 4 latents", variant="secondary")
                resync_btn = gr.Button("🎯 Resync context from GT")
                gr.Markdown("### Step with latent id (not Doom keys)")
                with gr.Row():
                    btn0 = gr.Button("Latent 0")
                    btn1 = gr.Button("Latent 1")
                with gr.Row():
                    btn2 = gr.Button("Latent 2")
                    btn3 = gr.Button("Latent 3")

        new_game.click(
            _init, inputs=[device, temperature, maskgit_steps], outputs=[status, pred, gt, stats]
        )
        probe_btn.click(_probe, outputs=[probe_img, stats])
        resync_btn.click(_resync, outputs=[pred, gt, stats])
        for btn, aid in [(btn0, 0), (btn1, 1), (btn2, 2), (btn3, 3)]:
            btn.click(lambda a=aid: _step(a), outputs=[pred, gt, stats])

        demo.load(
            _init,
            inputs=[device, temperature, maskgit_steps],
            outputs=[status, pred, gt, stats],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    build_ui().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
