#!/usr/bin/env python3
"""Interactive TinyWorlds Pong demo — sanity check for action-conditioned generation."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gradio as gr
import torch
from PIL import Image

from simulator.engine import EngineConfig, SurgeryWorldEngine

_engine = None
_last_latency = 0.0

# Classic TinyWorlds Pong uses 4 latent actions ( unsupervised ).
# Map UI buttons to action ids 0–3.
ACTIONS = {
    "noop": 0,
    "up": 1,
    "down": 2,
    "alt": 3,
}


def _resolve_run_root() -> Optional[str]:
    env = os.environ.get("PONG_RUN_ROOT")
    if env and os.path.isdir(env):
        return env
    results = os.path.join(ROOT, "results")
    if not os.path.isdir(results):
        return None
    # Prefer newest run that has a dynamics checkpoint
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


def _init(device: str, temperature: float) -> Tuple[str, Image.Image]:
    global _engine, _last_latency
    _last_latency = 0.0
    run_root = _resolve_run_root()
    if not run_root:
        blank = Image.new("RGB", (256, 256), (10, 10, 30))
        return "⚠️ No Pong training run found under results/.", blank

    vt = _latest_step(os.path.join(run_root, "video_tokenizer", "checkpoints"), "video_tokenizer_step_")
    lam = _latest_step(os.path.join(run_root, "latent_actions", "checkpoints"), "latent_actions_step_")
    dyn = _latest_step(os.path.join(run_root, "dynamics", "checkpoints"), "dynamics_step_")

    cfg = EngineConfig(
        device=device,
        temperature=temperature,
        dataset="PONG",
        preload_ratio=1.0,
        context_window=4,
        maskgit_steps=8,
        display_size=384,
        video_tokenizer_path=vt,
        latent_actions_path=lam,
        dynamics_path=dyn,
        use_latest_checkpoints=False,
    )
    _engine = SurgeryWorldEngine(cfg)
    frame = _engine.reset()
    status = (
        f"✅ **Pong world model loaded**\n\n"
        f"Run: `{run_root}`\n"
        f"Dynamics: `{os.path.basename(dyn)}`\n\n"
        "Use ↑ / ↓ — if training worked, the paddle/ball motion should respond to actions "
        "(latent codes are unsupervised; try all 4 buttons)."
    )
    return status, frame


def _step(action_name: str) -> Tuple[Image.Image, Optional[Image.Image], str]:
    global _last_latency
    if _engine is None:
        blank = Image.new("RGB", (256, 256), (10, 10, 30))
        return blank, None, "Press **New Game** first."
    result = _engine.step(ACTIONS[action_name])
    _last_latency = result.latency_ms
    stats = (
        f"Step **{result.step_index}** · action `{action_name}`=`{result.action_id}` · "
        f"{result.latency_ms:.0f} ms"
    )
    return result.frame, result.ground_truth_frame, stats


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="TinyWorlds Pong — Sanity Demo") as demo:
        gr.Markdown(
            """
            # TinyWorlds Pong
            Sanity check: original TinyWorlds-style stack trained on **PONG**.

            If action conditioning works, changing ↑/↓ should change the predicted frame
            (paddle / ball), not just copy the context.
            """
        )
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    gt = gr.Image(label="Ground truth continuation", type="pil", height=384)
                    pred = gr.Image(label="World model prediction", type="pil", height=384)
                stats = gr.Markdown("Press **New Game** to start.")
            with gr.Column(scale=2):
                status = gr.Markdown("")
                device = gr.Radio(
                    choices=["cuda", "cpu"],
                    value="cuda" if torch.cuda.is_available() else "cpu",
                    label="Device",
                )
                temperature = gr.Slider(0.0, 1.5, value=0.2, step=0.1, label="Temperature")
                new_game = gr.Button("🔄 New Game", variant="primary")
                gr.Markdown("### Actions (latent ids 0–3)")
                with gr.Row():
                    btn_noop = gr.Button("⏸ No-op (0)")
                    btn_up = gr.Button("↑ Up (1)")
                with gr.Row():
                    btn_down = gr.Button("↓ Down (2)")
                    btn_alt = gr.Button("↔ Alt (3)")

        new_game.click(_init, inputs=[device, temperature], outputs=[status, pred]).then(
            lambda: (None, "Game started — try ↑ / ↓."),
            outputs=[gt, stats],
        )
        for btn, name in [
            (btn_noop, "noop"),
            (btn_up, "up"),
            (btn_down, "down"),
            (btn_alt, "alt"),
        ]:
            btn.click(lambda n=name: _step(n), outputs=[pred, gt, stats])

        demo.load(
            _init,
            inputs=[device, temperature],
            outputs=[status, pred],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    build_ui().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
