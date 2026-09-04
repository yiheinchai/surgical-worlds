#!/usr/bin/env python3
"""Compare multiple dynamics checkpoints side-by-side (shared VT + LAM).

Supports multi-step free rollout: each action advances every checkpoint's
own context in parallel so you can keep acting and watch them diverge.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gradio as gr
import torch
from einops import repeat
from PIL import Image, ImageDraw

from datasets.data_utils import load_data_and_data_loaders
from simulator.frame_utils import tensor_frame_to_pil
from utils.inference_utils import load_models
from utils.utils import load_dynamics_from_checkpoint

DEFAULT_STEPS = [0, 10000, 20000, 30000, 38000]


def _blank(w: int = 256, h: int = 256) -> Image.Image:
    return Image.new("RGB", (w, h), (10, 10, 30))


def _label(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, 30), fill=(0, 0, 0))
    draw.text((8, 7), text, fill=(240, 240, 240))
    return out


def _hstrip(frames: List[Image.Image]) -> Image.Image:
    w, h = frames[0].size
    out = Image.new("RGB", (w * len(frames), h), (0, 0, 0))
    for i, im in enumerate(frames):
        out.paste(im, (i * w, 0))
    return out


def _parse_steps(raw: Optional[str]) -> List[int]:
    if not raw:
        return list(DEFAULT_STEPS)
    out = []
    for part in raw.split(","):
        part = part.strip().lower().replace("k", "000")
        if part:
            out.append(int(part))
    return out


class MultiDynComparer:
    def __init__(
        self,
        run_root: str,
        dyn_steps: List[int],
        device: str = "cuda",
        context_window: int = 4,
        maskgit_steps: int = 8,
        temperature: float = 0.0,
        display_size: int = 256,
        preload_ratio: float = 0.05,
    ):
        self.device = device
        self.context_window = context_window
        self.maskgit_steps = maskgit_steps
        self.temperature = temperature
        self.display_size = display_size
        self.preload_ratio = preload_ratio
        self.run_root = run_root

        vt = self._latest(os.path.join(run_root, "video_tokenizer", "checkpoints"), "video_tokenizer_step_")
        lam = self._latest(os.path.join(run_root, "latent_actions", "checkpoints"), "latent_actions_step_")
        self.video_tokenizer, self.latent_action_model, _ = load_models(
            vt, lam, self._dyn_path(dyn_steps[0]), device, use_actions=True
        )
        self.video_tokenizer.eval()
        self.latent_action_model.eval()

        self.dynamics: Dict[int, torch.nn.Module] = {}
        missing = []
        for step in dyn_steps:
            path = self._dyn_path(step)
            if not os.path.isdir(path):
                missing.append(step)
                continue
            model, _ = load_dynamics_from_checkpoint(path, device)
            model.eval()
            self.dynamics[step] = model
        if not self.dynamics:
            raise FileNotFoundError(f"No dynamics checkpoints found; missing={missing}")
        self.steps = sorted(self.dynamics.keys())
        self.missing = missing

        # Per-checkpoint rolling contexts (free rollout).
        self.contexts: Dict[int, torch.Tensor] = {}
        self.action_histories: Dict[int, List[int]] = {}
        self.gt_full: Optional[torch.Tensor] = None
        self.seed_index = 0
        self.rollout_step = 0

    @staticmethod
    def _latest(ckpt_dir: str, prefix: str) -> str:
        best = None
        for name in os.listdir(ckpt_dir):
            if not name.startswith(prefix):
                continue
            step = int(name.rsplit("_", 1)[-1])
            if best is None or step > best[0]:
                best = (step, name)
        if best is None:
            raise FileNotFoundError(ckpt_dir)
        return os.path.join(ckpt_dir, best[1])

    def _dyn_path(self, step: int) -> str:
        return os.path.join(self.run_root, "dynamics", "checkpoints", f"dynamics_step_{step}")

    def _to_pil(self, frame_chw: torch.Tensor) -> Image.Image:
        return tensor_frame_to_pil(frame_chw, upscale=True, display_size=self.display_size)

    def new_context(self, seed_index: Optional[int] = None) -> Tuple[Image.Image, Image.Image]:
        frames_to_load = self.context_window + 64
        _, _, loader, _, _ = load_data_and_data_loaders(
            dataset="PICODOOM",
            batch_size=1,
            num_frames=frames_to_load,
            preload_ratio=self.preload_ratio,
        )
        idx = seed_index if seed_index is not None else random.randint(0, len(loader.dataset) - 1)
        full = loader.dataset[idx][0].unsqueeze(0).to(self.device)
        self.gt_full = full
        seed_ctx = full[:, : self.context_window].clone()
        self.contexts = {s: seed_ctx.clone() for s in self.steps}
        self.action_histories = {s: [] for s in self.steps}
        self.seed_index = idx
        self.rollout_step = 0
        ctx_pil = self._to_pil(seed_ctx[0, -1])
        # Initial strip: GT context last frame repeated as baseline row
        strip = self._current_strip(include_gt=True, use_last_ctx=True)
        return ctx_pil, strip

    def _action_latent(self, step: int, action_id: int) -> torch.Tensor:
        context = self.contexts[step]
        hist = self.action_histories[step] + [action_id]
        recent = hist[-self.context_window :]
        recent_tensor = repeat(
            torch.tensor(recent, device=self.device, dtype=torch.long), "i -> 1 i"
        )
        action_latent = self.latent_action_model.quantizer.get_latents_from_indices(recent_tensor)
        if len(recent) < self.context_window:
            pad_count = self.context_window - len(recent) + 1
            gt_pad = self.latent_action_model.encode(context[:, :pad_count])
            action_latent = torch.cat(
                [self.latent_action_model.quantizer(gt_pad), action_latent], dim=1
            )
        return action_latent

    def _predict_one(self, step: int, action_id: int) -> torch.Tensor:
        """Return next frame tensor [1,1,C,H,W] without mutating state."""
        context = self.contexts[step]
        dyn = self.dynamics[step]
        video_indices = self.video_tokenizer.tokenize(context)
        video_latents = self.video_tokenizer.quantizer.get_latents_from_indices(video_indices)
        action_latent = self._action_latent(step, action_id)

        def idx_to_latents(idx):
            return self.video_tokenizer.quantizer.get_latents_from_indices(idx, dim=-1)

        with torch.amp.autocast("cuda", enabled=self.device.startswith("cuda"), dtype=torch.bfloat16):
            next_latents = dyn.forward_inference(
                context_latents=video_latents,
                prediction_horizon=1,
                num_steps=self.maskgit_steps,
                index_to_latents_fn=idx_to_latents,
                conditioning=action_latent,
                temperature=self.temperature,
            )
        next_frames = self.video_tokenizer.detokenize(next_latents)
        return next_frames[:, -1:]

    def _current_strip(self, include_gt: bool = True, use_last_ctx: bool = False) -> Image.Image:
        cells = []
        if include_gt and self.gt_full is not None:
            if use_last_ctx:
                gt_frame = self.gt_full[0, self.context_window - 1]
                cells.append(_label(self._to_pil(gt_frame), "GT context"))
            else:
                gt_idx = self.context_window + self.rollout_step - 1
                if 0 <= gt_idx < self.gt_full.shape[1]:
                    cells.append(_label(self._to_pil(self.gt_full[0, gt_idx]), "GT next"))
                else:
                    cells.append(_label(_blank(self.display_size, self.display_size), "GT (end)"))
        for s in self.steps:
            frame = self.contexts[s][0, -1]
            cells.append(_label(self._to_pil(frame), f"dyn {s}"))
        return _hstrip(cells)

    @torch.inference_mode()
    def step_all(self, action_id: int) -> Tuple[Image.Image, str]:
        """Apply the same latent action to every checkpoint and advance each context."""
        if not self.contexts:
            self.new_context()
        t0 = time.perf_counter()
        action_id = int(action_id)
        preds = {}
        for s in self.steps:
            new_frame = self._predict_one(s, action_id)
            preds[s] = new_frame
            self.contexts[s] = torch.cat([self.contexts[s], new_frame], dim=1)[
                :, -self.context_window :
            ]
            self.action_histories[s].append(action_id)
        self.rollout_step += 1

        strip = self._current_strip(include_gt=True, use_last_ctx=False)

        # Metrics vs GT and early vs late
        gt_idx = self.context_window + self.rollout_step - 1
        mae_str = "n/a"
        early_late = 0.0
        if self.gt_full is not None and gt_idx < self.gt_full.shape[1]:
            gt_pil = self._to_pil(self.gt_full[0, gt_idx])
            gt_t = torch.tensor(list(gt_pil.getdata()), dtype=torch.float32)
            maes = []
            tensors = []
            for s in self.steps:
                pil = self._to_pil(preds[s][0, 0])
                t = torch.tensor(list(pil.getdata()), dtype=torch.float32)
                tensors.append(t)
                maes.append(float(torch.mean(torch.abs(t - gt_t))))
            mae_str = " · ".join(f"{s}:{m:.1f}" for s, m in zip(self.steps, maes))
            if len(tensors) >= 2:
                early_late = float(torch.mean(torch.abs(tensors[0] - tensors[-1])))

        ms = (time.perf_counter() - t0) * 1000
        hist = self.action_histories[self.steps[0]]
        hist_str = ",".join(str(a) for a in hist[-12:])
        warn = ""
        if self.rollout_step >= 4:
            warn = "\n\n⚠️ Free-rollout drift usually grows after a few steps — compare MAE trend across ckpts."
        note = (
            f"Seed `{self.seed_index}` · rollout step **{self.rollout_step}** · "
            f"just took latent **{action_id}** · {ms:.0f} ms\n\n"
            f"Action history: `{hist_str}`\n\n"
            f"MAE to GT (this step): {mae_str}\n\n"
            f"|dyn@{self.steps[0]} − dyn@{self.steps[-1]}| Δ ≈ **{early_late:.1f}**"
            f"{warn}"
        )
        if self.missing:
            note += f"\n\n⚠️ missing ckpts skipped: {self.missing}"
        return strip, note

    def resync_from_gt(self) -> Tuple[Image.Image, str]:
        """Reset all ckpt contexts to GT at current rollout depth (teacher-force)."""
        if self.gt_full is None:
            return _blank(), "No GT loaded."
        end = min(self.context_window + self.rollout_step, self.gt_full.shape[1])
        start = max(0, end - self.context_window)
        ctx = self.gt_full[:, start:end].clone()
        self.contexts = {s: ctx.clone() for s in self.steps}
        self.action_histories = {s: [] for s in self.steps}
        # Keep rollout_step so GT column stays aligned; or reset? Keep depth.
        strip = self._current_strip(include_gt=True, use_last_ctx=(self.rollout_step == 0))
        return strip, (
            f"Resynced all checkpoints to GT frames [{start}:{end}] "
            f"at rollout depth {self.rollout_step}. Action histories cleared."
        )


_comparer: Optional[MultiDynComparer] = None


def _resolve_run_root() -> str:
    env = os.environ.get("PICODOOM_RUN_ROOT") or os.environ.get("NG_RUN_ROOT_DIR")
    if env and os.path.isdir(env):
        return env
    raise FileNotFoundError("Set PICODOOM_RUN_ROOT")


def _ensure(device: str, temperature: float, maskgit_steps: int) -> MultiDynComparer:
    global _comparer
    steps = _parse_steps(os.environ.get("DEMO_DYN_STEPS"))
    need_new = (
        _comparer is None
        or _comparer.device != device
        or abs(_comparer.temperature - temperature) > 1e-6
        or _comparer.maskgit_steps != int(maskgit_steps)
    )
    if need_new:
        _comparer = MultiDynComparer(
            run_root=_resolve_run_root(),
            dyn_steps=steps,
            device=device,
            temperature=temperature,
            maskgit_steps=int(maskgit_steps),
            preload_ratio=float(os.environ.get("DEMO_PRELOAD_RATIO", "0.05")),
        )
    return _comparer


def _load(device: str, temperature: float, maskgit_steps: int):
    c = _ensure(device, temperature, maskgit_steps)
    ctx, strip = c.new_context()
    status = (
        f"✅ Loaded VT+LAM + **{len(c.steps)}** dynamics in parallel: `{c.steps}`\n\n"
        "Press a **Latent** button to take that action on **all** checkpoints at once. "
        "Keep pressing to roll forward. **New seed** resets; **Resync from GT** "
        "teacher-forces contexts back to real frames at the current depth."
    )
    return status, ctx, strip, "Ready — pick a latent action."


def _act(action_id: int, device: str, temperature: float, maskgit_steps: int):
    c = _ensure(device, temperature, maskgit_steps)
    if not c.contexts:
        c.new_context()
    strip, note = c.step_all(int(action_id))
    return strip, note


def _new_seed(device: str, temperature: float, maskgit_steps: int):
    c = _ensure(device, temperature, maskgit_steps)
    ctx, strip = c.new_context()
    return ctx, strip, f"New seed `{c.seed_index}` — rollout reset to 0."


def _resync(device: str, temperature: float, maskgit_steps: int):
    c = _ensure(device, temperature, maskgit_steps)
    strip, note = c.resync_from_gt()
    return strip, note


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="TinyWorlds PicoDoom — Checkpoint Compare") as demo:
        gr.Markdown(
            """
            # Dynamics checkpoint compare (multi-step)
            Same latent action → **all** checkpoints advance together.

            Row: `GT | dyn@0 | dyn@10k | dyn@20k | dyn@30k | latest`
            """
        )
        status = gr.Markdown("")
        with gr.Row():
            context = gr.Image(label="Seed context (last frame)", type="pil", height=256)
            compare = gr.Image(label="GT vs dynamics (current rollout step)", type="pil", height=280)
        stats = gr.Markdown("")
        with gr.Row():
            device = gr.Radio(
                choices=["cuda", "cpu"],
                value="cuda" if torch.cuda.is_available() else "cpu",
                label="Device",
            )
            temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature")
            maskgit_steps = gr.Slider(1, 16, value=8, step=1, label="MaskGIT steps")
        with gr.Row():
            load_btn = gr.Button("Load / reload models", variant="secondary")
            seed_btn = gr.Button("🎲 New seed")
            resync_btn = gr.Button("🎯 Resync all from GT")
        gr.Markdown("### Take action (applies to every checkpoint)")
        with gr.Row():
            btn0 = gr.Button("Latent 0", variant="primary")
            btn1 = gr.Button("Latent 1", variant="primary")
            btn2 = gr.Button("Latent 2", variant="primary")
            btn3 = gr.Button("Latent 3", variant="primary")

        load_btn.click(_load, inputs=[device, temperature, maskgit_steps], outputs=[status, context, compare, stats])
        seed_btn.click(_new_seed, inputs=[device, temperature, maskgit_steps], outputs=[context, compare, stats])
        resync_btn.click(_resync, inputs=[device, temperature, maskgit_steps], outputs=[compare, stats])
        btn0.click(lambda d, t, m: _act(0, d, t, m), inputs=[device, temperature, maskgit_steps], outputs=[compare, stats])
        btn1.click(lambda d, t, m: _act(1, d, t, m), inputs=[device, temperature, maskgit_steps], outputs=[compare, stats])
        btn2.click(lambda d, t, m: _act(2, d, t, m), inputs=[device, temperature, maskgit_steps], outputs=[compare, stats])
        btn3.click(lambda d, t, m: _act(3, d, t, m), inputs=[device, temperature, maskgit_steps], outputs=[compare, stats])

        demo.load(_load, inputs=[device, temperature, maskgit_steps], outputs=[status, context, compare, stats])
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
