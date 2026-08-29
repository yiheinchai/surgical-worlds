"""
Genie 3-style surgery world model engine.

Maintains rolling context and autoregressively generates the next surgical
frame conditioned on user-selected latent instrument actions.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from einops import repeat
from PIL import Image

from datasets.data_utils import load_data_and_data_loaders
from simulator.frame_utils import tensor_frame_to_pil, tensor_sequence_to_pil_list
from utils.inference_utils import load_models
from utils.utils import find_latest_checkpoint


@dataclass
class EngineConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    context_window: int = 4
    prediction_horizon: int = 1
    maskgit_steps: int = 8
    temperature: float = 0.4
    amp: bool = torch.cuda.is_available()
    tf32: bool = torch.cuda.is_available()
    compile: bool = False
    dataset: str = "LAPAROSCOPIC"
    preload_ratio: Optional[float] = None
    video_tokenizer_path: Optional[str] = None
    latent_actions_path: Optional[str] = None
    dynamics_path: Optional[str] = None
    use_latest_checkpoints: bool = True


@dataclass
class StepResult:
    frame: Image.Image
    action_id: int
    step_index: int
    latency_ms: float
    session_frames: int
    ground_truth_frame: Optional[Image.Image] = None


@dataclass
class SessionState:
    context_frames: torch.Tensor  # [1, T_ctx, C, H, W]
    ground_truth_frames: Optional[torch.Tensor] = None  # [1, T_full, C, H, W]
    seed_index: int = 0
    action_history: List[int] = field(default_factory=list)
    step_count: int = 0
    all_frames: List[Image.Image] = field(default_factory=list)


class SurgeryWorldEngine:
    """Playable surgery world model — Genie 3 architecture for laparoscopy."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self._setup_device()
        self._load_models()
        self.session: Optional[SessionState] = None
        self.n_actions = self.latent_action_model.quantizer.codebook_size

    def _setup_device(self) -> None:
        if self.config.tf32 and self.config.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _resolve_checkpoint(self, path: Optional[str], model_type: str) -> str:
        if path and os.path.exists(path):
            return path
        if self.config.use_latest_checkpoints:
            found = find_latest_checkpoint(os.getcwd(), model_type)
            if found:
                return found
        raise FileNotFoundError(
            f"No {model_type} checkpoint found. Train with "
            f"`python scripts/full_train.py --config configs/surgical_training.yaml` "
            f"or enable demo mode."
        )

    def _load_models(self) -> None:
        vt_path = self._resolve_checkpoint(self.config.video_tokenizer_path, "video_tokenizer")
        lam_path = self._resolve_checkpoint(self.config.latent_actions_path, "latent_actions")
        dyn_path = self._resolve_checkpoint(self.config.dynamics_path, "dynamics")

        self.video_tokenizer, self.latent_action_model, self.dynamics_model = load_models(
            vt_path, lam_path, dyn_path, self.config.device, use_actions=True
        )

        if self.config.compile:
            self.video_tokenizer = torch.compile(self.video_tokenizer, mode="reduce-overhead")
            self.latent_action_model = torch.compile(self.latent_action_model, mode="reduce-overhead")
            self.dynamics_model = torch.compile(self.dynamics_model, mode="reduce-overhead")

        self.video_tokenizer.eval()
        self.latent_action_model.eval()
        self.dynamics_model.eval()

    def reset(self, seed_index: Optional[int] = None) -> Image.Image:
        """Start a new simulated procedure from a real surgical context window."""
        overrides = {}
        if self.config.preload_ratio is not None:
            overrides["preload_ratio"] = self.config.preload_ratio

        # Load extra frames so we can show ground-truth continuation for comparison.
        frames_to_load = self.config.context_window + 16
        _, _, data_loader, _, _ = load_data_and_data_loaders(
            dataset=self.config.dataset,
            batch_size=1,
            num_frames=frames_to_load,
            **overrides,
        )

        idx = seed_index if seed_index is not None else random.randint(0, len(data_loader.dataset) - 1)
        full_sequence = data_loader.dataset[idx][0].unsqueeze(0).to(self.config.device)
        context = full_sequence[:, : self.config.context_window]

        self.session = SessionState(
            context_frames=context,
            ground_truth_frames=full_sequence,
            seed_index=idx,
            action_history=[],
            step_count=0,
            all_frames=tensor_sequence_to_pil_list(context, upscale=True),
        )
        return self.session.all_frames[-1]

    def _build_action_latent(self, action_id: int, context_frames: torch.Tensor) -> torch.Tensor:
        self.session.action_history.append(action_id)
        recent = self.session.action_history[-self.config.context_window :]
        recent_tensor = repeat(
            torch.tensor(recent, device=self.config.device, dtype=torch.long),
            "i -> 1 i",
        )
        action_latent = self.latent_action_model.quantizer.get_latents_from_indices(recent_tensor)

        if self.config.prediction_horizon > 1:
            action_latent = repeat(
                action_latent, "b 1 a -> b ph a", ph=self.config.prediction_horizon
            )

        if len(recent) < self.config.context_window:
            pad_count = self.config.context_window - len(recent) + 1
            gt_pad = self.latent_action_model.encode(context_frames[:, :pad_count])
            action_latent = torch.cat(
                [self.latent_action_model.quantizer(gt_pad), action_latent], dim=1
            )
        return action_latent

    @torch.inference_mode()
    def step(self, action_id: int) -> StepResult:
        """
        Execute one simulation step: user action → next surgical frame.
        This is the core Genie 3 interaction loop.
        """
        if self.session is None:
            self.reset()

        t0 = time.perf_counter()
        context = self.session.context_frames

        video_indices = self.video_tokenizer.tokenize(context)
        video_latents = self.video_tokenizer.quantizer.get_latents_from_indices(video_indices)
        action_latent = self._build_action_latent(action_id, context)

        def idx_to_latents(idx):
            return self.video_tokenizer.quantizer.get_latents_from_indices(idx, dim=-1)

        autocast_dtype = torch.bfloat16 if self.config.amp else None
        with torch.amp.autocast(
            self.config.device.split(":")[0],
            enabled=self.config.amp and self.config.device.startswith("cuda"),
            dtype=autocast_dtype,
        ):
            next_latents = self.dynamics_model.forward_inference(
                context_latents=video_latents,
                prediction_horizon=self.config.prediction_horizon,
                num_steps=self.config.maskgit_steps,
                index_to_latents_fn=idx_to_latents,
                conditioning=action_latent,
                temperature=self.config.temperature,
            )

        next_frames = self.video_tokenizer.detokenize(next_latents)
        new_frame = next_frames[:, -self.config.prediction_horizon :]

        # Roll context window forward (autoregressive, like Genie 3)
        self.session.context_frames = torch.cat(
            [self.session.context_frames, new_frame], dim=1
        )[:, -self.config.context_window :]

        pil_frame = tensor_frame_to_pil(new_frame[0, -1], upscale=True)
        self.session.all_frames.append(pil_frame)
        self.session.step_count += 1

        gt_frame = None
        if self.session.ground_truth_frames is not None:
            gt_idx = self.config.context_window + self.session.step_count - 1
            if gt_idx < self.session.ground_truth_frames.shape[1]:
                gt_frame = tensor_frame_to_pil(
                    self.session.ground_truth_frames[0, gt_idx], upscale=True
                )

        latency_ms = (time.perf_counter() - t0) * 1000
        return StepResult(
            frame=pil_frame,
            action_id=action_id,
            step_index=self.session.step_count,
            latency_ms=latency_ms,
            session_frames=len(self.session.all_frames),
            ground_truth_frame=gt_frame,
        )

    def get_session_gif_frames(self) -> List[Image.Image]:
        if self.session is None:
            return []
        return list(self.session.all_frames)

    @property
    def is_ready(self) -> bool:
        return self.session is not None
