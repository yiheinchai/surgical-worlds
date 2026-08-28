#!/usr/bin/env python3
"""
Interactive surgical world model simulation.

Extends TinyWorlds inference for laparoscopic / robotic laparoscopic videos.
Uses Genie-style latent actions to steer instrument motion during simulation.
"""

import os
import random
import sys

import torch
from einops import repeat

from datasets.data_utils import load_data_and_data_loaders
from utils.config import InferenceConfig, load_config
from utils.inference_utils import (
    get_action_latent,
    load_models,
    sample_random_action,
    visualize_inference,
)
from utils.surgical_inference_utils import (
    describe_simulation_mode,
    get_action_legend,
    print_action_legend,
    prompt_surgical_action,
)
from utils.utils import find_latest_checkpoint


def _resolve_checkpoints(args: InferenceConfig, use_latent_actions: bool) -> None:
    base_dir = os.getcwd()

    def missing(path) -> bool:
        return (path is None) or (not os.path.exists(path))

    if args.use_latest_checkpoints or missing(args.video_tokenizer_path):
        args.video_tokenizer_path = find_latest_checkpoint(base_dir, "video_tokenizer")
    if use_latent_actions and (args.use_latest_checkpoints or missing(args.latent_actions_path)):
        args.latent_actions_path = find_latest_checkpoint(base_dir, "latent_actions")
    if args.use_latest_checkpoints or missing(args.dynamics_path):
        args.dynamics_path = find_latest_checkpoint(base_dir, "dynamics")


def main() -> None:
    config_path = os.path.join(os.getcwd(), "configs", "surgical_inference.yaml")
    args: InferenceConfig = load_config(InferenceConfig, default_config_path=config_path)

    # Read surgical-specific overrides from yaml (not in InferenceConfig dataclass)
    from omegaconf import OmegaConf
    raw_cfg = OmegaConf.load(config_path)
    surgery_type = raw_cfg.get("surgery_type", "laparoscopic")
    show_action_legend = bool(raw_cfg.get("show_action_legend", True))

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    use_latent_actions = args.use_actions or args.use_gt_actions or args.use_interactive_mode
    _resolve_checkpoints(args, use_latent_actions)

    print(f"Video tokenizer: {args.video_tokenizer_path}")
    if use_latent_actions:
        print(f"Latent actions:  {args.latent_actions_path}")
    print(f"Dynamics:        {args.dynamics_path}")

    for path, name in [
        (args.video_tokenizer_path, "video_tokenizer"),
        (args.latent_actions_path if use_latent_actions else "ok", "latent_actions"),
        (args.dynamics_path, "dynamics"),
    ]:
        if name == "latent_actions" and not use_latent_actions:
            continue
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name} checkpoint: {path}")

    video_tokenizer, latent_action_model, dynamics_model = load_models(
        args.video_tokenizer_path,
        args.latent_actions_path,
        args.dynamics_path,
        args.device,
        use_actions=use_latent_actions,
    )

    if args.compile:
        video_tokenizer = torch.compile(video_tokenizer, mode="reduce-overhead")
        if use_latent_actions:
            latent_action_model = torch.compile(latent_action_model, mode="reduce-overhead")
        dynamics_model = torch.compile(dynamics_model, mode="reduce-overhead")

    frames_to_load = args.context_window + args.generation_steps * args.prediction_horizon
    data_overrides = {"preload_ratio": args.preload_ratio} if args.preload_ratio is not None else {}
    _, _, data_loader, _, _ = load_data_and_data_loaders(
        dataset=args.dataset,
        batch_size=1,
        num_frames=frames_to_load,
        **data_overrides,
    )

    random_idx = random.randint(0, len(data_loader.dataset) - 1)
    ground_truth_frames = data_loader.dataset[random_idx][0].unsqueeze(0).to(args.device)
    context_frames = ground_truth_frames[:, : args.context_window]
    generated_frames = context_frames.clone()

    n_actions = latent_action_model.quantizer.codebook_size if use_latent_actions else 0
    inferred_actions = []
    legend = get_action_legend(surgery_type, n_actions) if use_latent_actions else {}

    describe_simulation_mode(
        surgery_type=surgery_type,
        n_actions=n_actions,
        generation_steps=args.generation_steps,
        context_window=args.context_window,
    )
    if show_action_legend and use_latent_actions:
        print_action_legend(legend)

    max_possible = ground_truth_frames.shape[1] - args.context_window
    effective_steps = (
        min(args.generation_steps, max_possible)
        if args.teacher_forced
        else args.generation_steps
    )

    for step in range(effective_steps):
        print(f"\n--- Simulating frame {step + 1}/{effective_steps} ---")

        if args.teacher_forced:
            context_frames = ground_truth_frames[
                :, step : step + args.context_window
            ]
        else:
            context_frames = generated_frames[:, -args.context_window :]

        video_indices = video_tokenizer.tokenize(context_frames)
        video_latents = video_tokenizer.quantizer.get_latents_from_indices(video_indices)

        if args.use_interactive_mode and use_latent_actions:
            action_id = prompt_surgical_action(step, n_actions, legend, default_action=0)
            sampled_action_index = torch.tensor([action_id], device=args.device)
            inferred_actions.append(sampled_action_index)
            recent = inferred_actions[-args.context_window :]
            recent_tensor = repeat(torch.tensor(recent, device=args.device), "i -> 1 i")
            action_latent = latent_action_model.quantizer.get_latents_from_indices(recent_tensor)
            if args.prediction_horizon > 1:
                action_latent = repeat(action_latent, "b 1 a -> b ph a", ph=args.prediction_horizon)
            if len(recent) < args.context_window:
                gt_pad = latent_action_model.encode(
                    context_frames[:, : args.context_window - len(recent) + 1]
                )
                action_latent = torch.cat(
                    [latent_action_model.quantizer(gt_pad), action_latent], dim=1
                )
        else:
            _, action_latent = get_action_latent(
                args, inferred_actions, n_actions, context_frames, latent_action_model, step
            )

        def idx_to_latents(idx):
            return video_tokenizer.quantizer.get_latents_from_indices(idx, dim=-1)

        autocast_dtype = torch.bfloat16 if args.amp else None
        with torch.amp.autocast("cuda", enabled=args.amp, dtype=autocast_dtype):
            next_latents = dynamics_model.forward_inference(
                context_latents=video_latents,
                prediction_horizon=args.prediction_horizon,
                num_steps=10,
                index_to_latents_fn=idx_to_latents,
                conditioning=action_latent,
                temperature=args.temperature,
            )

        next_frames = video_tokenizer.detokenize(next_latents)
        generated_frames = torch.cat(
            [generated_frames, next_frames[:, -args.prediction_horizon :]], dim=1
        )

        if args.use_interactive_mode:
            latest = next_frames[0, -1].detach().cpu()
            latest = ((latest + 1) / 2).clamp(0, 1)
            print(f"  Predicted frame shape: {tuple(latest.shape)}")

    output_dir = os.path.join(os.getcwd(), "inference_results", "surgery_simulation")
    os.makedirs(output_dir, exist_ok=True)
    prev_cwd = os.getcwd()
    os.chdir(output_dir)
    try:
        visualize_inference(
            generated_frames,
            ground_truth_frames,
            inferred_actions,
            args.fps,
            use_actions=use_latent_actions,
        )
    finally:
        os.chdir(prev_cwd)


if __name__ == "__main__":
    main()
