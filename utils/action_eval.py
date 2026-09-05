"""Deterministic, video-only dynamics evaluation and action ablations."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fixed_clip_indices(n_windows: int, n_clips: int, seed: int) -> list[int]:
    """Choose a reproducible subset without relying on loader shuffle state."""
    if n_windows < 1 or n_clips < 1:
        raise ValueError("n_windows and n_clips must be positive")
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n_windows, generator=generator)[:min(n_windows, n_clips)].tolist()


@torch.no_grad()
def evaluate_action_ablations(
    dynamics,
    tokenizer,
    latent_action_model,
    clips: torch.Tensor,
    *,
    seed: int = 0,
    decoding_steps: int = 4,
) -> dict[str, float]:
    """Compare inferred, constant and shuffled video-inferred actions.

    Clips are ``[B,T,C,H,W]`` and contain no action labels. The copy baseline is
    measured in pixels. CE objectives separately report fully-hidden next-frame
    prediction and legacy partial-token reconstruction.
    """
    if clips.ndim != 5 or clips.shape[1] < 2:
        raise ValueError("clips must have shape [B,T,C,H,W] with T >= 2")
    tokens = tokenizer.tokenize(clips)
    latents = tokenizer.quantizer.get_latents_from_indices(tokens, dim=-1)
    inferred = latent_action_model.encode(clips)
    # A real inferred code, fixed across batch/time; never invent a labelled no-op.
    constant = inferred[:1, :1].expand_as(inferred).clone()
    generator = torch.Generator(device=inferred.device).manual_seed(seed)
    permutation = torch.randperm(inferred.numel() // inferred.shape[-1], generator=generator,
                                device=inferred.device)
    shuffled = inferred.reshape(-1, inferred.shape[-1])[permutation].reshape_as(inferred)

    metrics: dict[str, float] = {
        "copy_frame/l1": float(F.l1_loss(clips[:, -2], clips[:, -1])),
    }
    for name, actions in (("inferred_action", inferred), ("constant_action", constant),
                          ("shuffled_action", shuffled)):
        _, mask, full_loss = dynamics(latents, training=True, conditioning=actions,
                                      targets=tokens, objective_mode="next_frame")
        if not mask[:, -1].all() or mask[:, :-1].any():
            raise AssertionError("next-frame objective leaked or masked history tokens")
        metrics[f"{name}/fully_hidden_ce"] = float(full_loss)

        # Reset RNG per ablation so partial masks are identical.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            _, _, partial_loss = dynamics(latents, training=True, conditioning=actions,
                                          targets=tokens, objective_mode="legacy_maskgit")
        metrics[f"{name}/partial_reconstruction_ce"] = float(partial_loss)

        rollout = dynamics.forward_inference(
            latents[:, :-1], 1, decoding_steps,
            tokenizer.quantizer.get_latents_from_indices, conditioning=actions,
            temperature=0.0,
        )
        prediction = tokenizer.decoder(rollout[:, -1:])[:, 0]
        metrics[f"{name}/one_step_l1"] = float(F.l1_loss(prediction, clips[:, -1]))
    return metrics
