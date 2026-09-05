"""Bounded synthetic CPU diagnostic; it is an implementation check, not generalization."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from models.dynamics import DynamicsModel
from models.latent_actions import LatentActionModel
from models.video_tokenizer import VideoTokenizer
from utils.action_eval import evaluate_action_ablations


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(23)
    common = dict(frame_size=(16, 16), patch_size=4, embed_dim=24,
                  num_heads=4, hidden_dim=48, num_blocks=1)
    # Moving squares provide video motion but no externally supplied actions.
    clips = torch.full((4, 3, 3, 16, 16), -1.0)
    for b in range(4):
        for t in range(3):
            x = 2 + b + t
            clips[b, t, :, 4:8, x:x + 3] = 1.0
    tokenizer = VideoTokenizer(**common, latent_dim=2, num_bins=2)
    lam = LatentActionModel(**common, n_actions=4)
    dynamics = DynamicsModel(**common, latent_dim=2, num_bins=2, conditioning_dim=2)
    optimizer = torch.optim.AdamW(dynamics.parameters(), lr=2e-3)
    with torch.no_grad():
        tokens = tokenizer.tokenize(clips)
        latents = tokenizer.quantizer.get_latents_from_indices(tokens)
        actions = lam.encode(clips)
    initial = None
    for _ in range(12):
        optimizer.zero_grad()
        _, _, loss = dynamics(latents, targets=tokens, conditioning=actions,
                              objective_mode="next_frame")
        initial = float(loss.detach()) if initial is None else initial
        loss.backward()
        optimizer.step()
    metrics = evaluate_action_ablations(dynamics, tokenizer, lam, clips, seed=23, decoding_steps=2)
    print(json.dumps({"seed": 23, "updates": 12, "initial_train_ce": initial,
                      "final_train_ce": float(loss.detach()), "metrics": metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
