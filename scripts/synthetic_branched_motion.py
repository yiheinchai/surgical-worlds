"""Learnability probe on RGB-only translations; not PicoDoom gameplay training.

Each source texture has four successors with identical color statistics. The
training path receives only RGB pairs, never the transformation index. Validation
uses different source textures. Wraparound is artificial and limits transfer.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.bounded_picodoom import Experiment


def translated_pairs(textures):
    """Four possible next RGB frames for every identical first RGB frame."""
    shifts = [(0, 4), (0, -4), (4, 0), (-4, 0)]
    successors = torch.stack([torch.roll(textures, s, (-2, -1)) for s in shifts], 1)
    previous = textures[:, None].expand_as(successors)
    return torch.stack([previous, successors], 2).flatten(0, 1)


class BranchedExperiment(Experiment):
    def __init__(self, args):
        super().__init__(args)
        self.synthetic = {}
        provenance = {}
        for split, lo, hi, count in [('train', 3000, 46000, 64),
                                     ('validation', 47000, 53000, 32),
                                     ('test', 54000, 59785, 32)]:
            frames = np.linspace(lo, hi - 1, count, dtype=int)
            textures = self.data[frames].float().permute(0, 3, 1, 2) / 127.5 - 1
            self.synthetic[split] = translated_pairs(textures)
            self.starts[split] = np.arange(count * 4)
            provenance[split] = {'source_texture_frames': frames.tolist(),
                                 'branches_per_texture': 4}
        self.config.update(data_supervision='synthetic RGB translations only; no transformation labels',
                           interpretation='learnability diagnostic, not gameplay evidence',
                           learning_rate_comparison='both paired runs use 0.001',
                           textures=provenance, wraparound_translation_pixels=4)
        (self.out / 'config.json').write_text(json.dumps(self.config, indent=2))
        (self.out / 'source_starts.json').write_text(json.dumps(provenance, indent=2))
        if self.wandb:
            self.run.config.update(self.config, allow_val_change=True)

    def clips(self, split, indices):
        return self.synthetic[split][indices].to(self.device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--information-weight', type=float, default=0.)
    p.add_argument('--key-file', required=True)
    p.add_argument('--wandb-mode', choices=['online','offline'], default='online')
    p.add_argument('--init', choices=['checkpoint','scratch'], default='checkpoint')
    a = p.parse_args()
    args = SimpleNamespace(stage='lam', name=a.name, data='data/picodoom_frames.h5',
        checkpoints='data/checkpoints', output='results/bounded-20260905', device='cuda',
        seed=0, steps=3000, max_seconds=900, batch=16, train_clips=256, eval_clips=128,
        lr=.001, log_every=50, eval_every=500, init=a.init, objective='next_frame',
        information_weight=a.information_weight, lam_weights=None, dynamics_weights=None,
        pairwise_actions=False, test=False, rollouts=False, wandb_mode=a.wandb_mode,
        wandb_key_file=a.key_file)
    BranchedExperiment(args).train()


if __name__ == '__main__':
    main()
