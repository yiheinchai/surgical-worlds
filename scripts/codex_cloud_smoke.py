"""CPU smoke check for the cloud environment; no datasets, credentials, or training job."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from models.dynamics import DynamicsModel
from models.latent_actions import LatentActionModel
from models.video_tokenizer import VideoTokenizer
from datasets.datasets import PicoDoomDataset
from utils.config import DistributedConfig


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    common = dict(frame_size=(16, 16), patch_size=4, embed_dim=32,
                  num_heads=4, hidden_dim=64, num_blocks=1)
    video = torch.rand(2, 4, 3, 16, 16) * 2 - 1
    vt = VideoTokenizer(**common, latent_dim=3, num_bins=4)
    lam = LatentActionModel(**common, n_actions=4)
    dyn = DynamicsModel(**common, latent_dim=3, num_bins=4, conditioning_dim=2)
    vt_loss, recon = vt(video)
    vt_loss.backward()
    lam_loss, prediction = lam(video)
    lam_loss.backward()
    with torch.no_grad():
        tokens = vt.tokenize(video)
        latents = vt.quantizer.get_latents_from_indices(tokens)
        actions = lam.encode(video)
    logits, mask, loss = dyn(latents, targets=tokens, conditioning=actions,
                             objective_mode="next_frame")
    loss.backward()
    assert recon.shape == video.shape
    assert prediction.shape == video[:, 1:].shape
    assert logits.shape == (*tokens.shape, 64)
    assert mask[:, -1].all() and not mask[:, :-1].any()
    assert torch.isfinite(torch.stack([vt_loss, lam_loss, loss])).all()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in dyn.parameters())
    dyn.eval()
    with torch.no_grad():
        rollout = dyn.forward_inference(latents[:, :3], 1, 2,
            vt.quantizer.get_latents_from_indices, conditioning=actions)
    assert rollout.shape == latents.shape and torch.isfinite(rollout).all()
    print(json.dumps({'status': 'passed', 'device': 'cpu', 'torch': torch.__version__,
                      'video_tokenizer_loss': vt_loss.item(), 'lam_loss': lam_loss.item(),
                      'dynamics_loss': loss.item(), 'rollout_shape': list(rollout.shape)}))


if __name__ == '__main__':
    main()
