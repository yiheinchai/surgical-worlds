import h5py
import numpy as np
import pytest
import torch

from datasets.datasets import VideoHDF5Dataset
from models.dynamics import DynamicsModel
from models.latent_actions import LatentActionModel
from models.video_tokenizer import VideoTokenizer
from utils.action_eval import evaluate_action_ablations, fixed_clip_indices
from utils.training_utils import MicrobatchLossMean


def tiny_dynamics():
    torch.manual_seed(4)
    return DynamicsModel(frame_size=(8, 8), patch_size=4, embed_dim=12, num_heads=3,
                         hidden_dim=24, num_blocks=1, latent_dim=2, num_bins=2,
                         conditioning_dim=2)


def test_microbatch_logging_is_undivided_mean():
    parameter = torch.tensor(1.0, requires_grad=True)
    meter = MicrobatchLossMean(2)
    meter.backward_loss(parameter * 2).backward(retain_graph=True)
    meter.backward_loss(parameter * 6).backward()
    assert meter.mean == pytest.approx(4.0)
    assert parameter.grad.item() == pytest.approx(4.0)


def test_temporal_split_has_disjoint_source_frames(tmp_path):
    path = tmp_path / "frames.h5"
    frames = np.arange(40, dtype=np.uint8)[:, None, None, None].repeat(3, axis=3)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("frames", data=frames)
    kwargs = dict(video_path="unused", save_path=str(path), num_frames=3, fps=60,
                  split_policy="temporal_disjoint", validation_fraction=0.25)
    train = VideoHDF5Dataset(train=True, **kwargs)
    val = VideoHDF5Dataset(train=False, **kwargs)
    assert set(train.data[:, 0, 0, 0]).isdisjoint(set(val.data[:, 0, 0, 0]))
    assert len(train) > 0 and len(val) > 0


def test_temporal_split_rejects_empty_window_partition(tmp_path):
    path = tmp_path / "short.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("frames", data=np.zeros((8, 2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="empty validation window partition"):
        VideoHDF5Dataset("unused", save_path=str(path), train=False, num_frames=4,
                         fps=60, split_policy="temporal_disjoint")


def test_next_frame_is_fully_hidden_target_only_causal_and_finite():
    model = tiny_dynamics()
    indices = torch.randint(0, 4, (2, 3, 4))
    latents = torch.randn(2, 3, 4, 2)
    actions = torch.randn(2, 2, 2)
    logits, mask, loss = model(latents, targets=indices, conditioning=actions,
                               objective_mode="next_frame")
    changed_target = latents.clone()
    changed_target[:, -1] = torch.randn_like(changed_target[:, -1]) * 100
    logits_changed, _, _ = model(changed_target, targets=indices, conditioning=actions,
                                 objective_mode="next_frame")
    assert not mask[:, :-1].any() and mask[:, -1].all()
    assert torch.equal(logits[:, -1], logits_changed[:, -1])  # no clean target leakage
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0


def test_legacy_objective_remains_selectable():
    model = tiny_dynamics()
    latents = torch.randn(2, 3, 4, 2)
    targets = torch.randint(0, 4, (2, 3, 4))
    _, mask, loss = model(latents, targets=targets, conditioning=torch.randn(2, 2, 2),
                          objective_mode="legacy_maskgit")
    assert mask.shape == targets.shape and torch.isfinite(loss)


def test_action_t_conditions_following_frame_not_initial_frame():
    model = tiny_dynamics().eval()
    latents = torch.randn(1, 3, 4, 2)
    actions = torch.zeros(1, 2, 2)
    changed = actions.clone()
    changed[:, 0] = 10
    base, _, _ = model(latents, training=False, conditioning=actions)
    intervention, _, _ = model(latents, training=False, conditioning=changed)
    # Adaptive conditioning prepends zeros: a_0 aligns with z_1, never z_0.
    assert torch.equal(base[:, 0], intervention[:, 0])
    assert not torch.equal(base[:, 1], intervention[:, 1])


def test_video_only_ablation_is_deterministic_and_reports_separate_tasks():
    torch.manual_seed(8)
    common = dict(frame_size=(8, 8), patch_size=4, embed_dim=12, num_heads=3,
                  hidden_dim=24, num_blocks=1)
    tokenizer = VideoTokenizer(**common, latent_dim=2, num_bins=2)
    lam = LatentActionModel(**common, n_actions=4)
    dynamics = tiny_dynamics()
    clips = torch.rand(3, 3, 3, 8, 8) * 2 - 1
    first = evaluate_action_ablations(dynamics, tokenizer, lam, clips, seed=19, decoding_steps=2)
    second = evaluate_action_ablations(dynamics, tokenizer, lam, clips, seed=19, decoding_steps=2)
    assert first == second
    assert fixed_clip_indices(10, 4, 3) == fixed_clip_indices(10, 4, 3)
    for baseline in ("inferred_action", "constant_action", "shuffled_action"):
        assert f"{baseline}/fully_hidden_ce" in first
        assert f"{baseline}/partial_reconstruction_ce" in first
        assert f"{baseline}/one_step_l1" in first
    assert "copy_frame/l1" in first
