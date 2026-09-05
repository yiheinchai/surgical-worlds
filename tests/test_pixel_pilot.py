"""Guard science-critical isolation, solver determinism and motion learnability."""

import numpy as np
import pytest
import torch
from models.pixel_diffusion import PixelDenoiser, denoising_loss
from models.motion_codes import fit_codebook, assign_codes
from scripts.prepare_motion_codes import synthetic_diagnostic


def test_motion_bottleneck_transfers_without_training_labels():
    r = np.random.default_rng(91)
    train = r.integers(0, 256, (12, 64, 64, 3), dtype=np.uint8)
    val = r.integers(0, 256, (8, 64, 64, 3), dtype=np.uint8)
    x = synthetic_diagnostic(train, val, 0)
    assert x["validation_mapping_accuracy"] > 0.9
    assert x["all_branches_same_code_fraction"] == 0


def test_codebook_fitting_is_deterministic_and_inference_does_not_refit():
    r = np.random.default_rng(82)
    x = r.normal(size=(64, 8)).astype(np.float32)
    a = fit_codebook(x, 4, 3)
    b = fit_codebook(x, 4, 3)
    np.testing.assert_array_equal(a, b)
    copy = a.copy()
    assign_codes(r.normal(size=(16, 8)).astype(np.float32), a)
    np.testing.assert_array_equal(copy, a)


def test_diffusion_full_noise_generation_and_finite_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(3)
    m = PixelDenoiser(width=8)
    h = torch.rand(2, 4, 3, 16, 16) * 2 - 1
    y = torch.rand(2, 3, 16, 16) * 2 - 1
    a = torch.zeros(2, 4, dtype=torch.long)
    opt = torch.optim.Adam(m.parameters(), lr=0.001)
    for _ in range(3):
        opt.zero_grad()
        l, _ = denoising_loss(m, h, y, a)
        l.backward()
        assert all(
            p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()
        )
        opt.step()
    # The generation API has no target argument and starts from seeded noise.
    p = m.sample(h, a, steps=3, seed=9)
    q = m.sample(h, a, steps=3, seed=9)
    torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert torch.isfinite(p).all()
    assert (p - m.sample(h, a, steps=3, seed=10)).abs().max() > 1e-6
    assert (p - m.sample(h, a + 1, steps=3, seed=9)).abs().max() > 1e-6


def test_bounded_training_entrypoint_executes_and_saves(tmp_path, monkeypatch):
    import h5py, json
    from types import SimpleNamespace
    from scripts.train_pixel_dynamics import Pilot

    data = tmp_path / "data.h5"
    with h5py.File(data, "w") as f:
        f.create_dataset(
            "frames", shape=(54000, 16, 16, 3), dtype="uint8", fillvalue=100
        )
    codes = tmp_path / "codes.npz"
    np.savez(
        codes,
        centers=np.zeros((4, 32), dtype=np.float32),
        ids=np.zeros(54000, dtype=np.int64),
        stride=2,
    )
    a = SimpleNamespace(
        name="entrypoint",
        output=str(tmp_path),
        seed=0,
        device="cpu",
        data=str(data),
        codes=str(codes),
        eval_clips=4,
        width=8,
        lr=0.001,
        resume=None,
        key_file=None,
        conditioning="motion",
        steps=1,
        max_seconds=30,
        batch=2,
        context_noise=0.1,
        offset_noise=0.3,
        log_every=1,
        eval_every=10,
        sample_steps=2,
    )
    pilot = Pilot(a)
    monkeypatch.setattr(pilot, "evaluate", lambda step: {"step": step})
    pilot.train()
    status = json.loads((tmp_path / "entrypoint/status.json").read_text())
    assert status["status"] == "finished" and status["step"] == 1
    assert (tmp_path / "entrypoint/last.pt").exists()


def test_warp_residual_base_uses_only_history_and_requested_code():
    from models.pixel_diffusion import warp_frame

    frame = torch.zeros(1, 3, 16, 16)
    frame[:, :, 5:8, 5:8] = 1
    flow = torch.zeros(1, 2, 4, 4)
    flow[:, 0] = 3
    warped = warp_frame(frame, flow)
    torch.testing.assert_close(
        warped[:, :, 5:8, 8:11], torch.ones(1, 3, 3, 3), atol=1e-6, rtol=0
    )
    centers = torch.zeros(4, 4, 4, 2)
    centers[1, :, :, 0] = 3
    m = PixelDenoiser(4, 4, 8, 0.3, "warp_residual", centers)
    h = frame[:, None].expand(1, 4, 3, 16, 16)
    a = torch.ones(1, 4, dtype=torch.long)
    torch.testing.assert_close(m.prediction_base(h, a), warped)
    l, _ = denoising_loss(m, h, frame, a)
    l.backward()
    assert torch.isfinite(l)


def test_training_resume_preserves_random_stream_and_weights(tmp_path, monkeypatch):
    import h5py, copy
    from types import SimpleNamespace
    from scripts.train_pixel_dynamics import Pilot

    torch.set_num_threads(2)
    data = tmp_path / "resume.h5"
    with h5py.File(data, "w") as f:
        f.create_dataset(
            "frames", shape=(54000, 16, 16, 3), dtype="uint8", fillvalue=100
        )
    codes = tmp_path / "codes.npz"
    np.savez(
        codes,
        centers=np.zeros((4, 32), dtype=np.float32),
        ids=np.zeros(54000, dtype=np.int64),
        stride=2,
    )
    a = SimpleNamespace(
        name="full",
        output=str(tmp_path),
        seed=7,
        device="cpu",
        data=str(data),
        codes=str(codes),
        eval_clips=4,
        width=8,
        lr=0.001,
        resume=None,
        key_file=None,
        conditioning="motion",
        steps=2,
        max_seconds=60,
        batch=2,
        context_noise=0.1,
        offset_noise=0.3,
        log_every=1,
        eval_every=100,
        sample_steps=2,
    )
    full = Pilot(a)
    monkeypatch.setattr(full, "evaluate", lambda step: {})
    full.train()
    b = copy.copy(a)
    b.name = "part"
    b.steps = 1
    part = Pilot(b)
    monkeypatch.setattr(part, "evaluate", lambda step: {})
    part.train()
    c = copy.copy(a)
    c.name = "resume"
    c.resume = str(tmp_path / "part/last.pt")
    resumed = Pilot(c)
    monkeypatch.setattr(resumed, "evaluate", lambda step: {})
    resumed.train()
    for k, v in full.model.state_dict().items():
        torch.testing.assert_close(v, resumed.model.state_dict()[k], rtol=0, atol=0)


@pytest.mark.parametrize("generated_count", [2, 8])
def test_generated_context_never_refreshes_from_future_rgb(generated_count):
    from types import SimpleNamespace
    from scripts.train_pixel_dynamics import Pilot

    p = Pilot.__new__(Pilot)
    p.a = SimpleNamespace(generated_context_probability=1.0, conditioning="motion")
    p.device = torch.device("cpu")
    p.stride = 2
    p.max_generated_context = generated_count
    p.data = (torch.arange(40, dtype=torch.uint8) * 5)[:, None, None, None].expand(
        40, 8, 8, 3
    )
    p.ids = torch.arange(40)

    class FixedRng:
        def random(self):
            return 0.0

        def integers(self, *args):
            return generated_count if len(args) == 2 else 19

    p.rng = FixedRng()
    seen = []

    class FakeSampler:
        def sample(self, history, actions, **kwargs):
            seen.append((history.clone(), actions.clone()))
            return torch.full_like(history[:, -1], 0.777)

    p.ema = FakeSampler()
    history, target, actions, count = p.training_batch(np.array([2]))
    assert count == generated_count and len(seen) == generated_count
    torch.testing.assert_close(seen[1][0][:, -1], torch.full_like(target, 0.777))
    generated = min(4, generated_count)
    torch.testing.assert_close(
        history[:, -generated:], torch.full_like(history[:, -generated:], 0.777)
    )
    target_index = 2 + (4 + generated_count) * 2
    torch.testing.assert_close(
        target, torch.full_like(target, target_index * 5 / 127.5 - 1)
    )
    assert actions.tolist() == [[2 + 2 * (generated_count + i) for i in range(4)]]


def test_noise_feature_ablation_preserves_other_initialization_and_serialization():
    torch.manual_seed(41)
    legacy = PixelDenoiser(width=8)
    torch.manual_seed(41)
    fourier = PixelDenoiser(width=8, noise_features="fourier")
    for key, value in legacy.state_dict().items():
        if key != "frequencies":
            torch.testing.assert_close(value, fourier.state_dict()[key], rtol=0, atol=0)
    assert fourier.frequencies.abs().max() < 30
    restored = PixelDenoiser(width=8, noise_features="fourier")
    restored.load_state_dict(fourier.state_dict())
    h = torch.rand(1, 4, 3, 16, 16) * 2 - 1
    a = torch.zeros(1, 4, dtype=torch.long)
    torch.testing.assert_close(
        fourier.sample(h, a, 3, seed=9),
        restored.sample(h, a, 3, seed=9),
        rtol=0,
        atol=0,
    )
