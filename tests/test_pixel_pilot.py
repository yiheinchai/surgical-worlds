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


@pytest.mark.parametrize("event_count", [1, 2])
def test_bounded_training_entrypoint_executes_and_saves(
    tmp_path, monkeypatch, event_count
):
    import h5py, json
    from types import SimpleNamespace
    from scripts.train_pixel_dynamics import Pilot

    data = tmp_path / "data.h5"
    with h5py.File(data, "w") as f:
        f.create_dataset(
            "frames", shape=(54000, 16, 16, 3), dtype="uint8", fillvalue=100
        )
    codes = tmp_path / "codes.npz"
    metadata = (
        {}
        if event_count == 1
        else dict(
            event_count=event_count,
            flow_centers=np.zeros((4, 32), dtype=np.float32),
            event_pca_mean=np.zeros(192, dtype=np.float32),
            event_pca_components=np.zeros((24, 192), dtype=np.float32),
            event_centers=np.zeros((event_count, 24), dtype=np.float32),
        )
    )
    np.savez(
        codes,
        centers=np.zeros((4 * event_count, 32), dtype=np.float32),
        ids=np.zeros(54000, dtype=np.int64),
        stride=2,
        **metadata,
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
    checkpoint = torch.load(tmp_path / "entrypoint/last.pt", weights_only=True)
    restored = PixelDenoiser(**checkpoint["model_config"])
    restored.load_state_dict(checkpoint["ema"], strict=True)
    assert restored.event_count == event_count


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


def test_factorized_event_warm_start_preserves_motion_and_accepts_gradients():
    from models.residual_events import expand_motion_state
    from models.pixel_diffusion import Block

    torch.set_num_threads(2)
    torch.manual_seed(41)
    centers = torch.randn(4, 32) * 0.2
    original = PixelDenoiser(4, 4, 8, 0.0, "warp_residual", centers)
    # Make the residual branch nonzero so the equality check exercises conditioning.
    torch.nn.init.normal_(original.out[-1].weight, std=0.01)
    for block in original.modules():
        if isinstance(block, Block):
            torch.nn.init.normal_(block.c2.weight, std=0.01)
    factored = PixelDenoiser(
        12, 4, 8, 0.0, "warp_residual", centers.repeat_interleave(3, 0), event_count=3
    )
    original_state = {k: v.clone() for k, v in original.state_dict().items()}
    factored.load_state_dict(
        expand_motion_state(original.state_dict(), 4, 3), strict=True
    )
    h = torch.rand(2, 4, 3, 16, 16) * 2 - 1
    x = torch.randn(2, 3, 16, 16)
    sigma = torch.full((2,), 0.7)
    actions = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    expected = original(x, sigma, h, actions)
    for event in range(3):
        torch.testing.assert_close(
            factored(x, sigma, h, actions * 3 + event), expected, atol=2e-6, rtol=2e-6
        )
    for k, v in original.state_dict().items():
        torch.testing.assert_close(v, original_state[k], rtol=0, atol=0)
    loss = factored(x, sigma, h, actions * 3 + 1).square().mean()
    loss.backward()
    event_grad = factored.actions[0].weight.grad.reshape(32, 4, 7)[:, :, 4:]
    assert torch.isfinite(event_grad).all() and event_grad.abs().max() > 0


def test_event_prior_is_local_and_modal_event_is_neutral():
    from models.residual_events import make_event_prior

    bank = {
        "event_pca_mean": np.zeros(192, np.float32),
        "event_pca_components": np.ones((1, 192), np.float32),
        "event_centers": np.array([[0.0], [0.3], [-0.2]], np.float32),
    }
    prior = make_event_prior(bank)
    np.testing.assert_array_equal(prior[0], np.zeros_like(prior[0]))
    outside = prior.copy()
    outside[:, :, 32:58, 20:44] = 0
    np.testing.assert_array_equal(outside, np.zeros_like(outside))
    assert prior[1].max() > 0 and prior[2].min() < 0
    model = PixelDenoiser(
        12,
        4,
        8,
        0.0,
        "warp_residual",
        np.zeros((12, 32), np.float32),
        event_count=3,
        use_event_prior=True,
        event_prior=prior,
    )
    history = torch.zeros(1, 4, 3, 64, 64)
    modal = torch.zeros(1, 4, dtype=torch.long)
    torch.testing.assert_close(
        model.prediction_base(history, modal), torch.zeros_like(history[:, -1])
    )
    torch.testing.assert_close(
        model.prediction_base(history, modal + 1), torch.from_numpy(prior[1:2])
    )
    clone = PixelDenoiser(
        12, 4, 8, 0.0, "warp_residual", event_count=3, use_event_prior=True
    )
    clone.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(
        clone.prediction_base(history, modal + 1),
        model.prediction_base(history, modal + 1),
    )


def test_attention_warm_start_preserves_predictions_optimizer_and_rng():
    import copy
    from scripts.diagnostics.prepare_attention_pilot import expand_checkpoint

    torch.set_num_threads(2)
    torch.manual_seed(23)
    old = PixelDenoiser(width=8)
    torch.nn.init.normal_(old.out[-1].weight, std=0.02)
    opt = torch.optim.AdamW(old.parameters(), lr=2e-4)
    h = torch.randn(2, 4, 3, 16, 16)
    x = torch.randn(2, 3, 16, 16)
    a = torch.zeros(2, 4, dtype=torch.long)
    s = torch.ones(2)
    old(x, s, h, a).square().mean().backward()
    opt.step()
    ck = {
        "model": old.state_dict(),
        "ema": copy.deepcopy(old.state_dict()),
        "optimizer": opt.state_dict(),
        "model_config": {"width": 8},
        "step": 1,
        "rng_torch": torch.get_rng_state(),
    }
    before = torch.get_rng_state().clone()
    upgraded = expand_checkpoint(ck)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    torch.testing.assert_close(upgraded["rng_torch"], ck["rng_torch"], rtol=0, atol=0)
    new = PixelDenoiser(**upgraded["model_config"])
    new.load_state_dict(upgraded["model"], strict=True)
    torch.testing.assert_close(new(x, s, h, a), old(x, s, h, a), rtol=0, atol=0)
    new_opt = torch.optim.AdamW(new.parameters(), lr=2e-4)
    new_opt.load_state_dict(upgraded["optimizer"])
    for old_p, new_p in zip(old.parameters(), new.parameters()):
        for key, value in opt.state[old_p].items():
            torch.testing.assert_close(value, new_opt.state[new_p][key], rtol=0, atol=0)
    for _ in range(2):
        new_opt.zero_grad(set_to_none=True)
        new(x, s, h, a).square().mean().backward()
        assert torch.isfinite(new.spatial_attention.proj.weight.grad).all()
        assert new.spatial_attention.proj.weight.grad.abs().max() > 0
        new_opt.step()
    assert new.spatial_attention.qkv.weight.grad.abs().max() > 0
    assert torch.isfinite(new.spatial_attention.qkv.weight.grad).all()


def test_denoising_large_sigma_has_finite_gradients_and_reports_coverage():
    torch.set_num_threads(2)
    model = PixelDenoiser(width=8)
    history = torch.rand(4, 4, 3, 16, 16) * 2 - 1
    target = torch.rand(4, 3, 16, 16) * 2 - 1
    actions = torch.zeros(4, 4, dtype=torch.long)
    loss, metrics = denoising_loss(
        model, history, target, actions, sigma_location=10, sigma_max=20
    )
    assert torch.isfinite(loss)
    assert metrics["sigma_over_five_fraction"] == 1.0
    loss.backward()
    assert torch.isfinite(model.out[-1].weight.grad).all()
    assert model.out[-1].weight.grad.abs().max() > 0


def test_training_rejects_evaluation_only_capture(tmp_path):
    import h5py
    from types import SimpleNamespace
    from scripts.train_pixel_dynamics import Pilot

    path = tmp_path / "evaluation.h5"
    with h5py.File(path, "w") as f:
        f.attrs["evaluation_only"] = True
    args = SimpleNamespace(
        output=str(tmp_path), name="blocked", seed=0, device="cpu", data=str(path)
    )
    with pytest.raises(ValueError, match="Evaluation-only"):
        Pilot(args)


def test_control_probe_heldout_labels_do_not_change_calibration():
    from scripts.diagnostics.score_control_challenge import probe

    features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    labels = ["left", "right", "left", "right"]
    episodes = [0, 1, 2, 3]
    first = probe(features, labels, episodes, ["left", "right"])
    second = probe(
        features, labels[:2] + ["right", "left"], episodes, ["left", "right"]
    )
    assert first["calibration_centroids"] == second["calibration_centroids"]
    assert first["test_predictions"] == second["test_predictions"]
    assert first["accuracy"] == 1.0 and second["accuracy"] == 0.0
