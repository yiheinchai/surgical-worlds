"""One-step transfer to a fresh RGB episode; no refitting of motion codes."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import argparse, json, os, hashlib
import h5py, numpy as np, torch
from PIL import Image
from models.pixel_diffusion import PixelDenoiser, warp_frame
from models.motion_codes import rgb_flow, flow_descriptor, assign_codes, code_metrics
from scripts.train_pixel_dynamics import image_array


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--codes", default="results/readiness/motion/motion_codes.npz")
    p.add_argument("--name", required=True)
    p.add_argument("--output", default="results/readiness")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--clips", type=int, default=128)
    p.add_argument("--key-file")
    a = p.parse_args()
    out = Path(a.output) / a.name
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    with h5py.File(a.episode) as f:
        data = f["frames"][:]
        timestamps = f["capture_seconds"][:]
    centers = np.load(a.codes)["centers"]
    starts = np.linspace(90, len(data) - 4 * a.stride - 1, a.clips, dtype=int)
    ix = starts[:, None] + np.arange(5) * a.stride
    descriptors = np.stack(
        [
            [flow_descriptor(rgb_flow(data[t], data[t + a.stride])) for t in row[:4]]
            for row in ix
        ]
    )
    ids = assign_codes(descriptors.reshape(-1, 32), centers).reshape(-1, 4)
    ck = torch.load(a.checkpoint, map_location="cuda", weights_only=True)
    m = PixelDenoiser(**ck["model_config"]).cuda().eval()
    m.load_state_dict(ck["ema"], strict=True)
    config = {
        **vars(a),
        "checkpoint_step": ck["step"],
        "checkpoint_sha256": hashlib.sha256(
            Path(a.checkpoint).read_bytes()
        ).hexdigest(),
        "episode_sha256": hashlib.sha256(Path(a.episode).read_bytes()).hexdigest(),
        "trained_on_fresh_episode": False,
        "codes_refitted": False,
        "future_rgb_in_history": False,
        "controls": "Future-RGB inferred oracle; action ablations use identical sampler noise",
        "source_domain": "Fresh POOM capture; original TinyWorlds game/version provenance not confirmed",
    }
    run = None
    if a.key_file:
        import wandb

        os.environ["WANDB_API_KEY"] = Path(a.key_file).read_text().strip()
        run = wandb.init(
            entity="data2yihein-d",
            project="tinyworlds",
            group="picodoom-readiness-20260905",
            name=a.name,
            config=config,
            dir=str(out),
        )
    values = {}
    saved = []
    with torch.inference_mode():
        for j in range(0, len(ix), 16):
            x = (
                torch.from_numpy(data[ix[j : j + 16]])
                .cuda()
                .float()
                .permute(0, 1, 4, 2, 3)
                / 127.5
                - 1
            )
            h, y = x[:, :4], x[:, 4]
            actions = torch.from_numpy(ids[j : j + 16]).cuda()
            wrong = actions.clone()
            wrong[:, -1] = wrong[:, -1].roll(1)
            fixed = actions.clone()
            fixed[:, -1] = 6
            flow = (
                torch.from_numpy(centers[ids[j : j + 16, -1]])
                .cuda()
                .reshape(-1, 4, 4, 2)
                .permute(0, 3, 1, 2)
            )
            preds = {
                "inferred": m.sample(h, actions, 3, seed=781 + j, heun=False),
                "shuffled": m.sample(h, wrong, 3, seed=781 + j, heun=False),
                "modal_constant": m.sample(h, fixed, 3, seed=781 + j, heun=False),
                "copy": h[:, -1],
                "warp_only": warp_frame(h[:, -1], flow),
            }
            motion = (y - h[:, -1]).abs().mean(1, keepdim=True)
            for name, pred in preds.items():
                err = (pred - y).abs()
                values.setdefault(name + "_l1", []).extend(
                    err.mean((1, 2, 3)).cpu().tolist()
                )
                values.setdefault(name + "_motion_l1", []).extend(
                    (
                        (err * motion).sum((1, 2, 3))
                        / motion.expand_as(err).sum((1, 2, 3)).clamp_min(1e-6)
                    )
                    .cpu()
                    .tolist()
                )
            if not saved:
                saved = [h[:8, -1], y[:8], preds["inferred"][:8], preds["shuffled"][:8]]
    means = {k: float(np.mean(v)) for k, v in values.items()}
    means["shuffle_gap"] = means["shuffled_l1"] - means["inferred_l1"]
    result = {
        "config": config,
        "means": means,
        "per_clip": values,
        "source_starts": starts.tolist(),
        "target_intervals_seconds": (
            timestamps[ix[:, 4]] - timestamps[ix[:, 3]]
        ).tolist(),
        "code_usage": code_metrics(ids[:, -1], len(centers)),
        "motion_quantization_mse": float(
            np.mean((descriptors[:, -1] - centers[ids[:, -1]]) ** 2)
        ),
    }
    (out / "results.json").write_text(json.dumps(result, indent=2))
    Image.fromarray(
        np.concatenate([np.concatenate(image_array(row), axis=1) for row in saved])
    ).resize((1024, 512), Image.Resampling.NEAREST).save(out / "predictions.png")
    print(json.dumps(means), flush=True)
    if run:
        run.log({**means, "predictions": wandb.Image(str(out / "predictions.png"))})
        run.finish()


if __name__ == "__main__":
    main()
