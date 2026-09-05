from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import json, os, h5py, numpy as np, torch, wandb
from models.pixel_diffusion import PixelDenoiser, warp_frame
from models.motion_codes import flow_descriptor, rgb_flow, assign_codes

base = ROOT
out = base / "results/readiness/component-comparison-256"
out.mkdir(exist_ok=True)
os.environ["WANDB_API_KEY"] = Path(os.environ["WANDB_KEY_FILE"]).read_text().strip()
run = wandb.init(
    entity="data2yihein-d",
    project="tinyworlds",
    group="picodoom-readiness-20260905",
    name="component-comparison-256",
    dir=str(out),
    config={
        "cohort": "256 reused development validation clips",
        "sampler": "Euler 3, same private noise seed for all ablations",
        "labels": "RGB motion codes only",
        "training_updates_each": 5000,
        "latency": "not measured; another training process may be active",
    },
)
c = np.load(base / "results/readiness/motion/motion_codes.npz")
starts = np.linspace(47000, 52989, 256, dtype=int)
ix = starts[:, None] + np.arange(5) * 2
with h5py.File(base / "data/picodoom_frames.h5", "r") as f:
    data = f["frames"][:]
x = torch.from_numpy(data[ix]).float().permute(0, 1, 4, 2, 3) / 127.5 - 1
actions = torch.from_numpy(c["ids"][ix[:, :4]]).long()
centers = c["centers"]
results = {}
cases = [
    ("frame_offset0", "pixel-motion-noise01-seed0-5k-v2", 0.0, "frame"),
    ("frame_offset03", "pixel-motion-offset03-noise01-seed0-5k", 0.3, "frame"),
    ("warp_offset03", "pixel-warp-offset03-noise01-seed0-5k", 0.3, "warp_residual"),
    ("warp_offset0", "pixel-warp-offset0-noise01-seed0-5k", 0.0, "warp_residual"),
]
with torch.inference_mode():
    for name, folder, offset, mode in cases:
        ck = torch.load(
            base / "results/readiness" / folder / "last.pt",
            map_location="cuda",
            weights_only=True,
        )
        m = PixelDenoiser(8, 4, 32, offset, mode).cuda().eval()
        m.load_state_dict(ck["ema"])
        vals = {}
        for i in range(0, len(x), 16):
            h, y = x[i : i + 16, :4].cuda(), x[i : i + 16, 4].cuda()
            a = actions[i : i + 16].cuda()
            wrong = a.clone()
            wrong[:, -1] = a[:, -1].roll(1)
            fixed = a.clone()
            fixed[:, -1] = 6
            ps = {
                "inferred": m.sample(h, a, 3, seed=700 + i, heun=False),
                "shuffled": m.sample(h, wrong, 3, seed=700 + i, heun=False),
                "modal_constant": m.sample(h, fixed, 3, seed=700 + i, heun=False),
                "copy": h[:, -1],
                "warp_only": warp_frame(
                    h[:, -1],
                    torch.from_numpy(centers[a[:, -1].cpu().numpy()])
                    .cuda()
                    .reshape(-1, 4, 4, 2)
                    .permute(0, 3, 1, 2),
                ),
            }
            motion = (y - h[:, -1]).abs().mean(1, keepdim=True)
            for k, p in ps.items():
                err = (p - y).abs()
                vals.setdefault(k + "_l1", []).extend(
                    err.mean((1, 2, 3)).cpu().tolist()
                )
                vals.setdefault(k + "_motion_l1", []).extend(
                    (
                        (err * motion).sum((1, 2, 3))
                        / (motion.expand_as(err).sum((1, 2, 3)).clamp_min(1e-6))
                    )
                    .cpu()
                    .tolist()
                )
        means = {k: float(np.mean(v)) for k, v in vals.items()}
        dif = np.array(vals["shuffled_l1"]) - vals["inferred_l1"]
        block = dif.reshape(32, 8).mean(1)
        rng = np.random.default_rng(391)
        ci = np.quantile(
            rng.choice(block, (10000, 32), replace=True).mean(1), [0.025, 0.975]
        )
        results[name] = {
            "means": means,
            "shuffle_gap": float(dif.mean()),
            "gap_block_bootstrap_ci95": ci.tolist(),
            "per_clip": vals,
            "training_folder": folder,
        }
        print(name, json.dumps(means), "gap CI", ci.tolist(), flush=True)
        run.log({name + "/" + k: v for k, v in means.items()})
        del m, ck
        (out / "results.json").write_text(
            json.dumps(
                {
                    "source_starts": starts.tolist(),
                    "results": results,
                    "limitations": [
                        "Reused temporal development cohort, not independent episodes.",
                        "Bootstrap blocks of 8 clips may not capture all same-video dependence.",
                        "Constant control is TRAIN modal code 6; older pilot grids used code 0.",
                    ],
                },
                indent=2,
            )
        )
t = wandb.Table(
    columns=[
        "variant",
        "generated L1",
        "shuffled L1",
        "modal constant L1",
        "warp-only L1",
        "copy L1",
        "shuffle gap",
    ]
)
for k, v in results.items():
    r = v["means"]
    t.add_data(
        k,
        r["inferred_l1"],
        r["shuffled_l1"],
        r["modal_constant_l1"],
        r["warp_only_l1"],
        r["copy_l1"],
        v["shuffle_gap"],
    )
run.log({"component_comparison": t})
run.finish()
