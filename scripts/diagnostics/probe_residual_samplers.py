"""Matched residual-model sampler and context-stabilization ablation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import argparse, json, os
import h5py, numpy as np, torch
from PIL import Image, ImageDraw
from models.pixel_diffusion import PixelDenoiser
from models.motion_codes import rgb_flow, flow_descriptor, assign_codes
from scripts.train_pixel_dynamics import image_array


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--key-file")
    p.add_argument("--frames", type=int, default=64)
    a = p.parse_args()
    out = ROOT / "results/readiness" / a.name
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    ck = torch.load(a.checkpoint, map_location="cuda", weights_only=True)
    m = PixelDenoiser(**ck["model_config"]).cuda().eval()
    m.load_state_dict(ck["ema"])
    c = np.load(ROOT / "results/readiness/motion/motion_codes.npz")
    ids = c["ids"]
    centers = c["centers"]
    with h5py.File(ROOT / "data/picodoom_frames.h5") as f:
        data = f["frames"][:]
    starts = np.linspace(47000, 52989, 32, dtype=int)
    ix = starts[:, None] + np.arange(5) * 2
    x = torch.from_numpy(data[ix]).cuda().float().permute(0, 1, 4, 2, 3) / 127.5 - 1
    actions = torch.from_numpy(ids[ix[:, :4]]).cuda().long()
    ss = np.array([47000, 48500, 50000, 51500])
    ii = ss[:, None] + np.arange(4) * 2
    prompt = (
        torch.from_numpy(data[ii]).cuda().float().permute(0, 1, 4, 2, 3) / 127.5 - 1
    )
    run = None
    if a.key_file:
        import wandb

        os.environ["WANDB_API_KEY"] = Path(a.key_file).read_text().strip()
        run = wandb.init(
            entity="data2yihein-d",
            project="tinyworlds",
            group="picodoom-readiness-20260905",
            name=a.name,
            config={
                **vars(a),
                "checkpoint_step": ck["step"],
                "cohort": "Reused temporal development clips; no fresh holdouts",
                "noise": "Matched private seeds",
                "timing": "No latency claim; concurrent training may be present",
            },
            dir=str(out),
        )
    results = {}
    with torch.inference_mode():
        for steps, heun in [(3, False), (3, True), (8, False), (8, True)]:
            for stabilization in [0.0, 0.1]:
                name = f'{"heun" if heun else "euler"}{steps}-context{stabilization}'
                pred = m.sample(
                    x[:, :4],
                    actions,
                    steps,
                    seed=903,
                    heun=heun,
                    stabilization=stabilization,
                )
                wrong = actions.clone()
                wrong[:, -1] = wrong[:, -1].roll(1)
                shuffled = m.sample(
                    x[:, :4],
                    wrong,
                    steps,
                    seed=903,
                    heun=heun,
                    stabilization=stabilization,
                )
                result = {
                    "one_step_l1": float((pred - x[:, 4]).abs().mean()),
                    "shuffled_l1": float((shuffled - x[:, 4]).abs().mean()),
                    "copy_l1": float((x[:, -2] - x[:, -1]).abs().mean()),
                    "rollouts": {},
                }
                for mode in ["oracle", "1"]:
                    h = prompt.clone()
                    act = torch.from_numpy(ids[ii]).cuda().long()
                    frames = [image_array(h[:, -1])]
                    measured = []
                    requested = []
                    edges = []
                    for step in range(a.frames):
                        chosen = (
                            ids[ss + (3 + step) * 2]
                            if mode == "oracle"
                            else np.ones(4, dtype=int)
                        )
                        act[:, -1] = torch.as_tensor(chosen, device="cuda")
                        pred = m.sample(
                            h,
                            act,
                            steps,
                            seed=1024 + step,
                            heun=heun,
                            stabilization=stabilization,
                        )
                        arr = image_array(pred)
                        ds = np.stack(
                            [
                                flow_descriptor(rgb_flow(prev, nxt))
                                for prev, nxt in zip(frames[-1], arr)
                            ]
                        )
                        measured.append(assign_codes(ds, centers).tolist())
                        requested.append(chosen.tolist())
                        edges.append(
                            float((pred[..., 1:] - pred[..., :-1]).abs().mean())
                        )
                        frames.append(arr)
                        h = torch.cat([h[:, 1:], pred[:, None]], 1)
                        act = torch.cat([act[:, 1:], torch.zeros_like(act[:, :1])], 1)
                    ticks = [0, 1, 8, 32, a.frames]
                    canvas = Image.new("RGB", (256, 80 * len(ticks)))
                    draw = ImageDraw.Draw(canvas)
                    for j, t in enumerate(ticks):
                        canvas.paste(
                            Image.fromarray(np.concatenate(frames[t], axis=1)),
                            (0, j * 80 + 16),
                        )
                        draw.text((0, j * 80), f"{name} {mode} {t}", fill="white")
                    path = out / f"{name}-{mode}.png"
                    canvas.resize(
                        (512, canvas.height * 2), Image.Resampling.NEAREST
                    ).save(path)
                    result["rollouts"][mode] = {
                        "motion_agreement": float(
                            np.mean(np.array(measured) == requested)
                        ),
                        "edge_final16": float(np.mean(edges[-16:])),
                        "generated_codes": measured,
                        "requested_codes": requested,
                    }
                    if run:
                        run.log({f"{name}/{mode}": wandb.Image(str(path))})
                results[name] = result
                (out / "results.json").write_text(
                    json.dumps({"config": vars(a), "results": results}, indent=2)
                )
                print(
                    name,
                    json.dumps({k: v for k, v in result.items() if k != "rollouts"}),
                    "direct1",
                    result["rollouts"]["1"]["motion_agreement"],
                    flush=True,
                )
                if run:
                    run.log(
                        {
                            name + "/" + k: v
                            for k, v in result.items()
                            if k != "rollouts"
                        }
                    )
    if run:
        run.finish()


if __name__ == "__main__":
    main()
