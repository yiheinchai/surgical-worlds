"""Closed-loop pixel generation: only prompt RGB and specified latent codes.

Oracle mode obtains future controls from recorded RGB motion, but never feeds
future RGB into generated history. Fixed/switch modes consume no future controls.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, json, os, time
import h5py, numpy as np, torch, imageio.v2 as imageio
from PIL import Image, ImageDraw
from models.pixel_diffusion import PixelDenoiser
from models.motion_codes import rgb_flow, flow_descriptor, assign_codes
from scripts.train_pixel_dynamics import image_array


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--codes", default="results/readiness/motion/motion_codes.npz")
    p.add_argument("--data", default="data/picodoom_frames.h5")
    p.add_argument("--output", default="results/readiness")
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--sample-steps", type=int, default=8)
    p.add_argument("--stabilization", type=float, default=0.0)
    p.add_argument("--solver", choices=["euler", "heun"], default="heun")
    p.add_argument("--key-file")
    p.add_argument("--modes", default="oracle,0,1,2,3,4,5,6,7,switch")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    out = Path(a.output) / a.name
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    ck = torch.load(a.checkpoint, map_location=a.device, weights_only=True)
    m = (
        PixelDenoiser(
            ck["codes"],
            ck["history"],
            ck["width"],
            ck.get("offset_noise", 0.0),
            ck.get("prediction_mode", "frame"),
            noise_features=ck.get("noise_features", "legacy"),
        )
        .to(a.device)
        .eval()
    )
    m.load_state_dict(ck["model"])
    c = np.load(a.codes)
    ids = c["ids"]
    centers = c["centers"]
    stride = int(c["stride"])
    starts = np.array([47000, 48500, 50000, 51500])
    times = []
    with h5py.File(a.data, "r") as f:
        data = f["frames"][:]
    ix = starts[:, None] + np.arange(4) * stride
    prompt = (
        torch.from_numpy(data[ix]).to(a.device).float().permute(0, 1, 4, 2, 3) / 127.5
        - 1
    )
    run = None
    if a.key_file:
        os.environ["WANDB_API_KEY"] = Path(a.key_file).read_text().strip()
        import wandb

        run = wandb.init(
            entity="data2yihein-d",
            project="tinyworlds",
            group="picodoom-readiness-20260905",
            name=a.name,
            dir=str(out),
            config={
                **vars(a),
                "source_starts": starts.tolist(),
                "checkpoint_step": ck["step"],
                "future_rgb_in_history": False,
            },
        )
    reports = {}
    with torch.inference_mode():
        for mode in a.modes.split(","):
            h = prompt.clone()
            actions = torch.from_numpy(ids[ix]).long().to(a.device)
            frames = [image_array(h[:, -1])]
            l1 = []
            motionids = []
            requested = []
            edge = []
            spatial = []
            for step in range(a.frames):
                if mode == "oracle":
                    chosen = ids[starts + (3 + step) * stride]
                elif mode == "switch":
                    chosen = np.full(4, (step // 16) % len(centers))
                else:
                    chosen = np.full(4, int(mode))
                actions[:, -1] = torch.as_tensor(chosen, device=a.device)
                requested.append(chosen.tolist())
                if a.device == "cuda":
                    torch.cuda.synchronize()
                t = time.time()
                pred = m.sample(
                    h,
                    actions,
                    a.sample_steps,
                    seed=1024 + step,
                    stabilization=a.stabilization,
                    heun=a.solver == "heun",
                )
                if a.device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - t)
                arr = image_array(pred)
                ds = np.stack(
                    [flow_descriptor(rgb_flow(x, y)) for x, y in zip(frames[-1], arr)]
                )
                motionids.append(assign_codes(ds, centers).tolist())
                edge.append(float((pred[..., 1:] - pred[..., :-1]).abs().mean()))
                spatial.append(float(pred.std((-2, -1)).mean()))
                frames.append(arr)
                if mode == "oracle":
                    target = (
                        torch.from_numpy(data[starts + (4 + step) * stride])
                        .to(a.device)
                        .float()
                        .permute(0, 3, 1, 2)
                        / 127.5
                        - 1
                    )
                    l1.append(float((target - pred).abs().mean()))
                h = torch.cat([h[:, 1:], pred[:, None]], 1)
                actions = torch.cat(
                    [actions[:, 1:], torch.zeros_like(actions[:, :1])], 1
                )
            ims = [np.concatenate(f, axis=1) for f in frames]
            imageio.mimsave(out / f"{mode}.gif", ims, duration=100, loop=0)
            ticks = sorted(set([0, 1, 4, 16, 32, a.frames]))
            ticks = [x for x in ticks if x <= a.frames]
            grid = Image.new("RGB", (256, len(ticks) * 80))
            d = ImageDraw.Draw(grid)
            for j, tick in enumerate(ticks):
                grid.paste(Image.fromarray(ims[tick]), (0, j * 80 + 16))
                d.text((0, j * 80), f"{mode} step {tick}", fill="white")
            grid.resize((512, grid.height * 2), Image.Resampling.NEAREST).save(
                out / f"{mode}.png"
            )
            reports[mode] = {
                "oracle_target_l1": l1,
                "requested_codes": requested,
                "generated_motion_codes": motionids,
                "motion_code_agreement": float(
                    (np.array(motionids) == requested).mean()
                ),
                "edge_energy": edge,
                "spatial_std": spatial,
            }
            (out / "results.json").write_text(
                json.dumps(
                    {
                        "config": vars(a),
                        "source_starts": starts.tolist(),
                        "modes": reports,
                        "seconds_per_batch4": {
                            "median": float(np.median(times)),
                            "p95": float(np.quantile(times, 0.95)),
                        },
                    },
                    indent=2,
                )
            )
            print(
                mode,
                "motion agreement",
                reports[mode]["motion_code_agreement"],
                "end L1",
                l1[-1] if l1 else None,
                flush=True,
            )
            if run:
                run.log(
                    {
                        f"rollouts/{mode}": wandb.Video(str(out / f"{mode}.gif")),
                        f"contact_sheets/{mode}": wandb.Image(str(out / f"{mode}.png")),
                        f"{mode}/motion_code_agreement": reports[mode][
                            "motion_code_agreement"
                        ],
                    }
                )
    if run:
        run.finish()


if __name__ == "__main__":
    main()
