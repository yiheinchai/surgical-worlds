"""Compare event generation with the exact non-neural prior on identical clips."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import argparse, hashlib, json, os
import cv2, h5py, numpy as np, torch
from models.pixel_diffusion import warp_frame
from models.residual_events import infer_events
from scripts.train_pixel_dynamics import image_array


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-folder", required=True)
    p.add_argument("--codes", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--step", type=int, default=27000)
    p.add_argument("--output", default="results/readiness")
    p.add_argument("--data", default="data/picodoom_frames.h5")
    p.add_argument("--key-file")
    a = p.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    out = Path(a.output) / a.name
    out.mkdir(parents=True, exist_ok=False)
    folder = Path(a.model_folder)
    starts = np.array(json.loads((folder / "validation_starts.json").read_text()))
    neural = json.loads((folder / f"evaluation_{a.step:06}.json").read_text())
    bank = dict(np.load(a.codes))
    stride = int(bank["stride"])
    ec = int(bank["event_count"])
    last = starts + stride * 3
    with h5py.File(a.data) as f:
        before = f["frames"][last]
        after = f["frames"][last + stride]
    actions = bank["ids"][last]
    h = torch.from_numpy(before).float().permute(0, 3, 1, 2) / 127.5 - 1
    target = torch.from_numpy(after).float().permute(0, 3, 1, 2) / 127.5 - 1
    flow = (
        torch.from_numpy(bank["centers"][actions])
        .reshape(-1, 4, 4, 2)
        .permute(0, 3, 1, 2)
    )
    warp = warp_frame(h, flow)
    prior = torch.from_numpy(bank["event_prior"])[actions % ec]
    values = {}
    for name, prediction in [
        ("warp_only", warp),
        ("event_prior_only", (warp + prior).clamp(-1, 1)),
    ]:
        error = (prediction - target).abs()
        predicted = infer_events(before, image_array(prediction), actions // ec, bank)
        requested = actions % ec
        confusion = np.zeros((ec, ec), dtype=int)
        np.add.at(confusion, (requested, predicted), 1)
        totals = confusion.sum(1)
        recall = np.diag(confusion) / np.maximum(totals, 1)
        values[name] = {
            "l1": error.mean((1, 2, 3)).tolist(),
            "foreground_l1": error[:, :, 32:58, 20:44].mean((1, 2, 3)).tolist(),
            "event_confusion": confusion.tolist(),
            "event_accuracy": float(np.mean(requested == predicted)),
            "event_balanced_recall": float(recall[totals > 0].mean()),
            "nonmodal_recall": float(recall[1:][totals[1:] > 0].mean()),
        }
    summary = {
        f"{name}/{key}": float(np.mean(v[key]))
        for name, v in values.items()
        for key in [
            "l1",
            "foreground_l1",
            "event_accuracy",
            "event_balanced_recall",
            "nonmodal_recall",
        ]
    }
    for key in [
        "inferred_l1",
        "inferred_foreground_l1",
        "generated_event_balanced_recall",
        "generated_nonmodal_event_recall",
    ]:
        summary["neural/" + key] = neural["means"]["validation/" + key]
    result = {
        "config": vars(a),
        "summary": summary,
        "per_clip": values,
        "source_starts": starts.tolist(),
        "codebook_sha256": hashlib.sha256(Path(a.codes).read_bytes()).hexdigest(),
        "scope": "Exact same clips and train-fitted prior as the neural evaluation. Prior-only event agreement is imposed by construction; neither proves causal firing mechanics.",
    }
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(summary), flush=True)
    if a.key_file:
        import wandb

        os.environ["WANDB_API_KEY"] = Path(a.key_file).read_text().strip()
        run = wandb.init(
            entity="data2yihein-d",
            project="tinyworlds",
            group="picodoom-readiness-20260905",
            name=a.name,
            config=result["config"],
            dir=str(out),
        )
        run.log(summary)
        run.finish()


if __name__ == "__main__":
    main()
