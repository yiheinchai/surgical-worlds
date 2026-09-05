"""Prepare packed unsupervised motion/event controls and a neutral warm start."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import argparse, json, hashlib
import cv2, h5py, numpy as np, torch
from models.residual_events import infer_events, expand_motion_state, make_event_prior


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion", default="results/readiness/motion/motion_codes.npz")
    p.add_argument(
        "--events", default="results/readiness/residual-events/events_k4.npz"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="results/readiness/event-pilot")
    p.add_argument("--use-event-prior", action="store_true")
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    motion = np.load(a.motion)
    events = np.load(a.events)
    n_events = len(events["centers"])
    bank = {
        "flow_centers": motion["centers"],
        "event_pca_mean": events["pca_mean"],
        "event_pca_components": events["pca_components"],
        "event_centers": events["centers"],
    }
    with h5py.File("data/picodoom_frames.h5") as f:
        frames = f["frames"][:]
    stride = int(motion["stride"])
    ids = motion["ids"]
    event_ids = np.zeros(len(ids), dtype=np.int64)
    for start in range(0, len(frames) - stride, 512):
        stop = min(start + 512, len(frames) - stride)
        event_ids[start:stop] = infer_events(
            frames[start:stop],
            frames[start + stride : stop + stride],
            ids[start:stop],
            bank,
        )
    prior = make_event_prior(bank) if a.use_event_prior else None
    extra = {"event_prior": prior} if prior is not None else {}
    np.savez(
        out / "motion_events.npz",
        centers=np.repeat(motion["centers"], n_events, axis=0),
        ids=ids * n_events + event_ids,
        stride=stride,
        event_count=n_events,
        **bank,
        **extra,
    )
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    ck["model"] = expand_motion_state(ck["model"], 4, n_events)
    ck["ema"] = expand_motion_state(ck["ema"], 4, n_events)
    ck["model_config"]["codes"] *= n_events
    ck["model_config"]["event_count"] = n_events
    ck["model_config"]["use_event_prior"] = a.use_event_prior
    if prior is not None:
        ck["model"]["event_prior"] = torch.from_numpy(prior)
        ck["ema"]["event_prior"] = torch.from_numpy(prior.copy())
    ck["optimizer"]["state"] = {}
    torch.save(ck, out / "warm_start.pt")
    report = {
        "config": vars(a),
        "warm_start_step": ck["step"],
        "event_count": n_events,
        "event_prior": (
            "Train-fitted residual centroids impose a mean visual effect; this is not evidence of learned firing mechanics."
            if a.use_event_prior
            else None
        ),
        "training_event_counts": np.bincount(
            event_ids[300:46000], minlength=n_events
        ).tolist(),
        "validation_event_counts": np.bincount(
            event_ids[47000:53000], minlength=n_events
        ).tolist(),
        "optimizer": "Moments explicitly reset for warm start; not an exact training resume.",
        "supervision": "Both factor vocabularies fitted on training RGB only. No semantic or engine labels.",
        "event_roi": [32, 58, 20, 44],
        "event_bank_sha256": hashlib.sha256(Path(a.events).read_bytes()).hexdigest(),
    }
    (out / "preparation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
