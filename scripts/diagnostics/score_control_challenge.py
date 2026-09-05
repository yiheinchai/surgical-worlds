"""Score frozen RGB codes against evaluation-only buttons across fresh episodes.

Button labels calibrate a small diagnostic readout only. No action encoder,
codebook, diffusion model or training dataset is changed by this script.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import argparse, hashlib, json, os
import cv2, h5py, numpy as np
from models.motion_codes import rgb_flow, flow_descriptor, assign_codes
from models.residual_events import infer_events


def probe(features, labels, episodes, classes):
    """Fixed Hellinger-centroid readout; episodes 2/3 never affect calibration."""
    labels, episodes = np.asarray(labels), np.asarray(episodes)
    selected = np.isin(labels, classes)
    train = selected & (episodes < 2)
    test = selected & (episodes >= 2)
    centers = np.stack([features[train & (labels == c)].mean(0) for c in classes])
    predictions = np.asarray(classes)[
        ((features[test, None] - centers[None]) ** 2).sum(-1).argmin(1)
    ]
    targets = labels[test]
    confusion = np.array(
        [
            [int(np.sum((targets == c) & (predictions == p))) for p in classes]
            for c in classes
        ]
    )
    return {
        "classes": classes,
        "calibration_trials": int(train.sum()),
        "test_trials": int(test.sum()),
        "accuracy": float(np.mean(predictions == targets)),
        "chance_balanced": 1 / len(classes),
        "confusion": confusion.tolist(),
        "test_predictions": predictions.tolist(),
        "test_targets": targets.tolist(),
        "test_episodes": episodes[test].tolist(),
        "per_episode_accuracy": {
            str(ep): float(
                np.mean(
                    predictions[episodes[test] == ep] == targets[episodes[test] == ep]
                )
            )
            for ep in np.unique(episodes[test])
        },
        "calibration_centroids": centers.tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--challenge", default="results/evaluation_only/control_challenge_v1"
    )
    p.add_argument(
        "--codes", default="results/readiness/event-prior-pilot/motion_events.npz"
    )
    p.add_argument("--output", default="results/readiness/control-challenge-v1")
    p.add_argument("--key-file")
    a = p.parse_args()
    cv2.setNumThreads(2)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    folder = Path(a.challenge)
    bank = dict(np.load(a.codes))
    flow_centers = bank["flow_centers"]
    nm, ne = len(flow_centers), len(bank["event_centers"])
    features = {
        k: [] for k in ["motion", "motion_and_event", "pre_action_motion_and_event"]
    }
    labels, episodes, rows = [], [], []
    files = sorted(folder.glob("episode_*.h5"))
    if len(files) != 4:
        raise ValueError(
            "Four completed episodes are required for the prespecified split"
        )
    for path in files:
        ep = int(path.stem.split("_")[-1])
        with h5py.File(path) as f:
            assert f.attrs["evaluation_only"]
            trials = f["trials"][:]
        meta = json.loads(path.with_name(path.stem + "_labels.json").read_text())
        for trial, row in zip(trials, meta):
            histograms = {}
            transitions = {}
            for phase, indices in [
                ("pre", np.array([0, 2])),
                ("during", np.arange(8, 28, 2)),
            ]:
                before, after = trial[indices], trial[indices + 2]
                desc = np.stack(
                    [flow_descriptor(rgb_flow(x, y)) for x, y in zip(before, after)]
                )
                motions = assign_codes(desc, flow_centers)
                events = infer_events(before, after, motions, bank)
                mh = np.bincount(motions, minlength=nm) / len(motions)
                eh = np.bincount(events, minlength=ne) / len(events)
                histograms[phase] = (mh, eh)
                transitions[phase] = {
                    "motion_ids": motions.tolist(),
                    "event_ids": events.tolist(),
                    "mean_rgb_change_u8": float(
                        np.abs(before.astype(float) - after).mean()
                    ),
                }
            mh, eh = histograms["during"]
            features["motion"].append(np.sqrt(mh))
            features["motion_and_event"].append(np.sqrt(np.concatenate([mh, eh])))
            features["pre_action_motion_and_event"].append(
                np.sqrt(np.concatenate(histograms["pre"]))
            )
            labels.append(row["evaluation_control"])
            episodes.append(ep)
            rows.append(
                {
                    "episode": ep,
                    "trial": row["trial"],
                    "control": row["evaluation_control"],
                    "transitions": transitions,
                }
            )
    all_classes = ["idle", "left", "right", "forward", "backward", "fire"]
    scores = {}
    for name, feature in features.items():
        feature = np.stack(feature)
        for subset, classes in [
            ("all", all_classes),
            ("movement", all_classes[:-1]),
            ("fire_vs_idle", ["idle", "fire"]),
        ]:
            scores[name + "/" + subset] = probe(feature, labels, episodes, classes)
    result = {
        "config": vars(a),
        "scores": scores,
        "trials": rows,
        "codebook_sha256": hashlib.sha256(Path(a.codes).read_bytes()).hexdigest(),
        "data_manifest": json.loads((folder / "manifest.json").read_text()),
        "training_updates": 0,
        "codebook_refits": 0,
        "limitations": "Only two held-out live episodes, with correlated trials. No simulator-state snapshots; collision, enemies and death can obscure button effects. Raw scoring includes all trials. Calibration fits only a diagnostic code-to-button readout; button labels must not enter representation/dynamics training.",
    }
    (out / "results.json").write_text(json.dumps(result, indent=2))
    summary = {k: v["accuracy"] for k, v in scores.items()}
    print(json.dumps(summary), flush=True)
    if a.key_file:
        import wandb

        os.environ["WANDB_API_KEY"] = Path(a.key_file).read_text().strip()
        run = wandb.init(
            entity="data2yihein-d",
            project="tinyworlds",
            group="picodoom-readiness-20260905",
            name="control-challenge-v1",
            config={
                "purpose": "Evaluation-only, frozen representations; no model training",
                "test_episodes": [2, 3],
            },
            dir=str(out),
        )
        run.log(summary)
        artifact = wandb.Artifact(
            "picodoom-evaluation-only-controls-v1",
            type="evaluation-only-data",
            metadata={
                "NOT_FOR_TRAINING": True,
                "contains_evaluation_button_labels": True,
                "encoder_and_dynamics_updates": 0,
            },
        )
        for f in sorted(folder.iterdir()):
            if f.is_file():
                artifact.add_file(str(f), name="evaluation_only/" + f.name)
        artifact.add_file(str(out / "results.json"), name="scores.json")
        run.log_artifact(artifact)
        artifact.wait(timeout=300)
        (out / "published.json").write_text(
            json.dumps(
                {
                    "artifact_name": artifact.name,
                    "artifact_id": artifact.id,
                    "state": str(artifact.state),
                    "run_url": run.url,
                },
                indent=2,
            )
        )
        run.finish()


if __name__ == "__main__":
    main()
