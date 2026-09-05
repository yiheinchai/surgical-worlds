"""Explore RGB-only residual event codes; no engine or semantic training labels."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import argparse, json
import cv2, h5py, numpy as np
from PIL import Image, ImageDraw
from models.motion_codes import fit_codebook, assign_codes, code_metrics


def features(frames, ids, centers, starts, stride=2):
    yy, xx = np.mgrid[:64, :64].astype(np.float32)
    out = []
    for start in starts:
        flow = cv2.resize(
            centers[ids[start]].reshape(4, 4, 2),
            (64, 64),
            interpolation=cv2.INTER_LINEAR,
        )
        pred = cv2.remap(
            frames[start].astype(np.float32) / 127.5 - 1,
            xx - flow[:, :, 0],
            yy - flow[:, :, 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        residual = frames[start + stride].astype(np.float32) / 127.5 - 1 - pred
        # The spatial prior isolates foreground change from large camera displacement.
        # It is hand-selected geometry, not a firing label or engine measurement.
        crop = residual[32:58, 20:44]
        out.append(cv2.resize(crop, (8, 8), interpolation=cv2.INTER_AREA).reshape(-1))
    return np.asarray(out, np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/picodoom_frames.h5")
    p.add_argument("--codes", default="results/readiness/motion/motion_codes.npz")
    p.add_argument("--output", default="results/readiness/residual-events")
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(exist_ok=False, parents=True)
    cv2.setNumThreads(2)
    with h5py.File(a.data) as f:
        frames = f["frames"][:]
    c = np.load(a.codes)
    train = np.arange(300, 45998, 4)
    val = np.arange(47000, 52998, 4)
    starts = np.r_[train, val]
    f = features(frames, c["ids"], c["centers"], starts)
    mean = f[: len(train)].mean(0)
    # PCA fitted solely on training residuals; preserve amplitude (no sample norm).
    _, _, vt = np.linalg.svd(f[: len(train)] - mean, full_matrices=False)
    components = vt[:24]
    z = (f - mean) @ components.T
    results = {}
    for k in [4, 8]:
        centers = fit_codebook(z[: len(train)], k, 0)
        ids = assign_codes(z, centers)
        results[str(k)] = {
            "train": code_metrics(ids[: len(train)], k),
            "validation": code_metrics(ids[len(train) :], k),
        }
        canvas = Image.new("RGB", (64 * 18, 80 * k))
        draw = ImageDraw.Draw(canvas)
        for code in range(k):
            candidates = np.where(ids[len(train) :] == code)[0] + len(train)
            if not len(candidates):
                continue
            chosen = candidates[np.linspace(0, len(candidates) - 1, 9, dtype=int)]
            for j, index in enumerate(chosen):
                st = starts[index]
                canvas.paste(
                    Image.fromarray(np.concatenate([frames[st], frames[st + 2]], 1)),
                    (j * 128, code * 80 + 16),
                )
            draw.text(
                (0, code * 80), f"Unsupervised residual code {code}", fill="white"
            )
        canvas.save(out / f"events_k{k}.png")
        np.savez(
            out / f"events_k{k}.npz",
            centers=centers,
            pca_mean=mean,
            pca_components=components,
            source_starts=starts,
            ids=ids,
        )
    (out / "results.json").write_text(
        json.dumps(
            {
                "config": vars(a),
                "roi": [32, 58, 20, 44],
                "results": results,
                "limitations": [
                    "Codes represent observed residual changes, not identified player intentions.",
                    "Hand-selected foreground ROI; no engine action/state labels.",
                    "Existing temporal development range reused.",
                    "Vocabulary requires visual inspection and counterfactual validation before use.",
                ],
            },
            indent=2,
        )
    )
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
