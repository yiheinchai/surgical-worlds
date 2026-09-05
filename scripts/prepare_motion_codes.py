"""Fit motion vocabulary from RGB; transformation IDs used for evaluation only."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, json, time, itertools
import cv2, h5py, numpy as np
from models.motion_codes import (
    rgb_flow,
    flow_descriptor,
    fit_codebook,
    assign_codes,
    code_metrics,
)


def synthetic_diagnostic(train, val, seed):
    shifts = [(0, 4), (0, -4), (4, 0), (-4, 0)]

    def desc(textures):
        return np.stack(
            [
                flow_descriptor(rgb_flow(x, np.roll(x, s, (0, 1))))
                for x in textures
                for s in shifts
            ]
        )

    tr, va = desc(train), desc(val)
    c = fit_codebook(tr, 4, seed)
    ti, vi = assign_codes(tr, c), assign_codes(va, c)
    labels = np.tile(np.arange(4), len(train))
    mapping = max(
        itertools.permutations(range(4)),
        key=lambda p: (np.array(p)[ti] == labels).mean(),
    )
    return dict(
        seed=seed,
        validation_mapping_accuracy=float(
            (np.array(mapping)[vi] == np.tile(np.arange(4), len(val))).mean()
        ),
        train_mapping_accuracy=float((np.array(mapping)[ti] == labels).mean()),
        all_branches_same_code_fraction=float(
            (vi.reshape(-1, 4) == vi.reshape(-1, 4)[:, :1]).all(1).mean()
        ),
        validation_codes=vi.tolist(),
        centers=c.tolist(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/picodoom_frames.h5")
    p.add_argument("--output", default="results/readiness/motion")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--codes", type=int, default=8)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    with h5py.File(a.data, "r") as f:
        print("H5 keys", list(f), flush=True)
        data = f["frames"][:]
    t = time.time()
    cache = out / f"descriptors_stride{a.stride}.npy"
    if cache.exists():
        ds = np.load(cache)
    else:
        ds = np.stack(
            [
                flow_descriptor(rgb_flow(data[i], data[i + a.stride]))
                for i in range(len(data) - a.stride)
            ]
        )
        np.save(cache, ds)
    train = np.arange(300, 46000 - a.stride)
    val = np.arange(47000, 53000 - a.stride)
    c = fit_codebook(ds[train[::4]], a.codes, 0)
    ids = assign_codes(ds, c)
    np.savez(
        out / "motion_codes.npz",
        centers=c,
        ids=ids,
        descriptors=ds,
        stride=a.stride,
        train_fit_indices=train[::4],
    )
    report = dict(
        method="Classical RGB flow + train-only k-means; no action labels or pretrained flow weights",
        stride=a.stride,
        codes=a.codes,
        train=code_metrics(ids[train], a.codes),
        validation=code_metrics(ids[val], a.codes),
        preprocessing_seconds=time.time() - t,
        synthetic=[
            synthetic_diagnostic(
                data[np.linspace(3000, 45999, 64, dtype=int)],
                data[np.linspace(47000, 52999, 32, dtype=int)],
                s,
            )
            for s in range(3)
        ],
        limitations=[
            "Flow is not player intent.",
            "Temporal validation reused from prior campaign.",
            "Synthetic translations are a learnability check only.",
        ],
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "synthetic"}))
    print(
        "Synthetic accuracies",
        [x["validation_mapping_accuracy"] for x in report["synthetic"]],
        flush=True,
    )
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (8 * 192, a.codes * 80))
    draw = ImageDraw.Draw(canvas)
    for k in range(a.codes):
        choices = val[ids[val] == k]
        if not len(choices):
            continue
        for j, i in enumerate(choices[np.linspace(0, len(choices) - 1, 8, dtype=int)]):
            f = rgb_flow(data[i], data[i + a.stride])
            hsv = np.zeros_like(data[i])
            mag, ang = cv2.cartToPolar(f[..., 0], f[..., 1])
            hsv[..., 0] = ang * 90 / np.pi
            hsv[..., 1] = 255
            hsv[..., 2] = np.minimum(mag * 40, 255)
            tile = np.concatenate(
                [data[i], data[i + a.stride], cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)], 1
            )
            canvas.paste(Image.fromarray(tile), (j * 192, k * 80 + 16))
            draw.text((j * 192, k * 80), f"code {k} frame {i}", fill="white")
    canvas.save(out / "code_examples.png")


if __name__ == "__main__":
    main()
