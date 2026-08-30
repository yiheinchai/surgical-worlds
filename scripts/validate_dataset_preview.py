#!/usr/bin/env python3
"""Print dataset stats and save a native 128×128 preview montage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="data/surgical/train_robotic_frames.h5")
    parser.add_argument("--output", default="docs/inference_demos/dataset_preview_native_128.png")
    parser.add_argument("--num-frames", type=int, default=8)
    args = parser.parse_args()

    h5_path = Path(args.h5)
    manifest_path = h5_path.parent / "manifest.json"
    with h5py.File(h5_path) as h:
        n = len(h["frames"])
        indices = np.linspace(0, n - 1, args.num_frames, dtype=int)
        frames = [h["frames"][i] for i in indices]

    if manifest_path.exists():
        meta = json.loads(manifest_path.read_text()).get("metadata", {})
        print(json.dumps({"frames": n, **meta}, indent=2))

    tiles = []
    for i, rgb in zip(indices, frames):
        tile = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_NEAREST)
        cv2.putText(tile, f"#{i}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 220, 180), 1)
        tiles.append(tile)
    row1 = np.hstack(tiles[: len(tiles) // 2])
    row2 = np.hstack(tiles[len(tiles) // 2 :])
    montage = np.vstack([row1, row2])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
