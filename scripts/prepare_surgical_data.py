#!/usr/bin/env python3
"""Prepare laparoscopic / robotic surgical videos for SurgicalWorlds training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets.surgical_preprocessing import (
    discover_videos,
    read_video_frames,
    save_manifest,
    split_videos_by_ratio,
)
from datasets.surgical_datasets import LaparoscopicDataset, RoboticLaparoscopicDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare surgical videos for world model training")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a video file or directory of procedure videos (.mp4, .avi, .mov, .mkv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/surgical",
        help="Directory for cached .h5 frames and manifest",
    )
    parser.add_argument(
        "--surgery-type",
        choices=["laparoscopic", "robotic"],
        default="laparoscopic",
        help="Manual laparoscopic or robotic laparoscopic (da Vinci / EndoVis-style)",
    )
    parser.add_argument("--resolution", type=int, nargs=2, default=[128, 128], metavar=("H", "W"))
    parser.add_argument("--read-step", type=int, default=2, help="Frame subsampling during preprocessing")
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=10, help="Target FPS metadata for training configs")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild .h5 caches even if they already exist (required after preprocessing fixes)",
    )
    return parser.parse_args()


PREPROCESSING_VERSION = "center_crop_square_v1"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(input_path)
    train_videos, val_videos = split_videos_by_ratio(videos, train_ratio=args.train_ratio, seed=args.seed)

    manifest_path = output_dir / "manifest.json"
    save_manifest(
        manifest_path,
        train_videos=train_videos,
        val_videos=val_videos,
        metadata={
            "surgery_type": args.surgery_type,
            "resolution": list(args.resolution),
            "read_step": args.read_step,
            "fps": args.fps,
            "num_train_videos": len(train_videos),
            "num_val_videos": len(val_videos),
            "preprocessing_version": PREPROCESSING_VERSION,
            "center_crop_square": True,
            "circular_mask_robotic": args.surgery_type == "robotic",
        },
    )
    print(f"Saved manifest with {len(train_videos)} train / {len(val_videos)} val videos → {manifest_path}")

    resolution = tuple(args.resolution)
    dataset_cls = LaparoscopicDataset if args.surgery_type == "laparoscopic" else RoboticLaparoscopicDataset
    h5_name = "laparoscopic_frames.h5" if args.surgery_type == "laparoscopic" else "robotic_frames.h5"
    video_subdir = "laparoscopic" if args.surgery_type == "laparoscopic" else "robotic"
    video_root = output_dir / video_subdir
    video_root.mkdir(parents=True, exist_ok=True)

    # Link input videos into expected data layout for training loaders
    resolved_input = input_path.resolve()
    link_target = video_root / "videos"
    if link_target.exists() or link_target.is_symlink():
        link_target.unlink()
    link_target.symlink_to(resolved_input, target_is_directory=resolved_input.is_dir())

    layout_info = {
        "source": str(resolved_input),
        "linked_at": str(link_target.resolve()),
        "manifest": str(manifest_path.resolve()),
    }
    (output_dir / "layout.json").write_text(json.dumps(layout_info, indent=2))

    for split_name, train_flag in [("train", True), ("val", False)]:
        h5_path = output_dir / f"{split_name}_{h5_name}"
        if args.force and h5_path.exists():
            print(f"Removing stale cache {h5_path}")
            h5_path.unlink()
        print(f"\nBuilding {split_name} cache → {h5_path}")
        ds = dataset_cls(
            video_path=str(input_path),
            save_path=str(h5_path),
            train=train_flag,
            num_frames=4,
            resolution=resolution,
            fps=args.fps,
            preload_ratio=1.0,
            preprocess_read_step=args.read_step,
            max_frames_per_video=args.max_frames_per_video,
            manifest_path=str(manifest_path),
        )
        print(f"  {split_name}: {len(ds.data)} frames")

    print("\nDone. Next steps:")
    print("  export PYTHONPATH=\"$(pwd):$PYTHONPATH\"")
    if args.surgery_type == "laparoscopic":
        print("  python scripts/full_train.py --config configs/surgical_training.yaml -- dataset=LAPAROSCOPIC")
    else:
        print("  python scripts/full_train.py --config configs/crcd_crisp_128_training.yaml -- dataset=ROBOTIC_LAPAROSCOPIC")


if __name__ == "__main__":
    main()
