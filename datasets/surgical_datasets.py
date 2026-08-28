"""Dataset classes for laparoscopic and robotic surgical world modeling."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from datasets.surgical_preprocessing import (
    discover_videos,
    load_manifest,
    read_video_frames,
    split_videos_by_ratio,
)


class SurgicalMultiVideoDataset(Dataset):
    """
    World-model dataset built from one or more laparoscopic / robotic surgery videos.

    Supports:
    - Single .mp4 file or directory of procedure videos
    - Cached .h5 frame stores
    - Train/val split by video (prevents frame leakage across procedures)
    - Manifest-based reproducible splits
    """

    def __init__(
        self,
        video_source: str,
        transform=None,
        save_path: Optional[str] = None,
        train: bool = True,
        num_frames: int = 4,
        resize_to: Tuple[int, int] = (128, 128),
        fps: int = 10,
        sequence_stride: Optional[int] = None,
        fraction_of_dataset: float = 1.0,
        load_chunk_size: int = 1000,
        preload_ratio: Optional[float] = None,
        preprocess_read_step: int = 2,
        preprocess_slice: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
        crop_borders: bool = True,
        circular_mask: bool = False,
        color_normalize: bool = True,
        manifest_path: Optional[str] = None,
        surgery_type: str = "laparoscopic",
        max_frames_per_video: Optional[int] = None,
        train_ratio: float = 0.9,
        split_seed: int = 42,
    ) -> None:
        self.transform = transform
        self.train = train
        self.num_frames = num_frames
        self.fps = fps
        self.frame_skip = max(
            1, sequence_stride if sequence_stride is not None else max(1, 25 // fps)
        )
        self.fraction_of_dataset = float(fraction_of_dataset)
        self.resize_to = resize_to
        self.surgery_type = surgery_type

        if save_path and os.path.exists(save_path):
            self.data, self.frame_to_video = self._load_h5(save_path, preload_ratio, load_chunk_size)
        else:
            self.data, self.frame_to_video = self._build_from_videos(
                video_source=video_source,
                save_path=save_path,
                preprocess_read_step=preprocess_read_step,
                preprocess_slice=preprocess_slice,
                crop_borders=crop_borders,
                circular_mask=circular_mask,
                color_normalize=color_normalize,
                manifest_path=manifest_path,
                max_frames_per_video=max_frames_per_video,
                train_ratio=train_ratio,
                split_seed=split_seed,
            )

        if len(self.data) == 0:
            raise ValueError("Surgical dataset is empty after preprocessing.")

    def _load_h5(
        self,
        save_path: str,
        preload_ratio: Optional[float],
        load_chunk_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(save_path, "r") as h5_file:
            frames_dset = h5_file["frames"]
            video_ids_dset = h5_file["video_ids"] if "video_ids" in h5_file else None
            total = len(frames_dset)
            n_frames = int(
                total if preload_ratio is None else max(0, min(total, int(total * preload_ratio)))
            )

            data: List[np.ndarray] = []
            video_ids: List[int] = []
            for i in tqdm(range(0, n_frames, load_chunk_size), desc=f"Loading {n_frames} frames"):
                chunk_end = min(i + load_chunk_size, n_frames)
                data.extend(frames_dset[i:chunk_end][:])
                if video_ids_dset is not None:
                    video_ids.extend(video_ids_dset[i:chunk_end][:])

        frame_to_video = np.array(video_ids if video_ids else [0] * len(data), dtype=np.int32)
        return np.array(data), frame_to_video

    def _build_from_videos(
        self,
        video_source: str,
        save_path: Optional[str],
        preprocess_read_step: int,
        preprocess_slice: Optional[Tuple[Union[int, float], Union[int, float]]],
        crop_borders: bool,
        circular_mask: bool,
        color_normalize: bool,
        manifest_path: Optional[str],
        max_frames_per_video: Optional[int],
        train_ratio: float,
        split_seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if manifest_path and os.path.exists(manifest_path):
            manifest = load_manifest(manifest_path)
            train_videos = [Path(p) for p in manifest["train_videos"]]
            val_videos = [Path(p) for p in manifest["val_videos"]]
        else:
            all_videos = discover_videos(video_source)
            train_videos, val_videos = split_videos_by_ratio(
                all_videos, train_ratio=train_ratio, seed=split_seed
            )

        selected_videos = train_videos if self.train else val_videos
        if not selected_videos:
            selected_videos = train_videos if train_videos else val_videos

        all_frames: List[np.ndarray] = []
        frame_to_video: List[int] = []

        for video_idx, video_path in enumerate(
            tqdm(selected_videos, desc=f"Preprocessing {'train' if self.train else 'val'} videos")
        ):
            frames = read_video_frames(
                video_path=video_path,
                resize_to=self.resize_to,
                read_step=preprocess_read_step,
                slice_spec=preprocess_slice,
                crop_borders=crop_borders,
                circular_mask=circular_mask,
                color_normalize=color_normalize,
                max_frames=max_frames_per_video,
            )
            all_frames.append(frames)
            frame_to_video.extend([video_idx] * len(frames))

        stacked = np.concatenate(all_frames, axis=0)

        if save_path:
            print(f"Saving preprocessed surgical frames to {save_path}")
            with h5py.File(save_path, "w") as h5_file:
                h5_file.create_dataset("frames", data=stacked, compression="lzf")
                h5_file.create_dataset("video_ids", data=np.array(frame_to_video, dtype=np.int32))
                meta = h5_file.create_group("metadata")
                meta.attrs["surgery_type"] = self.surgery_type
                meta.attrs["resize_to"] = json.dumps(list(self.resize_to))
                meta.attrs["num_videos"] = len(selected_videos)

        return stacked, np.array(frame_to_video, dtype=np.int32)

    def __len__(self) -> int:
        max_valid_index = int(
            (len(self.data) - (self.num_frames * self.frame_skip)) * self.fraction_of_dataset
        )
        return max(0, max_valid_index)

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError(f"Index {index} out of bounds for dataset of length {len(self)}")

        frame_sequence = self.data[
            index : index + (self.num_frames * self.frame_skip) : self.frame_skip
        ]
        if len(frame_sequence) != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {len(frame_sequence)}")

        frame_sequence = frame_sequence.astype(np.float32) / 255.0

        if self.transform:
            transformed = [self.transform(frame) for frame in frame_sequence]
            frame_sequence = torch.stack(transformed, dim=0)
        else:
            frame_sequence = torch.from_numpy(frame_sequence).permute(0, 3, 1, 2)

        return frame_sequence, 0


class LaparoscopicDataset(SurgicalMultiVideoDataset):
    """Manual laparoscopic surgery videos (e.g., Cholec80, HeiChole, PhaKIR)."""

    def __init__(
        self,
        video_path: str,
        transform=None,
        save_path: Optional[str] = None,
        train: bool = True,
        num_frames: int = 4,
        resolution: Tuple[int, int] = (128, 128),
        fps: int = 10,
        preload_ratio: Optional[float] = 1.0,
        preprocess_read_step: int = 2,
        max_frames_per_video: Optional[int] = None,
        manifest_path: Optional[str] = None,
    ):
        super().__init__(
            video_source=video_path,
            transform=transform,
            save_path=save_path,
            train=train,
            num_frames=num_frames,
            resize_to=resolution,
            fps=fps,
            preload_ratio=preload_ratio,
            preprocess_read_step=preprocess_read_step,
            crop_borders=True,
            circular_mask=False,
            color_normalize=True,
            manifest_path=manifest_path,
            surgery_type="laparoscopic",
            max_frames_per_video=max_frames_per_video,
        )


class RoboticLaparoscopicDataset(SurgicalMultiVideoDataset):
    """Robotic laparoscopic surgery videos (e.g., da Vinci, EndoVis/RARP)."""

    def __init__(
        self,
        video_path: str,
        transform=None,
        save_path: Optional[str] = None,
        train: bool = True,
        num_frames: int = 4,
        resolution: Tuple[int, int] = (128, 128),
        fps: int = 10,
        preload_ratio: Optional[float] = 1.0,
        preprocess_read_step: int = 2,
        max_frames_per_video: Optional[int] = None,
        manifest_path: Optional[str] = None,
    ):
        super().__init__(
            video_source=video_path,
            transform=transform,
            save_path=save_path,
            train=train,
            num_frames=num_frames,
            resize_to=resolution,
            fps=fps,
            preload_ratio=preload_ratio,
            preprocess_read_step=preprocess_read_step,
            crop_borders=True,
            circular_mask=True,
            color_normalize=True,
            manifest_path=manifest_path,
            surgery_type="robotic_laparoscopic",
            max_frames_per_video=max_frames_per_video,
        )
