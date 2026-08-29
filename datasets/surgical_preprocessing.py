"""Preprocessing utilities for laparoscopic and robotic surgical videos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


def crop_black_borders(
    frame: np.ndarray,
    threshold: int = 15,
    margin: int = 2,
) -> np.ndarray:
    """Remove letterbox/pillarbox black borders common in laparoscopic feeds."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    if not mask.any():
        return frame

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return frame

    y0 = max(0, int(rows[0]) - margin)
    y1 = min(frame.shape[0], int(rows[-1]) + margin + 1)
    x0 = max(0, int(cols[0]) - margin)
    x1 = min(frame.shape[1], int(cols[-1]) + margin + 1)
    return frame[y0:y1, x0:x1]


def center_crop_square(frame: np.ndarray) -> np.ndarray:
    """Crop to a centered square so downstream resize does not squash circles into ovals."""
    h, w = frame.shape[:2]
    size = min(h, w)
    if h == w:
        return frame
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return frame[y0 : y0 + size, x0 : x0 + size]


def apply_circular_mask(frame: np.ndarray, fill_value: int = 0) -> np.ndarray:
    """Mask pixels outside the endoscopic circular viewport."""
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    radius = min(h, w) // 2
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, thickness=-1)
    masked = frame.copy()
    masked[mask == 0] = fill_value
    return masked


def normalize_laparoscopic_color(frame: np.ndarray) -> np.ndarray:
    """Apply mild CLAHE on the L channel to reduce glare and improve tissue contrast."""
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def preprocess_surgical_frame(
    frame_bgr: np.ndarray,
    resize_to: Tuple[int, int],
    crop_borders: bool = True,
    circular_mask: bool = False,
    color_normalize: bool = True,
) -> np.ndarray:
    """Convert a raw BGR frame into a normalized RGB surgical training frame."""
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if crop_borders:
        frame = crop_black_borders(frame)
    out_w, out_h = resize_to
    if out_w == out_h:
        # Widescreen laparoscopic feeds (e.g. 848x480) must be square before resize,
        # otherwise a true circular mask becomes an oval in the final tensor.
        frame = center_crop_square(frame)
    if circular_mask:
        frame = apply_circular_mask(frame)
    if color_normalize:
        frame = normalize_laparoscopic_color(frame)
    frame = cv2.resize(frame, resize_to, interpolation=cv2.INTER_AREA)
    return frame


def read_video_frames(
    video_path: Union[str, Path],
    resize_to: Tuple[int, int],
    read_step: int = 1,
    slice_spec: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
    crop_borders: bool = True,
    circular_mask: bool = False,
    color_normalize: bool = True,
    max_frames: Optional[int] = None,
) -> np.ndarray:
    """Read and preprocess frames from a single surgical video file."""
    video_path = str(video_path)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(read_step))
    frames: List[np.ndarray] = []

    for frame_idx in range(0, total_frames, step):
        if max_frames is not None and len(frames) >= max_frames:
            break
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frames.append(
            preprocess_surgical_frame(
                frame_bgr,
                resize_to=resize_to,
                crop_borders=crop_borders,
                circular_mask=circular_mask,
                color_normalize=color_normalize,
            )
        )

    capture.release()
    if not frames:
        raise ValueError(f"No frames read from {video_path}")

    stacked = np.stack(frames, axis=0)
    if slice_spec is None:
        return stacked

    start, end = slice_spec
    n = len(stacked)
    if isinstance(start, float) or isinstance(end, float):
        s = 0 if start is None else int(n * max(0.0, min(1.0, float(start))))
        e = n if end is None else int(n * max(0.0, min(1.0, float(end))))
    else:
        s = 0 if start is None else int(start)
        e = n if end is None else int(end)
    s = max(0, min(n, s))
    e = max(s, min(n, e))
    return stacked[s:e]


def discover_videos(
    source: Union[str, Path],
    extensions: Sequence[str] = (".mp4", ".avi", ".mov", ".mkv", ".webm"),
) -> List[Path]:
    """Find video files in a directory or return a single file path."""
    source_path = Path(source)
    if source_path.is_file():
        return [source_path]
    if not source_path.is_dir():
        raise FileNotFoundError(f"Video source not found: {source}")

    videos: List[Path] = []
    for ext in extensions:
        videos.extend(sorted(source_path.rglob(f"*{ext}")))
    if not videos:
        raise FileNotFoundError(f"No videos found under {source_path}")
    return videos


def split_videos_by_ratio(
    video_paths: Sequence[Path],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """Split videos (not frames) into train/validation sets to avoid leakage."""
    if len(video_paths) == 1:
        return list(video_paths), []

    rng = np.random.default_rng(seed)
    paths = list(video_paths)
    rng.shuffle(paths)
    split_idx = max(1, int(len(paths) * train_ratio))
    if split_idx >= len(paths):
        split_idx = len(paths) - 1
    return paths[:split_idx], paths[split_idx:]


def save_manifest(
    manifest_path: Union[str, Path],
    train_videos: Iterable[Path],
    val_videos: Iterable[Path],
    metadata: Optional[dict] = None,
) -> None:
    """Persist train/val video lists for reproducible dataset preparation."""
    payload = {
        "train_videos": [str(p) for p in train_videos],
        "val_videos": [str(p) for p in val_videos],
        "metadata": metadata or {},
    }
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2))


def load_manifest(manifest_path: Union[str, Path]) -> dict:
    return json.loads(Path(manifest_path).read_text())
