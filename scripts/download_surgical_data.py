#!/usr/bin/env python3
"""
Download surgical video data on remote machines (Vast.ai, cloud GPUs).

Vast.ai instances typically have very fast datacenter bandwidth — much faster
than uploading from a home connection. Use this script on the remote instance
to fetch surgical data before training.

Supported sources:
  demo         — synthetic laparoscopic video (no external data)
  url          — direct download of .tar.gz / .zip / .tar archive (Cholec80 mirror, etc.)
  huggingface  — pull videos from a HuggingFace dataset repo
  cholect50    — download CholecT50 frame sequences from HF and stitch to MP4

Environment variables (used by vastai/onstart.sh):
  DATA_SOURCE          demo | url | huggingface | cholect50
  DATA_DOWNLOAD_URL    direct archive URL (for source=url)
  HF_DATASET_REPO      e.g. orena-dkfz/lapchole-focus-vqa (for source=huggingface)
  HF_TOKEN             HuggingFace token for gated datasets
  MAX_VIDEOS           limit videos downloaded (default: all / 10 for hf)
  SURGERY_TYPE         laparoscopic | robotic
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import cv2
import numpy as np
from tqdm import tqdm


def _run(cmd: List[str], **kwargs) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def download_demo(output_dir: Path) -> Path:
    """Generate synthetic demo video."""
    repo_root = Path(__file__).resolve().parents[1]
    demo_mp4 = output_dir / "demo" / "laparoscopic_demo.mp4"
    _run([sys.executable, str(repo_root / "scripts" / "generate_demo_surgical_video.py"),
          "--output", str(demo_mp4), "--frames", "600"])
    return demo_mp4


def download_url(url: str, output_dir: Path) -> Path:
    """Download and extract a direct archive URL (fast on Vast.ai)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name or "surgical_data.archive"
    archive = output_dir / filename

    print(f"Downloading {url}")
    print(f"  → {archive}")

    # Prefer aria2c for multi-connection download if available
    if shutil.which("aria2c"):
        _run([
            "aria2c", "-x", "16", "-s", "16", "-k", "1M",
            "-d", str(output_dir), "-o", filename, url,
        ])
    else:
        _run(["curl", "-L", "--progress-bar", "-o", str(archive), url])

    extract_dir = output_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)

    print(f"Extracting {archive}...")
    if archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(extract_dir)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(extract_dir)
    elif archive.suffix == ".tar":
        with tarfile.open(archive, "r") as tf:
            tf.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive.suffix}")

    print(f"Extracted to {extract_dir}")
    return extract_dir


def _png_folder_to_mp4(png_dir: Path, output_mp4: Path, fps: int = 1) -> None:
    """Stitch a folder of numbered PNG frames into an MP4."""
    pngs = sorted(png_dir.glob("*.png"))
    if not pngs:
        pngs = sorted(png_dir.glob("*.jpg"))
    if not pngs:
        return

    sample = cv2.imread(str(pngs[0]))
    h, w = sample.shape[:2]
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for png in pngs:
        frame = cv2.imread(str(png))
        if frame is not None:
            writer.write(frame)
    writer.release()


def download_cholect50(output_dir: Path, max_videos: Optional[int] = 10) -> Path:
    """
    Download CholecT50 frame sequences from HuggingFace (Voxel51/cholect50)
    and stitch each video folder into MP4 for our pipeline.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError("pip install huggingface_hub")

    raw_dir = output_dir / "cholect50_raw"
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading CholecT50 from HuggingFace (Voxel51/cholect50)...")
    local = Path(snapshot_download(
        repo_id="Voxel51/cholect50",
        repo_type="dataset",
        allow_patterns=["**/VID*/*.png", "**/videos/**"],
    ))

    # Find video frame directories (VID01, VID02, ...)
    vid_dirs = sorted(
        d for d in local.rglob("VID*")
        if d.is_dir() and any(d.glob("*.png"))
    )
    if not vid_dirs:
        # Try nested videos/ structure
        videos_root = local / "videos"
        if videos_root.exists():
            vid_dirs = sorted(d for d in videos_root.iterdir() if d.is_dir())

    if not vid_dirs:
        raise FileNotFoundError(
            f"No VID*/ frame folders found under {local}. "
            "Dataset layout may have changed."
        )

    if max_videos:
        vid_dirs = vid_dirs[:max_videos]

    print(f"Stitching {len(vid_dirs)} videos to MP4...")
    for vid_dir in tqdm(vid_dirs):
        out_mp4 = video_dir / f"{vid_dir.name}.mp4"
        if not out_mp4.exists():
            _png_folder_to_mp4(vid_dir, out_mp4, fps=1)

    return video_dir


def download_huggingface(
    repo_id: str,
    output_dir: Path,
    token: Optional[str] = None,
    max_videos: Optional[int] = 10,
    pattern: str = "videos/**",
) -> Path:
    """Download video files from a HuggingFace dataset repo."""
    from huggingface_hub import hf_hub_download, list_repo_files

    video_dir = output_dir / "hf_videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    print(f"Listing files in {repo_id}...")
    all_files = list_repo_files(repo_id, repo_type="dataset", token=token)
    video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    video_files = [f for f in all_files if f.lower().endswith(video_exts)]

    if pattern and pattern != "*":
        prefix = pattern.rstrip("*").rstrip("/")
        video_files = [f for f in video_files if f.startswith(prefix)]

    if max_videos:
        video_files = video_files[:max_videos]

    if not video_files:
        raise FileNotFoundError(f"No video files found in {repo_id} matching {pattern}")

    print(f"Downloading {len(video_files)} videos from HuggingFace...")
    for vf in tqdm(video_files):
        dest_name = Path(vf).name
        dest = video_dir / dest_name
        if dest.exists():
            continue
        path = hf_hub_download(
            repo_id=repo_id,
            filename=vf,
            repo_type="dataset",
            token=token,
            local_dir=video_dir,
        )
        # hf_hub_download preserves directory structure; flatten if needed
        downloaded = Path(path)
        if downloaded != dest and downloaded.exists():
            shutil.move(str(downloaded), str(dest))

    return video_dir


def find_video_root(path: Path) -> Path:
    """Find directory containing video files after extraction."""
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = [p for p in path.rglob("*") if p.suffix.lower() in exts]
    if not videos:
        raise FileNotFoundError(f"No video files found under {path}")
    # Prefer shallowest common parent with multiple videos
    parents = {}
    for v in videos:
        parents[v.parent] = parents.get(v.parent, 0) + 1
    best = max(parents, key=parents.get)
    print(f"Found {parents[best]} videos in {best}")
    return best


def prepare_data(video_source: Path, surgery_type: str, read_step: int) -> None:
    """Run prepare_surgical_data.py on downloaded videos."""
    repo_root = Path(__file__).resolve().parents[1]
    _run([
        sys.executable,
        str(repo_root / "scripts" / "prepare_surgical_data.py"),
        "--input", str(video_source),
        "--surgery-type", surgery_type,
        "--read-step", str(read_step),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Download surgical data for remote training")
    parser.add_argument(
        "--source",
        choices=["demo", "url", "huggingface", "cholect50"],
        default=os.environ.get("DATA_SOURCE", "demo"),
    )
    parser.add_argument("--url", default=os.environ.get("DATA_DOWNLOAD_URL"))
    parser.add_argument("--hf-repo", default=os.environ.get("HF_DATASET_REPO", "orena-dkfz/lapchole-focus-vqa"))
    parser.add_argument("--hf-pattern", default=os.environ.get("HF_PATTERN", "videos/**"))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--output-dir", default="data/surgical/downloads")
    parser.add_argument("--surgery-type", default=os.environ.get("SURGERY_TYPE", "laparoscopic"))
    parser.add_argument("--max-videos", type=int, default=int(os.environ.get("MAX_VIDEOS", "10")))
    parser.add_argument("--read-step", type=int, default=int(os.environ.get("READ_STEP", "2")))
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "demo":
        video_source = download_demo(output_dir)
    elif args.source == "url":
        if not args.url:
            raise ValueError("Set DATA_DOWNLOAD_URL or pass --url for source=url")
        extracted = download_url(args.url, output_dir)
        video_source = find_video_root(extracted)
    elif args.source == "cholect50":
        video_source = download_cholect50(output_dir, max_videos=args.max_videos)
    elif args.source == "huggingface":
        video_source = download_huggingface(
            args.hf_repo, output_dir,
            token=args.hf_token,
            max_videos=args.max_videos,
            pattern=args.hf_pattern,
        )
    else:
        raise ValueError(f"Unknown source: {args.source}")

    if not args.skip_prepare:
        prepare_data(video_source, args.surgery_type, args.read_step)

    print(f"\nDone. Data ready for training (dataset=LAPAROSCOPIC).")
    print(f"Video source: {video_source}")


if __name__ == "__main__":
    main()
