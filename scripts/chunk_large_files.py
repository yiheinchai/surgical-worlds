#!/usr/bin/env python3
"""Split large files into GitHub-safe chunks (default 90 MB, under 100 MB limit)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


DEFAULT_CHUNK_MB = 90


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def chunk_file(input_path: Path, output_dir: Path, chunk_mb: int) -> Path:
    chunk_size = chunk_mb * 1024 * 1024
    file_size = input_path.stat().st_size
    num_chunks = math.ceil(file_size / chunk_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.name

    for i in range(num_chunks):
        chunk_path = output_dir / f"{stem}.part{i:04d}"
        with open(input_path, "rb") as src, open(chunk_path, "wb") as dst:
            src.seek(i * chunk_size)
            dst.write(src.read(chunk_size))
        size_mb = chunk_path.stat().st_size / (1024 * 1024)
        if size_mb > 100:
            raise ValueError(
                f"Chunk {chunk_path} is {size_mb:.1f} MB — increase chunk count or lower chunk_mb"
            )

    manifest = {
        "original_name": stem,
        "original_size": file_size,
        "sha256": sha256_file(input_path),
        "chunk_mb": chunk_mb,
        "num_chunks": num_chunks,
        "chunks": [f"{stem}.part{i:04d}" for i in range(num_chunks)],
    }
    manifest_path = output_dir / f"{stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Split {input_path} ({file_size / 1e6:.1f} MB) → {num_chunks} chunks in {output_dir}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Split large files for GitHub upload")
    parser.add_argument("input", type=str, help="File to split (e.g. data/surgical/train.h5)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <input_dir>/chunks/)",
    )
    parser.add_argument("--chunk-mb", type=int, default=DEFAULT_CHUNK_MB)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "chunks"
    chunk_file(input_path, output_dir, args.chunk_mb)


if __name__ == "__main__":
    main()
