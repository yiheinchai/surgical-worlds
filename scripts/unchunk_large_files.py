#!/usr/bin/env python3
"""Reassemble chunked files split by chunk_large_files.py."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unchunk(manifest_path: Path, output_path: Path | None = None) -> Path:
    manifest = json.loads(manifest_path.read_text())
    chunk_dir = manifest_path.parent
    out = output_path or chunk_dir.parent / manifest["original_name"]

    with open(out, "wb") as dst:
        for chunk_name in manifest["chunks"]:
            chunk_path = chunk_dir / chunk_name
            if not chunk_path.exists():
                raise FileNotFoundError(f"Missing chunk: {chunk_path}")
            dst.write(chunk_path.read_bytes())

    actual_hash = sha256_file(out)
    if actual_hash != manifest["sha256"]:
        out.unlink(missing_ok=True)
        raise ValueError(
            f"Checksum mismatch after reassembly.\n"
            f"  expected: {manifest['sha256']}\n"
            f"  got:      {actual_hash}"
        )

    print(f"Reassembled {out} ({out.stat().st_size / 1e6:.1f} MB) — checksum OK")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Reassemble chunked files")
    parser.add_argument("manifest", type=str, help="Path to .manifest.json")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()
    unchunk(Path(args.manifest), Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
