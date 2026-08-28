#!/usr/bin/env python3
"""Pre-push check: fail if any tracked file exceeds GitHub's 100 MB limit."""

from __future__ import annotations

import subprocess
import sys

GITHUB_LIMIT_MB = 100
WARN_MB = 50


def main() -> int:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        check=True,
    )
    paths = result.stdout.decode().split("\0")
    errors = []
    warnings = []

    for path in paths:
        if not path:
            continue
        try:
            size = __import__("os").path.getsize(path)
        except OSError:
            continue
        mb = size / (1024 * 1024)
        if mb > GITHUB_LIMIT_MB:
            errors.append(f"  {path}: {mb:.1f} MB (exceeds {GITHUB_LIMIT_MB} MB limit)")
        elif mb > WARN_MB:
            warnings.append(f"  {path}: {mb:.1f} MB (consider chunking)")

    if warnings:
        print("Warnings:")
        print("\n".join(warnings))
        print(f"\nChunk with: python scripts/chunk_large_files.py <file>\n")

    if errors:
        print("ERROR: Files too large for GitHub:")
        print("\n".join(errors))
        print("\nFix: python scripts/chunk_large_files.py <file>")
        return 1

    print(f"OK — no tracked files exceed {GITHUB_LIMIT_MB} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
