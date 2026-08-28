#!/usr/bin/env python3
"""Upload trained checkpoints to HuggingFace Hub for remote simulator access."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def find_latest_checkpoint(base_dir: Path, model_type: str) -> Path | None:
    from utils.utils import find_latest_checkpoint as _find
    try:
        return Path(_find(str(base_dir), model_type))
    except Exception:
        return None


def upload_checkpoints(
    repo_id: str,
    results_dir: Path,
    token: str | None = None,
    private: bool = False,
) -> str:
    api = HfApi(token=token)
    create_repo(repo_id, repo_type="model", exist_ok=True, private=private, token=token)

    staging = Path("/tmp/surgical_worlds_ckpt_upload")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    uploaded = []
    for model_type in ("video_tokenizer", "latent_actions", "dynamics"):
        ckpt = find_latest_checkpoint(results_dir, model_type)
        if ckpt is None:
            print(f"Warning: no {model_type} checkpoint found")
            continue
        dest = staging / model_type / "checkpoints" / ckpt.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ckpt, dest)
        uploaded.append(model_type)
        print(f"Staged {model_type}: {ckpt}")

    if not uploaded:
        raise FileNotFoundError(f"No checkpoints found under {results_dir}")

    api.upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="model",
        commit_message="SurgicalWorlds trained checkpoints",
    )
    url = f"https://huggingface.co/{repo_id}"
    print(f"Uploaded to {url}")
    return url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.environ.get("HF_MODEL_REPO", "yiheinchai/surgical-worlds-model"))
    parser.add_argument("--results-dir", default=os.environ.get("NG_RUN_ROOT_DIR", "results"))
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    results = Path(args.results_dir)
    if not results.exists():
        # Search newest results subfolder
        candidates = sorted(Path("results").glob("surgical-worlds-*"), key=lambda p: p.stat().st_mtime)
        results = candidates[-1] if candidates else Path("results")

    token = os.environ.get("HF_TOKEN")
    url = upload_checkpoints(args.repo, results, token=token, private=args.private)

    status = Path("/workspace/TRAINING_STATUS.json")
    import json
    status.write_text(json.dumps({"hf_model_url": url, "status": "checkpoints_uploaded"}, indent=2))
    print(f"Status written to {status}")


if __name__ == "__main__":
    main()
