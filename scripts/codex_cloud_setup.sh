#!/usr/bin/env bash
# Self-contained: paste this file into Codex Cloud's setup-script field.
# It also works before the task branch has been checked out from a cached main.
set -euo pipefail

cloud_repo_root="$(git rev-parse --show-toplevel)"
cloud_python="${CLOUD_PYTHON:-python3.12}"
if ! command -v "$cloud_python" >/dev/null 2>&1; then
  cloud_python=python3
fi
"$cloud_python" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 14), "Use Python 3.10–3.13 (recommended: 3.12)"'
cloud_venv="$cloud_repo_root/.venv"
if [[ ! -x "$cloud_venv/bin/python" ]]; then
  "$cloud_python" -m venv "$cloud_venv"
fi

if command -v uv >/dev/null 2>&1; then
  cloud_installer=(uv pip install --python "$cloud_venv/bin/python")
else
  cloud_installer=("$cloud_venv/bin/python" -m pip install)
fi

if [[ "$(uname -s)" == Linux ]]; then
  "${cloud_installer[@]}" torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
else
  "${cloud_installer[@]}" torch==2.8.0 torchvision==0.23.0
fi
"${cloud_installer[@]}" \
  'numpy>=1.26,<3' 'einops>=0.8,<1' 'h5py>=3.11,<4' \
  omegaconf==2.3.0 'opencv-python-headless>=4.8,<5' \
  'matplotlib>=3.8,<4' 'tqdm>=4.66,<5' 'wandb>=0.15,<1' \
  'huggingface_hub>=0.23,<2' 'requests>=2.31,<3' 'pytest>=8,<10'

"$cloud_venv/bin/python" - <<'PY'
import cv2, einops, h5py, matplotlib, omegaconf, pytest, torch, torchvision, wandb
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
print(f"Cloud dependencies ready: torch={torch.__version__}, Python environment installed")
PY

if [[ -f "$cloud_repo_root/scripts/codex_cloud_smoke.py" ]]; then
  cd "$cloud_repo_root"
  WANDB_MODE=disabled MPLBACKEND=Agg "$cloud_venv/bin/python" scripts/codex_cloud_smoke.py
fi
