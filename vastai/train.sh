#!/usr/bin/env bash
# SurgicalWorlds training runner for Vast.ai
set -euo pipefail
WORK_DIR="${WORK_DIR:-/workspace/surgical-worlds}"
cd "$WORK_DIR"
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"
export NG_RUN_ROOT_DIR="${NG_RUN_ROOT_DIR:-/workspace/results/surgical-worlds-$(date +%Y%m%d_%H%M%S)}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/vastai_training.yaml}"
DATASET="${DATASET:-LAPAROSCOPIC}"
PRELOAD_RATIO="${PRELOAD_RATIO:-0.1}"
NPROC="${NPROC_PER_NODE:-1}"
EXTRA_ARGS=()
if [ -z "${WANDB_API_KEY:-}" ]; then EXTRA_ARGS+=(use_wandb=false); fi
echo "=== SurgicalWorlds Training ==="
echo "Config: $TRAINING_CONFIG Dataset: $DATASET Preload: $PRELOAD_RATIO GPUs: $NPROC"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true
mkdir -p "$NG_RUN_ROOT_DIR"
python3 scripts/full_train.py --config "$TRAINING_CONFIG" -- dataset="$DATASET" preload_ratio="$PRELOAD_RATIO" "${EXTRA_ARGS[@]}"
echo "=== Training complete ==="
