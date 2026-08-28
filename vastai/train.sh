#!/usr/bin/env bash
# SurgicalWorlds training runner for Vast.ai — checkpoint-resumable.
set -euo pipefail

WORK_DIR="${WORK_DIR:-/workspace/surgical-worlds}"
cd "$WORK_DIR"

export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"
export NG_RUN_ROOT_DIR="${NG_RUN_ROOT_DIR:-/workspace/results/surgical-worlds-$(date +%Y%m%d_%H%M%S)}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAINING_CONFIG="${TRAINING_CONFIG:-configs/quick_training.yaml}"
DATASET="${DATASET:-LAPAROSCOPIC}"
PRELOAD_RATIO="${PRELOAD_RATIO:-0.1}"
NPROC="${NPROC_PER_NODE:-1}"

EXTRA_ARGS=()
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=disabled
    EXTRA_ARGS+=(use_wandb=false)
fi

echo "=== SurgicalWorlds Training ==="
echo "Config:        $TRAINING_CONFIG"
echo "Dataset:       $DATASET"
echo "Preload ratio: $PRELOAD_RATIO"
echo "GPUs:          $NPROC"
echo "Run root:      $NG_RUN_ROOT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || true

mkdir -p "$NG_RUN_ROOT_DIR"

set +e
if [ "$NPROC" -gt 1 ]; then
    torchrun --nproc_per_node="$NPROC" --standalone scripts/full_train.py \
        --config "$TRAINING_CONFIG" -- dataset="$DATASET" preload_ratio="$PRELOAD_RATIO" "${EXTRA_ARGS[@]}"
else
    python3 scripts/full_train.py \
        --config "$TRAINING_CONFIG" -- dataset="$DATASET" preload_ratio="$PRELOAD_RATIO" "${EXTRA_ARGS[@]}"
fi
RC=$?
set -e

if [ "$RC" -ne 0 ]; then
    echo "=== Training FAILED (exit $RC) ==="
    exit "$RC"
fi

echo "=== Training complete $(date -Is) ==="
echo "Checkpoints: $NG_RUN_ROOT_DIR"
