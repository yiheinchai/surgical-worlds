#!/usr/bin/env bash
# Launch a Vast.ai GPU instance for SurgicalWorlds training.
# Requires: pip install vastai && vastai set api-key YOUR_KEY
#
# Usage:
#   export VASTAI_API_KEY=...
#   export WANDB_API_KEY=...          # optional
#   bash vastai/launch.sh             # interactive offer selection
#   bash vastai/launch.sh 12345678    # specific offer ID
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/yiheinchai/surgical-worlds.git}"
IMAGE="${VAST_IMAGE:-pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime}"
DISK_GB="${DISK_GB:-80}"   # 80GB+ if downloading Cholec80-scale archives
MIN_GPU_RAM="${MIN_GPU_RAM:-24}"   # GB — 24GB+ recommended for 128x128 training
MAX_PRICE="${MAX_PRICE:-0.60}"     # $/hr

if ! command -v vastai &>/dev/null; then
    echo "Installing vastai CLI..."
    pip install -q vastai
fi

if [ -z "${VASTAI_API_KEY:-}" ]; then
    echo "Set VASTAI_API_KEY (get from https://cloud.vast.ai/cli/)"
    exit 1
fi

vastai set api-key "$VASTAI_API_KEY"

OFFER_ID="${1:-}"
if [ -z "$OFFER_ID" ]; then
    echo "Searching for GPU offers (>=${MIN_GPU_RAM}GB VRAM, <=\$${MAX_PRICE}/hr)..."
    vastai search offers \
        "gpu_ram >= ${MIN_GPU_RAM} dph <= ${MAX_PRICE} reliability > 0.95 cuda_vers >= 12.0" \
        --order dph \
        --limit 10
    echo ""
    read -rp "Enter offer ID to launch: " OFFER_ID
fi

# Onstart fetches the script from GitHub (stays under 16KB limit)
ONSTART_CMD="curl -fsSL https://raw.githubusercontent.com/yiheinchai/surgical-worlds/main/vastai/onstart.sh | bash"

ENV_FLAGS=""
[ -n "${WANDB_API_KEY:-}" ]     && ENV_FLAGS="$ENV_FLAGS -e WANDB_API_KEY=${WANDB_API_KEY}"
[ -n "${HF_TOKEN:-}" ]           && ENV_FLAGS="$ENV_FLAGS -e HF_TOKEN=${HF_TOKEN}"
[ -n "${DATA_SOURCE:-}" ]        && ENV_FLAGS="$ENV_FLAGS -e DATA_SOURCE=${DATA_SOURCE}"
[ -n "${DATA_DOWNLOAD_URL:-}" ]  && ENV_FLAGS="$ENV_FLAGS -e DATA_DOWNLOAD_URL=${DATA_DOWNLOAD_URL}"
[ -n "${HF_DATASET_REPO:-}" ]    && ENV_FLAGS="$ENV_FLAGS -e HF_DATASET_REPO=${HF_DATASET_REPO}"
[ -n "${MAX_VIDEOS:-}" ]         && ENV_FLAGS="$ENV_FLAGS -e MAX_VIDEOS=${MAX_VIDEOS}"
[ -n "${DATASET:-}" ]            && ENV_FLAGS="$ENV_FLAGS -e DATASET=${DATASET}"
[ -n "${SURGERY_TYPE:-}" ]       && ENV_FLAGS="$ENV_FLAGS -e SURGERY_TYPE=${SURGERY_TYPE}"
[ -n "${PRELOAD_RATIO:-}" ]      && ENV_FLAGS="$ENV_FLAGS -e PRELOAD_RATIO=${PRELOAD_RATIO}"

echo "Creating Vast.ai instance from offer $OFFER_ID..."
vastai create instance "$OFFER_ID" \
    --image "$IMAGE" \
    --disk "$DISK_GB" \
    --ssh --direct \
    --env "$ENV_FLAGS" \
    --onstart-cmd "$ONSTART_CMD"

echo ""
echo "Instance launching. Monitor at https://cloud.vast.ai/instances/"
echo "SSH: vastai ssh <instance_id>"
echo "Logs on instance: /workspace/surgical-worlds-onstart.log"
