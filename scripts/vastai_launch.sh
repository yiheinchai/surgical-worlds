#!/usr/bin/env bash
# Launch a Vast.ai GPU instance for SurgicalWorlds training.
#
# Prerequisites:
#   pip install vastai
#   vastai set api-key YOUR_VAST_API_KEY
#
# Usage:
#   ./scripts/vastai_launch.sh                  # search & launch cheapest RTX 4090
#   ./scripts/vastai_launch.sh 12345678         # launch specific offer ID
#   MIN_GPU_RAM=24 ./scripts/vastai_launch.sh     # require 24GB+ VRAM

set -euo pipefail

MIN_GPU_RAM="${MIN_GPU_RAM:-16}"
DISK_GB="${DISK_GB:-50}"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime}"
ONSTART_URL="${ONSTART_URL:-https://raw.githubusercontent.com/yiheinchai/surgical-worlds/main/vastai/onstart.sh}"

if ! command -v vastai &>/dev/null; then
  echo "Install Vast.ai CLI: pip install vastai"
  echo "Then: vastai set api-key YOUR_KEY"
  exit 1
fi

OFFER_ID="${1:-}"

if [ -z "$OFFER_ID" ]; then
  echo "Searching for GPU offers (>= ${MIN_GPU_RAM}GB VRAM)..."
  vastai search offers \
    "gpu_ram >= $MIN_GPU_RAM num_gpus = 1 reliability > 0.95 dph < 1.0" \
    --order dph --limit 5
  echo ""
  echo "Copy an OFFER_ID from above and re-run:"
  echo "  ./scripts/vastai_launch.sh <OFFER_ID>"
  exit 0
fi

# Build env flags for secrets (never hardcode keys)
ENV_FLAGS=""
[ -n "${WANDB_API_KEY:-}" ] && ENV_FLAGS="$ENV_FLAGS -e WANDB_API_KEY=$WANDB_API_KEY"
[ -n "${HF_TOKEN:-}" ]       && ENV_FLAGS="$ENV_FLAGS -e HF_TOKEN=$HF_TOKEN"
[ -n "${DATASET:-}" ]        && ENV_FLAGS="$ENV_FLAGS -e DATASET=$DATASET"
[ -n "${PRELOAD_RATIO:-}" ]  && ENV_FLAGS="$ENV_FLAGS -e PRELOAD_RATIO=$PRELOAD_RATIO"

ONSTART_CMD="curl -fsSL '$ONSTART_URL' | bash"

echo "Launching instance from offer $OFFER_ID..."
vastai create instance "$OFFER_ID" \
  --image "$IMAGE" \
  --disk "$DISK_GB" \
  --ssh --direct \
  --env "$ENV_FLAGS" \
  --onstart-cmd "$ONSTART_CMD"

echo ""
echo "Instance launching. Monitor with: vastai show instances"
echo "SSH in with: vastai ssh <INSTANCE_ID>"
echo "Training log: /workspace/vastai_train.log"
