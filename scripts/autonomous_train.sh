#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# SurgicalWorlds — Fully Autonomous Training Pipeline
#
# One command to: find GPU → download data → train → upload → launch simulator
#
# Required secret: VASTAI_API_KEY
# Optional:        HF_TOKEN (upload checkpoints), WANDB_API_KEY (logging)
#
# Usage:
#   export VASTAI_API_KEY=your_key
#   bash scripts/autonomous_train.sh
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── Config ──────────────────────────────────────────────────────────────────
export DATA_SOURCE="${DATA_SOURCE:-cholect50}"
export MAX_VIDEOS="${MAX_VIDEOS:-10}"
export TRAINING_MODE="${TRAINING_MODE:-quick}"   # quick | full
export PRELOAD_RATIO="${PRELOAD_RATIO:-0.3}"
export DISK_GB="${DISK_GB:-80}"
export MIN_GPU_RAM="${MIN_GPU_RAM:-24}"
export MAX_PRICE="${MAX_PRICE:-0.70}"

if [ "$TRAINING_MODE" = "quick" ]; then
  export TRAINING_CONFIG="configs/quick_training.yaml"
else
  export TRAINING_CONFIG="configs/vastai_training.yaml"
fi
export TRAINING_CONFIG  # used by vastai/train.sh via onstart

# ── Prerequisites ───────────────────────────────────────────────────────────
if [ -z "${VASTAI_API_KEY:-}" ]; then
  echo "ERROR: Set VASTAI_API_KEY to launch autonomous training."
  echo "  Get one at: https://cloud.vast.ai/cli/"
  echo "  Then run:   export VASTAI_API_KEY=your_key && bash scripts/autonomous_train.sh"
  exit 1
fi

pip install -q vastai 2>/dev/null || true
vastai set api-key "$VASTAI_API_KEY"

# ── Auto-select cheapest suitable GPU ───────────────────────────────────────
OFFER_ID="${1:-}"
if [ -z "$OFFER_ID" ]; then
  echo "Searching for GPU (>=${MIN_GPU_RAM}GB, <=\$${MAX_PRICE}/hr)..."
  OFFER_ID=$(vastai search offers \
    "gpu_ram >= ${MIN_GPU_RAM} dph <= ${MAX_PRICE} reliability > 0.95 cuda_vers >= 12.0" \
    --order dph --limit 1 --raw 2>/dev/null | python3 -c "
import sys, json
raw = sys.stdin.read().strip()
if not raw: sys.exit(1)
data = json.loads(raw)
offers = data if isinstance(data, list) else data.get('offers', [])
if not offers: sys.exit(1)
print(offers[0]['id'])
" 2>/dev/null || true)

  if [ -z "$OFFER_ID" ]; then
    echo "Could not auto-select offer. Showing top 5:"
    vastai search offers \
      "gpu_ram >= ${MIN_GPU_RAM} dph <= ${MAX_PRICE} reliability > 0.95" \
      --order dph --limit 5
    echo ""
    echo "Re-run with offer ID: bash scripts/autonomous_train.sh <OFFER_ID>"
    exit 1
  fi
  echo "Auto-selected offer: $OFFER_ID"
fi

# ── Launch instance ─────────────────────────────────────────────────────────
ONSTART_CMD="curl -fsSL https://raw.githubusercontent.com/yiheinchai/surgical-worlds/main/vastai/onstart.sh | bash"

ENV_FLAGS="-e DATA_SOURCE=${DATA_SOURCE} -e MAX_VIDEOS=${MAX_VIDEOS}"
ENV_FLAGS="$ENV_FLAGS -e TRAINING_CONFIG=${TRAINING_CONFIG}"
ENV_FLAGS="$ENV_FLAGS -e PRELOAD_RATIO=${PRELOAD_RATIO}"
ENV_FLAGS="$ENV_FLAGS -e TRAINING_MODE=${TRAINING_MODE}"
[ -n "${HF_TOKEN:-}" ]         && ENV_FLAGS="$ENV_FLAGS -e HF_TOKEN=${HF_TOKEN}"
[ -n "${WANDB_API_KEY:-}" ]    && ENV_FLAGS="$ENV_FLAGS -e WANDB_API_KEY=${WANDB_API_KEY}"
[ -n "${HF_MODEL_REPO:-}" ]    && ENV_FLAGS="$ENV_FLAGS -e HF_MODEL_REPO=${HF_MODEL_REPO}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Launching SurgicalWorlds autonomous training"
echo "  Mode:     $TRAINING_MODE ($TRAINING_CONFIG)"
echo "  Data:     $DATA_SOURCE ($MAX_VIDEOS videos)"
echo "  Offer:    $OFFER_ID"
echo "═══════════════════════════════════════════════════════════"
echo ""

RESULT=$(vastai create instance "$OFFER_ID" \
  --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime \
  --disk "$DISK_GB" \
  --ssh --direct \
  --env "$ENV_FLAGS" \
  --onstart-cmd "$ONSTART_CMD" 2>&1)

echo "$RESULT"

INSTANCE_ID=$(echo "$RESULT" | python3 -c "
import sys, re
text = sys.stdin.read()
m = re.search(r'\"new_contract\":\s*(\d+)', text) or re.search(r'instance[_ ]?id[:\s]+(\d+)', text, re.I)
if m: print(m.group(1))
" 2>/dev/null || true)

# Save launch info locally
cat > /tmp/surgical_worlds_launch.json <<EOF
{
  "offer_id": "$OFFER_ID",
  "instance_id": "$INSTANCE_ID",
  "training_mode": "$TRAINING_MODE",
  "data_source": "$DATA_SOURCE",
  "launched_at": "$(date -Iseconds)"
}
EOF

echo ""
echo "Instance launched!"
[ -n "$INSTANCE_ID" ] && echo "  Instance ID: $INSTANCE_ID"
echo "  Monitor:     vastai show instances"
echo "  SSH:         vastai ssh $INSTANCE_ID"
echo "  Train log:   /workspace/vastai_train.log (on instance)"
echo "  Status:      /workspace/TRAINING_STATUS.json (on instance, after training)"
echo "  Simulator:   starts automatically with public Gradio URL when done"
echo ""
echo "When you're back, run:"
echo "  vastai ssh $INSTANCE_ID"
echo "  cat /workspace/TRAINING_STATUS.json"
