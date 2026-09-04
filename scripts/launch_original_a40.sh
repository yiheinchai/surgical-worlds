#!/usr/bin/env bash
# Launch exact original TinyWorlds training on a ≥40GB GPU (A40 preferred)
# with Weights & Biases train/val logging.
#
# Prerequisites on this agent machine:
#   export VASTAI_API_KEY=...
#   export WANDB_API_KEY=...          # https://wandb.ai/authorize
#   optional: export WANDB_ENTITY=your_team_or_user
#   optional: export DATASET=PICODOOM|PONG|ZELDA|SONIC  (default PICODOOM)
#
# Usage:
#   bash scripts/launch_original_a40.sh
#   bash scripts/launch_original_a40.sh 47876011   # specific offer
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${VASTAI_API_KEY:-}" ] && [ -f "${HOME}/.config/vastai/vast_api_key" ]; then
  export VASTAI_API_KEY="$(cat "${HOME}/.config/vastai/vast_api_key")"
fi
[ -n "${VASTAI_API_KEY:-}" ] || { echo "Set VASTAI_API_KEY"; exit 1; }
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "${HOME}/.config/wandb/api_key" ]; then
  export WANDB_API_KEY="$(cat "${HOME}/.config/wandb/api_key")"
fi
[ -n "${WANDB_API_KEY:-}" ] || {
  echo "WANDB_API_KEY is required for monitoring."
  echo "Get one at https://wandb.ai/authorize then: export WANDB_API_KEY=..."
  exit 1
}

vastai set api-key "$VASTAI_API_KEY" >/dev/null

OFFER_ID="${1:-}"
if [ -z "$OFFER_ID" ]; then
  PREFER_GPU="$PREFER_GPU" STRICT_GPU="$STRICT_GPU" bash scripts/pick_a40_offer.sh
  OFFER_ID="$(python3 -c 'import json; d=json.load(open("/tmp/selected_vast_offer.json")); print(d.get("selected",d)["id"])')"
fi

IMAGE="${VAST_IMAGE:-pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime}"
DISK_GB="${DISK_GB:-80}"
DATASET="${DATASET:-PICODOOM}"
GIT_BRANCH="${GIT_BRANCH:-cursor/a40-original-tinyworlds-wandb-19ae}"
GIT_REPO="${GIT_REPO:-https://github.com/yiheinchai/surgical-worlds.git}"
WANDB_PROJECT="${WANDB_PROJECT:-tinyworlds}"
WANDB_ENTITY="${WANDB_ENTITY:-data1yihein}"
RUN_GROUP="original-${DATASET}-$(date -u +%Y%m%d_%H%M%S)"
# Prefer exact A40 when STRICT_GPU=1; otherwise bottleneck-aware best ≥40GB.
STRICT_GPU="${STRICT_GPU:-1}"
PREFER_GPU="${PREFER_GPU:-A40}"

# Dataset file patterns for AlmondGod/tinyworlds HF dataset repo
case "$DATASET" in
  PICODOOM) HF_PATTERN='*picodoom*' ;;
  PONG) HF_PATTERN='*pong*' ;;
  ZELDA) HF_PATTERN='*zelda*' ;;
  SONIC) HF_PATTERN='*sonic*' ;;
  POLE_POSITION) HF_PATTERN='*pole*' ;;
  *) echo "Unknown DATASET=$DATASET"; exit 1 ;;
esac

# Onstart: clone branch, install deps, download data, start full_train with wandb.
# Kept compact for Vast's onstart size limit.
read -r -d '' ONSTART <<EOF || true
#!/bin/bash
set -euo pipefail
exec > >(tee -a /workspace/original_tinyworlds_onstart.log) 2>&1
echo "[onstart] \$(date -u) beginning original TinyWorlds setup"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git ffmpeg >/dev/null
pip install -q --upgrade pip
cd /workspace
if [ ! -d tinyworlds/.git ]; then
  git clone --branch ${GIT_BRANCH} ${GIT_REPO} tinyworlds
fi
cd tinyworlds
pip install -q -r requirements.txt wandb huggingface_hub

export PYTHONPATH=/workspace/tinyworlds:\${PYTHONPATH:-}
export WANDB_API_KEY='${WANDB_API_KEY}'
export WANDB_PROJECT='${WANDB_PROJECT}'
export WANDB_RUN_GROUP='${RUN_GROUP}'
export WANDB_VAL_BATCHES=4
${WANDB_ENTITY:+export WANDB_ENTITY='${WANDB_ENTITY}'}

python scripts/download_assets.py datasets --pattern '${HF_PATTERN}' || true
# HF layout may nest; ensure data/ has the expected frames h5
find . -name '*_frames.h5' -o -name '*frames.h5' | head

mkdir -p /workspace/logs
nohup env PYTHONPATH=/workspace/tinyworlds WANDB_API_KEY=\$WANDB_API_KEY WANDB_PROJECT=\$WANDB_PROJECT WANDB_RUN_GROUP=\$WANDB_RUN_GROUP WANDB_VAL_BATCHES=4 ${WANDB_ENTITY:+WANDB_ENTITY=$WANDB_ENTITY} \
  python -u scripts/full_train.py --config configs/training.yaml -- --dataset=${DATASET} use_wandb=true wandb_project=${WANDB_PROJECT} \
  > /workspace/logs/full_train.log 2>&1 &
echo \$! > /workspace/full_train.pid
echo "[onstart] training pid \$(cat /workspace/full_train.pid)"
echo "[onstart] wandb group ${RUN_GROUP} project ${WANDB_PROJECT}"
EOF

ENV_FLAGS="-e WANDB_API_KEY=${WANDB_API_KEY} -e WANDB_PROJECT=${WANDB_PROJECT} -e WANDB_RUN_GROUP=${RUN_GROUP} -e DATASET=${DATASET}"
[ -n "${WANDB_ENTITY}" ] && ENV_FLAGS="$ENV_FLAGS -e WANDB_ENTITY=${WANDB_ENTITY}"

echo "Creating instance from offer $OFFER_ID (disk=${DISK_GB}G, dataset=${DATASET})..."
CREATE_OUT=$(vastai create instance "$OFFER_ID" \
  --image "$IMAGE" \
  --disk "$DISK_GB" \
  --ssh --direct \
  --env "$ENV_FLAGS" \
  --onstart-cmd "$ONSTART" \
  --raw 2>&1) || {
  echo "$CREATE_OUT"
  exit 1
}
echo "$CREATE_OUT"
INSTANCE_ID=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]) if sys.argv[1].strip().startswith('{') else json.load(open('/dev/stdin'));
print(d.get('new_contract') or d.get('id') or '')" "$CREATE_OUT" 2>/dev/null || true)
if [ -z "$INSTANCE_ID" ]; then
  # vastai sometimes prints plain text
  INSTANCE_ID=$(echo "$CREATE_OUT" | python3 -c "import sys,re; t=sys.stdin.read(); m=re.search(r'(\d{6,})', t); print(m.group(1) if m else '')")
fi

mkdir -p /tmp
cat > /tmp/original_tinyworlds_launch.json <<JSON
{
  "instance_id": "${INSTANCE_ID}",
  "offer_id": "${OFFER_ID}",
  "dataset": "${DATASET}",
  "wandb_project": "${WANDB_PROJECT}",
  "wandb_group": "${RUN_GROUP}",
  "wandb_entity": "${WANDB_ENTITY}",
  "git_branch": "${GIT_BRANCH}",
  "launched_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
cp /tmp/original_tinyworlds_launch.json "$REPO_ROOT/ORIGINAL_A40_LAUNCH.json"
echo "Launch record: $REPO_ROOT/ORIGINAL_A40_LAUNCH.json"
echo "Instance: $INSTANCE_ID"
echo "W&B project: ${WANDB_PROJECT}  group: ${RUN_GROUP}"
echo "Monitor: https://wandb.ai/${WANDB_ENTITY:-home}/${WANDB_PROJECT}"
echo "Train log on box: /workspace/logs/full_train.log"
