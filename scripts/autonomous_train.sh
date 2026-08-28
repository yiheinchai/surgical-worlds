#!/usr/bin/env bash
# SurgicalWorlds — Fully Autonomous Training Pipeline
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export DATA_SOURCE="${DATA_SOURCE:-}"
export MAX_VIDEOS="${MAX_VIDEOS:-10}"
export TRAINING_MODE="${TRAINING_MODE:-quick}"
export PRELOAD_RATIO="${PRELOAD_RATIO:-0.3}"
export DISK_GB="${DISK_GB:-80}"
export MIN_GPU_RAM="${MIN_GPU_RAM:-24}"
export MAX_PRICE="${MAX_PRICE:-0.70}"
if [ "$TRAINING_MODE" = "quick" ]; then
  export TRAINING_CONFIG="configs/quick_training.yaml"
  export DATA_SOURCE="${DATA_SOURCE:-demo}"
else
  export TRAINING_CONFIG="configs/vastai_training.yaml"
  export DATA_SOURCE="${DATA_SOURCE:-cholect50}"
fi
[ -z "${VASTAI_API_KEY:-}" ] && echo "Set VASTAI_API_KEY" && exit 1
pip install -q vastai 2>/dev/null || true
vastai set api-key "$VASTAI_API_KEY"
OFFER_ID="${1:-}"
if [ -z "$OFFER_ID" ]; then
  OFFER_ID=$(vastai search offers "gpu_ram >= ${MIN_GPU_RAM} dph <= ${MAX_PRICE} reliability > 0.95 cuda_vers >= 12.0" --order dph --limit 1 --raw 2>/dev/null | python3 -c "import sys,json; d=json.loads(sys.stdin.read() or '[]'); o=d if isinstance(d,list) else d.get('offers',[]); print(o[0]['id'] if o else '')" 2>/dev/null || true)
  [ -z "$OFFER_ID" ] && echo "No offer found" && exit 1
fi
ONSTART_CMD="curl -fsSL https://raw.githubusercontent.com/yiheinchai/surgical-worlds/main/vastai/onstart.sh | bash"
ENV_FLAGS="-e DATA_SOURCE=${DATA_SOURCE} -e MAX_VIDEOS=${MAX_VIDEOS} -e TRAINING_CONFIG=${TRAINING_CONFIG} -e PRELOAD_RATIO=${PRELOAD_RATIO} -e TRAINING_MODE=${TRAINING_MODE}"
RESULT=$(vastai create instance "$OFFER_ID" --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime --disk "$DISK_GB" --ssh --direct --env "$ENV_FLAGS" --onstart-cmd "$ONSTART_CMD" 2>&1)
echo "$RESULT"
INSTANCE_ID=$(echo "$RESULT" | python3 -c "import sys,re,ast; t=sys.stdin.read(); m=re.search(r'new_contract[\"\\']?\\s*[:=]\\s*(\\d+)',t); print(m.group(1) if m else '')" 2>/dev/null || true)
printf '{"instance_id":"%s","training_mode":"%s","data_source":"%s"}\n' "$INSTANCE_ID" "$TRAINING_MODE" "$DATA_SOURCE" > /tmp/surgical_worlds_launch.json
echo "Instance ID: $INSTANCE_ID (data=$DATA_SOURCE)"
[ -n "$INSTANCE_ID" ] && bash "$REPO_ROOT/scripts/wait_for_health.sh" "$INSTANCE_ID" 20 || true
echo "Monitor: bash scripts/monitor_training.sh $INSTANCE_ID"
