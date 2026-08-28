#!/usr/bin/env bash
# Post-training: upload checkpoints + launch interactive simulator with public URL.
set -euo pipefail

WORK_DIR="${WORK_DIR:-/workspace/surgical-worlds}"
cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:$PYTHONPATH"

LOG="/workspace/post_train.log"
STATUS="/workspace/TRAINING_STATUS.json"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "=== Post-training: upload + simulator ==="

# Upload checkpoints to HuggingFace if token available
if [ -n "${HF_TOKEN:-}" ]; then
  log "Uploading checkpoints to HuggingFace..."
  python3 scripts/upload_checkpoints.py \
    --repo "${HF_MODEL_REPO:-yiheinchai/surgical-worlds-model}" \
    2>&1 | tee -a "$LOG" || log "HF upload failed (non-fatal)"
else
  log "No HF_TOKEN — checkpoints remain on instance at results/"
fi

# Write status file
python3 - <<'PY'
import json, os
from pathlib import Path
from datetime import datetime, timezone

status = {
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "status": "training_complete",
    "checkpoints": str(os.environ.get("NG_RUN_ROOT_DIR", "results")),
    "simulator": "starting",
}
Path("/workspace/TRAINING_STATUS.json").write_text(json.dumps(status, indent=2))
print("Wrote TRAINING_STATUS.json")
PY

# Launch Gradio simulator with public share link (runs in background)
log "Starting interactive surgery simulator..."
nohup python3 app/surgery_simulator.py \
  --host 0.0.0.0 \
  --port 7860 \
  --share \
  > /workspace/simulator.log 2>&1 &

SIM_PID=$!
echo "$SIM_PID" > /workspace/simulator.pid
log "Simulator PID: $SIM_PID"

# Wait for Gradio public URL in logs (up to 120s)
for i in $(seq 1 60); do
  if grep -q "https://.*gradio.live" /workspace/simulator.log 2>/dev/null; then
    URL=$(grep -o 'https://[^ ]*gradio.live' /workspace/simulator.log | head -1)
    log "🎮 PLAY HERE: $URL"
    python3 -c "
import json
from pathlib import Path
p = Path('/workspace/TRAINING_STATUS.json')
d = json.loads(p.read_text())
d['simulator_url'] = '$URL'
d['simulator_local'] = 'http://0.0.0.0:7860'
d['status'] = 'ready'
p.write_text(json.dumps(d, indent=2))
"
    break
  fi
  sleep 2
done

log "=== Ready for interaction ==="
log "Local:  http://<instance-ip>:7860"
log "Status: /workspace/TRAINING_STATUS.json"
log "Logs:   /workspace/simulator.log"
