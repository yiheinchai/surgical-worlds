#!/usr/bin/env bash
# Post-training: upload checkpoints + launch interactive simulator with public URL.
set -euo pipefail

WORK_DIR="${WORK_DIR:-/workspace/surgical-worlds}"
cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}"

LOG="/workspace/post_train.log"
STATUS="/workspace/TRAINING_STATUS.json"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "=== Post-training: upload + simulator ==="

# Ensure run root points at the trained checkpoint tree
if [ -z "${NG_RUN_ROOT_DIR:-}" ] || [ ! -d "${NG_RUN_ROOT_DIR}" ]; then
  NG_RUN_ROOT_DIR=$(python3 - <<'PY'
import glob, os
runs = sorted(glob.glob("results/*/video_tokenizer"), key=os.path.getmtime)
if runs:
    print(os.path.dirname(runs[-1]))
PY
)
  export NG_RUN_ROOT_DIR
fi
if [ -n "${NG_RUN_ROOT_DIR:-}" ]; then
  log "Using run root: $NG_RUN_ROOT_DIR"
else
  log "WARNING: could not detect NG_RUN_ROOT_DIR — simulator may fall back to demo mode"
fi

# Final checkpoint upload (mandatory if credentials present)
if [ -n "${HF_TOKEN:-}" ] || [ -n "${GITHUB_TOKEN:-}" ]; then
  log "Uploading all checkpoints off-instance..."
  if python3 scripts/upload_checkpoints.py --sync-all 2>&1 | tee -a "$LOG"; then
    log "Checkpoint upload complete"
  else
    log "ERROR: checkpoint upload failed — checkpoints may be lost if instance stops!"
    exit 1
  fi
else
  log "CRITICAL: No HF_TOKEN or GITHUB_TOKEN on this instance."
  log "Checkpoints are NOT being uploaded off-instance. Use agent_pull_checkpoints.sh from the agent VM,"
  log "or set HF_TOKEN before launching future Vast.ai runs."
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
export NG_RUN_ROOT_DIR="${NG_RUN_ROOT_DIR:-}"
nohup env NG_RUN_ROOT_DIR="${NG_RUN_ROOT_DIR:-}" PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}" \
  python3 app/surgery_simulator.py \
  --host 0.0.0.0 \
  --port 7860 \
  --share \
  > /workspace/simulator.log 2>&1 &

SIM_PID=$!
echo "$SIM_PID" > /workspace/simulator.pid
log "Simulator PID: $SIM_PID"

# Wait for Gradio public URL (retry up to ~15 min — gradio.live tunnel can be flaky)
URL=""
for attempt in $(seq 1 30); do
  if grep -q "https://.*gradio.live" /workspace/simulator.log 2>/dev/null; then
    URL=$(grep -o 'https://[^ ]*gradio.live' /workspace/simulator.log | head -1)
    break
  fi
  if grep -q "Could not create share link" /workspace/simulator.log 2>/dev/null; then
    log "Gradio share failed (attempt $attempt/30) — restarting simulator..."
    kill "$(cat /workspace/simulator.pid 2>/dev/null)" 2>/dev/null || true
    sleep 5
    nohup env NG_RUN_ROOT_DIR="${NG_RUN_ROOT_DIR:-}" PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}" \
      python3 app/surgery_simulator.py --host 0.0.0.0 --port 7860 --share \
      >> /workspace/simulator.log 2>&1 &
    echo $! > /workspace/simulator.pid
    sleep 30
  else
    sleep 2
  fi
done

if [ -n "$URL" ]; then
  log "🎮 PLAY HERE: $URL"
  python3 -c "
import json, os
from pathlib import Path
p = Path('/workspace/TRAINING_STATUS.json')
d = json.loads(p.read_text()) if p.exists() else {}
d['simulator_url'] = '$URL'
d['simulator_local'] = 'http://0.0.0.0:7860'
d['status'] = 'ready'
d['phase'] = 'ready'
d['run_root'] = os.environ.get('NG_RUN_ROOT_DIR', '')
p.write_text(json.dumps(d, indent=2))
"
else
  log "WARNING: Could not obtain gradio.live URL after retries."
  log "Trying cloudflared tunnel as fallback..."
  CF=/tmp/cloudflared
  if [ ! -x "$CF" ]; then
    curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o "$CF" && chmod +x "$CF" || true
  fi
  if [ -x "$CF" ]; then
    nohup "$CF" tunnel --url http://127.0.0.1:7860 > /workspace/cloudflared_tunnel.log 2>&1 &
    sleep 8
    URL=$(grep -o 'https://[^ ]*trycloudflare.com' /workspace/cloudflared_tunnel.log 2>/dev/null | head -1 || true)
    if [ -n "$URL" ]; then
      log "🎮 PLAY HERE (cloudflared): $URL"
    fi
  fi
  if [ -z "${URL:-}" ]; then
    log "Simulator is running locally on port 7860 — use SSH port-forward or local play instructions."
  fi
  python3 -c "
import json, os
from pathlib import Path
p = Path('/workspace/TRAINING_STATUS.json')
d = json.loads(p.read_text()) if p.exists() else {}
d['simulator_local'] = 'http://0.0.0.0:7860'
d['status'] = 'ready_no_public_url' if not '${URL:-}' else 'ready'
d['phase'] = 'ready'
d['run_root'] = os.environ.get('NG_RUN_ROOT_DIR', '')
if '${URL:-}':
    d['simulator_url'] = '${URL:-}'
    d['simulator_tunnel'] = 'cloudflared'
else:
    d['simulator_note'] = 'gradio.live tunnel unavailable — use local play or SSH port-forward'
p.write_text(json.dumps(d, indent=2))
"
fi

log "=== Ready for interaction ==="
log "Local:  http://<instance-ip>:7860"
log "Status: /workspace/TRAINING_STATUS.json"
log "Logs:   /workspace/simulator.log"
