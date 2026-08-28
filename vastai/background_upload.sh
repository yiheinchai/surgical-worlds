#!/usr/bin/env bash
# Background checkpoint sync — Vast.ai wipes disk when instances stop.
# Runs during training; uploads after each new checkpoint and on an interval.
set -euo pipefail

WORK_DIR="${WORK_DIR:-/workspace/surgical-worlds}"
cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}"

LOG="${CKPT_UPLOAD_LOG:-/workspace/checkpoint_upload.log}"
INTERVAL="${CKPT_UPLOAD_INTERVAL_SEC:-1800}"  # 30 minutes
PIDFILE="/workspace/checkpoint_upload.pid"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  log "Checkpoint uploader already running (pid $(cat "$PIDFILE"))"
  exit 0
fi

if [ -z "${HF_TOKEN:-}" ] && [ -z "${GITHUB_TOKEN:-}" ]; then
  log "WARNING: No HF_TOKEN or GITHUB_TOKEN — checkpoints will NOT be uploaded off-instance!"
  log "Set HF_TOKEN (recommended) or GITHUB_TOKEN before launching Vast.ai training."
  exit 0
fi

log "Starting background checkpoint uploader (interval=${INTERVAL}s)"

(
  while true; do
    log "Syncing checkpoints..."
    if python3 scripts/upload_checkpoints.py --sync-all 2>&1 | tee -a "$LOG"; then
      log "Checkpoint sync OK"
    else
      log "Checkpoint sync failed (will retry)"
    fi
    sleep "$INTERVAL"
  done
) >> "$LOG" 2>&1 &

echo $! > "$PIDFILE"
log "Background uploader started (pid $(cat "$PIDFILE"))"
