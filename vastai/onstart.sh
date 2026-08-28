#!/usr/bin/env bash
# Vast.ai onstart script — runs on every container start (including spot resume).
# Fetches this script from GitHub and launches surgical world model training.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
LOG="/workspace/vastai_train.log"
REPO_DIR="${REPO_DIR:-/workspace/surgical-worlds}"
REPO_URL="${REPO_URL:-https://github.com/yiheinchai/surgical-worlds.git}"
BRANCH="${BRANCH:-main}"
STATUS_SCRIPT=""

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

write_status() {
  local phase="$1"
  shift
  if [ -n "$STATUS_SCRIPT" ] && [ -f "$STATUS_SCRIPT" ]; then
    bash "$STATUS_SCRIPT" "$phase" "$@" 2>&1 | tee -a "$LOG"
  else
    echo "[STATUS] phase=$phase $*" | tee -a "$LOG"
  fi
}

on_fail() {
  local code=$?
  write_status failed exit_code="$code" message="onstart_or_training_failed"
  log "FAILED with exit code $code — see $LOG"
  exit "$code"
}
trap on_fail ERR

log "=== SurgicalWorlds Vast.ai onstart ==="
write_status starting instance="${INSTANCE_ID:-unknown}"
nvidia-smi || true

write_status setup step=system_deps
apt-get update -qq && apt-get install -y -qq git ffmpeg aria2 libgl1 libglib2.0-0 2>/dev/null || true

write_status setup step=git_clone
if [ -d "$REPO_DIR/.git" ]; then
  log "Updating existing repo at $REPO_DIR"
  git -C "$REPO_DIR" fetch origin "$BRANCH"
  git -C "$REPO_DIR" checkout "$BRANCH"
  git -C "$REPO_DIR" pull origin "$BRANCH"
else
  log "Cloning $REPO_URL → $REPO_DIR"
  git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
STATUS_SCRIPT="$REPO_DIR/vastai/write_status.sh"

write_status setup step=pip_install
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q torch torchvision --upgrade 2>/dev/null || true

if [ ! -f "data/surgical/train_laparoscopic_frames.h5" ]; then
  DATA_SOURCE="${DATA_SOURCE:-demo}"
  write_status downloading source="$DATA_SOURCE"
  log "Downloading surgical data (source=$DATA_SOURCE)"
  python3 scripts/download_surgical_data.py \
    --source "$DATA_SOURCE" \
    ${DATA_DOWNLOAD_URL:+--url "$DATA_DOWNLOAD_URL"} \
    ${HF_DATASET_REPO:+--hf-repo "$HF_DATASET_REPO"} \
    --surgery-type "${SURGERY_TYPE:-laparoscopic}" \
    --max-videos "${MAX_VIDEOS:-10}" \
    --read-step "${READ_STEP:-2}"
fi

DATASET="${DATASET:-LAPAROSCOPIC}"
PRELOAD_RATIO="${PRELOAD_RATIO:-0.1}"

write_status training dataset="$DATASET" config="${TRAINING_CONFIG:-configs/quick_training.yaml}"
log "Starting training: dataset=$DATASET preload_ratio=$PRELOAD_RATIO"
export DATASET PRELOAD_RATIO TRAINING_CONFIG
bash vastai/train.sh 2>&1 | tee -a "$LOG"

write_status post_train
log "Training finished — starting post-training"
bash vastai/post_train.sh 2>&1 | tee -a "$LOG"

write_status ready message="pipeline_complete"
