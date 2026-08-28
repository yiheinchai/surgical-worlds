#!/usr/bin/env bash
# Vast.ai onstart script — runs on every container start (including spot resume).
# Fetches this script from GitHub and launches surgical world model training.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
LOG="/workspace/vastai_train.log"
REPO_DIR="${REPO_DIR:-/workspace/surgical-worlds}"
REPO_URL="${REPO_URL:-https://github.com/yiheinchai/surgical-worlds.git}"
BRANCH="${BRANCH:-main}"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "=== SurgicalWorlds Vast.ai onstart ==="
nvidia-smi || true

# System deps (aria2 for fast multi-connection downloads on Vast.ai)
apt-get update -qq && apt-get install -y -qq git ffmpeg aria2 libgl1 libglib2.0-0 2>/dev/null || true

# Clone or update repo
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
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# Python deps (upgrades torch to version in requirements.txt)
pip install -q --upgrade pip
pip install -q -r requirements.txt
# Gradio not needed on training instances
pip install -q torch torchvision --upgrade 2>/dev/null || true

# Prepare demo data if no surgical data uploaded
if [ ! -f "data/surgical/train_laparoscopic_frames.h5" ]; then
  DATA_SOURCE="${DATA_SOURCE:-demo}"
  log "Downloading surgical data (source=$DATA_SOURCE) — Vast.ai has fast datacenter bandwidth"
  python3 scripts/download_surgical_data.py \
    --source "$DATA_SOURCE" \
    ${DATA_DOWNLOAD_URL:+--url "$DATA_DOWNLOAD_URL"} \
    ${HF_DATASET_REPO:+--hf-repo "$HF_DATASET_REPO"} \
    --surgery-type "${SURGERY_TYPE:-laparoscopic}" \
    --max-videos "${MAX_VIDEOS:-10}" \
    --read-step "${READ_STEP:-2}"
fi

# Training overrides from env
DATASET="${DATASET:-LAPAROSCOPIC}"
PRELOAD_RATIO="${PRELOAD_RATIO:-0.1}"
WANDB_PROJECT="${WANDB_PROJECT:-surgical-worlds}"

log "Starting training: dataset=$DATASET preload_ratio=$PRELOAD_RATIO"
export DATASET PRELOAD_RATIO
bash vastai/train.sh 2>&1 | tee -a "$LOG"
