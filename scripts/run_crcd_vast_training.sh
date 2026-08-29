#!/usr/bin/env bash
# Train SurgicalWorlds on CRCD-dVRK-LeRobot on a running Vast.ai instance.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/surgical-worlds}"
LOG="/workspace/crcd_train.log"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "=== CRCD robotic training pipeline ==="
cd "$REPO_DIR"

# Clean stale demo data so we fetch CRCD
rm -f data/surgical/train_laparoscopic_frames.h5 data/surgical/val_laparoscopic_frames.h5
rm -f data/surgical/train_robotic_frames.h5 data/surgical/val_robotic_frames.h5

log "Downloading CRCD-dVRK-LeRobot (~2.1 GB)..."
python3 scripts/download_surgical_data.py \
  --source huggingface \
  --hf-repo morozovdd/CRCD-dVRK-LeRobot \
  --hf-pattern "videos/" \
  --surgery-type robotic \
  --max-videos 18 \
  --read-step "${READ_STEP:-10}" \
  --output-dir data/surgical/downloads \
  --skip-prepare

VIDEO_DIR="data/surgical/downloads/hf_videos"
if [ ! -d "$VIDEO_DIR" ] || [ -z "$(find "$VIDEO_DIR" -maxdepth 1 -name '*.mp4' -print -quit)" ]; then
  VIDEO_DIR=$(find data/surgical/downloads -path '*/hf_videos/*.mp4' | head -1 | xargs dirname)
fi
if [ -z "$VIDEO_DIR" ] || [ ! -d "$VIDEO_DIR" ]; then
  log "ERROR: no CRCD MP4 files found under data/surgical/downloads"
  exit 1
fi
log "Preparing HDF5 from $VIDEO_DIR (read_step=${READ_STEP:-10}, max_frames=${MAX_FRAMES_PER_VIDEO:-2500})"
python3 scripts/prepare_surgical_data.py \
  --input "$VIDEO_DIR" \
  --surgery-type robotic \
  --read-step "${READ_STEP:-10}" \
  --max-frames-per-video "${MAX_FRAMES_PER_VIDEO:-2500}"

export DATASET=ROBOTIC_LAPAROSCOPIC
export TRAINING_CONFIG="${TRAINING_CONFIG:-configs/quick_training.yaml}"
export PRELOAD_RATIO="${PRELOAD_RATIO:-0.08}"
export NG_RUN_ROOT_DIR="${NG_RUN_ROOT_DIR:-/workspace/results/crcd_$(date +%Y%m%d_%H%M%S)}"
export WANDB_MODE=disabled

log "Training dataset=$DATASET config=$TRAINING_CONFIG preload=$PRELOAD_RATIO"
log "Run root: $NG_RUN_ROOT_DIR"
nvidia-smi || true

bash vastai/train.sh 2>&1 | tee -a "$LOG"
bash vastai/post_train.sh 2>&1 | tee -a "$LOG"

log "=== DONE ==="
cat /workspace/TRAINING_STATUS.json 2>/dev/null || true
