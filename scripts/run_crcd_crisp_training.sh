#!/usr/bin/env bash
# CRCD full-dataset training for crisp native 128×128 output.
# Strategy: all 18 CRCD videos, denser sampling, full preload — moderate epochs.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/surgical-worlds}"
LOG="/workspace/crcd_crisp_train.log"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "=== CRCD crisp 128×128 training pipeline ==="
cd "$REPO_DIR"

# --- Data: all CRCD episodes, center-crop preprocessing, rebuild caches ---
export MAX_VIDEOS="${MAX_VIDEOS:-18}"
export READ_STEP="${READ_STEP:-5}"
export HF_REPO="${HF_REPO:-morozovdd/CRCD-dVRK-LeRobot}"

log "Downloading up to $MAX_VIDEOS CRCD videos (read_step=$READ_STEP)..."
python3 scripts/download_surgical_data.py \
  --source huggingface \
  --hf-repo "$HF_REPO" \
  --hf-pattern "videos/" \
  --surgery-type robotic \
  --max-videos "$MAX_VIDEOS" \
  --read-step "$READ_STEP" \
  --output-dir data/surgical/downloads \
  --skip-prepare

VIDEO_DIR="data/surgical/downloads/hf_videos"
if [ ! -d "$VIDEO_DIR" ] || [ -z "$(find "$VIDEO_DIR" -maxdepth 1 -name '*.mp4' -print -quit)" ]; then
  VIDEO_DIR=$(find data/surgical/downloads -path '*/hf_videos/*.mp4' | head -1 | xargs dirname)
fi
if [ -z "$VIDEO_DIR" ] || [ ! -d "$VIDEO_DIR" ]; then
  log "ERROR: no CRCD MP4 files found"
  exit 1
fi

N_VIDS=$(find "$VIDEO_DIR" -maxdepth 1 -name '*.mp4' | wc -l)
log "Preparing HDF5 from $VIDEO_DIR ($N_VIDS videos, force rebuild with center_crop_square)"
python3 scripts/prepare_surgical_data.py \
  --input "$VIDEO_DIR" \
  --surgery-type robotic \
  --read-step "$READ_STEP" \
  --resolution 128 128 \
  --force

python3 - <<'PY'
import h5py, json
from pathlib import Path
for split in ("train", "val"):
    p = Path(f"data/surgical/{split}_robotic_frames.h5")
    with h5py.File(p) as h:
        print(f"{split}: {len(h['frames'])} frames")
manifest = json.loads(Path("data/surgical/manifest.json").read_text())
print("manifest:", json.dumps(manifest.get("metadata", {}), indent=2))
PY

# --- Training: full data in memory, crisp-128 configs ---
export DATASET=ROBOTIC_LAPAROSCOPIC
export TRAINING_CONFIG="${TRAINING_CONFIG:-configs/crcd_crisp_128_training.yaml}"
export PRELOAD_RATIO="${PRELOAD_RATIO:-1.0}"
export NG_RUN_ROOT_DIR="${NG_RUN_ROOT_DIR:-/workspace/results/crcd_crisp_128_$(date +%Y%m%d_%H%M%S)}"
export WANDB_MODE=disabled

log "Training dataset=$DATASET config=$TRAINING_CONFIG preload=$PRELOAD_RATIO"
log "Run root: $NG_RUN_ROOT_DIR"
nvidia-smi || true

bash vastai/train.sh 2>&1 | tee -a "$LOG"
bash vastai/post_train.sh 2>&1 | tee -a "$LOG"

log "=== DONE ==="
cat /workspace/TRAINING_STATUS.json 2>/dev/null || true
