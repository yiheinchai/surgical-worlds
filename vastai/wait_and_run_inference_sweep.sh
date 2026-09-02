#!/usr/bin/env bash
# Wait for crisp dynamics training to finish, then run GPU inference sweep + push to GitHub.
set -euo pipefail

WORK_DIR="${WORK_DIR:-/workspace/surgical-worlds}"
cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}"

LOG="/workspace/inference_sweep.log"
SWEEP_OUT="$WORK_DIR/docs/inference_sweep"
TRAIN_LOG="${TRAIN_LOG:-/workspace/crcd_crisp_train.log}"
GITHUB_BRANCH="${GITHUB_BRANCH:-cursor/crcd-crisp-128-training-19ae}"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "=== Inference sweep watchdog started ==="

# --- Wait for training ---
log "Waiting for dynamics to reach 30000 steps..."
while true; do
  PROGRESS=$(grep -oP '\d+/30000' "$TRAIN_LOG" 2>/dev/null | tail -1 || true)
  STEP="${PROGRESS%%/*}"
  if [ "${STEP:-0}" -ge 30000 ] 2>/dev/null; then
    log "Training complete: $PROGRESS"
    break
  fi
  if ! tmux has-session -t crcd-crisp-train 2>/dev/null; then
    log "Training tmux ended at $PROGRESS — proceeding"
    break
  fi
  log "Still training: ${PROGRESS:-unknown} (sleep 120s)"
  sleep 120
done

# Allow final checkpoint write
sleep 30

# --- Resolve run root ---
RUN_ROOT=$(ls -td results/*/dynamics 2>/dev/null | head -1 | xargs dirname)
if [ -z "$RUN_ROOT" ] || [ ! -d "$RUN_ROOT/dynamics/checkpoints" ]; then
  log "ERROR: could not find run root under results/"
  exit 1
fi
log "Run root: $RUN_ROOT"

DYN_STEP=$(ls "$RUN_ROOT/dynamics/checkpoints" | grep -oP 'dynamics_step_\K\d+' | sort -n | tail -1)
log "Latest dynamics checkpoint: dynamics_step_$DYN_STEP"

# --- GPU inference sweep (side-by-side GT|WM) ---
log "Starting batch inference experiments on GPU..."
python3 scripts/batch_inference_experiments.py \
  --run-root "$RUN_ROOT" \
  --output-dir "$SWEEP_OUT" \
  --device cuda \
  --dynamics-step "$DYN_STEP" \
  2>&1 | tee -a "$LOG"

# --- 3-panel sanity videos (tokenizer vs dynamics) ---
SANITY_DIR="$SWEEP_OUT/sanity"
mkdir -p "$SANITY_DIR"
VT="$RUN_ROOT/video_tokenizer/checkpoints/video_tokenizer_step_14000"
LAM="$RUN_ROOT/latent_actions/checkpoints/latent_actions_step_7000"
DYN="$RUN_ROOT/dynamics/checkpoints/dynamics_step_${DYN_STEP}"

for SEED in 0 3 25 100; do
  for MODE in gt fixed; do
  EXTRA=""
  [ "$MODE" = "fixed" ] && EXTRA="--action-mode fixed --fixed-action 4"
  python3 scripts/render_sanity_video.py \
    --output "$SANITY_DIR/sanity_seed${SEED}_${MODE}.mp4" \
    --video-tokenizer-path "$VT" \
    --latent-actions-path "$LAM" \
    --dynamics-path "$DYN" \
    --device cuda \
    --seed "$SEED" \
    --steps 12 \
    --action-mode "$MODE" \
    --temperature 0.0 \
    $EXTRA \
    2>&1 | tee -a "$LOG"
  done
done

# --- README ---
python3 - <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone

out = Path("docs/inference_sweep")
manifest = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").exists() else {}
cats = sorted({e.get("category", "?") for e in manifest.get("experiments", []) if "error" not in e})
lines = [
    "# CRCD Inference Sweep",
    "",
    f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
    f"Dynamics checkpoint: `dynamics_step_{manifest.get('dynamics_step_default', '?')}`",
    f"Experiments: {manifest.get('experiment_count', '?')} ({len([e for e in manifest.get('experiments', []) if 'error' not in e])} succeeded)",
    "",
    "## Layout",
    "",
    "**Side-by-side demos** (`by_*/*.mp4`): left = ground truth, right = autoregressive world model.",
    "",
    "**Sanity videos** (`sanity/*.mp4`): GT | tokenizer recon | dynamics (3 columns).",
    "",
    "## Categories",
    "",
]
for c in cats:
    n = len(list((out / c).glob("*.mp4"))) if (out / c).exists() else 0
    lines.append(f"- **{c}/** — {n} videos")
lines += [
    "",
    "## Quick picks",
    "",
    "| File | What it tests |",
    "|------|----------------|",
    "| `by_seed/seed0003_left_then_right.mp4` | Baseline on seed 3 |",
    "| `by_checkpoint/dyn30000_left_then_right.mp4` | Final checkpoint |",
    "| `by_context/ctx2_left_then_right.mp4` | Context window 2 vs trained 4 |",
    "| `by_horizon/ph2_left_then_right.mp4` | Predict 2 frames per step |",
    "| `sanity/sanity_seed003_gt.mp4` | Tokenizer OK + best-case dynamics |",
    "| `sanity/sanity_seed003_fixed.mp4` | Same with forced action 4 |",
    "",
    "See `manifest.json` for full experiment metadata.",
]
(out / "README.md").write_text("\n".join(lines) + "\n")
print("Wrote README.md")
PY

# --- Push to GitHub ---
log "Pushing to GitHub..."
if [ -n "${GITHUB_TOKEN:-}" ]; then
  git remote set-url yiheinchai "https://x-access-token:${GITHUB_TOKEN}@github.com/yiheinchai/surgical-worlds.git" 2>/dev/null \
    || git remote add yiheinchai "https://x-access-token:${GITHUB_TOKEN}@github.com/yiheinchai/surgical-worlds.git"
  git fetch yiheinchai "$GITHUB_BRANCH" 2>/dev/null || true
  git checkout -B "$GITHUB_BRANCH" 2>/dev/null || git checkout "$GITHUB_BRANCH"
  git add docs/inference_sweep/
  git add scripts/batch_inference_experiments.py scripts/render_sanity_video.py simulator/engine.py 2>/dev/null || true
  if git diff --cached --quiet; then
    log "Nothing new to commit"
  else
    git -c user.email="agent@cursor.com" -c user.name="Cursor Agent" \
      commit -m "Add CRCD inference sweep experiments (post-training batch)"
    git push yiheinchai "$GITHUB_BRANCH"
    log "Pushed to yiheinchai/surgical-worlds branch $GITHUB_BRANCH"
  fi
else
  log "No GITHUB_TOKEN — sweep saved locally at $SWEEP_OUT"
  log "Agent will need to pull and push manually"
  tar czf /workspace/inference_sweep.tar.gz -C "$WORK_DIR/docs" inference_sweep
  log "Archive: /workspace/inference_sweep.tar.gz"
fi

log "=== Inference sweep complete ==="
