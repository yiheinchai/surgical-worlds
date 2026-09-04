# Original PicoDoom TinyWorlds — Agent Handoff

**Status:** training stopped early; GPUs destroyed; checkpoints backed up and released.  
**Goal for next agents:** fix **latent-action collapse** and/or **dynamics objective/rollout mismatch**, then retrain.

---

## TL;DR

1. Trained original AlmondGod TinyWorlds recipe on **PicoDoom** (`64²`, patch `4`, `n_actions=4`).
2. **VT** looks healthy. **LAM** collapses to mostly **action 0** (action 3 dead). **Dyn** val CE plateaus ~12k–44k; free-rollout looks best around **dyn@20k**, not later.
3. Root issue is **not** “need 32 actions.” With `n_actions=4`, unsupervised LAM still collapses; Dyn then trains on near-constant conditioning.
4. Checkpoints + this doc are in-repo / GitHub Release for follow-up work.

---

## Artifacts

| What | Where |
|---|---|
| Git branch | `cursor/a40-original-tinyworlds-wandb-19ae` on `yiheinchai/surgical-worlds` |
| Handoff doc | `docs/ORIGINAL_PICODOOM_HANDOFF.md` (this file) |
| Run monitor JSON | `ORIGINAL_TRAIN_MONITOR.json` |
| Checkpoint release | GitHub Release tag `original-picodoom-20260904` (see repo Releases) |
| Local backup (agent VM) | `/agent/checkpoint-backups/49801668/FINAL/` |

### Checkpoint layout (after extracting release)

```text
results/2026_09_03_23_40_55/
  video_tokenizer/checkpoints/video_tokenizer_step_{0..37500}
  latent_actions/checkpoints/latent_actions_step_{0..9500}
  dynamics/checkpoints/dynamics_step_{0..44000}
```

**Recommended eval set:**
- VT: `video_tokenizer_step_37500`
- LAM: `latent_actions_step_9500` (collapsed — use for diagnosis)
- Dyn: `dynamics_step_20000` (best perceptual in side-by-side demo) and `dynamics_step_44000` (last)

---

## WandB

Entity: `data2yihein-d` · Project: `tinyworlds` · Group: `original-PICODOOM-20260903_234048`

| Stage | Run | State | URL |
|---|---|---|---|
| Video tokenizer | `48l9ulop` | finished | https://wandb.ai/data2yihein-d/tinyworlds/runs/48l9ulop |
| Latent actions | `4d5anb7o` | finished | https://wandb.ai/data2yihein-d/tinyworlds/runs/4d5anb7o |
| Dynamics | `b9lhcat1` | crashed (instance destroy) | https://wandb.ai/data2yihein-d/tinyworlds/runs/b9lhcat1 |

Failed dyn launches (compile/OOM before successful microbatch run): `9ltwufjs`, `rklllf2u`.

API key for this agent env: `~/.config/wandb/api_key` (account `data2yihein`).

---

## Config actually used

- Dataset: `PICODOOM`
- `preload_ratio=0.05` (**not** yaml default `0.005`, which loads **0 samples** — upstream empty-dataset bug)
- Frame `64`, patch `4`, `n_actions=4`, context `4`
- VT: `embed_dim=32`, `num_blocks=4`, 40k updates → finished @ 37500 ckpt
- LAM: `embed_dim=32`, `num_blocks=2`, 10k updates
- Dyn: `embed_dim=32`, `num_blocks=8`, planned 300k; stopped ~**44k**
- Dyn launch overrides on Blackwell 48GB: `compile=false`, `batch_size_per_gpu=125`, `gradient_accumulation_steps=4` (effective batch 500)
- Param scale: VT ~0.14M, LAM ~0.07M, Dyn ~0.17M (~0.4M total)

Training instance (destroyed): Vast `49801668` RTX PRO 5000 Blackwell ~$0.69/hr.  
Demo instance (destroyed): Vast `49855427` RTX 3060 ~$0.09/hr.

---

## Results summary

### Video tokenizer — OK
- Train/val loss ≈ **0.025 / 0.025**
- Codebook usage → **100%**
- Not the bottleneck

### Latent actions — COLLAPSED (primary bug)
WandB `action_distribution`:
- Early: action **1** dominates (~0.5k–4k)
- After ~4k: action **0** dominates to the end
- Action **3** dies; `unique_actions=3`, usage 0.75, entropy ~0.5 (uniform-4 would be ~1.39)

Offline probe on PicoDoom with final LAM ckpt:
- Usage ≈ **80% / 17% / 3% / 0%** for actions 0–3
- FSQ cells `(0,0),(1,0),(0,1),(1,1)` → ids 0–3; **(1,1) unused** (dim saturates)
- **Identity baseline** (copy prev frame) L1 ≈ 0.151 **beats** LAM recon ≈ 0.156
- Encoded actions beat a zero action vector only **~34%** of the time
- Codes are **not** clean Doom controls; action 0 is a dump bucket, not a clean noop

Mechanism (see `models/latent_actions.py`):
- Loss = next-frame recon + weak continuous variance penalty
- `keep_rate=0.0` masks patches but first frame stays visible → identity-like solutions win
- No discrete entropy / codebook usage loss → FSQ corner collapse

### Dynamics — early win, then plateau
- Val CE: **~7.1 → ~4.0 by ~12k**, then flat **~3.75–4.24 through 44k**
- Train ~1.0 with persistent **~3.0 train/val gap**
- Same collapsed action stream (`unique_actions=3`) while training
- Interactive/multi-ckpt demo: **dyn@20k** consistently looked best; **30k/42k** muddier despite sometimes better MAE
- Free rollout collapses after a few steps (MaskGIT hole-filling train ≠ multi-step decode)

---

## Demo code added this run

- `app/picodoom_simulator.py` — single-ckpt Gradio play
- `app/picodoom_ckpt_compare.py` — multi-ckpt parallel rollout with Latent 0–3 buttons
- `simulator/engine.py` — `preview_next`, `resync_from_ground_truth` helpers

These are for diagnosis; GPUs are gone.

---

## Suggested next experiments (priority)

1. **Fix LAM collapse (do this first)**  
   - Add discrete codebook usage / entropy regularizer  
   - Prevent FSQ dim saturation  
   - Make identity fail (stronger masking, residual prediction, or stop giving full previous frame)  
   - Keep `n_actions=4` unless diagnosis says otherwise  
   - Success metric: flat-ish action histogram, all 4 codes used, beat identity baseline, action-swap changes frames a lot

2. **Align Dyn objective with play**  
   - Mask **future** frame(s) only  
   - Log teacher-forced + **N-step free-rollout** metrics; early-stop on rollout, not CE  
   - Action dropout / wrong-action negatives so Dyn cannot ignore conditioning

3. **Data**  
   - Raise `preload_ratio` (0.05 was thin); never use `0.005` without fixing empty-load bug  
   - Optionally stream full H5

4. **Scale model only after 1–2**  
   - Tiny width (~0.17M Dyn) may be a ceiling later; scaling first mostly memorizes the wrong task

---

## Repro sketch

```bash
# data
python scripts/download_assets.py datasets --pattern '*picodoom*'
# or HF dataset AlmondGod/TinyWorlds → picodoom_frames.h5

# extract release checkpoints under results/
export PYTHONPATH=.
# diagnose LAM actions
python -c "from utils.utils import load_latent_actions_from_checkpoint; ..."

# compare dyn ckpts (needs GPU + data + Gradio)
python app/picodoom_ckpt_compare.py --share
```

---

## Open questions for next agent

- Can a small LAM change restore 4-way usage without changing `n_actions`?
- Does residual-frame prediction remove the identity attractor?
- With a healthy LAM, does Dyn val/rollout keep improving past 20k?
- Is PicoDoom’s visual delta distribution too continuous for 4 hard codes without aux losses?

---

## Contact / provenance

- Cloud agent conversation: Surgical world model / original PicoDoom training  
- Destroyed Vast IDs: train `49801668`, demo `49855427`  
- Stop reason: user halt after val plateau + action-collapse diagnosis
