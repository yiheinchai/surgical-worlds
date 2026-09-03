# CRCD crisp 128×128 training

Next-run recipe optimized for **sharp native 128×128** with **more data, fewer epochs** than the full surgical configs.

## What changed

| Area | Before (quick run) | Crisp 128 run |
|------|-------------------|---------------|
| Videos | 4 | **18** (full CRCD) |
| `read_step` | 10 | **5** (2× denser sampling) |
| `preload_ratio` | 0.08 | **1.0** (all frames in memory) |
| Preprocessing | widescreen→square squash | **`center_crop_square`** |
| Video tokenizer steps | 4,000 | **15,000** |
| Latent actions steps | 500 | **8,000** |
| Dynamics steps | 4,000 | **30,000** |
| Recon loss | smooth L1 only | **L1 + gradient** (sharper) |
| `embed_dim` | 48 | **64** |

## Launch on Vast.ai

```bash
cd /workspace/surgical-worlds
bash scripts/run_crcd_crisp_training.sh
```

Environment overrides:
- `MAX_VIDEOS=18` — all CRCD episodes
- `READ_STEP=5` — frame sampling density
- `PRELOAD_RATIO=1.0` — use full HDF5

## Validate dataset before training

```bash
python3 scripts/validate_dataset_preview.py
# → docs/inference_demos/dataset_preview_native_128.png
```

Check `video_tokenizer_recon_step_*.png` during stage 1 — recon should match ground truth at native 128×128 before continuing to dynamics.

## Config files

- `configs/crcd_crisp_128_training.yaml` — master config
- `configs/crcd_crisp_128_video_tokenizer.yaml`
- `configs/crcd_crisp_128_latent_actions.yaml`
- `configs/crcd_crisp_128_dynamics.yaml`
