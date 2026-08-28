# Training on Vast.ai

Run SurgicalWorlds training on cheap cloud GPUs via [Vast.ai](https://vast.ai).

## 1. Setup Vast.ai CLI

```bash
pip install vastai
vastai set api-key YOUR_VAST_API_KEY   # from https://cloud.vast.ai/cli/
```

## 2. Launch training (one command)

```bash
# Search for GPUs and show cheapest offers
./scripts/vastai_launch.sh

# Launch on a specific offer
./scripts/vastai_launch.sh 12345678

# With W&B logging
WANDB_API_KEY=your_key ./scripts/vastai_launch.sh 12345678
```

The instance will automatically:
1. Clone `yiheinchai/surgical-worlds` from GitHub
2. Install dependencies
3. Generate demo surgical data (if no data uploaded)
4. Run full 3-stage training (`configs/surgical_training.yaml`)

Training log: `/workspace/vastai_train.log`

## 3. SSH into instance

```bash
vastai show instances
vastai ssh <INSTANCE_ID>
tail -f /workspace/vastai_train.log
```

## 4. Train on your own surgical videos

Upload videos to the instance, then prepare data before training:

```bash
# On the Vast.ai instance
cd /workspace/surgical-worlds
# Upload videos to /workspace/videos/ via scp or vastai copy
python3 scripts/prepare_surgical_data.py --input /workspace/videos/
DATASET=LAPAROSCOPIC PRELOAD_RATIO=1.0 bash vastai/onstart.sh
```

Or set env vars at launch time:

```bash
DATASET=LAPAROSCOPIC PRELOAD_RATIO=0.5 WANDB_API_KEY=xxx ./scripts/vastai_launch.sh <OFFER_ID>
```

## 5. Spot / interruptible instances

The `vastai/onstart.sh` script re-runs on every container start (including spot resume). Training checkpoints are saved under `results/` — increase `preload_ratio` gradually as training stabilizes.

## Recommended GPUs

| GPU | VRAM | Notes |
|-----|------|-------|
| RTX 4090 | 24 GB | Good balance of cost/speed |
| RTX 3090 | 24 GB | Cheaper, slightly slower |
| A100 40GB | 40 GB | Faster dynamics training |
| L40S | 48 GB | Good for larger batch sizes |

Set minimum VRAM: `MIN_GPU_RAM=24 ./scripts/vastai_launch.sh`

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WANDB_API_KEY` | — | Weights & Biases logging |
| `HF_TOKEN` | — | HuggingFace token for dataset download |
| `DATASET` | `LAPAROSCOPIC` | `LAPAROSCOPIC` or `ROBOTIC_LAPAROSCOPIC` |
| `PRELOAD_RATIO` | `0.1` | Fraction of dataset to use (increase over time) |
| `REPO_URL` | `yiheinchai/surgical-worlds` | Git repo to clone |
| `BRANCH` | `main` | Git branch |

## Cost estimate

At ~$0.30–0.50/hr for an RTX 4090:
- Video tokenizer stage (~60k steps): ~4–8 hrs
- Latent actions (~20k steps): ~2–4 hrs  
- Dynamics (~400k steps): ~24–48 hrs

Use `preload_ratio=0.05` for initial debugging runs to validate the pipeline cheaply.
