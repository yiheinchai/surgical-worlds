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
3. **Download surgical data on the remote machine** (fast datacenter bandwidth)
4. Run full 3-stage training (`configs/vastai_training.yaml`)

Training log: `/workspace/vastai_train.log`

## 3. Download surgical data on the remote instance (recommended)

Vast.ai machines have very fast download speeds — **don't upload from home**. Let the instance fetch data directly.

### Option A — CholecT50 from HuggingFace (10 videos, no registration)

```bash
DATA_SOURCE=cholect50 MAX_VIDEOS=10 bash vastai/launch.sh <OFFER_ID>
```

Downloads laparoscopic cholecystectomy frame sequences from `Voxel51/cholect50` on HuggingFace, stitches to MP4, and prepares training HDF5.

### Option B — Direct archive URL (Cholec80, HeiChole mirror, your cloud storage)

After you get dataset access (CAMMA form, Synapse, etc.), host or use a direct download link:

```bash
DATA_SOURCE=url \
DATA_DOWNLOAD_URL="https://your-storage.com/cholec80.tar.gz" \
DISK_GB=120 \
bash vastai/launch.sh <OFFER_ID>
```

Supports `.tar.gz`, `.zip`, `.tar`. Uses `aria2c` (16 connections) for maximum speed.

### Option C — HuggingFace video dataset

```bash
DATA_SOURCE=huggingface \
HF_DATASET_REPO="orena-dkfz/lapchole-focus-vqa" \
HF_TOKEN=your_hf_token \
MAX_VIDEOS=20 \
bash vastai/launch.sh <OFFER_ID>
```

### Option D — Demo only (pipeline test)

```bash
DATA_SOURCE=demo bash vastai/launch.sh <OFFER_ID>
```

### Manual download on a running instance

```bash
vastai ssh <INSTANCE_ID>
cd /workspace/surgical-worlds
python3 scripts/download_surgical_data.py --source cholect50 --max-videos 20
python3 scripts/download_surgical_data.py --source url --url "https://..."
```

## 4. SSH into instance

```bash
vastai show instances
vastai ssh <INSTANCE_ID>
tail -f /workspace/vastai_train.log
```

## 5. Train on your own surgical videos (alternative to remote download)

If you already have videos on the instance (e.g. from a completed download):

```bash
cd /workspace/surgical-worlds
python3 scripts/prepare_surgical_data.py --input /workspace/surgical/downloads/extracted/videos/
PRELOAD_RATIO=1.0 bash vastai/train.sh
```

## 6. Spot / interruptible instances

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
| `DATA_SOURCE` | `demo` | `demo`, `url`, `huggingface`, `cholect50` |
| `DATA_DOWNLOAD_URL` | — | Direct archive URL (for `DATA_SOURCE=url`) |
| `HF_DATASET_REPO` | `orena-dkfz/lapchole-focus-vqa` | HuggingFace dataset repo |
| `HF_TOKEN` | — | HuggingFace token for gated datasets |
| `MAX_VIDEOS` | `10` | Limit videos downloaded |
| `DISK_GB` | `80` | Disk space (use 120+ for Cholec80 archives) |
| `WANDB_API_KEY` | — | Weights & Biases logging |
| `DATASET` | `LAPAROSCOPIC` | `LAPAROSCOPIC` or `ROBOTIC_LAPAROSCOPIC` |
| `PRELOAD_RATIO` | `0.1` | Fraction of dataset to use (increase over time) |

## Cost estimate

At ~$0.30–0.50/hr for an RTX 4090:
- Video tokenizer stage (~60k steps): ~4–8 hrs
- Latent actions (~20k steps): ~2–4 hrs  
- Dynamics (~400k steps): ~24–48 hrs

Use `preload_ratio=0.05` for initial debugging runs to validate the pipeline cheaply.
