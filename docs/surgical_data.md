# Surgical Video Data

See [SURGICAL_WORLDS.md](../SURGICAL_WORLDS.md) for the full guide.

Place laparoscopic or robotic surgical procedure videos under `data/surgical/` for training.

## Supported Open Datasets

| Dataset | Type | Access |
|---------|------|--------|
| [Cholec80](https://camma.unistra.fr/datasets/) | Manual laparoscopic | Request form (CC BY-NC-SA 4.0) |
| [HeiChole](https://www.synapse.org/heichole) | Manual laparoscopic | Synapse registration |
| [PhaKIR](https://arxiv.org/abs/2511.06549) | Manual laparoscopic | Zenodo request |
| [EndoVis / SAR-RARP](https://endovissub2019-robotic-scene-segmentation.grand-challenge.org/) | Robotic laparoscopic | Challenge registration |

## Prepare Your Videos

```bash
python scripts/generate_demo_surgical_video.py   # optional synthetic demo
python scripts/prepare_surgical_data.py --input /path/to/videos/
```

## Large Files & GitHub 100 MB Limit

GitHub rejects any single file over **100 MB**. Surgical datasets (Cholec80 ~70 GB) must never be committed directly.

Raw `.h5` / `.mp4` files are **gitignored**. To upload a large preprocessed HDF5 via GitHub:

```bash
# Split into 90 MB chunks (under the 100 MB limit)
python scripts/chunk_large_files.py data/surgical/train_laparoscopic_frames.h5

# Reassemble on Vast.ai or another machine
python scripts/unchunk_large_files.py data/surgical/chunks/train_laparoscopic_frames.h5.manifest.json

# Pre-push safety check
python scripts/check_github_file_sizes.py
```

For training, upload videos directly to your Vast.ai instance instead of via GitHub.
