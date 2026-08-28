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
