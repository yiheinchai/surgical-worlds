# SurgicalWorlds — Genie 3 for Surgery

**Play simulated laparoscopic surgeries.** Control instruments, watch the world model generate the next frame — the same interaction paradigm as [Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/), applied to surgical video.

Built on [TinyWorlds](https://github.com/AlmondGod/tinyworlds), a minimal autoregressive world model based on Google DeepMind's [Genie Architecture](https://arxiv.org/pdf/2402.15391).

## Quick Start — Play Now

```bash
git clone https://github.com/yiheinchai/surgical-worlds.git
cd surgical-worlds
pip install -r requirements.txt
export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 scripts/play_surgery.py
```

Open **http://localhost:7860** — click instrument buttons to perform a simulated surgery.

## Train on Real Surgical Video

```bash
python3 scripts/prepare_surgical_data.py --input /path/to/videos/
python3 scripts/full_train.py --config configs/surgical_training.yaml
python3 scripts/play_surgery.py
```

## License

MIT (TinyWorlds). Surgical datasets have separate licenses. See [LICENSE](LICENSE).
