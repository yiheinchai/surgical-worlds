# SurgicalWorlds — Genie 3 for Surgery

**Play simulated laparoscopic surgeries.** Control instruments, watch the world model generate the next frame — the same interaction paradigm as [Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/), applied to surgical video.

## The Product

```
┌──────────────────────────────────────────────────────────────┐
│                   🏥 SurgicalWorlds UI                        │
│  ┌────────────────────────────┐  ┌─────────────────────────┐ │
│  │                            │  │  ✊ Grasp   🖐 Release  │ │
│  │   Laparoscopic Viewport    │  │  ← → ↑ ↓  Instrument   │ │
│  │   (live generated frames)  │  │  📷 Camera   ⏸ Hold     │ │
│  │                            │  │  🔄 New Procedure        │ │
│  └────────────────────────────┘  └─────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
         ▲                                    │
         │ next frame                         │ user action
         │                                    ▼
    ┌─────────┐    ┌──────────────┐    ┌─────────────┐
    │ Dynamics│◄───│ Latent Action│◄───│  User Input │
    │  Model  │    │   Tokenizer  │    │  (buttons)  │
    └────┬────┘    └──────────────┘    └─────────────┘
         │
    ┌────▼────┐
    │  Video  │
    │Tokenizer│
    └─────────┘
```

**Genie 3 interaction loop:**
1. User sees current surgical frame in the viewport
2. User presses an instrument control (grasp, move left, camera pan…)
3. Action maps to a latent motion token
4. Dynamics model predicts the next frame autoregressively
5. New frame appears — repeat

No action labels needed during training. The model learns instrument motions from video alone.

## Quick Start — Play Now

```bash
cd /agent/tinyworlds
pip install -r requirements.txt
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Launch the playable simulator (demo mode works immediately)
python3 scripts/play_surgery.py
```

Open **http://localhost:7860** — click instrument buttons to perform a simulated surgery.

- **Demo mode** (default): synthetic laparoscopic scene with instrument physics — playable without training
- **Model mode**: enable "Use trained world model" after training for real Genie-style frame generation

## Train on Real Surgical Video

```bash
# 1. Prepare laparoscopic videos (Cholec80, HeiChole, your own .mp4s)
python3 scripts/prepare_surgical_data.py --input /path/to/videos/

# 2. Train all three stages
python3 scripts/full_train.py --config configs/surgical_training.yaml

# 3. Play with the trained world model
python3 scripts/play_surgery.py
# → enable "Use trained world model" in the UI
```

## Controls

### Manual Laparoscopic

| Control | Action |
|---------|--------|
| ⏸ Hold | Maintain instrument position |
| ✊ Grasp | Close jaws / grasp tissue |
| 🖐 Release | Open jaws |
| ← → ↑ ↓ | Move instrument |
| 📷 Camera | Pan / retract camera |

### Robotic Laparoscopic (da Vinci / EndoVis)

Switch surgery type to **robotic** for dual-arm controls (left grasp, right grasp, retract, cautery).

## Architecture (TinyWorlds / Genie)

Built on [TinyWorlds](https://github.com/AlmondGod/tinyworlds):

| Module | Purpose |
|--------|---------|
| Video Tokenizer | FSQ VAE — frames → discrete tokens |
| Action Tokenizer | Infers latent instrument motion between frames |
| Dynamics Model | MaskGIT-style next-frame prediction |

## Project Layout

```
app/surgery_simulator.py     ← Main product (Gradio UI)
scripts/play_surgery.py      ← Launcher
simulator/
  engine.py                  ← Real world model engine
  demo_engine.py             ← Immediate-play demo
  controls.py                ← Genie 3-style action mappings
configs/
  surgical_training.yaml     ← Training pipeline
  simulator.yaml             ← Simulator settings
```

## Supported Video Sources

| Dataset | Type | Access |
|---------|------|--------|
| [Cholec80](https://camma.unistra.fr/datasets/) | Manual laparoscopic | Request form |
| [HeiChole](https://www.synapse.org/heichole) | Multi-center laparoscopic | Synapse |
| [EndoVis / SAR-RARP](https://endovissub2019-robotic-scene-segmentation.grand-challenge.org/) | Robotic | Challenge registration |

## Requirements

- Python 3.10+
- GPU recommended for model mode (demo mode runs on CPU)
- ~8GB+ VRAM for 128×128 surgical training

## License

MIT (TinyWorlds). Surgical datasets have separate licenses.
