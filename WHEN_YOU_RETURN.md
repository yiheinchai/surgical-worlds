# When You're Back — SurgicalWorlds

Training runs fully autonomously on Vast.ai. When it's done, you'll have an interactive surgery simulator ready.

## Find your simulator

SSH into the instance and check status:

```bash
vastai show instances          # find your instance ID
vastai ssh <INSTANCE_ID>
cat /workspace/TRAINING_STATUS.json
```

The status file contains:
- `simulator_url` — public Gradio link (play in browser from anywhere)
- `simulator_local` — local URL on the instance
- `hf_model_url` — HuggingFace checkpoint link (if HF_TOKEN was set)

## Play locally (if you downloaded checkpoints)

```bash
git clone https://github.com/yiheinchai/surgical-worlds.git
cd surgical-worlds
pip install -r requirements.txt
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Download checkpoints from HuggingFace (if uploaded)
huggingface-cli download yiheinchai/surgical-worlds-model --local-dir checkpoints/

python3 scripts/play_surgery.py
# Enable "Use trained world model" in the UI
```

## What ran automatically

1. Downloaded CholecT50 laparoscopic videos (10 procedures)
2. Trained all 3 stages (quick mode, ~3-6 hrs on RTX 4090)
3. Uploaded checkpoints to HuggingFace (if token provided)
4. Launched Gradio surgery simulator with public URL

## Re-launch training

```bash
export VASTAI_API_KEY=your_key
bash scripts/autonomous_train.sh
```

For full-quality training (24-48+ hrs):
```bash
TRAINING_MODE=full MAX_VIDEOS=50 bash scripts/autonomous_train.sh
```
