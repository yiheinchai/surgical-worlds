# CRCD Inference Sweep

Generated: 2026-09-03 03:19 UTC
Dynamics checkpoint: `dynamics_step_29000`
Experiments: 69 (54 succeeded, 15 failed)

## Layout

**Side-by-side demos** (`by_*/*.mp4`): left = ground truth, right = autoregressive world model.

**Sanity videos** (`sanity/*.mp4`): GT | tokenizer recon | dynamics (3 columns).

## Categories

- **by_action/** — 11 videos
- **by_checkpoint/** — 4 videos
- **by_context/** — 14 videos
- **by_horizon/** — 12 videos
- **by_length/** — 3 videos
- **by_maskgit/** — 2 videos
- **by_seed/** — 16 videos
- **by_temperature/** — 2 videos
- **sanity/** — 8 videos

## Notes

- Trained context length = 4; `prediction_horizon > 1` experiments fail (action broadcast shape mismatch).
- Latest dynamics checkpoint: **dynamics_step_29000** (training interrupted at ~29498/30000).

See `manifest.json` for full experiment metadata.

## Playback note

Videos are **H.264 / yuv420p**. An earlier upload used OpenCV `mp4v` (MPEG-4 Part 2), which many browsers and GitHub's player render as a solid green frame.
