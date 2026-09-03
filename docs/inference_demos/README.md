# CRCD GPU Inference Demos

Side-by-side world model inference on **CRCD-dVRK-LeRobot** (RTX 3090, ~1s/step).

| File | Scripted actions |
|------|------------------|
| [crcd_circle_instruments.mp4](crcd_circle_instruments.mp4) | Left grasp → retract → right grasp → camera (24 steps) |
| [crcd_dual_grasp_sweep.mp4](crcd_dual_grasp_sweep.mp4) | Dual grasp sweep (18 steps) |
| [crcd_cautery_pass.mp4](crcd_cautery_pass.mp4) | Cautery + retract + camera (18 steps) |
| [crcd_camera_orbit.mp4](crcd_camera_orbit.mp4) | Camera orbit pattern (18 steps) |
| [crcd_left_then_right.mp4](crcd_left_then_right.mp4) | Tool left ×8, then right ×8 (16 steps) |
| [crcd_sanity_gt_action.mp4](crcd_sanity_gt_action.mp4) | **Sanity check:** GT \| tokenizer recon \| dynamics (GT action), seed 3, 12 steps |
| [crcd_sanity_action4.mp4](crcd_sanity_action4.mp4) | **Sanity check:** GT \| tokenizer recon \| dynamics (forced action 4), seed 3, 12 steps |

**Layout (action demos):** left = ground truth CRCD frame, right = world model prediction.

**Layout (sanity videos):** three columns — ground truth | tokenizer encode/decode of GT | dynamics prediction (`dynamics_step_19000`, teacher-forced GT context).

**Checkpoints:** `dynamics_step_3800`, `latent_actions_step_500`, `video_tokenizer_step_1000`

**Regenerate:**
```bash
python scripts/render_action_demo_videos.py --device cuda --output-dir inference_results/action_demos
```
