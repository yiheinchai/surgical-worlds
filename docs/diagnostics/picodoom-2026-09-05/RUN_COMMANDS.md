# Reproducing the bounded experiments

Use a fresh results directory. The JSON configuration saved beside every run is
the authoritative record of its arguments; split references resolve to shared
content-hashed JSON files. The scripts below read only RGB video. They do not
provision a GPU or authorize a new rental.

The measured environment was Python 3.12.14, Torch 2.11.0+cu128, CUDA 12.8,
RTX 4090. Package versions and data/checkpoint SHA-256 values are recorded in
`bounded_gpu/experiment_provenance.json`. The existing CPU setup and smoke remain
available; a CUDA runtime is required for these measured GPU runs.

Data: `data/picodoom_frames.h5` from the linked original handoff. Place the three
original `model_state_dict.pt` files beneath `data/checkpoints`, preserving the
`video_tokenizer/checkpoints/video_tokenizer_step_37500`,
`latent_actions/checkpoints/latent_actions_step_9500`, and
`dynamics/checkpoints/dynamics_step_44000` directories. Do not load pickled
optimizer/training states.

Store the W&B credential outside the repository. These examples use a path,
never a literal key. Offline mode preserves W&B histories and media when network
logging is unreliable; sync completed run directories afterward. The initial
runs were online, and later runs switched to offline after a logging failure.

```bash
export OMP_NUM_THREADS=4 MPLBACKEND=Agg
export WANDB_MODE=offline
PICODOOM_WANDB_KEY_FILE=/secure/path/wandb_api_key

# Same initial weights, batches, optimizer, seed and step budget per comparison.
for seed in 0 1; do
  .venv/bin/python scripts/bounded_picodoom.py --stage lam --name lam_pair_reference_s${seed} --seed "$seed" --steps 1000 --wandb-mode offline --wandb-key-file "$PICODOOM_WANDB_KEY_FILE"
  .venv/bin/python scripts/bounded_picodoom.py --stage lam --name lam_pair_information_s${seed} --seed "$seed" --steps 1000 --information-weight 0.01 --wandb-mode offline --wandb-key-file "$PICODOOM_WANDB_KEY_FILE"
  .venv/bin/python scripts/bounded_picodoom.py --stage dynamics --name dynamics_legacy_reference_s${seed} --seed "$seed" --steps 1000 --objective legacy_maskgit --wandb-mode offline --wandb-key-file "$PICODOOM_WANDB_KEY_FILE"
  .venv/bin/python scripts/bounded_picodoom.py --stage dynamics --name dynamics_next_frame_s${seed} --seed "$seed" --steps 1000 --objective next_frame --wandb-mode offline --wandb-key-file "$PICODOOM_WANDB_KEY_FILE"
done

.venv/bin/python scripts/bounded_picodoom.py --stage lam --name lam_pair_reference_5000_s0 --steps 5000 --eval-every 1000 --wandb-mode offline --wandb-key-file "$PICODOOM_WANDB_KEY_FILE"
.venv/bin/python scripts/bounded_picodoom.py --stage lam --name lam_pair_information_5000_s0 --steps 5000 --eval-every 1000 --information-weight 0.01 --wandb-mode offline --wandb-key-file "$PICODOOM_WANDB_KEY_FILE"
.venv/bin/python scripts/bounded_picodoom.py --stage dynamics --name dynamics_next_frame_5000_s0 --steps 5000 --eval-every 1000 --rollouts --wandb-mode offline --wandb-key-file "$PICODOOM_WANDB_KEY_FILE"

.venv/bin/python scripts/synthetic_branched_motion.py --name synthetic_branched_reference --wandb-mode offline --key-file "$PICODOOM_WANDB_KEY_FILE"
.venv/bin/python scripts/synthetic_branched_motion.py --name synthetic_branched_information --information-weight 0.01 --wandb-mode offline --key-file "$PICODOOM_WANDB_KEY_FILE"
.venv/bin/python scripts/probe_synthetic_codes.py --key-file "$PICODOOM_WANDB_KEY_FILE"

# Fresh-initialization control, same fixed RGB fixture and update budget.
.venv/bin/python scripts/synthetic_branched_motion.py --name synthetic_scratch_reference --init scratch --wandb-mode offline --key-file "$PICODOOM_WANDB_KEY_FILE"
.venv/bin/python scripts/synthetic_branched_motion.py --name synthetic_scratch_information --init scratch --information-weight 0.01 --wandb-mode offline --key-file "$PICODOOM_WANDB_KEY_FILE"
.venv/bin/python scripts/probe_synthetic_codes.py --prefix synthetic_scratch --name synthetic_scratch_code_semantics --key-file "$PICODOOM_WANDB_KEY_FILE"
.venv/bin/python scripts/probe_tokenizer_baselines.py --key-file "$PICODOOM_WANDB_KEY_FILE"
.venv/bin/python scripts/probe_action_semantics.py --name action_semantics_5000_validation --models reference=results/bounded-20260905/lam_pair_reference_5000_s0/last_weights.pt information=results/bounded-20260905/lam_pair_information_5000_s0/last_weights.pt --key-file "$PICODOOM_WANDB_KEY_FILE"
.venv/bin/python scripts/probe_generation.py --name generation_validation --models original=original next1000=results/bounded-20260905/dynamics_next_frame_s0/last_weights.pt next5000=results/bounded-20260905/dynamics_next_frame_5000_s0/last_weights.pt --key-file "$PICODOOM_WANDB_KEY_FILE"
```

The real experiment also benchmarked 100 LAM updates, and ran the semantics probe
at 1,000 steps. Final-test runs use `bounded_picodoom.py --steps 0 --test` with
the appropriate `--lam-weights` or `--dynamics-weights` path. Exact final-test
configurations accompany their metrics. Final-test outcomes must not be used to
tune this recipe; future adaptive work needs a new test set. The late fresh-initialization control used only synthetic training and validation textures; it did not evaluate on the PicoDoom final-test partition.

## Explicit one-pass deployment option

`DynamicsModel.forward_inference(..., decoding_mode="one_pass")` reproduces the
measured direct fully hidden prediction. It requires `prediction_horizon=1`.
Keep `temperature=0` for deterministic argmax. Longer predictions must advance
one frame at a time. The default remains `decoding_mode="maskgit"`.

The standard inference script accepts OmegaConf overrides
`decoding_mode=one_pass prediction_horizon=1 temperature=0`. Its existing RGB
history re-encoding and interaction plumbing remain separate from this option;
it is not a complete playable PicoDoom frontend. `maskgit_steps` now exposes the
previous fixed 10 iterations, preserving the old default. Setting an iterative
step count to 1 is not equivalent to the explicit one-pass path.
