# Cloud continuation: unsupervised surgical world models

## User's objective

Develop an unsupervised video-only world model with discovered discrete actions, following DeepMind's Genie. The user tried simulated surgery, real robotic laparoscopic video, Pong, and then a longer PicoDoom run. Reconstructions improved but dynamics and control remained poor. **The user explicitly rejected switching the research approach to supervised action/kinematic training.** They requested cloud continuation so development can continue while their PC is off.

This branch starts from the actual PicoDoom experiment branch, commit `22b7aefa691228d4a189eb4bd5c3d9267f5190ed`, and adds the current diagnosis plus cloud setup. The original local main was `693e45c`. Preserve previous checkpoints as baselines.

## Read first

1. `docs/diagnostics/picodoom-2026-09-05/README.md` — findings and the revised video-only experiment plan.
2. `docs/diagnostics/picodoom-2026-09-05/probe_results.json` — CPU float32 checkpoint probes.
3. `docs/diagnostics/picodoom-2026-09-05/audit_metrics.json` — sanitized live W&B histories and data/masking checks. Analyze it programmatically; avoid printing the full histories.
4. `docs/ORIGINAL_PICODOOM_HANDOFF.md` — checkpoint locations and historical context. Its claimed three-point train/validation gap is a logging artifact; its “held-out validation” is not actually held out; a balanced histogram alone is not a success criterion. See the newer diagnosis.

## Confirmed findings

- Dynamics training logs loss after division by gradient accumulation (4), while validation logs undivided CE. Corrected train and validation losses both plateau near four.
- PicoDoom train and validation loaders receive identical frames because `disable_test_split=True` remains the default. The loaded prefix comprises 2,689 frames and 2,681 overlapping windows. Split episodes, or separate contiguous ranges with a context gap; do not randomly split overlapping windows.
- Random temporal anchors make fully hidden next frames effectively absent during training. Holding scoring positions, context, and actions fixed, dynamics@44k CE changes from 3.780 with partial target visibility to 6.426 with the target entirely hidden.
- LAM has strongly imbalanced codes, but some predictive information. The variance floor applies before quantization and is already inactive while codes collapse. The 32-clip probe yielded counts 80/11/5/0. Code 0 is vector (-1,-1), not a physical no-op.
- LAM is frozen during dynamics training, so longer dynamics training cannot repair its encoder.
- The LAM's smooth-L1 + spatial-gradient objective beats copying while ordinary L1 is worse than copying. Report both the actual objective and motion-sensitive/rollout measures.
- On eight in-sample one-step probes, dynamics@44k beats @20k but both lose to copying the previous frame. This does not establish the best long-rollout checkpoint.
- No fix has been trained yet. Do not report these diagnostic findings as a completed solution.

## First cloud task: make the next experiment interpretable

Implement and validate the smallest coherent changes necessary for the next video-only experiment:

1. Fix undivided microbatch-averaged training-loss logging. Apply it consistently wherever accumulation can distort reported losses.
2. Add an explicit disjoint validation policy for game videos with no overlapping source frames across the split. Preserve clear behavior for existing callers and reject empty partitions with useful messages.
3. Add an explicit dynamics training mode with clean causal history and a wholly masked next frame, scoring only the next-frame target tokens. Keep a selectable legacy mode for checkpoint comparisons. Check action alignment, causality, target isolation, and finite gradients with meaningful tests.
4. Add deterministic video-only evaluation of copy-frame, inferred-action, constant-action, and shuffled-action baselines. Evaluate fully hidden one-step prediction separately from partially masked token reconstruction. Make short rollouts reproducible by fixing clips and seeds.
5. Add a small CPU-capable configuration/smoke experiment. Synthetic videos are acceptable for diagnostics, but no action labels may be consumed during training. Do not equate tiny-set memorization with generalization or claim useful controls solely from entropy.
6. Update the evidence record with commands, numerical results, limitations, and the next LAM experiment. Return a reviewable diff.

If that scope is completed within the task, prepare the two-frame LAM ablation described in the diagnosis. Keep architecture changes separate from masking and logging corrections; do not combine VQ, a new optimizer, bigger models, and new losses into one uninterpretable run.

This handoff authorizes development, tests, and bounded CPU diagnostics in the Codex Cloud task. It does not establish a paid GPU budget or authorize an unbounded recurring job. Preserve a runnable next-run recipe if GPU compute is needed.

## Setup and commands

The environment setup installs Python dependencies into `.venv`. It has no dependency on a local Mac, mounted folders, W&B authentication, or a running Vast instance. All earlier Vast training/demo instances were destroyed.

```bash
bash scripts/codex_cloud_setup.sh
WANDB_MODE=disabled MPLBACKEND=Agg .venv/bin/python scripts/codex_cloud_smoke.py
.venv/bin/python -m pytest -q
```

The first command is idempotent. Paste its **full contents** into the cloud setup field, because cached environments can initially check out the repository's default branch, which may not contain this setup file. Maintenance can run `if [ -f scripts/codex_cloud_setup.sh ]; then bash scripts/codex_cloud_setup.sh; fi` after the task branch is checked out.

Set non-secret environment variables `WANDB_MODE=disabled`, `MPLBACKEND=Agg`, and `OMP_NUM_THREADS=2`. W&B credentials are not needed for the exported evidence. If live W&B access is later required, use the cloud environment's credential mechanisms; do not copy a credential into this document. Cloud secrets are available during setup only under the documented environment behavior.

The checked-in setup uses CPU PyTorch on Linux. Treat accelerator availability as something to inspect, not assume. A cloud coding task is a bounded task that returns a diff, not a permanent GPU training server or an automatically recurring research loop.

## External artifacts and reading

- [W&B group](https://wandb.ai/data2yihein-d/tinyworlds/groups/original-PICODOOM-20260903_234048): VT `48l9ulop`, LAM `4d5anb7o`, dynamics `b9lhcat1`.
- [Checkpoint release](https://github.com/yiheinchai/surgical-worlds/releases/tag/original-picodoom-20260904), archive `original-picodoom-checkpoints-20260904.tar.gz`, approximately 52 MB. SHA-256: `81e61101aaa03e15aca58f762a08f338f856e2f95ab01a4013e87ef7465bcbc7`.
- [PicoDoom HDF5](https://huggingface.co/datasets/AlmondGod/TinyWorlds/blob/main/picodoom_frames.h5): 59,785 RGB 64×64 frames, approximately 662 MB. Source frame sampling for the audit is recorded in `probe_results.json`.
- [Genie paper](https://arxiv.org/html/2402.15391v1): raw-video LAM; discrete VQ codes; history-only forward decoder; stopped action gradients from dynamics; additive action embeddings. Start with this reference to keep the user's goal intact.
- [Oniris](https://francesco215.github.io/autoregressive_diffusion/) and [Open-Dreamer](https://next-state.github.io/open-dreamer/): useful lessons on objective and generation mismatch, not established cures for this LAM collapse.
- [Official Codex Cloud environments](https://learn.chatgpt.com/docs/environments/cloud-environment).
