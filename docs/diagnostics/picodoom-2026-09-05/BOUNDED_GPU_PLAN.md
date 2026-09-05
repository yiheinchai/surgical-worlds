# Bounded GPU investigation — 5 September 2026

The user authorized small Vast experiments within the existing $8.66 credit and
requested live W&B observability. This does not authorize replenishing credit.
The first tranche targets at most $2 including setup and transfers. A rented
instance has a provider-side deletion deadline and will be deleted sooner after
results are retrieved. Credentials stay outside the repository and artifacts.

## Provenance

The preceding cloud task was recovered using `codex cloud apply` from task
`task_e_6a9b62b2575c8330b8efe888874f8e93`. Its reported commit `cf2ad11` was not
published on GitHub; the recovered diff is retained as a separate local commit.
The starting repository is `c22a6ea`. All earlier checkpoints remain unchanged.

## Common controls

- RGB video only. Four anonymous binary-FSQ codes; no labelled controls,
  kinematics, rewards, or text supervision.
- 64x64 input, patch 4, width 32. Existing VT@37500 remains frozen.
- Two-frame LAM decoder sees only frame 1 and its inferred code.
- Source ranges: training [300,46000), validation [47000,53000), final test
  [54000,59785). Clips within a partition are nonoverlapping. The source gap
  exceeds the context length. These are temporal partitions, not verified
  independent episodes. The prior training prefix is confined to training.
- LAM and dynamics configurations and per-run frame starts are saved exactly.
- AdamW with explicit learning rate, weight decay .01, clip norm 1, batch 16,
  no gradient accumulation, float32/TF32. No compile or optimizer changes within
  paired comparisons. Resuming weights resets optimizer state and is recorded.
- Fixed optimizer-step budgets, validation every 500 steps, logs every 50.
  **No validation-based early stopping.** Wall-time caps are interruption limits;
  an interrupted run must not be described as completing its step budget.
- Validation guides iteration. Final test is opt-in and reserved for the
  selected recipe and its predeclared reference comparison.

## Initial comparisons and adaptive follow-ups

1. Benchmark 100 optimizer steps and verify live metrics and artifact upload.
2. Two-frame LAM: 1,000 steps from the old LAM checkpoint. Measure inferred vs
   shuffled and modal-constant codes, copy baseline, and per-target exhaustive
   best-of-four code. The latter uses the target as an explicitly labelled
   diagnostic oracle, not a deployable controller.
3. Dynamics: compare 1,000 steps of legacy masking and fully hidden next-frame
   masking from the same dynamics@44000 weights, frozen old LAM, identical
   batch sampling and optimizer settings. No LAM change is combined with this.
4. Choose follow-ups from the measured failure: encoder vs decoder gap, discrete
   collapse vs nuisance coding, optimization vs data scarcity, or rollout error.
   A soft discrete-information regularizer is available as an isolated ablation,
   with weight zero in the reference. It is not an established remedy.
5. Repeat the useful comparison with another seed before making a recommendation.
   Integrate the LAM with dynamics only after isolated results justify it.

## Observability and interpretation

W&B group: `picodoom-bounded-20260905`, entity `data2yihein-d`, project `tinyworlds`.
Every run logs undivided loss, each discrete code frequency, entropy in nats,
continuous variance and saturation, gradient norms and histograms, weights,
throughput, peak VRAM, system telemetry, train/validation metrics, comparison
grids, intervention grids, and optional 16-step rollout videos. JSONL metrics,
configurations, exact splits, best/last weights, images, and final status are
also saved locally and uploaded as W&B artifacts.

Paired gaps are positive when inferred actions beat their baseline. Report
per-clip mean and descriptive standard error; nearby clips are correlated, so
these are not formal independent-sample confidence intervals. Discrete entropy
does not establish action semantics. The temporal tokenizer is decoded with
its causal history in the new runner; the prior helper's standalone-frame decode
must be assessed separately before comparing numerical pixel metrics.

Free rollouts use ground-truth-video-inferred actions as oracle diagnostic
inputs, without feeding target pixels/tokens to dynamics. They are explicitly
distinguished from anonymous-code interventions and human play. Success means
useful prediction and repeatable plausible interventions on held-out video,
not merely lower training loss or different outputs. A bigger run requires a
documented go/no-go decision based on those measurements.

## Follow-ups fixed before final-test evaluation

The initial 1,000-step LAM comparison was repeated with seed 1. The information
penalty increased entropy and the shuffled-action gap but slightly worsened
inferred prediction in both seeds. A video-derived semantics probe found code
interventions dominated by global color changes. Therefore:

- Extend the seed-0 LAM pair and next-frame dynamics to 5,000 steps with the same
  learning rate and initial weights. Evaluate every 1,000 steps, no early stop.
- Run a separate synthetic learnability pair: 64 training and 32 validation RGB
  textures, four wraparound translations per identical first frame, 3,000 steps,
  learning rate .001 in both variants. No transformation labels enter training.
  This is deliberately artificial and does not establish PicoDoom control.
- Compare original, next-frame@1k, and next-frame@5k with one-pass and existing
  16-iteration deterministic MaskGIT generation. Compare cached latent history
  with history re-encoded from rendered RGB. Four fixed validation starts, 64
  steps each. Timing uses batch 1 and excludes user-interface/network latency.
- Repeat the 1,000-step dynamics objective comparison with seed 1.
- Final test is fixed to LAM reference@5k versus information@5k, and original
  dynamics versus next-frame@5k, all **last** weights. No test-based selection or
  further recipe tuning. These are evaluation-only runs with zero optimizer
  steps. The test is temporal, not verified episode-disjoint.
