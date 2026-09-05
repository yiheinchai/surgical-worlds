# Samplers, generated histories, and sparse events

**Stopped at the user's request. See [the final handoff](USER_STOP_HANDOFF.md).**

This is a development investigation, not a claim of playable PicoDoom or approval
of a full run. All training still uses RGB and train-fitted unsupervised controls.

## Sampler behavior matters separately from loss

For the same width-64 warp-residual checkpoint at 20k updates, a 32-clip one-step
sample and four-scene, 64-frame direct turn probe gave:

| Solver | History stabilization | One-step L1 | Turn-code agreement | Final edge energy |
|---|---:|---:|---:|---:|
| Euler 3 | 0 | 0.09860 | 0.04297 | 0.02223 |
| Euler 3 | 0.1 | 0.10057 | 0.03516 | 0.02534 |
| Heun 3 | 0 | 0.25033 | 0.67969 | 0.51966 |
| Heun 3 | 0.1 | 0.25452 | 0.66406 | 0.53159 |
| Euler 8 | 0 | 0.10705 | 0.60156 | 0.03731 |
| Euler 8 | 0.1 | 0.10873 | 0.66016 | 0.04196 |
| Heun 8 | 0 | 0.12252 | 0.87109 | 0.06450 |
| Heun 8 | 0.1 | 0.12334 | 0.89453 | 0.06859 |

Heun 3 produced noisy images: its high motion agreement is not a success. Heun 8
retained textured surfaces and directional changes longer, but room geometry
still deformed and the weapon could disappear. The motion-centroid prior itself
imposes motion, so agreement does not establish correct game mechanics. Runs
overlapped other GPU work; these observations do not establish inference latency.
[W&B sampler comparison](https://wandb.ai/data2yihein-d/tinyworlds/runs/qedxv1ps).

## Matching the training sampler did not help

Both branches started from the same width-64 checkpoint at 20k, trained another
5k updates with generated histories on half the batches, up to eight generated
frames. One used Euler 3 histories; the other Heun 8 with stabilization 0.1. Both
were evaluated using Heun 8 / 0.1 for 128 generated frames.

| Generated training history | Oracle motion agreement | Final oracle L1 | Direct code 1 | Direct code 5 | Switched codes |
|---|---:|---:|---:|---:|---:|
| Euler 3 | 0.92578 | 0.33729 | 0.89648 | 0.88281 | 0.86328 |
| Heun 8 / 0.1 | 0.86133 | 0.35444 | 0.82031 | 0.81836 | 0.80078 |

The more expensive history generator did not improve this experiment. The Euler
branch is therefore the continuation baseline. Neither branch preserved reliable
room geometry. Wider capacity and longer generated histories improved some
metrics earlier, but are not sufficient evidence for full-run readiness.

## Sparse foreground events: information exists, first neural pilot fails

After removing the train-fitted motion warp, cluster residual RGB in a fixed
foreground region (rows 32:58, columns 20:44), using train-only PCA and k-means.
Four prototypes include a common small residual, a positive flash-like change,
and negative/recovery-like changes. These are observed effects, not identified
player intentions. Pack eight motion codes and four event codes with factorized
conditioning. The motion warm start is initially unchanged for every event ID;
optimizer moments were explicitly reset in this experiment.

After 2k further updates (27k total), on 256 development clips, inferred foreground
L1 was 0.141192 and event-only-shuffled L1 was 0.141035: a gap of -0.000158. The
event signal was not useful to the model. Its event agreement of 87.5% mainly
reflects the common event, so evaluation now includes balanced recall, nonmodal
recall, the modal baseline and the full confusion matrix.

A separate CPU check found that adding train-fitted event prototypes to the motion
warp improved foreground MSE from 0.052611 to 0.045268 with correct codes, while
shuffled prototypes gave 0.054303. This motivates a bounded learned-prior pilot,
not a firing claim. Its prior adds each event prototype relative to the modal
prototype only inside the tapered foreground region. Any resulting flash is
partly imposed by construction and must be compared with that prior alone.

## Current controlled follow-ups

- Continue the motion baseline from 25k to 50k to measure the effect of training
  duration; inspect 256-frame direct and oracle rollouts.
- Test the event prior for 2k updates, with event-only ablations and diagnostic
  pulse schedules. Pulse schedules are manually chosen latent-code probes.
- Starting from the same 25k motion checkpoint, add one zero-output residual
  spatial attention block at the 16x16 bottleneck and train to 35k. Compare against
  the convolution-only checkpoint at 35k. Existing weights, EMA, optimizer
  moments and saved RNG state are retained; only new attention parameters start
  with empty optimizer state. This tests a specific missing global spatial
  operation, inspired by DIAMOND, not a new architecture sweep.

- Measure warm batch-one inference latency with no other listed GPU compute process.
  Timing excludes UI, network and logging; compare Euler 3, Euler 8 and Heun 8.
- From the same 25k motion checkpoint, test log-sigma location -0.4 and maximum
  sigma 20 (DIAMOND defaults), versus the pilot's -1.2 / 5. The sampler remains
  unchanged. Train 10k updates and compare with the 35k convolution baseline.
  This isolates training-noise coverage; direct rollout quality remains decisive.
- Compare the candidate models on fresh episode 0 with the same Heun-8 sampler.
  Episodes 1 and 2 remain reserved until development choices are frozen.

## Additional references supplied through the research index

The user supplied [nik-55/world-models](https://github.com/nik-55/world-models), a
research index. Relevant primary sources were checked on September 5, 2026:

- [SWIRL](https://arxiv.org/pdf/2602.06130) alternates forward and inverse models
  using identifiability and likelihood objectives. Its visual setting uses
  language actions and a pretrained 7B VLM with supervised warm-up; it is not a
  drop-in RGB-only latent-action solution. The mutual-information idea is useful
  inspiration, but could reward visual shortcuts if used without fidelity tests.
- [minWM](https://github.com/shengshu-ai/minWM) exposes staged causal distillation
  and self-rollout training. Its provided data/control paths use camera poses and
  captions, with 1.3B/8B backbones; adopting them directly would change our setup.
- [PERSIST](https://francelico.github.io/persist.github.io/) directly addresses
  spatial memory through latent 3D state. Its authors explicitly identify 3D
  training annotations as a limitation. This is evidence for testing persistent
  geometry, not evidence that its current recipe satisfies our video-only rule.
- [DIAMOND's blocks](https://github.com/eloialonso/diamond/blob/main/src/models/blocks.py)
  use spatial attention in the middle block. Our attention test isolates that
  component with a neutral warm start; it is not a complete DIAMOND reproduction.

## Durability and validation

30 local CPU tests passed in 202.15 seconds on the loaded local host. The newest local snapshot contains 433
individually SHA256-verified files (66,844,010-byte archive); see
`current_evidence_archive.json`. Numerical records are in `evidence/`.
W&B artifact `picodoom-readiness-checkpoints-v2:v0` is COMMITTED, with 282 selected
files totaling 299,036,357 bytes, including five selected EMA checkpoints and
three fresh RGB episodes. Episode 0 has been used for development; episodes 1 and
2 remain uninspected holdouts. Later active experiments require further snapshots.
