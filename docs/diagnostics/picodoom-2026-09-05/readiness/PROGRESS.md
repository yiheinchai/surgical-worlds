# Readiness progress — investigation remains active

The previous campaign diagnosed failure; it did not justify a full playable-game
training run. This continuation has improved one-step generation and identified a
useful error-recovery mechanism, but direct long-horizon control remains inadequate.
Do not describe these results as a playable model or a completed readiness gate.

## Completed comparisons

All training consumes RGB only. The new action bottleneck uses classical Farneback
flow followed by training-only K-means. It is a deliberate alternative to the failed
neural LAM, not a faithful reproduction of Genie's learned VQ encoder. The optional
warped-image prior explicitly encodes the RGB-derived motion centroid, so strong
one-step motion-code agreement is partly imposed by the model construction.

On a shared 256-clip reused development sample with Euler-3 and shared sampler noise:

| 5k-update variant | Inferred L1 | Shuffled L1 | Modal constant L1 | Motion-weighted L1 |
|---|---:|---:|---:|---:|
| Frame EDM, no offset | 0.215429 | 0.217950 | 0.211071 | 0.248057 |
| Frame EDM, offset 0.3 | 0.112663 | 0.114977 | 0.110569 | 0.181751 |
| Warp-residual EDM, offset 0.3 | 0.099221 | 0.113021 | 0.105293 | 0.170222 |
| Warp-residual EDM, no offset | 0.098486 | 0.112491 | 0.105329 | 0.170203 |
| Copy previous frame | 0.111139 | — | — | 0.248850 |
| Motion-centroid warp alone | 0.102142 | — | — | 0.189306 |

For the last model, paired shuffled-minus-inferred L1 is 0.014005, with a
block-bootstrap 95% interval [0.011203, 0.016911]. Blocks of eight nearby clips do
not remove all dependence from one video. These are development results, not an
independent final test. The constant control here is train-modal code 6; older
64-clip pilot grids used code 0. Cohorts and samplers differ from some earlier
pilot logs; compare the matched table rather than mixing their means.

The motion codebook transfers four synthetic translation branches to unseen
textures at 100% accuracy for three clustering seeds, versus chance performance
of the old LAM. This establishes motion information, not intent or game mechanics.
An additional public upstream 2.15M-parameter LAM checkpoint also failed the
branching test (28.125%) and did not beat copying on ordinary real-video L1.

## Error accumulation

Starting both runs from the same 5k checkpoint, train 10k further updates either
with clean history or with one/two generated history frames on half the batches.
At 128 generated frames, the latter retains more visible texture. For oracle
future-RGB controls, mean edge energy over the final 16 frames rises from 0.022145
to 0.028910. Target L1 at the final frame is slightly worse, 0.225460 vs 0.218251.
Sharper images are not necessarily more faithful geometry.

Direct repeated turn-like codes still fail: generated-motion code agreement over
128 frames is only 1.6%/5.9% for codes 1/5 with generated-history training. Near-static
code 6 is easy (99.4%). This is a central remaining readiness failure. Oracle-control
rollouts use future-video-inferred control IDs but never refresh history with future
RGB. All direct fixed/switch-code rollouts use only the initial prompt and integer
controls. See the saved per-frame reports and videos before interpreting metrics.

## Ongoing targeted work

- Matched wider-model run to test whether capacity improves long-horizon control.
- DIAMOND-inspired lower-frequency noise-embedding ablation; changing the embedding
  alone did not remove drift. Legacy checkpoint behavior stays selectable.
- Fresh rendered POOM episodes with capture timestamps and no stored engine state
  or action labels. Original TinyWorlds source-game/version provenance is missing,
  so treat source equivalence as unverified and report this domain limitation.
- Foreground residual-event vocabulary probe, trained without semantic labels, to
  investigate firing/interaction information missing from flow-only navigation.

## Reproducibility and observability

[W&B group](https://wandb.ai/data2yihein-d/tinyworlds/groups/picodoom-readiness-20260905).
[Matched component comparison](https://wandb.ai/data2yihein-d/tinyworlds/runs/n1l70p4b).
[Generated-history rollouts](https://wandb.ai/data2yihein-d/tinyworlds/runs/vma2dr3s).

Training saves undivided interval loss, gradient norms, update throughput, peak
CUDA memory, parameter/gradient histograms, checkpoints, source hashes, one-step
prediction grids, direct intervention grids and per-clip metrics. Rollouts save
GIFs, contact sheets and per-frame metrics. Exact CPU checkpoint-resume reproduction
is tested, including RNG state. All 25 CPU tests passed after adding the Fourier
ablation and testing complete generated-history replacement. Two recovered 15k checkpoints passed strict weights-only local loading.

The 261,507,784-byte checkpoint snapshot contains 215 files, individually SHA256
verified after extraction; its archive hash and scope are recorded alongside this
report. Numerical/provenance evidence is copied into `evidence/`. Later runs require
an additional snapshot. No top-up or recurring job was created. The active paid
instance retains a single-occurrence provider cleanup deadline.


### Fresh episode development probe

The width-64 model (15k updates, last 10k with generated history) obtains inferred
L1 0.134445, shuffled 0.161441, modal constant 0.147888, and copy 0.142680 on 128
clips from fresh episode 0 at stride 2. Warping alone has better ordinary L1
(0.129764), while learned generation has better motion-weighted L1 (0.282016 vs
0.309810). This is mixed evidence, not a universal improvement. No fresh episodes
were used for training; episodes 1 and 2 are still reserved. The capture smoke
measured a median 33.18 ms interval and produced visible gameplay/muzzle animation.
