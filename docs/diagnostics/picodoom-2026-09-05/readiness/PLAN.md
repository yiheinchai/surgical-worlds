# Readiness investigation (continuation)

The user explicitly requested continued iteration until a larger video-only run
is justified for playable PicoDoom. The previous bounded campaign established
failure modes, not readiness. The remaining authorized credit was $7.4224284.
No top-up is authorized. The active instance has a one-occurrence provider cleanup
deadline; each training invocation also has a wall-clock and step limit.

## Working hypothesis and first experiment

The old LAM conflates scene content with motion. A global entropy regularizer
balanced codes without making them useful. The old token generator also discards
visual detail and becomes unstable in open-loop generation. Diagnose and repair
these separately:

1. Fit a small motion vocabulary from classical RGB optical flow. This uses no
   learned optical-flow weights, engine actions, camera poses, labels or text.
   K-means is fitted exclusively to training transitions. This changes the LAM
   mechanism deliberately; it is an unsupervised motion bottleneck baseline,
   not a faithful reproduction of Genie's jointly learned VQ LAM. Flow can encode
   exogenous events and does not identify player intent by itself.
2. Re-run the identical four-branch translation challenge on held-out PicoDoom
   textures with three clustering seeds. Transformation IDs are used only to
   score a train-selected permutation after fitting, never for action learning.
3. Train a small raw-pixel conditional EDM from scratch using these codes. Avoid
   dependence on the existing lossy tokenizer. Compare history corruption,
   action ablations, direct code interventions, and open-loop rollouts. Every
   frame in a rollout must be generated from previous generated frames and the
   supplied latent controls, with no ground-truth frame refresh.
4. Iterate from measured failures, including alternative action representations,
   generative training objectives, longer memory or improved video coverage as
   needed. Do not declare readiness just because this first pilot improves loss.

## Evidence requirements

- Motion consistency on unseen source textures, per-code usage and entropy,
  no-motion pairs, appearance perturbations, and across-scene interventions.
- Fully hidden next-frame generation; paired inferred/shuffled/constant controls
  with shared sampler noise; copy-frame baseline, motion-weighted error, and
  per-sample records for uncertainty estimates.
- Direct player-code rollouts with repeated and switched controls, and visual
  checks for scene geometry, weapon, enemies, hallucinated resets and drift.
  Future-video-inferred control rollouts must be called oracle-control rollouts.
- Record actual update throughput, GPU memory, sampler latency and account spend.
- W&B: loss, gradient and parameter histograms, code frequencies, intervention
  grids, one-step predictions and rollout media; local JSONL and checkpoints.
- The former temporal validation and final-test ranges have already been used.
  They are development evidence now, not a fresh unbiased final test. Full-run
  readiness needs independently held-out episodes or explicit data limitations.

## Reading the supplied repositories

| Reference | What is useful here | What it does not establish for this project |
|---|---|---|
| [Oasis](https://github.com/etched-ai/open-oasis), [sampler](https://github.com/etched-ai/open-oasis/blob/master/generate.py) | Explicit history noise-level conditioning during autoregressive diffusion; sliding context; continuous image latents. | Released 500M weights and inference code use action streams. No latent-action discovery training recipe is provided. |
| [MineWorld](https://github.com/microsoft/mineworld) | Benchmarks for action following, visual-action autoregression, diagonal decoding as a speed option. | Requires action inputs; Minecraft weights are not a PicoDoom control solution. README currently says checkpoints were taken down in May 2025. |
| [Open-Dreamer](https://github.com/next-state/open-dreamer) | Full tokenizer/dynamics training pipeline, causal tokenization, latent statistics and episode handling. | Its data records include actions; replacing them with useful RGB-inferred codes remains our responsibility. |
| [LingBot-World](https://github.com/robbyant/lingbot-world) | Memory/consistency design and separation of camera-control interfaces from generation. | Supplied models use camera poses/actions and text. The released setup is far larger than this exploratory budget. |
| [LingBot v2](https://github.com/robbyant/lingbot-world-v2) | Causal streaming, KV caching, local attention plus persistent context. | Released 14B inference is not a small-budget video-only latent-action training recipe. README's long-horizon claims need independent evaluation on our task. |

Additional primary references: [Genie](https://arxiv.org/abs/2402.15391) for
unsupervised discrete controls; [DiLA](https://arxiv.org/abs/2605.15725) for separating
motion-relevant structure from content; [DIAMOND](https://github.com/eloialonso/diamond)
for pixel-space diffusion and training on model-produced contexts;
[GameNGen](https://arxiv.org/abs/2408.14837) for noisy-history stabilization;
[EDM](https://arxiv.org/abs/2206.00364) for preconditioning and sampling. Our pilot is
locally implemented, not a reproduction of any of these systems. The supervised
action requirements of DIAMOND/GameNGen are replaced by RGB-derived motion codes.
