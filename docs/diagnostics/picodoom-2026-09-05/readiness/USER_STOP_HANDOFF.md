# Stopped at the user's request

The user requested: "ok can you stop and wrap up the worknow" on September 5,
2026. All experiments and dependent queues were cancelled. Do not resume training,
collection, benchmarks or paid compute without a new user request. The earlier
readiness objective was not achieved; this is an honest stopping record.

## What the experiments established

The old neural latent-action model could obtain a balanced code histogram while
encoding appearance instead of useful controls. A train-fitted RGB optical-flow
vocabulary transfers four synthetic translation branches to unseen textures at
100% accuracy across three clustering seeds. It is a useful unsupervised motion
baseline, not a faithful reproduction of Genie's neural latent-action model, and
does not identify player intent by itself.

On the matched 256-clip development comparison at 5k updates, warp-residual pixel
diffusion achieved L1 0.098486, versus 0.112491 with shuffled controls, 0.105329
with the modal constant control, 0.111139 for copying and 0.102142 for warping
alone. The paired shuffle gap was 0.014005, with a block-bootstrap interval
[0.011203, 0.016911]. Reused samples from one video do not constitute an
independent final test.

Eight-step residual sampling retained substantially more motion and texture in
rollouts than three-step sampling. Increasing model width and training with
generated histories helped some measures; matching the expensive Heun sampler
during generated-history training did not help. Longer training alone also
plateaued on one-step development error. At 50k updates, the 256-frame rollout
retained recognizable walls and weapons, but room geometry changed. Oracle
motion agreement was 0.876953, with final-frame L1 0.458306. Direct repeated
turn-code agreement was 0.818359/0.843750 for codes 1/5. Motion agreement is partly
imposed by the motion prior and must not be mistaken for correct game mechanics.

The first factorized event pilot ignored its event code: event-only shuffling
changed foreground L1 by -0.000158. Adding train-fitted foreground residual
prototypes gave a positive shuffle gap of 0.006691 and balanced event recall
0.883964. However, the exact prior alone achieved perfect event-code agreement
and lower foreground L1 (0.121356 versus the neural model's 0.137809). The neural
model therefore did not establish learned firing mechanics; visible flash-like
effects are partly supplied by construction.

Fresh rendered POOM episode 0 showed a positive motion shuffle gap, but learned
generation did not beat the motion warp on ordinary L1. Fresh episodes 1 and 2
remain uninspected holdouts. The original recording's game-version and frame-rate
provenance is incomplete, so transfer results have an explicit domain limitation.

## Work interrupted or never run

- Spatial-attention continuation stopped at **32,596 updates**, before the
  planned matched 35k comparison and 256-frame rollout evaluation. Its full
  optimizer checkpoint and EMA were saved. Do not claim an attention result.
- The broader training-noise comparison, isolated inference benchmarks and new
  matched fresh-episode comparisons were queued but **did not run**.
- The evaluation-only keyboard challenge saved three of four planned episodes;
  its frozen-code scoring **did not run**. Button labels are evaluation-only,
  stored separately, and the training loader rejects evaluation-only captures.
- The interrupted trainer initially wrote `finished` because KeyboardInterrupt
  bypassed its exception handler. The saved status was corrected to `user_stopped`
  with the original value preserved, and the handler was repaired for future use.
  Saved RNG state at interruption is not a claim of bit-exact continuation through
  an incomplete microstep.

## Compute evidence

The 8,093,315-parameter width-64 model completed 25,000 additional updates in
2,518.17 seconds at batch 32, including periodic evaluation: about 9.93 updates
per second. At the rented instance rate of $0.371111/hour including its disk, that
segment cost approximately $0.26 in instance time. At unchanged throughput,
100,000 updates would take roughly 2.8 hours and $1.04, excluding setup, extra
evaluation, transfers and other models. This is an extrapolation for this small
pilot, **not a cost estimate or recommendation for achieving playable Doom**.
No isolated inference-speed result was obtained before the stop request.

## Validation and recovery

The full local CPU suite passed 30 tests before the final evaluation-only capture
guard and diagnostic scorer were added. This covers target isolation, disjoint
splits, undivided losses, deterministic ablations, RNG resume, generated histories,
event warm starts, localized priors, identity attention migration and finite
high-noise gradients. Additional targeted checks and cleanup verification are
recorded in the final delivery manifest. The 215-file, 433-file and 36-file local
snapshots were individually SHA256 verified.

W&B artifacts are committed:

- `picodoom-readiness-checkpoints-v2:v0`: 282 selected files, 299,036,357 bytes;
  selected EMA checkpoints and all three fresh RGB episodes.
- `picodoom-readiness-checkpoints-v3:v0`: 104 selected files, 302,810,108 bytes;
  the completed 50k motion and 27k event pilots, including checkpoints.
- `picodoom-readiness-user-stop:v0`: 15 selected files, 168,393,739 bytes;
  the partial attention checkpoint and exact event-prior comparison.

[W&B experiment group](https://wandb.ai/data2yihein-d/tinyworlds/groups/picodoom-readiness-20260905)
and [draft PR](https://github.com/yiheinchai/surgical-worlds/pull/3).

The reviewable branch preserves the legacy tokenizer/LAM/MaskGIT objectives and
all negative results. No supervised engine actions, kinematics or text entered
training. No credit top-up or recurring job was created. Vast instance 49967727 was destroyed and verified absent at 20:15:59 UTC; the
single-occurrence cleanup job 714 was deleted. The account reported $5.148592
remaining, a $3.515259 decrease from the initial $8.663851. See `cleanup.json`.

## Remaining scientific decision

A full playable run is not yet justified. Before authorizing one, a resumed
investigation would need to finish a controlled geometry comparison, establish
action consistency on the evaluation-only challenge, demonstrate event effects
beyond their explicit prior, and then freeze choices before using fresh holdouts.
Those are outstanding requirements, not jobs left running.
