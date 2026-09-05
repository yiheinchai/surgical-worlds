# Surgical world models

## Research requirement

The user wants **unsupervised, video-only latent-action discovery like DeepMind Genie**. Do not replace this goal with supervised robot actions, kinematics, action labels, or text conditioning. Test fidelity, next-frame prediction, and control consistency separately.

Read `docs/CODEX_CLOUD_HANDOFF.md` before continuing this project. The evidence record is `docs/diagnostics/picodoom-2026-09-05/README.md`; it corrects misleading claims in the older PicoDoom handoff. Follow the current user's steering when it differs from either document.

## Environment

- Cloud setup: `bash scripts/codex_cloud_setup.sh` (Python 3.12 recommended; CPU PyTorch).
- Use `.venv/bin/python`; shell activation from setup does not persist between phases.
- Smoke check: `WANDB_MODE=disabled MPLBACKEND=Agg .venv/bin/python scripts/codex_cloud_smoke.py`.
- Tests added during the continuation: `.venv/bin/python -m pytest -q`.
- Keep CPU diagnostics small. Existing pipeline defaults are large CUDA training jobs; do not run `full_train.py` with defaults as an environment check.
- Store numerical evidence and conclusions in the repository. Large data and checkpoint downloads belong in ignored `data/` or `results/` directories.
- Never put credentials in code, prompts, logs, commits, or diagnostic artifacts. Existing sanitized W&B exports allow work without a live credential.

## Quota awareness

Before work exceeding roughly 10 tool calls or significant code generation, read `~/.codex/usage-status.md` if available. If a fresh snapshot shows 7d usage above 95%, report it and ask the user whether to continue or defer. If missing or several hours old, state that current usage is unknown and continue within the authorized task. `5h=?` is normal for this plan.

## Notion

Do not automatically capture work in Notion. Use Notion only when the user explicitly requests a Notion search, page, or update.
