#!/usr/bin/env bash
# Pull checkpoint archives from a running Vast.ai instance to the agent VM.
# Survives instance destruction — run hourly while training is active.
#
# Usage:
#   export VASTAI_API_KEY=...
#   bash scripts/agent_pull_checkpoints.sh [INSTANCE_ID]
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INSTANCE_ID="${1:-}"

if [ -z "${VASTAI_API_KEY:-}" ] && [ -f "$HOME/.config/vastai/vast_api_key" ]; then
  export VASTAI_API_KEY="$(cat "$HOME/.config/vastai/vast_api_key")"
fi
[ -n "${VASTAI_API_KEY:-}" ] || { echo "Set VASTAI_API_KEY"; exit 1; }

if [ -z "$INSTANCE_ID" ] && [ -f /tmp/surgical_worlds_launch.json ]; then
  INSTANCE_ID=$(python3 -c "import json; print(json.load(open('/tmp/surgical_worlds_launch.json')).get('instance_id',''))")
fi
INSTANCE_ID="${INSTANCE_ID:-49067833}"

BACKUP_ROOT="${CKPT_BACKUP_ROOT:-/agent/checkpoint-backups}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
DEST="$BACKUP_ROOT/$INSTANCE_ID/$STAMP"
mkdir -p "$DEST"

SSH_INFO=$(vastai ssh-url "$INSTANCE_ID" 2>&1)
HOST=$(echo "$SSH_INFO" | sed 's|ssh://root@||;s|:.*||')
PORT=$(echo "$SSH_INFO" | sed 's|.*:||')
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

echo "Pulling checkpoints from instance $INSTANCE_ID → $DEST"

ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=30 -p "$PORT" root@"$HOST" \
  'cd /workspace/surgical-worlds 2>/dev/null || cd /workspace; tar czf - results 2>/dev/null || tar czf - surgical-worlds/results 2>/dev/null' \
  | tar xzf - -C "$DEST"

ln -sfn "$DEST" "$BACKUP_ROOT/$INSTANCE_ID/latest"

echo "Backup complete: $DEST"
du -sh "$DEST"
