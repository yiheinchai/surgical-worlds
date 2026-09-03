#!/usr/bin/env bash
# Pull inference sweep archive from Vast after post-training batch completes.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
INSTANCE_ID="${1:-49067833}"
DEST="${2:-/agent/tinyworlds/docs/inference_sweep}"

if [ -z "${VASTAI_API_KEY:-}" ] && [ -f "$HOME/.config/vastai/vast_api_key" ]; then
  export VASTAI_API_KEY="$(cat "$HOME/.config/vastai/vast_api_key")"
fi

SSH_INFO=$(vastai ssh-url "$INSTANCE_ID" 2>&1)
HOST=$(echo "$SSH_INFO" | sed 's|ssh://root@||;s|:.*||')
PORT=$(echo "$SSH_INFO" | sed 's|.*:||')
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

mkdir -p "$DEST"

# Prefer live directory; fall back to tarball
if ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -p "$PORT" root@"$HOST" \
    'test -d /workspace/surgical-worlds/docs/inference_sweep'; then
  echo "Pulling inference_sweep directory..."
  rsync -az -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no -p $PORT" \
    root@"$HOST":/workspace/surgical-worlds/docs/inference_sweep/ "$DEST/"
elif ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -p "$PORT" root@"$HOST" \
    'test -f /workspace/inference_sweep.tar.gz'; then
  echo "Pulling inference_sweep.tar.gz..."
  scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -P "$PORT" \
    root@"$HOST":/workspace/inference_sweep.tar.gz /tmp/inference_sweep.tar.gz
  tar xzf /tmp/inference_sweep.tar.gz -C "$(dirname "$DEST")"
else
  echo "No inference sweep found on instance yet"
  exit 1
fi

echo "Pulled to $DEST"
find "$DEST" -name '*.mp4' | wc -l | xargs echo "MP4 count:"
