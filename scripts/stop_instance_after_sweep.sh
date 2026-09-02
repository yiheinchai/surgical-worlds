#!/usr/bin/env bash
# Poll Vast instance until inference sweep finishes, then stop (not destroy).
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
INSTANCE_ID="${1:-49067833}"
POLL_SEC="${POLL_SEC:-180}"

if [ -z "${VASTAI_API_KEY:-}" ] && [ -f "$HOME/.config/vastai/vast_api_key" ]; then
  export VASTAI_API_KEY="$(cat "$HOME/.config/vastai/vast_api_key)"
fi
[ -n "${VASTAI_API_KEY:-}" ] || { echo "Set VASTAI_API_KEY"; exit 1; }

vastai set api-key "$VASTAI_API_KEY" >/dev/null
SSH_INFO=$(vastai ssh-url "$INSTANCE_ID" 2>&1)
HOST=$(echo "$SSH_INFO" | sed 's|ssh://root@||;s|:.*||')
PORT=$(echo "$SSH_INFO" | sed 's|.*:||')
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

echo "[$(date -Iseconds)] Watching instance $INSTANCE_ID for inference sweep completion..."

while true; do
  if ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p "$PORT" root@"$HOST" \
      'test -f /workspace/inference_sweep_complete.txt || grep -q "Inference sweep complete" /workspace/inference_sweep.log 2>/dev/null' 2>/dev/null; then
    echo "[$(date -Iseconds)] Sweep complete — stopping instance $INSTANCE_ID (preserve disk)"
    vastai stop instance "$INSTANCE_ID"
    echo "[$(date -Iseconds)] Stop requested. Instance disk/data retained."
    exit 0
  fi

  STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('actual_status','?'))" 2>/dev/null || echo "?")
  if [ "$STATUS" = "stopped" ] || [ "$STATUS" = "exited" ]; then
    echo "[$(date -Iseconds)] Instance already $STATUS"
    exit 0
  fi

  PROGRESS=$(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p "$PORT" root@"$HOST" \
    'grep -oP "\d+/30000" /workspace/crcd_crisp_train.log 2>/dev/null | tail -1; tail -1 /workspace/inference_sweep.log 2>/dev/null' 2>/dev/null || echo "unreachable")
  echo "[$(date -Iseconds)] Waiting... train/sweep: $PROGRESS"
  sleep "$POLL_SEC"
done
