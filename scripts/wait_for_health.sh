#!/usr/bin/env bash
# Poll monitor_training.sh until training starts, fails, or timeout.
set -euo pipefail
INSTANCE_ID="${1:?Instance ID required}"
TIMEOUT_MIN="${2:-15}"
INTERVAL_SEC=30
DEADLINE=$(($(date +%s) + TIMEOUT_MIN * 60))
echo "Waiting up to ${TIMEOUT_MIN}m for training to start or fail..."
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if bash "$(dirname "$0")/monitor_training.sh" "$INSTANCE_ID"; then exit 0; fi
  [ $? -eq 1 ] && exit 1
  sleep "$INTERVAL_SEC"
done
echo "Timeout after ${TIMEOUT_MIN}m"; exit 2
