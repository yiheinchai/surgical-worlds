#!/usr/bin/env bash
# Poll monitor_training.sh until training starts, fails, or timeout.
# Called automatically by autonomous_train.sh after launch.
#
# Usage: bash scripts/wait_for_health.sh <INSTANCE_ID> [TIMEOUT_MINUTES]
set -euo pipefail

INSTANCE_ID="${1:?Instance ID required}"
TIMEOUT_MIN="${2:-15}"
INTERVAL_SEC=30
DEADLINE=$(($(date +%s) + TIMEOUT_MIN * 60))

echo "Waiting up to ${TIMEOUT_MIN}m for training to start or fail (every ${INTERVAL_SEC}s)..."
echo ""

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  echo "--- $(date -Iseconds) ---"
  if bash "$(dirname "$0")/monitor_training.sh" "$INSTANCE_ID"; then
    echo ""
    echo "Health check passed — training appears to be running or complete."
    exit 0
  fi
  RC=$?
  if [ "$RC" -eq 1 ]; then
    echo ""
    echo "Health check FAILED — do not wait on this instance. Fix and relaunch."
    exit 1
  fi
  echo ""
  sleep "$INTERVAL_SEC"
done

echo ""
echo "Timeout after ${TIMEOUT_MIN}m — still in SETUP. Training may still be downloading data."
echo "Keep monitoring: bash scripts/monitor_training.sh $INSTANCE_ID"
exit 2
