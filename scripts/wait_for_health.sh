#!/usr/bin/env bash
# Poll monitor_training.sh until training starts, fails, or timeout.
set -euo pipefail
INSTANCE_ID="${1:?Instance ID required}"
TIMEOUT_MIN="${2:-15}"
DEADLINE=$(($(date +%s) + TIMEOUT_MIN * 60))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  echo "--- $(date -Iseconds) ---"
  bash "$(dirname "$0")/monitor_training.sh" "$INSTANCE_ID" && exit 0
  RC=$?; [ "$RC" -eq 1 ] && echo "FAILED — relaunch needed" && exit 1
  sleep 30
done
echo "Timeout — still in SETUP. Run: bash scripts/monitor_training.sh $INSTANCE_ID"
exit 2
