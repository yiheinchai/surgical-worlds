#!/usr/bin/env bash
# Check if Vast.ai surgical training is healthy — not just "instance running".
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
INSTANCE_ID="${1:-}"

if [ -z "${VASTAI_API_KEY:-}" ]; then
  if [ -f "$HOME/.config/vastai/vast_api_key" ]; then
    export VASTAI_API_KEY="$(cat "$HOME/.config/vastai/vast_api_key")"
  else
    echo "Set VASTAI_API_KEY first"; exit 1
  fi
fi

if [ -z "$INSTANCE_ID" ] && [ -f /tmp/surgical_worlds_launch.json ]; then
  INSTANCE_ID=$(python3 -c "import json; print(json.load(open('/tmp/surgical_worlds_launch.json')).get('instance_id',''))")
fi
INSTANCE_ID="${INSTANCE_ID:?Instance ID required}"

vastai set api-key "$VASTAI_API_KEY" >/dev/null

echo "═══ Instance $INSTANCE_ID ═══"
vastai show instance "$INSTANCE_ID" 2>&1 | head -20

RAW=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null)
LOGS=$(vastai logs "$INSTANCE_ID" 2>&1 || true)
LOG_FILE=$(mktemp)
echo "$LOGS" > "$LOG_FILE"

VERDICT=$(python3 - "$RAW" "$LOG_FILE" <<'PY'
import json, re, sys
d = json.loads(sys.argv[1])
logs = open(sys.argv[2]).read()
status = d.get("actual_status", "?")
gpu = float(d.get("gpu_util") or 0)
disk = float(d.get("disk_usage") or 0)
markers = re.findall(r'\[STATUS\]\s+(.+)', logs)
phase = ""
if markers:
    for part in markers[-1].split():
        if part.startswith("phase="): phase = part.split("=",1)[1]
for pat in [r"unbound variable", r"Traceback", r"ModuleNotFoundError", r"phase=failed"]:
    if re.search(pat, logs, re.I):
        print(f"FAILED|Error in logs: {pat}")
        sys.exit(0)
if "phase=ready" in logs:
    print("READY|Training complete")
    sys.exit(0)
if re.search(r"epoch|loss|full_train", logs, re.I) and gpu > 1:
    print(f"TRAINING|GPU {gpu:.0f}%")
    sys.exit(0)
if phase == "training" or "full_train" in logs:
    print(f"TRAINING|phase={phase or 'training'} GPU {gpu:.0f}%")
    sys.exit(0)
if phase in ("downloading","setup","starting") or "Cloning" in logs:
    print(f"SETUP|phase={phase or 'setup'}")
    sys.exit(0)
if status == "running" and "Cloning into" in logs and "pip install" not in logs:
    print("FAILED|Setup stopped after git clone — onstart likely crashed")
    sys.exit(0)
print(f"UNKNOWN|status={status} GPU={gpu:.0f}%")
PY
)
rm -f "$LOG_FILE"

TYPE="${VERDICT%%|*}"; MSG="${VERDICT#*|}"
echo ""; echo "═══ Verdict ═══"
case "$TYPE" in
  READY|TRAINING|POST_TRAIN) echo "  ✅ $TYPE — $MSG" ;;
  SETUP) echo "  ⏳ SETUP — $MSG" ;;
  FAILED) echo "  ❌ FAILED — $MSG" ;;
  *) echo "  ❓ $TYPE — $MSG" ;;
esac
echo ""; echo "═══ Recent logs ═══"; echo "$LOGS" | tail -15
case "$TYPE" in READY|TRAINING|POST_TRAIN) exit 0;; SETUP|UNKNOWN) exit 2;; *) exit 1;; esac
