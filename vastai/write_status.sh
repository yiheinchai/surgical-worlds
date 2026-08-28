#!/usr/bin/env bash
# Write /workspace/TRAINING_STATUS.json and echo a grep-friendly log marker.
set -euo pipefail

STATUS_FILE="${STATUS_FILE:-/workspace/TRAINING_STATUS.json}"
PHASE="${1:-unknown}"
shift || true

python3 - "$STATUS_FILE" "$PHASE" "$@" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path

status_file, phase = sys.argv[1], sys.argv[2]
extra = {}
for arg in sys.argv[3:]:
    if "=" in arg:
        k, v = arg.split("=", 1)
        extra[k] = v

data = {}
p = Path(status_file)
if p.exists():
    try:
        data = json.loads(p.read_text())
    except Exception:
        pass

data["phase"] = phase
data["updated_at"] = datetime.now(timezone.utc).isoformat()
data.update(extra)
p.write_text(json.dumps(data, indent=2))

parts = [f"phase={phase}"]
for k, v in extra.items():
    parts.append(f"{k}={v}")
print(f"[STATUS] {' '.join(parts)}")
PY
