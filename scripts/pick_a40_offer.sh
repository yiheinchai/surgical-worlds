#!/usr/bin/env bash
# Pick a Vast.ai offer for exact original TinyWorlds (bs=350/500 needs ≥40GB).
# Prefers machines that won't bottleneck the GPU (PCIe, CPU, RAM, disk, net).
#
# Usage:
#   export VASTAI_API_KEY=...
#   bash scripts/pick_a40_offer.sh              # prefer A40 / 48GB class
#   GPU_NAME=RTX_A6000 bash scripts/pick_a40_offer.sh
#   PREFER_GPU=A40 bash scripts/pick_a40_offer.sh --create   # also create instance
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
if [ -z "${VASTAI_API_KEY:-}" ] && [ -f "${HOME}/.config/vastai/vast_api_key" ]; then
  export VASTAI_API_KEY="$(cat "${HOME}/.config/vastai/vast_api_key")"
fi
[ -n "${VASTAI_API_KEY:-}" ] || { echo "Set VASTAI_API_KEY"; exit 1; }
vastai set api-key "$VASTAI_API_KEY" >/dev/null

PREFER_GPU="${PREFER_GPU:-A40}"
MIN_GPU_RAM_MB="${MIN_GPU_RAM_MB:-40000}"
MIN_CPU="${MIN_CPU:-8}"
MIN_RAM_MB="${MIN_RAM_MB:-32000}"
MIN_DISK_GB="${MIN_DISK_GB:-80}"
MIN_INET_DOWN="${MIN_INET_DOWN:-100}"
MIN_PCIE="${MIN_PCIE:-8}"
MIN_REL="${MIN_REL:-0.95}"
MAX_DPH="${MAX_DPH:-1.0}"
CREATE="${1:-}"

python3 - <<'PY' "$PREFER_GPU" "$MIN_GPU_RAM_MB" "$MIN_CPU" "$MIN_RAM_MB" "$MIN_DISK_GB" "$MIN_INET_DOWN" "$MIN_PCIE" "$MIN_REL" "$MAX_DPH" "$CREATE"
import json, os, subprocess, sys

prefer, min_vram, min_cpu, min_ram, min_disk, min_down, min_pcie, min_rel, max_dph, create = sys.argv[1:]
min_vram, min_cpu, min_ram = float(min_vram), float(min_cpu), float(min_ram)
min_disk, min_down, min_pcie = float(min_disk), float(min_down), float(min_pcie)
min_rel, max_dph = float(min_rel), float(max_dph)

query = (
    f"gpu_ram>={min_vram/1024:.0f} num_gpus=1 reliability>{min_rel} "
    f"rentable=True inet_down>{min_down} cpu_cores_effective>={min_cpu} "
    f"dph<{max_dph}"
)
raw = subprocess.check_output(
    ["vastai", "search", "offers", query, "--order", "dph_total", "--limit", "50", "--raw"],
    text=True,
)
offers = json.loads(raw)

# Exact original TinyWorlds needs ~40–48GB; skip 24GB cards even if listed.
keep = []
for o in offers:
    vram = float(o.get("gpu_ram") or 0)
    disk = float(o.get("disk_space") or 0)
    ram = float(o.get("cpu_ram") or 0)
    pcie = float(o.get("pcie_bw") or 0)
    if vram < float(min_vram):
        continue
    if disk < min_disk:
        continue
    if ram < min_ram:
        continue
    if pcie and pcie < min_pcie:
        continue
    keep.append(o)

def score(o):
    dph = float(o.get("dph_total") or 9e9)
    dlp = float(o.get("dlperf") or 1.0)
    pcie = float(o.get("pcie_bw") or 1.0)
    cpu = float(o.get("cpu_cores_effective") or 1.0)
    rel = float(o.get("reliability2") or 0.9)
    name = (o.get("gpu_name") or "")
    # Primary: dollars per unit DLPerf (lower better). Soft bonuses for host quality.
    cost_eff = dph / max(dlp, 1.0)
    prefer_bonus = -0.05 if prefer.lower() in name.lower().replace(" ", "_") or prefer.lower() in name.lower() else 0.0
    # Penalize weak host-side bottlenecks relative to GPU.
    bottleneck = 0.0
    if pcie < 12:
        bottleneck += 0.03
    if cpu < 12:
        bottleneck += 0.02
    if float(o.get("inet_down") or 0) < 200:
        bottleneck += 0.01
    return cost_eff + prefer_bonus + bottleneck, dph, -dlp

keep.sort(key=score)
if not keep:
    print("No offers matched filters.", file=sys.stderr)
    sys.exit(2)

print("Top offers (cost/DLPerf, host-bottleneck aware):")
for o in keep[:8]:
    dph = float(o["dph_total"])
    dlp = float(o.get("dlperf") or 0)
    print(
        f"  id={o['id']}  ${dph:.3f}/hr  {o.get('gpu_name')}  "
        f"vram={o.get('gpu_ram')}  pcie={o.get('pcie_bw')}  "
        f"cpu={o.get('cpu_cores_effective')}  ram={o.get('cpu_ram')}  "
        f"disk={o.get('disk_space')}  down={o.get('inet_down')}  "
        f"dlp={dlp:.1f}  $/dlp={dph/max(dlp,1):.4f}  "
        f"rel={float(o.get('reliability2') or 0):.3f}  {o.get('geolocation')}"
    )

best = keep[0]
print(f"\nSELECTED_OFFER_ID={best['id']}")
print(f"SELECTED_GPU={best.get('gpu_name')}")
print(f"SELECTED_DPH={best.get('dph_total')}")
open("/tmp/selected_vast_offer.json", "w").write(json.dumps(best, indent=2))

if create == "--create":
    print("Pass --create to scripts/launch_original_a40.sh instead.")
PY
