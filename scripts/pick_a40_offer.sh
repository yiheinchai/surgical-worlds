#!/usr/bin/env bash
# Pick a Vast.ai offer for exact original TinyWorlds (bs=350/500 needs ≥40GB).
# Scores by effective $/throughput after penalizing host bottlenecks that leave
# the GPU idle (weak PCIe, few CPUs, low disk BW, partial GPU, etc.).
#
# Usage:
#   export VASTAI_API_KEY=...
#   bash scripts/pick_a40_offer.sh
#   PREFER_GPU=A40 STRICT_GPU=1 bash scripts/pick_a40_offer.sh
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
if [ -z "${VASTAI_API_KEY:-}" ] && [ -f "${HOME}/.config/vastai/vast_api_key" ]; then
  export VASTAI_API_KEY="$(cat "${HOME}/.config/vastai/vast_api_key")"
fi
[ -n "${VASTAI_API_KEY:-}" ] || { echo "Set VASTAI_API_KEY"; exit 1; }
vastai set api-key "$VASTAI_API_KEY" >/dev/null

PREFER_GPU="${PREFER_GPU:-A40}"
STRICT_GPU="${STRICT_GPU:-0}"          # 1 = only prefer-gpu name
MIN_GPU_RAM_MB="${MIN_GPU_RAM_MB:-40000}"
MIN_CPU="${MIN_CPU:-8}"
MIN_RAM_MB="${MIN_RAM_MB:-32000}"
MIN_DISK_GB="${MIN_DISK_GB:-80}"
MIN_INET_DOWN="${MIN_INET_DOWN:-50}"
MIN_PCIE="${MIN_PCIE:-10}"             # GB/s measured PCIe; <10 often starves GPU
MIN_DISK_BW="${MIN_DISK_BW:-500}"      # MB/s; HDF5 preload / checkpoint IO
MIN_REL="${MIN_REL:-0.95}"
MAX_DPH="${MAX_DPH:-1.2}"
MIN_GPU_FRAC="${MIN_GPU_FRAC:-1.0}"    # reject fractional/shared GPUs
REQUIRE_VERIFIED="${REQUIRE_VERIFIED:-1}"

python3 - <<'PY' \
  "$PREFER_GPU" "$STRICT_GPU" "$MIN_GPU_RAM_MB" "$MIN_CPU" "$MIN_RAM_MB" \
  "$MIN_DISK_GB" "$MIN_INET_DOWN" "$MIN_PCIE" "$MIN_DISK_BW" "$MIN_REL" \
  "$MAX_DPH" "$MIN_GPU_FRAC" "$REQUIRE_VERIFIED"
import json, subprocess, sys
from pathlib import Path

(
    prefer, strict, min_vram, min_cpu, min_ram, min_disk, min_down,
    min_pcie, min_disk_bw, min_rel, max_dph, min_gpu_frac, require_verified,
) = sys.argv[1:]
min_vram = float(min_vram)
min_cpu = float(min_cpu)
min_ram = float(min_ram)
min_disk = float(min_disk)
min_down = float(min_down)
min_pcie = float(min_pcie)
min_disk_bw = float(min_disk_bw)
min_rel = float(min_rel)
max_dph = float(max_dph)
min_gpu_frac = float(min_gpu_frac)
strict = strict == "1"
require_verified = require_verified == "1"

# Vast search language uses GB for gpu_ram.
query = (
    f"gpu_ram>={min_vram/1024:.0f} num_gpus=1 reliability>{min_rel} "
    f"rentable=True inet_down>={min_down} cpu_cores_effective>={min_cpu} "
    f"dph<{max_dph} gpu_frac>={min_gpu_frac}"
)
raw = subprocess.check_output(
    ["vastai", "search", "offers", query, "--order", "dph_total", "--limit", "80", "--raw"],
    text=True,
)
offers = json.loads(raw)

# Consumer 24GB cards sometimes appear with inflated gpu_ram in search; require real ≥40GB.
# Also reject weird "RTX 4090 with 48GB" listings (usually dual-GPU mislabels) unless
# gpu_total_ram matches a known 48GB SKU name.
KNOWN_48GB = {
    "a40", "rtx a6000", "a6000", "l40", "l40s", "rtx 6000 ada", "6000 ada",
    "rtx pro 5000", "rtx 5000 ada", "a100", "a100 sxm4", "a100 pcie",
    "q rtx 8000", "rtx 8000", "rtx a5000",
}

def is_plausible_vram(o):
    name = (o.get("gpu_name") or "").lower()
    vram = float(o.get("gpu_ram") or 0)
    if vram < min_vram:
        return False
    if "4090" in name and vram > 30000:
        return False  # 4090 is 24GB; 48GB listing is not trustworthy
    if "3090" in name and vram > 30000:
        return False
    if vram >= 40000 and not any(k in name for k in KNOWN_48GB):
        # allow unknown 40GB+ professional cards, but flag later
        pass
    return True

def hard_reject_reason(o):
    if not is_plausible_vram(o):
        return "implausible_vram"
    if float(o.get("disk_space") or 0) < min_disk:
        return "disk"
    if float(o.get("cpu_ram") or 0) < min_ram:
        return "cpu_ram"
    pcie = float(o.get("pcie_bw") or 0)
    if pcie and pcie < min_pcie:
        return f"pcie<{min_pcie}"
    disk_bw = float(o.get("disk_bw") or 0)
    if disk_bw and disk_bw < min_disk_bw:
        return f"disk_bw<{min_disk_bw}"
    if float(o.get("gpu_frac") or 1) < min_gpu_frac:
        return "shared_gpu"
    if int(o.get("num_gpus") or 1) != 1:
        return "multi_gpu"
    if require_verified and str(o.get("verification") or "").lower() not in ("verified", "true", "1"):
        # verification field can be "verified" or boolean-ish
        ver = o.get("verification")
        if ver not in (True, "verified", "Verified"):
            return "unverified"
    if strict:
        pname = prefer.lower().replace("_", " ")
        if pname not in (o.get("gpu_name") or "").lower():
            return "not_preferred_gpu"
    return None

kept, rejected = [], {}
for o in offers:
    reason = hard_reject_reason(o)
    if reason:
        rejected[reason] = rejected.get(reason, 0) + 1
        continue
    kept.append(o)

def effective_throughput(o):
    """Estimate usable training throughput vs advertised DLPerf.

    TinyWorlds is activation/PCIe heavy (large batches of video tokens). When
    host PCIe, disk, or CPU can't feed the GPU, realized TFLOP/s collapses.
    """
    dlp = max(float(o.get("dlperf") or 1.0), 1.0)
    pcie = float(o.get("pcie_bw") or 1.0)
    disk_bw = float(o.get("disk_bw") or 1.0)
    cpu = float(o.get("cpu_cores_effective") or 1.0)
    ram_gb = float(o.get("cpu_ram") or 1.0) / 1024.0
    gpu_mem_bw = float(o.get("gpu_mem_bw") or 500.0)
    lanes = float(o.get("gpu_lanes") or 8)

    # Soft caps: each factor in (0,1], geometric mean-ish.
    # PCIe gen4 x16 ~ 25GB/s; measured pcie_bw is Vast's probe.
    f_pcie = min(1.0, pcie / 20.0)          # full credit by ~20 GB/s
    f_disk = min(1.0, disk_bw / 2000.0)     # NVMe-ish
    f_cpu = min(1.0, cpu / 16.0)            # dataloader workers
    f_ram = min(1.0, ram_gb / 48.0)         # preload headroom
    f_lanes = min(1.0, lanes / 16.0)
    f_gmem = min(1.0, gpu_mem_bw / 500.0)

    # For this workload PCIe + CPU matter most; disk matters at ckpt/data load.
    feed = (f_pcie ** 0.35) * (f_cpu ** 0.25) * (f_disk ** 0.15) * (f_ram ** 0.10) * (f_lanes ** 0.10) * (f_gmem ** 0.05)
    feed = max(0.15, min(1.0, feed))
    return dlp * feed, feed, {
        "f_pcie": round(f_pcie, 3),
        "f_cpu": round(f_cpu, 3),
        "f_disk": round(f_disk, 3),
        "f_ram": round(f_ram, 3),
        "f_lanes": round(f_lanes, 3),
        "feed": round(feed, 3),
    }

def rank_key(o):
    dph = float(o.get("dph_total") or 9e9)
    eff, feed, _ = effective_throughput(o)
    # dollars per effective DLPerf unit (lower better)
    cost = dph / max(eff, 1e-3)
    name = (o.get("gpu_name") or "").lower()
    prefer_hit = prefer.lower().replace("_", " ") in name
    # Small preference nudge only — never override a clearly better machine.
    nudge = -0.0005 if prefer_hit else 0.0
    rel = float(o.get("reliability2") or 0.9)
    # Prefer verified high-reliability slightly
    cost -= 0.0002 * max(0.0, rel - 0.98)
    return (cost + nudge, dph, -eff)

kept.sort(key=rank_key)

print(f"Offers searched: {len(offers)}  kept: {len(kept)}  rejected: {rejected}")
if not kept:
    print("No offers matched filters.", file=sys.stderr)
    sys.exit(2)

print("\nTop offers (lowest $/effective-throughput; feed factor shows host bottleneck):")
print(f"{'id':>10} {'$/hr':>7} {'gpu':14} {'pcie':>5} {'diskBW':>7} {'cpu':>4} {'feed':>5} {'effDLP':>7} {'$/eff':>8} {'where'}")
rows = []
for o in kept[:10]:
    dph = float(o["dph_total"])
    eff, feed, factors = effective_throughput(o)
    row = {
        "id": o["id"],
        "dph": dph,
        "gpu": o.get("gpu_name"),
        "pcie": o.get("pcie_bw"),
        "disk_bw": o.get("disk_bw"),
        "cpu": o.get("cpu_cores_effective"),
        "ram_mb": o.get("cpu_ram"),
        "disk": o.get("disk_space"),
        "inet_down": o.get("inet_down"),
        "dlperf": o.get("dlperf"),
        "eff_dlp": eff,
        "feed": feed,
        "factors": factors,
        "cost_per_eff": dph / max(eff, 1e-3),
        "rel": o.get("reliability2"),
        "geo": o.get("geolocation"),
        "verification": o.get("verification"),
        "gpu_frac": o.get("gpu_frac"),
        "gpu_lanes": o.get("gpu_lanes"),
        "cpu_name": o.get("cpu_name"),
        "disk_name": o.get("disk_name"),
        "mobo": o.get("mobo_name"),
    }
    rows.append(row)
    print(
        f"{o['id']:>10} {dph:7.3f} {(o.get('gpu_name') or '')[:14]:14} "
        f"{float(o.get('pcie_bw') or 0):5.1f} {float(o.get('disk_bw') or 0):7.0f} "
        f"{float(o.get('cpu_cores_effective') or 0):4.0f} {feed:5.2f} {eff:7.1f} "
        f"{dph/max(eff,1e-3):8.4f} {o.get('geolocation')}"
    )

best = kept[0]
eff, feed, factors = effective_throughput(best)
print(f"\nSELECTED_OFFER_ID={best['id']}")
print(f"SELECTED_GPU={best.get('gpu_name')}")
print(f"SELECTED_DPH={best.get('dph_total')}")
print(f"SELECTED_FEED={feed:.3f}  factors={factors}")
if feed < 0.7:
    print("WARNING: selected host feed factor <0.7 — GPU may be partially starved.")

out = {
    "selected": best,
    "effective_throughput": eff,
    "feed_factor": feed,
    "factors": factors,
    "ranking": rows,
    "rejected_counts": rejected,
}
Path("/tmp/selected_vast_offer.json").write_text(json.dumps(out, indent=2))
PY
