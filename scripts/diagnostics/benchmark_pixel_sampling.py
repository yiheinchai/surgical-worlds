"""Measure isolated batch-one sampling cost, separately from visual quality."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import argparse
import hashlib
import json
import os
import subprocess
import time
import h5py
import numpy as np
import torch
from models.pixel_diffusion import PixelDenoiser


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--data", default="data/picodoom_frames.h5")
    p.add_argument("--output", default="results/readiness")
    p.add_argument("--repetitions", type=int, default=30)
    p.add_argument("--key-file")
    a = p.parse_args()
    if not 5 <= a.repetitions <= 100:
        raise ValueError("Use 5 to 100 bounded repetitions")
    out = Path(a.output) / a.name
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    ck = torch.load(a.checkpoint, map_location="cuda", weights_only=True)
    config = (
        ck["model_config"]
        if "model_config" in ck
        else {
            k: ck[k]
            for k in [
                "codes",
                "history",
                "width",
                "offset_noise",
                "prediction_mode",
                "noise_features",
                "event_count",
                "use_event_prior",
                "bottleneck_attention",
            ]
            if k in ck
        }
    )
    model = PixelDenoiser(**config).cuda().eval()
    model.load_state_dict(ck.get("ema", ck["model"]), strict=True)
    with h5py.File(a.data) as f:
        frames = f["frames"][47000:47008:2]
    history = (
        torch.from_numpy(frames).cuda().float().permute(0, 3, 1, 2)[None] / 127.5 - 1
    )
    actions = torch.zeros(1, 4, dtype=torch.long, device="cuda")

    def processes():
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        rows = [line for line in r.stdout.splitlines() if line.strip()]
        # nvidia-smi reports host PIDs on Vast, while Python sees container PIDs.
        # This process already owns a CUDA context; require it to be the sole
        # listed compute context, rather than incorrectly comparing namespaces.
        if len(rows) != 1:
            raise RuntimeError(
                "Expected one GPU compute process; isolated latency measurement deferred"
            )
        return rows

    before = processes()
    results = {}
    with torch.inference_mode():
        for solver, steps in [("euler", 3), ("euler", 8), ("heun", 8)]:
            for i in range(5):
                model.sample(
                    history,
                    actions,
                    steps,
                    seed=i,
                    stabilization=0.1,
                    heun=solver == "heun",
                )
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline_memory = torch.cuda.memory_allocated()
            wall, gpu = [], []
            for i in range(a.repetitions):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                t = time.perf_counter()
                start.record()
                model.sample(
                    history,
                    actions,
                    steps,
                    seed=100 + i,
                    stabilization=0.1,
                    heun=solver == "heun",
                )
                end.record()
                torch.cuda.synchronize()
                wall.append((time.perf_counter() - t) * 1000)
                gpu.append(start.elapsed_time(end))
            results[f"{solver}{steps}"] = {
                "wall_ms": wall,
                "gpu_ms": gpu,
                "wall_median_ms": float(np.median(wall)),
                "wall_p95_ms": float(np.percentile(wall, 95)),
                "median_fps": 1000 / float(np.median(wall)),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "incremental_peak_bytes": torch.cuda.max_memory_allocated()
                - baseline_memory,
            }
    result = {
        "config": vars(a),
        "model_config": config,
        "checkpoint_sha256": hashlib.sha256(
            Path(a.checkpoint).read_bytes()
        ).hexdigest(),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "batch": 1,
        "precision": "float32",
        "gpu_processes_before": before,
        "gpu_processes_after": processes(),
        "results": results,
        "scope": "Warm sampling only, four fixed RGB history frames, no UI/network/decode/logging overhead; latency is not a quality assessment.",
    }
    (out / "results.json").write_text(json.dumps(result, indent=2))
    summary = {
        f"{k}/{metric}": v[metric]
        for k, v in results.items()
        for metric in [
            "wall_median_ms",
            "wall_p95_ms",
            "median_fps",
            "peak_allocated_bytes",
        ]
    }
    print(json.dumps(summary), flush=True)
    if a.key_file:
        import wandb

        os.environ["WANDB_API_KEY"] = Path(a.key_file).read_text().strip()
        run = wandb.init(
            entity="data2yihein-d",
            project="tinyworlds",
            group="picodoom-readiness-20260905",
            name=a.name,
            config=result["config"],
            dir=str(out),
        )
        run.log(summary)
        run.finish()


if __name__ == "__main__":
    main()
