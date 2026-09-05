"""Insert neutral spatial attention while retaining model/RNG/optimizer state."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import argparse
import copy
import hashlib
import json
import torch
from models.pixel_diffusion import PixelDenoiser


def expand_checkpoint(checkpoint):
    ck = copy.deepcopy(checkpoint)
    config = dict(ck["model_config"])
    if config.get("bottleneck_attention", False):
        raise ValueError("Checkpoint already contains attention")
    # fork_rng prevents this migration from changing the caller's RNG stream.
    with torch.random.fork_rng():
        torch.manual_seed(731)
        old = PixelDenoiser(**config)
        old.load_state_dict(ck["model"], strict=True)
        config["bottleneck_attention"] = True
        new = PixelDenoiser(**config)
    old_names = [n for n, _ in old.named_parameters()]
    new_names = [n for n, _ in new.named_parameters()]
    assert new_names[: len(old_names)] == old_names
    extra = {
        k: v.clone()
        for k, v in new.state_dict().items()
        if k.startswith("spatial_attention.")
    }
    for key in ("model", "ema"):
        ck[key].update(extra)
        new.load_state_dict(ck[key], strict=True)
    groups = ck["optimizer"]["param_groups"]
    if len(groups) != 1 or groups[0]["params"] != list(range(len(old_names))):
        raise ValueError("Expected a single optimizer group in model parameter order")
    groups[0]["params"] = list(range(len(new_names)))
    ck["model_config"] = config
    return ck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    ck = expand_checkpoint(
        torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    )
    torch.save(ck, out)
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": a.checkpoint,
                "source_sha256": hashlib.sha256(
                    Path(a.checkpoint).read_bytes()
                ).hexdigest(),
                "step": ck["step"],
                "change": "One zero-output residual spatial attention block at the 16x16 bottleneck",
                "retained": "All existing weights, EMA, AdamW moments, and saved training RNG state",
                "new_optimizer_state": "Attention parameters start with empty AdamW moments",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
