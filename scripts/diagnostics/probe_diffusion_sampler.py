from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import json, torch, h5py, numpy as np, time
from PIL import Image
from models.pixel_diffusion import PixelDenoiser
from scripts.train_pixel_dynamics import image_array

base = ROOT
out = base / "results/readiness/sampler-probe-5k"
out.mkdir(exist_ok=True)
ck = torch.load(
    base / "results/readiness/pixel-motion-noise01-seed0-5k-v2/last.pt",
    map_location="cuda",
    weights_only=True,
)
c = np.load(base / "results/readiness/motion/motion_codes.npz")
starts = np.linspace(47000, 52989, 32, dtype=int)
ix = starts[:, None] + np.arange(5) * 2
with h5py.File(base / "data/picodoom_frames.h5", "r") as f:
    data = f["frames"][:]
x = torch.from_numpy(data[ix]).cuda().float().permute(0, 1, 4, 2, 3) / 127.5 - 1
h, y = x[:, :4], x[:, 4]
a = torch.from_numpy(c["ids"][ix[:, :4]]).long().cuda()
m = PixelDenoiser(8, 4, 32).cuda().eval()
results = {}
with torch.no_grad():
    for state in ["model", "ema"]:
        m.load_state_dict(ck[state])
        row = {}
        torch.manual_seed(71)
        for sigma in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 5.0]:
            s = torch.full((len(y),), sigma, device="cuda")
            noisy = y + torch.randn_like(y) * sigma
            pred = m(noisy, s, h, a)
            row[f"denoise_{sigma}"] = float((pred - y).abs().mean())
        for steps in [4, 8, 16, 32]:
            for heun in [False, True]:
                torch.cuda.synchronize()
                t = time.time()
                pred = m.sample(h, a, steps, seed=71, heun=heun)
                torch.cuda.synchronize()
                key = f"sample_{steps}_heun{heun}"
                row[key] = {
                    "l1": float((pred - y).abs().mean()),
                    "seconds": time.time() - t,
                    "saturation": float((pred.abs() > 0.95).float().mean()),
                }
                if state == "ema":
                    Image.fromarray(np.concatenate(image_array(pred[:8]), 1)).save(
                        out / (key + ".png")
                    )
        results[state] = row
        print(state, json.dumps(row), flush=True)
(out / "results.json").write_text(json.dumps(results, indent=2))
