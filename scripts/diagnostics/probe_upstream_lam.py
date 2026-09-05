from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import json, torch, h5py, numpy as np
from models.latent_actions import LatentActionModel
from scripts.synthetic_branched_motion import translated_pairs

base = ROOT
out = base / "results/readiness/upstream-lam-probe"
out.mkdir(exist_ok=True)
x = torch.load(
    base / "data/upstream/picodoom_latent_actions_step_4500_2025_09_18_23_02_09.pth",
    map_location="cpu",
    weights_only=True,
)
c = x["config"]
kw = {
    k: c[k]
    for k in [
        "n_actions",
        "patch_size",
        "embed_dim",
        "num_heads",
        "hidden_dim",
        "num_blocks",
    ]
}
m = LatentActionModel(frame_size=(64, 64), **kw).cuda().eval()
m.load_state_dict(x["model"], strict=True)
with h5py.File(base / "data/picodoom_frames.h5") as f:
    data = f["frames"][:]
res = {
    "config": c,
    "strict_load": True,
    "parameters": sum(p.numel() for p in m.parameters()),
}
preds = {}
with torch.inference_mode():
    for split, lo, hi, n in [
        ("train", 3000, 46000, 64),
        ("validation", 47000, 53000, 32),
    ]:
        tex = (
            torch.from_numpy(data[np.linspace(lo, hi - 1, n, dtype=int)])
            .float()
            .permute(0, 3, 1, 2)
            / 127.5
            - 1
        )
        pairs = translated_pairs(tex)
        labels = []
        for batch in pairs.split(16):
            labels.extend(
                m.quantizer.get_indices_from_latents(m.encode(batch.cuda()))
                .cpu()
                .reshape(-1)
                .tolist()
            )
        preds[split] = np.array(labels).reshape(n, 4)
    mapping = np.array(
        [
            np.bincount(
                np.tile(np.arange(4), 64)[preds["train"].reshape(-1) == k], minlength=4
            ).argmax()
            for k in range(c["n_actions"])
        ]
    )
    v = preds["validation"]
    res["synthetic"] = {
        "train_mapping_accuracy": float(
            (mapping[preds["train"]] == np.arange(4)).mean()
        ),
        "validation_mapping_accuracy": float((mapping[v] == np.arange(4)).mean()),
        "same_code_all_branches": float((v == v[:, :1]).all(1).mean()),
        "codes": v.tolist(),
    }
    starts = np.linspace(47000, 52990, 128, dtype=int)
    ix = starts[:, None] + np.arange(4) * 2
    clips = torch.from_numpy(data[ix]).float().permute(0, 1, 4, 2, 3) / 127.5 - 1
    metrics = {k: [] for k in ["inferred_l1", "shuffled_l1", "copy_l1"]}
    all_ids = []
    for x in clips.split(16):
        x = x.cuda()
        q = m.encode(x)
        pred = m.decoder(x, q, training=False)
        wrong = m.decoder(x, q.roll(1, 0), training=False)
        for k, p in [
            ("inferred_l1", pred[:, -1]),
            ("shuffled_l1", wrong[:, -1]),
            ("copy_l1", x[:, -2]),
        ]:
            metrics[k].extend((p - x[:, -1]).abs().mean((1, 2, 3)).cpu().tolist())
        all_ids.extend(
            m.quantizer.get_indices_from_latents(q).cpu().reshape(-1).tolist()
        )
    res["real"] = {k: float(np.mean(v)) for k, v in metrics.items()}
    res["real"]["code_counts"] = np.bincount(all_ids, minlength=c["n_actions"]).tolist()
(out / "results.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res), flush=True)
