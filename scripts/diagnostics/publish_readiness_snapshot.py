"""Publish an explicit, completed evidence snapshot with per-file SHA256 records."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import argparse, hashlib, json, os
import wandb


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--folder", action="append", default=[])
    p.add_argument("--weight", action="append", default=[])
    p.add_argument("--fresh-episodes")
    p.add_argument("--key-file", required=True)
    a = p.parse_args()
    out = ROOT / "results/readiness" / ("publish-" + a.name)
    out.mkdir(exist_ok=False)
    files = set()
    for name in a.folder:
        folder = ROOT / "results/readiness" / name
        for f in folder.rglob("*"):
            if (
                f.is_file()
                and "wandb" not in f.relative_to(folder).parts
                and f.suffix in (".json", ".jsonl", ".png", ".gif", ".py", ".npz")
            ):
                files.add(f)
    for path in a.weight:
        files.add((ROOT / path).resolve())
    if a.fresh_episodes:
        folder = ROOT / a.fresh_episodes
        files.update(folder.glob("*.h5"))
        files.add(folder / "manifest.json")
    manifest = {
        str(f.relative_to(ROOT)): {"bytes": f.stat().st_size, "sha256": digest(f)}
        for f in sorted(files)
    }
    mp = out / "sha256_manifest.json"
    mp.write_text(json.dumps(manifest, indent=2))
    os.environ["WANDB_API_KEY"] = Path(a.key_file).read_text().strip()
    run = wandb.init(
        entity="data2yihein-d",
        project="tinyworlds",
        group="picodoom-readiness-20260905",
        name="publish-" + a.name,
        dir=str(out),
        config={
            "files": len(files),
            "bytes": sum(x["bytes"] for x in manifest.values()),
            "scope": "Explicit completed directories and selected weights. Fresh episodes 1 and 2 remain uninspected by the investigator.",
            "labels": "RGB only; no engine-state or action-label exports.",
        },
    )
    artifact = wandb.Artifact(
        a.name,
        type="model",
        metadata={
            "files": len(files),
            "manifest_sha256": digest(mp),
            "status": "Research evidence; full-run readiness not yet established",
        },
    )
    artifact.add_file(str(mp), name="sha256_manifest.json")
    for f in sorted(files):
        artifact.add_file(str(f), name=str(f.relative_to(ROOT)))
    run.log_artifact(artifact)
    artifact.wait(timeout=300)
    result = {
        "artifact_name": artifact.name,
        "artifact_id": artifact.id,
        "state": str(artifact.state),
        "run_url": run.url,
        "files": len(files),
        "bytes": sum(x["bytes"] for x in manifest.values()),
        "manifest_sha256": digest(mp),
    }
    (out / "published.json").write_text(json.dumps(result, indent=2))
    run.summary.update(result)
    print(json.dumps(result), flush=True)
    run.finish()


if __name__ == "__main__":
    main()
