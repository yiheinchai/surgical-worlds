"""Collect rendered RGB episodes through the public POOM keyboard interface.

The recorder exports pixels and capture timestamps only. Keyboard commands and
engine state are never stored as training labels. These are fresh POOM episodes;
the original TinyWorlds video has no source provenance proving exact equivalence.
Requires the optional playwright package and its Chromium headless shell.
"""

from pathlib import Path
import argparse, base64, io, json, time, hashlib
import h5py, numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

GAME_URL = "https://html-classic.itch.zone/html/7204900/index.html"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/fresh_poom")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seconds", type=int, default=90)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=3101)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    metadata = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        try:
            for ep in range(a.episodes):
                page = browser.new_page(viewport={"width": 640, "height": 640})
                page.goto(GAME_URL, wait_until="domcontentloaded", timeout=60000)
                page.locator("#p8_start_button").click()
                page.wait_for_timeout(25000)
                canvas = page.locator("canvas")
                canvas.click()
                for _ in range(3):
                    page.keyboard.press("x", delay=250)
                    page.wait_for_timeout(2200)
                page.wait_for_timeout(5000)
                rng = np.random.default_rng(a.seed + ep)
                frames = []
                times = []
                held = set()
                previous = None
                stuck = 0
                started = time.perf_counter()
                for i in range(a.seconds * a.fps):
                    # Bounded exploration, without reading hidden engine state. Long key holds
                    # are required because the game samples keyboard state once per frame.
                    if i % a.fps == 0:
                        desired = set()
                        if rng.random() < 0.75:
                            desired.add("ArrowUp")
                        if rng.random() < 0.4 or stuck > 4:
                            desired.add(rng.choice(["ArrowLeft", "ArrowRight"]))
                        if rng.random() < 0.55:
                            desired.add("x")
                        if rng.random() < 0.2:
                            desired.add("z")
                        for key in held - desired:
                            page.keyboard.up(key)
                        for key in desired - held:
                            page.keyboard.down(str(key))
                        held = desired
                    raw = base64.b64decode(
                        canvas.evaluate('(c)=>c.toDataURL("image/png")').split(",", 1)[
                            1
                        ]
                    )
                    arr = np.asarray(
                        Image.open(io.BytesIO(raw))
                        .convert("RGB")
                        .resize((64, 64), Image.Resampling.BOX)
                    )
                    elapsed = time.perf_counter() - started
                    frames.append(arr)
                    times.append(elapsed)
                    if previous is not None:
                        stuck = (
                            stuck + 1
                            if np.abs(arr.astype(float) - previous).mean() < 0.25
                            else 0
                        )
                    previous = arr.astype(float)
                    delay = (i + 1) / a.fps - (time.perf_counter() - started)
                    if delay > 0:
                        page.wait_for_timeout(delay * 1000)
                for key in held:
                    page.keyboard.up(str(key))
                frames = np.stack(frames)
                path = out / f"episode_{ep:03}.h5"
                with h5py.File(path, "w") as f:
                    f.create_dataset(
                        "frames", data=frames, compression="gzip", compression_opts=1
                    )
                    f.create_dataset("capture_seconds", data=times)
                    f.attrs["source_url"] = "https://freds72.itch.io/poom"
                    f.attrs["game_url"] = GAME_URL
                    f.attrs["episode_id"] = ep
                    f.attrs["requested_capture_fps"] = a.fps
                    f.attrs["supervision"] = (
                        "Rendered RGB only; no engine state or action labels"
                    )
                    f.attrs["downsampling"] = (
                        "Rendered canvas PNG -> RGB64 via Pillow BOX"
                    )
                    f.attrs["source_equivalence"] = (
                        "Exact original TinyWorlds recording game/version not documented"
                    )
                selected = frames[np.linspace(0, len(frames) - 1, 12, dtype=int)]
                Image.fromarray(np.concatenate(selected, axis=1)).resize(
                    (1536, 128), Image.Resampling.NEAREST
                ).save(out / f"episode_{ep:03}.png")
                row = {
                    "episode": ep,
                    "frames": len(frames),
                    "elapsed_seconds": times[-1],
                    "median_frame_interval": float(np.median(np.diff(times))),
                    "p95_frame_interval": float(np.quantile(np.diff(times), 0.95)),
                    "mean_frame_change_u8": float(
                        np.abs(np.diff(frames.astype(np.float32), axis=0)).mean()
                    ),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                metadata.append(row)
                (out / "manifest.json").write_text(
                    json.dumps(
                        {
                            "config": vars(a),
                            "episodes": metadata,
                            "attribution": "POOM by freds72 and Paranoid Cactus; https://freds72.itch.io/poom; game assets CC BY-NC 4.0",
                            "holdout_policy": "Episode 0: development. Episodes 1 and 2: retain as uninspected holdouts until development choices are frozen. No fresh frames used to train the current pilots.",
                        },
                        indent=2,
                    )
                )
                print(json.dumps(row), flush=True)
                page.close()
        finally:
            browser.close()


if __name__ == "__main__":
    main()
