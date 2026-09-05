"""Evaluation-only keyboard challenge; never an action-labelled training set."""

from pathlib import Path
import argparse, base64, hashlib, io, json, time
import h5py, numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

URL = "https://html-classic.itch.zone/html/7204900/index.html"
CONTROLS = {
    "idle": None,
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "forward": "ArrowUp",
    "backward": "ArrowDown",
    "fire": "x",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/evaluation_only/control_challenge_v1")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--cycles", type=int, default=3)
    a = p.parse_args()
    if not 1 <= a.episodes <= 4 or not 1 <= a.cycles <= 3:
        raise ValueError("Challenge is bounded to four episodes and three cycles")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    all_rows = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        try:
            for ep in range(a.episodes):
                page = browser.new_page(viewport={"width": 640, "height": 640})
                page.goto(URL, wait_until="domcontentloaded", timeout=60000)
                page.locator("#p8_start_button").click()
                page.wait_for_timeout(25000)
                canvas = page.locator("canvas")
                canvas.click()
                for _ in range(3):
                    page.keyboard.press("x", delay=250)
                    page.wait_for_timeout(2200)
                page.wait_for_timeout(5000)
                # Distinct visible viewpoints, without reading or setting engine state.
                if ep:
                    page.keyboard.press("ArrowRight", delay=ep * 300)
                page.keyboard.press("ArrowUp", delay=900)
                rng = np.random.default_rng(6100 + ep)
                frames, rows, preview = [], [], []
                for cycle in range(a.cycles):
                    for control in rng.permutation(list(CONTROLS)):
                        key = CONTROLS[control]
                        trial, times = [], []
                        started = time.perf_counter()
                        for i in range(30):
                            if i == 6 and key:
                                page.keyboard.down(key)
                            raw = base64.b64decode(
                                canvas.evaluate('(c)=>c.toDataURL("image/png")').split(
                                    ",", 1
                                )[1]
                            )
                            arr = np.asarray(
                                Image.open(io.BytesIO(raw))
                                .convert("RGB")
                                .resize((64, 64), Image.Resampling.BOX)
                            )
                            trial.append(arr)
                            times.append(time.perf_counter() - started)
                            delay = (i + 1) / 30 - (time.perf_counter() - started)
                            if delay > 0:
                                page.wait_for_timeout(delay * 1000)
                        if key:
                            page.keyboard.up(key)
                        index = len(frames)
                        frames.append(np.stack(trial))
                        rows.append(
                            {
                                "episode": ep,
                                "trial": index,
                                "cycle": cycle,
                                "evaluation_control": str(control),
                                "key_pressed_after_frame": 5,
                                "capture_seconds": times,
                            }
                        )
                        preview.append(
                            np.concatenate([trial[5], trial[10], trial[-1]], axis=1)
                        )
                    # Additional visual exploration between cycles; not part of scored labels.
                    page.keyboard.press(
                        str(rng.choice(["ArrowLeft", "ArrowRight"])), delay=600
                    )
                    page.keyboard.press("ArrowUp", delay=1300)
                path = out / f"episode_{ep:03}.h5"
                with h5py.File(path, "w") as f:
                    f.create_dataset(
                        "trials",
                        data=np.stack(frames),
                        compression="gzip",
                        compression_opts=1,
                    )
                    f.attrs["evaluation_only"] = True
                    f.attrs["purpose"] = (
                        "Frozen representation/control scoring only; prohibited for training or fitting latent codebooks"
                    )
                    f.attrs["source_url"] = "https://freds72.itch.io/poom"
                (out / f"episode_{ep:03}_labels.json").write_text(
                    json.dumps(rows, indent=2)
                )
                Image.fromarray(np.concatenate(preview, axis=0)).save(
                    out / f"episode_{ep:03}.png"
                )
                all_rows.append(
                    {
                        "episode": ep,
                        "trials": len(rows),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
                (out / "manifest.json").write_text(
                    json.dumps(
                        {
                            "config": vars(a),
                            "episodes": all_rows,
                            "split": "Episodes 0/1 calibrate an evaluation-only code-to-control probe; episodes 2/3 score it. No encoder, codebook or dynamics updates.",
                            "limitations": "Different live states, not matched simulator snapshots; collisions, NPCs and death can make button effects ambiguous. These must remain in the reported raw score.",
                            "attribution": "POOM by freds72 and Paranoid Cactus; game assets CC BY-NC 4.0",
                        },
                        indent=2,
                    )
                )
                print(json.dumps(all_rows[-1]), flush=True)
                page.close()
        finally:
            browser.close()


if __name__ == "__main__":
    main()
