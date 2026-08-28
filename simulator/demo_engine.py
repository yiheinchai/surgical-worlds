"""
Demo surgery engine — playable immediately without trained checkpoints.

Uses a lightweight physics state (instrument x/y) over a synthetic laparoscopic
scene so users can experience Genie 3-style interaction while models train.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from simulator.engine import StepResult


@dataclass
class InstrumentState:
    x: float = 0.45
    y: float = 0.50
    jaw_open: float = 1.0  # 1=open, 0=closed
    camera_pan: float = 0.0


# Action effects on instrument state (dx, dy, jaw_delta, camera_delta)
ACTION_DELTAS = {
    0: (0.0, 0.0, 0.0, 0.0),       # idle
    1: (0.0, 0.0, -0.25, 0.0),     # grasp
    2: (0.0, 0.0, 0.25, 0.0),      # release
    3: (-0.04, 0.0, 0.0, 0.0),     # left
    4: (0.04, 0.0, 0.0, 0.0),      # right
    5: (0.0, -0.04, 0.0, 0.0),     # up
    6: (0.0, 0.04, 0.0, 0.0),      # down
    7: (0.0, 0.0, 0.0, 0.08),      # camera
}


class DemoSurgeryEngine:
    """
    Immediate-play demo simulator.
    Renders synthetic laparoscopic frames steered by user actions.
  Switch to SurgeryWorldEngine after training for real world-model generation.
    """

    def __init__(self, resolution: int = 512, surgery_type: str = "laparoscopic"):
        self.resolution = resolution
        self.surgery_type = surgery_type
        self.state = InstrumentState()
        self.step_count = 0
        self.all_frames: List[Image.Image] = []
        self.n_actions = 8
        self._t = 0

    def reset(self, seed_index: Optional[int] = None) -> Image.Image:
        self.state = InstrumentState(
            x=0.35 + 0.1 * (seed_index or 0) % 3,
            y=0.45,
        )
        self.step_count = 0
        self.all_frames = []
        self._t = 0
        frame = self._render()
        self.all_frames.append(frame)
        return frame

    def _render(self) -> Image.Image:
        """Render synthetic laparoscopic viewport at current instrument state."""
        w = h = self.resolution
        img = Image.new("RGB", (w, h), (15, 8, 8))
        draw = ImageDraw.Draw(img)

        # Tissue field
        for i in range(12):
            bx = int(w * (0.15 + 0.65 * ((i * 37) % 100) / 100))
            by = int(h * (0.15 + 0.65 * ((i * 53) % 100) / 100))
            br = 25 + (i * 7) % 30
            color = (70 + i * 5, 35 + i * 2, 30)
            draw.ellipse([bx - br, by - br, bx + br, by + br], fill=color)

        # Specular highlight
        draw.ellipse([w // 2 - 30, h // 4, w // 2 + 30, h // 4 + 20], fill=(220, 200, 180))

        # Circular viewport mask
        mask = Image.new("L", (w, h), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse([8, 8, w - 8, h - 8], fill=255)

        # Instrument
        tx = int(self.state.x * w)
        ty = int(self.state.y * h)
        jaw = self.state.jaw_open
        angle = -0.4 + self.state.camera_pan
        jaw_spread = int(18 * jaw)

        dx = int(70 * math.cos(angle))
        dy = int(70 * math.sin(angle))
        draw.line([(tx, ty), (tx + dx, ty + dy)], fill=(210, 210, 220), width=6)
        draw.line([(tx, ty), (tx + dx - jaw_spread, ty + dy + 14)], fill=(200, 200, 210), width=4)
        draw.line([(tx, ty), (tx + dx + jaw_spread, ty + dy - 14)], fill=(200, 200, 210), width=4)
        draw.ellipse([tx - 8, ty - 8, tx + 8, ty + 8], fill=(180, 180, 190))

        # Apply circular mask
        bg = Image.new("RGB", (w, h), (0, 0, 0))
        bg.paste(img, mask=mask)

        # HUD overlay
        hud = ImageDraw.Draw(bg)
        hud.text((16, h - 28), f"Demo mode | step {self.step_count}", fill=(180, 180, 200))
        return bg

    def step(self, action_id: int) -> StepResult:
        t0 = time.perf_counter()
        action_id = max(0, min(action_id, self.n_actions - 1))
        dx, dy, jaw_d, cam_d = ACTION_DELTAS.get(action_id, (0, 0, 0, 0))

        self.state.x = float(np.clip(self.state.x + dx, 0.12, 0.88))
        self.state.y = float(np.clip(self.state.y + dy, 0.12, 0.88))
        self.state.jaw_open = float(np.clip(self.state.jaw_open + jaw_d, 0.0, 1.0))
        self.state.camera_pan = float(np.clip(self.state.camera_pan + cam_d, -0.6, 0.6))
        self._t += 1
        self.step_count += 1

        frame = self._render()
        self.all_frames.append(frame)

        return StepResult(
            frame=frame,
            action_id=action_id,
            step_index=self.step_count,
            latency_ms=(time.perf_counter() - t0) * 1000,
            session_frames=len(self.all_frames),
        )

    def get_session_gif_frames(self) -> List[Image.Image]:
        return list(self.all_frames)

    @property
    def is_ready(self) -> bool:
        return self.step_count >= 0
