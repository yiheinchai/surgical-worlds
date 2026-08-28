#!/usr/bin/env python3
"""Generate a synthetic laparoscopic demo video for pipeline testing without licensed data."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def draw_laparoscopic_scene(
    frame_idx: int,
    width: int,
    height: int,
    tool_x: float,
    tool_y: float,
    tool_angle: float,
) -> np.ndarray:
    """Render a stylized laparoscopic viewport with tissue and instrument."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Tissue background gradient (reddish-brown)
    yy, xx = np.mgrid[0:height, 0:width]
    tissue = (40 + 30 * np.sin(xx / 40.0 + frame_idx * 0.02)).astype(np.uint8)
    tissue2 = (25 + 20 * np.cos(yy / 35.0 - frame_idx * 0.015)).astype(np.uint8)
    img[:, :, 0] = np.clip(tissue + 20, 0, 255)
    img[:, :, 1] = np.clip(tissue2, 0, 255)
    img[:, :, 2] = np.clip(tissue // 3, 0, 255)

    # Vignette / circular endoscopic viewport
    cx, cy = width // 2, height // 2
    radius = min(width, height) // 2 - 4
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, thickness=-1)
    img[mask == 0] = 0

    # Fatty tissue blobs
    for i, (bx, by, br) in enumerate([(120, 100, 35), (400, 280, 45), (250, 350, 30)]):
        offset = int(8 * np.sin(frame_idx * 0.05 + i))
        cv2.circle(img, (bx + offset, by - offset), br, (90, 55, 45), thickness=-1)

    # Grasping forceps (two jaws)
    tx = int(tool_x * width)
    ty = int(tool_y * height)
    jaw_len = 60
    dx = int(jaw_len * np.cos(tool_angle))
    dy = int(jaw_len * np.sin(tool_angle))
    cv2.line(img, (tx, ty), (tx + dx, ty + dy), (200, 200, 210), 4)
    cv2.line(img, (tx, ty), (tx + dx - 8, ty + dy + 12), (200, 200, 210), 3)
    cv2.line(img, (tx, ty), (tx + dx - 8, ty + dy - 12), (200, 200, 210), 3)
    cv2.circle(img, (tx, ty), 6, (180, 180, 190), thickness=-1)

    # Specular highlight (common in laparoscopy)
    cv2.ellipse(img, (cx + 40, cy - 30), (25, 12), -20, 0, 360, (255, 255, 255), -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def generate_demo_video(output_path: Path, num_frames: int, fps: int, width: int, height: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Simulate instrument trajectory through tissue
    for t in range(num_frames):
        phase = t / max(1, num_frames - 1)
        tool_x = 0.35 + 0.35 * np.sin(phase * 2 * np.pi)
        tool_y = 0.45 + 0.15 * np.cos(phase * 3 * np.pi)
        tool_angle = -0.5 + 0.8 * np.sin(phase * 4 * np.pi)
        frame_rgb = draw_laparoscopic_scene(t, width, height, tool_x, tool_y, tool_angle)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    print(f"Generated {num_frames}-frame demo video → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic laparoscopic demo video")
    parser.add_argument("--output", type=str, default="data/surgical/demo/laparoscopic_demo.mp4")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    generate_demo_video(Path(args.output), args.frames, args.fps, args.width, args.height)


if __name__ == "__main__":
    main()
