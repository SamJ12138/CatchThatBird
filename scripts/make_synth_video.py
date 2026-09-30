"""Generate a synthetic test clip: static grey background with a small dark
blob moving back and forth across the car ROI from data/roi.json.

YOLO will not call the blob a bird. The clip exists to push every pipeline
stage (MOG2 -> gate -> contours -> crop -> YOLO) through a run of
`main.py --source <clip> --headless`.

    python scripts/make_synth_video.py [--out data/samples/synth_blob.mp4]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=ROOT / "data" / "samples" / "synth_blob.mp4")
    p.add_argument("--roi", type=Path, default=ROOT / "data" / "roi.json")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--passes", type=int, default=3, help="left-right sweeps across the ROI")
    p.add_argument("--noise", type=float, default=2.0, help="per-pixel Gaussian noise sigma")
    args = p.parse_args()

    # ROI is stored in the coordinates of the frame it was drawn on; scale it.
    roi = json.loads(args.roi.read_text())
    sx = args.width / roi["frame_width"]
    sy = args.height / roi["frame_height"]
    rx, ry = int(roi["x"] * sx), int(roi["y"] * sy)
    rw, rh = int(roi["w"] * sx), int(roi["h"] * sy)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    if not writer.isOpened():
        raise SystemExit(f"cv2.VideoWriter could not open {args.out}")

    rng = np.random.default_rng(0)
    background = np.full((args.height, args.width, 3), 128, dtype=np.uint8)
    axes = (14, 9)  # ellipse half-axes: ~400 px area, above motion_min_area=200
    margin = axes[0] + 4
    n_frames = int(args.seconds * args.fps)

    for i in range(n_frames):
        t = i / n_frames
        # Triangle wave across the ROI width, gentle sine bob vertically.
        phase = (t * args.passes) % 1.0
        u = 2 * phase if phase < 0.5 else 2 * (1 - phase)
        cx = int(rx + margin + u * (rw - 2 * margin))
        cy = int(ry + rh / 2 + (rh / 4) * math.sin(2 * math.pi * 2 * t))

        frame = background.copy()
        if args.noise > 0:
            noise = rng.normal(0, args.noise, frame.shape)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        cv2.ellipse(frame, (cx, cy), axes, 0, 0, 360, (40, 40, 40), -1, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    print(
        f"Wrote {args.out} ({n_frames} frames, {args.width}x{args.height}@{args.fps}); "
        f"blob sweeps ROI x={rx} y={ry} w={rw} h={rh} (scaled from "
        f"{roi['frame_width']}x{roi['frame_height']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
