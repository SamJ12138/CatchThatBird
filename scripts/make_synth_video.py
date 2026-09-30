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
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def scaled_roi(roi_path: Path, width: int, height: int) -> tuple[int, int, int, int]:
    """ROI from roi_path, scaled from the frame it was drawn on to width x height."""
    roi = json.loads(roi_path.read_text())
    sx = width / roi["frame_width"]
    sy = height / roi["frame_height"]
    return (int(roi["x"] * sx), int(roi["y"] * sy),
            int(roi["w"] * sx), int(roi["h"] * sy))


def make_video(
    out: Path,
    *,
    seconds: float = 20.0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    passes: int = 3,
    noise: float = 2.0,
    roi: Optional[tuple[int, int, int, int]] = None,
) -> Path:
    """Write the clip to `out`. `roi` (x, y, w, h) in output-frame pixels is
    the region the blob sweeps; default is the centre half of the frame."""
    rx, ry, rw, rh = roi if roi is not None else (width // 4, height // 4, width // 2, height // 2)

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {out}")

    rng = np.random.default_rng(0)
    background = np.full((height, width, 3), 128, dtype=np.uint8)
    axes = (14, 9)  # ellipse half-axes: ~400 px area, above motion_min_area=200
    margin = axes[0] + 4
    n_frames = int(seconds * fps)

    for i in range(n_frames):
        t = i / n_frames
        # Triangle wave across the ROI width, gentle sine bob vertically.
        phase = (t * passes) % 1.0
        u = 2 * phase if phase < 0.5 else 2 * (1 - phase)
        cx = int(rx + margin + u * (rw - 2 * margin))
        cy = int(ry + rh / 2 + (rh / 4) * math.sin(2 * math.pi * 2 * t))

        frame = background.copy()
        if noise > 0:
            jitter = rng.normal(0, noise, frame.shape)
            frame = np.clip(frame.astype(np.float32) + jitter, 0, 255).astype(np.uint8)
        cv2.ellipse(frame, (cx, cy), axes, 0, 0, 360, (40, 40, 40), -1, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    return out


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
    roi = scaled_roi(args.roi, args.width, args.height)
    make_video(
        args.out, seconds=args.seconds, width=args.width, height=args.height,
        fps=args.fps, passes=args.passes, noise=args.noise, roi=roi,
    )
    n_frames = int(args.seconds * args.fps)
    rx, ry, rw, rh = roi
    print(
        f"Wrote {args.out} ({n_frames} frames, {args.width}x{args.height}@{args.fps}); "
        f"blob sweeps ROI x={rx} y={ry} w={rw} h={rh} (scaled from {args.roi.name})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
