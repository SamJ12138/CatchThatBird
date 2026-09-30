"""Generate a synthetic test clip over a noisy grey background, inside the car
ROI from data/roi.json (or data/roi.example.json on a fresh clone).

Default (--bird): a photo of a real house sparrow (public domain, see
data/samples/assets/CREDITS.md) flies into the ROI, perches for about 13 s
and flies off, so the real YOLOv8n logs one visit. Output: data/samples/synth_bird.mp4.

--no-bird: a small dark blob sweeps back and forth instead. YOLO never calls
it a bird; it pushes every pipeline stage through a run without producing an
event. Output: data/samples/synth_blob.mp4.

    python scripts/make_synth_video.py [--no-bird] [--out PATH]
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
BIRD_IMAGE = ROOT / "data" / "samples" / "assets" / "house_sparrow_female.png"
SPRITE_ROI_FRACTION = 0.25  # sprite width as a fraction of the ROI width

# (seconds, x, y): the sprite's top-left corner, as fractions of the free
# space inside the ROI (0 = left/top edge, 1 = right/bottom edge). Linear in
# between; absent before the first and after the last keyframe. The first
# 3 s are empty so MOG2 (60-frame warm-up) learns the plain background.
# The flights are short and fall between the 1 Hz gated frames (t = 2, 3,
# 4 ... s with the default config), so the clip is one visit.
BIRD_KEYFRAMES = (
    (3.2, 0.00, 0.10),   # flies in from the top left
    (3.7, 0.35, 0.70),   # lands
    (17.1, 0.35, 0.70),  # perched 13.4 s
    (17.6, 1.00, 0.00),  # flies out at the top right, then gone
)
# The perched bird only bobs by a couple of pixels. MOG2 absorbs a bird that
# holds that still within about a second, so it passes the motion gate only
# right after landing: the visit is logged, but visit_frames and last_seen
# understate how long it stayed (docs/observations.md, P9).
BOB_PX = 2  # perched birds are not perfectly still


def default_roi_path(data_dir: Path = ROOT / "data") -> Path:
    """data/roi.json if this machine has one, else the tracked example."""
    own = data_dir / "roi.json"
    return own if own.exists() else data_dir / "roi.example.json"


def default_out(bird: bool) -> Path:
    return ROOT / "data" / "samples" / ("synth_bird.mp4" if bird else "synth_blob.mp4")


def scaled_roi(roi_path: Path, width: int, height: int) -> tuple[int, int, int, int]:
    """ROI from roi_path, scaled from the frame it was drawn on to width x height."""
    roi = json.loads(roi_path.read_text())
    sx = width / roi["frame_width"]
    sy = height / roi["frame_height"]
    return (int(roi["x"] * sx), int(roi["y"] * sy),
            int(roi["w"] * sx), int(roi["h"] * sy))


def sprite_size(roi: tuple[int, int, int, int]) -> tuple[int, int]:
    """(w, h) of the bird sprite for this ROI, keeping the photo's aspect ratio."""
    h0, w0 = cv2.imread(str(BIRD_IMAGE), cv2.IMREAD_UNCHANGED).shape[:2]
    w = max(8, round(roi[2] * SPRITE_ROI_FRACTION))
    return w, round(w * h0 / w0)


def bird_position(t: float, roi: tuple[int, int, int, int],
                  sprite_wh: tuple[int, int]) -> Optional[tuple[int, int]]:
    """Top-left corner of the sprite at `t` seconds, or None when it is not in view."""
    if t < BIRD_KEYFRAMES[0][0] or t > BIRD_KEYFRAMES[-1][0]:
        return None
    for (t0, x0, y0), (t1, x1, y1) in zip(BIRD_KEYFRAMES, BIRD_KEYFRAMES[1:]):
        if t0 <= t <= t1:
            u = (t - t0) / (t1 - t0)
            fx, fy = x0 + u * (x1 - x0), y0 + u * (y1 - y0)
            perched = (x0, y0) == (x1, y1)
            break
    rx, ry, rw, rh = roi
    sw, sh = sprite_wh
    x = rx + round(fx * (rw - sw))
    y = ry + round(fy * (rh - sh - BOB_PX)) + BOB_PX
    if perched:
        y -= round(BOB_PX * abs(math.sin(2 * math.pi * 1.5 * t)))
    return x, y


def _composite(frame: np.ndarray, sprite: np.ndarray, x: int, y: int) -> None:
    h, w = sprite.shape[:2]
    alpha = sprite[:, :, 3:4].astype(np.float32) / 255.0
    region = frame[y:y + h, x:x + w].astype(np.float32)
    frame[y:y + h, x:x + w] = (sprite[:, :, :3] * alpha + region * (1 - alpha)).astype(np.uint8)


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
    bird: bool = True,
) -> Path:
    """Write the clip to `out`. `roi` (x, y, w, h) in output-frame pixels is
    where the bird (or the blob) moves; default is the centre half of the frame."""
    rx, ry, rw, rh = roi if roi is not None else (width // 4, height // 4, width // 2, height // 2)

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {out}")

    # cv2.randn is ~10x faster than numpy's Gaussian generator; seeded, so the
    # clip is the same on every run.
    cv2.setRNGSeed(0)
    jitter = np.empty((height, width, 3), dtype=np.int16)
    background = np.full((height, width, 3), 128, dtype=np.uint8)
    axes = (14, 9)  # blob ellipse half-axes: ~400 px area, above motion_min_area=200
    margin = axes[0] + 4
    n_frames = int(seconds * fps)
    if bird:
        sprite_wh = sprite_size((rx, ry, rw, rh))
        sprite = cv2.resize(cv2.imread(str(BIRD_IMAGE), cv2.IMREAD_UNCHANGED), sprite_wh,
                            interpolation=cv2.INTER_AREA)

    for i in range(n_frames):
        frame = background.copy()
        if bird:
            t = i / fps
            pos = bird_position(t, (rx, ry, rw, rh), sprite_wh)
            if pos is not None:
                _composite(frame, sprite, *pos)
        else:
            # Triangle wave across the ROI width, gentle sine bob vertically.
            t = i / n_frames
            phase = (t * passes) % 1.0
            u = 2 * phase if phase < 0.5 else 2 * (1 - phase)
            cx = int(rx + margin + u * (rw - 2 * margin))
            cy = int(ry + rh / 2 + (rh / 4) * math.sin(2 * math.pi * 2 * t))
            cv2.ellipse(frame, (cx, cy), axes, 0, 0, 360, (40, 40, 40), -1, cv2.LINE_AA)
        if noise > 0:
            cv2.randn(jitter, 0, noise)
            frame = np.clip(frame.astype(np.int16) + jitter, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bird", action=argparse.BooleanOptionalAction, default=True,
                   help="a real bird photo (default) or, with --no-bird, a dark blob")
    p.add_argument("--out", type=Path, default=None,
                   help="output file (default: data/samples/synth_bird.mp4, "
                        "or synth_blob.mp4 with --no-bird)")
    p.add_argument("--roi", type=Path, default=None,
                   help="ROI file (default: data/roi.json, else data/roi.example.json)")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--passes", type=int, default=3, help="--no-bird: left-right sweeps across the ROI")
    p.add_argument("--noise", type=float, default=2.0, help="per-pixel Gaussian noise sigma")
    args = p.parse_args()
    if args.roi is None:
        args.roi = default_roi_path()
    if args.out is None:
        args.out = default_out(args.bird)

    # ROI is stored in the coordinates of the frame it was drawn on; scale it.
    roi = scaled_roi(args.roi, args.width, args.height)
    make_video(
        args.out, seconds=args.seconds, width=args.width, height=args.height,
        fps=args.fps, passes=args.passes, noise=args.noise, roi=roi, bird=args.bird,
    )
    n_frames = int(args.seconds * args.fps)
    rx, ry, rw, rh = roi
    what = ("a house sparrow photo lands, perches and leaves" if args.bird
            else "a dark blob sweeps")
    print(
        f"Wrote {args.out} ({n_frames} frames, {args.width}x{args.height}@{args.fps}); "
        f"{what} inside ROI x={rx} y={ry} w={rw} h={rh} (scaled from {args.roi.name})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
