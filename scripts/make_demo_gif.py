"""Render docs/demo.gif from the pipeline: the input clip and the annotated
preview side by side, around the first visit.

    python scripts/make_demo_gif.py                 # data/samples clip -> docs/demo.gif
    python scripts/make_demo_gif.py --source clip.mp4 --annotated annotated.mp4 \\
        --events events.jsonl --out demo.gif        # reuse an earlier run

Source clip: data/samples/demo.mp4 if it exists, else synth_bird.mp4
(made with scripts/make_synth_video.py if missing). Without --annotated and
--events, main.py runs on the clip (--headless --annotate-out) in a
temporary directory, so data/events.jsonl and data/snapshots are untouched.
The run is paced at the clip's frame rate, like a camera, so the HUD shows
live numbers. A paced run skips frames while YOLO is busy; for those, both
panels show the last processed frame (the annotated video's .frames.json
lists which ones were processed), as the live preview would.

The GIF covers 2 s before the first visit's `ts` to 2 s after its
`last_seen`, read from events.jsonl. The clip position of `ts` is
(frame_seq - 1) / fps, the media time the pipeline stamped. If that is longer
than 12 s, the perch (first to last confirmation) plays faster, with an "Nx"
label, so the GIF fits; landing and take-off stay at 1x.

Encoding: ffmpeg from imageio-ffmpeg (requirements-dev.txt), palettegen then
paletteuse. Tries 480 px panels at 10 fps, then 8 fps, then 400 px at 8 fps,
and keeps the first result of 5 MB or less (else the smallest one up to 8 MB).
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples"
MARGIN_S = 2.0
MAX_SECONDS = 12.0
TARGET_BYTES = 5 * 1024 * 1024
HARD_LIMIT_BYTES = 8 * 1024 * 1024
LADDER = ((480, 10), (480, 8), (400, 8))   # (panel width px, fps), in order of preference
CAPTION_PX = 24


class DemoError(Exception):
    """Ends the script with exit code 1 and a one-line message."""


@dataclass(frozen=True)
class Segment:
    """Clip seconds: GIF start, first and last confirmation, GIF end."""
    start: float
    first: float
    last: float
    end: float


@dataclass(frozen=True)
class OutFrame:
    t: float                  # clip seconds of the source frame shown
    label: Optional[str]      # speed label ("2x") while sped up


def choose_source(samples: Path = SAMPLES) -> Path:
    demo = samples / "demo.mp4"
    return demo if demo.exists() else samples / "synth_bird.mp4"


def first_visit(events_path: Path) -> dict[str, Any]:
    """The visit with the earliest `ts` (lines are in close order)."""
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if not events:
        raise DemoError(f"no visit in {events_path}: the pipeline logged nothing to show")
    return min(events, key=lambda e: datetime.fromisoformat(e["ts"]))


def segment(event: dict[str, Any], fps: float, clip_seconds: float,
            margin: float = MARGIN_S) -> Segment:
    first = (event["frame_seq"] - 1) / fps
    stay = (datetime.fromisoformat(event["last_seen"])
            - datetime.fromisoformat(event["ts"])).total_seconds()
    last = first + stay
    return Segment(max(0.0, first - margin), first, last, min(clip_seconds, last + margin))


def plan(seg: Segment, fps: float, max_seconds: float = MAX_SECONDS) -> tuple[list[OutFrame], int]:
    """Output frames and the perch speed-up factor (smallest integer that fits)."""
    edges = (seg.first - seg.start) + (seg.end - seg.last)
    perch = seg.last - seg.first
    if edges >= max_seconds:
        raise DemoError(f"landing and take-off alone take {edges:.1f} s, over {max_seconds:g} s")
    factor = max(1, math.ceil(perch / (max_seconds - edges) - 1e-9))
    label = f"{factor}x" if factor > 1 else None
    frames: list[OutFrame] = []
    step = 1.0 / fps
    eps = 1e-9

    def run(t0: float, t1: float, speed: int, lab: Optional[str]) -> None:
        n = math.ceil((t1 - t0) / (step * speed) - eps)
        frames.extend(OutFrame(round(t0 + i * step * speed, 6), lab) for i in range(n))

    run(seg.start, seg.first, 1, None)
    run(seg.first, seg.last, factor, label)
    run(seg.last, seg.end, 1, None)
    return frames, factor


def annotated_seqs(annotated: Path) -> Optional[list[int]]:
    """Frame seqs of the annotated video's frames, from main.py's sidecar
    (None without one: then frame i is source frame i)."""
    sidecar = annotated.with_name(annotated.name + ".frames.json")
    if not sidecar.exists():
        return None
    return json.loads(sidecar.read_text(encoding="utf-8"))["frame_seqs"]


def align(indices: list[int], seqs: Optional[list[int]]) -> list[tuple[int, int]]:
    """(source frame index, annotated frame index) to show for each wanted
    source frame index: the last processed frame at or before it."""
    if seqs is None:
        return [(i, i) for i in indices]
    out = []
    for i in indices:
        k = max(0, bisect.bisect_right(seqs, i + 1) - 1)   # seq = index + 1
        out.append((seqs[k] - 1, k))
    return out


def read_frames(path: Path, indices: list[int], width: int) -> dict[int, np.ndarray]:
    """Frames at `indices` (ascending, repeats allowed), resized to `width`."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise DemoError(f"cannot open {path}")
    wanted = set(indices)
    out: dict[int, np.ndarray] = {}
    i = 0
    try:
        while i <= max(wanted):
            ok, img = cap.read()
            if not ok:
                break
            if i in wanted:
                h = round(img.shape[0] * width / img.shape[1])
                out[i] = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
            i += 1
    finally:
        cap.release()
    missing = wanted - set(out)
    if missing:
        raise DemoError(f"{path} has no frame {min(missing)} (it ends after {i} frames)")
    return out


def caption(panel: np.ndarray, title: str, label: Optional[str]) -> np.ndarray:
    bar = np.full((CAPTION_PX, panel.shape[1], 3), 32, dtype=np.uint8)
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.5
    cv2.putText(bar, title, (8, 17), font, scale, (235, 235, 235), 1, cv2.LINE_AA)
    if label:
        (w, _), _ = cv2.getTextSize(label, font, scale, 1)
        cv2.putText(bar, label, (panel.shape[1] - w - 8, 17), font, scale, (0, 215, 255), 1,
                    cv2.LINE_AA)
    return np.vstack([bar, panel])


def compose(left: np.ndarray, right: np.ndarray, label: Optional[str]) -> np.ndarray:
    return np.hstack([caption(left, "input", label), caption(right, "detection", label)])


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as e:
        raise DemoError("imageio-ffmpeg is missing: pip install -r requirements-dev.txt") from e
    return imageio_ffmpeg.get_ffmpeg_exe()


def encode_gif(frames_dir: Path, fps: int, out: Path) -> None:
    """Two passes: an optimal 256-colour palette for the whole clip, then the
    frames mapped onto it. -loop 0 repeats forever."""
    ff = ffmpeg_exe()
    pattern = str(frames_dir / "f%05d.png")
    palette = frames_dir / "palette.png"
    for cmd in (
        [ff, "-v", "error", "-y", "-framerate", str(fps), "-i", pattern,
         "-vf", "palettegen=stats_mode=full", str(palette)],
        [ff, "-v", "error", "-y", "-framerate", str(fps), "-i", pattern, "-i", str(palette),
         "-lavfi", "paletteuse=dither=sierra2_4a", "-loop", "0", str(out)],
    ):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise DemoError(f"ffmpeg failed: {result.stderr.strip()[:500]}")


def render(source: Path, annotated: Path, frames: list[OutFrame], clip_fps: float,
           width: int, fps: int, out: Path, work: Path) -> tuple[int, int]:
    """Write the GIF; returns its (width, height)."""
    pairs = align([round(f.t * clip_fps) for f in frames], annotated_seqs(annotated))
    left = read_frames(source, [s for s, _ in pairs], width)
    right = read_frames(annotated, [a for _, a in pairs], width)
    frames_dir = work / f"frames_{width}_{fps}"
    frames_dir.mkdir()
    for n, (f, (s, a)) in enumerate(zip(frames, pairs)):
        image = compose(left[s], right[a], f.label)
        cv2.imwrite(str(frames_dir / f"f{n:05d}.png"), image)
    encode_gif(frames_dir, fps, out)
    return image.shape[1], image.shape[0]


def pipeline_command(source: Path, work: Path) -> list[str]:
    """main.py on `source`, writing the annotated video, events and logs under `work`."""
    return [sys.executable, str(ROOT / "main.py"), "--source", str(source), "--headless",
            "--yes", "--annotate-out", str(work / "annotated.mp4"),
            "--data-root", str(work), "--log-dir", str(work / "logs")]


def run_pipeline(source: Path, work: Path) -> tuple[Path, Path]:
    print(f"Running the pipeline on {source} ...", flush=True)
    result = subprocess.run(pipeline_command(source, work), capture_output=True, text=True)
    if result.returncode != 0:
        raise DemoError(f"main.py exited {result.returncode}: "
                        f"{result.stderr.strip().splitlines()[-1:]}")
    return work / "annotated.mp4", work / "data" / "events.jsonl"


def clip_info(path: Path) -> tuple[float, float]:
    """(fps, seconds)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise DemoError(f"cannot open {path}")
    fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0 or n <= 0:
        raise DemoError(f"{path}: cannot read its frame rate or length")
    return fps, n / fps


def shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Render docs/demo.gif from the pipeline")
    p.add_argument("--source", type=Path, default=None,
                   help="input clip (default: data/samples/demo.mp4, else synth_bird.mp4)")
    p.add_argument("--annotated", type=Path, default=None,
                   help="annotated video of a run on --source, with its .frames.json "
                        "(default: run main.py)")
    p.add_argument("--events", type=Path, default=None,
                   help="events.jsonl of that run (required with --annotated)")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "demo.gif")
    args = p.parse_args(argv)

    try:
        if (args.annotated is None) != (args.events is None):
            raise DemoError("--annotated and --events go together")
        source = args.source or choose_source()
        if not source.exists() and source == SAMPLES / "synth_bird.mp4":
            subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synth_video.py")],
                           check=True)
        if not source.exists():
            raise DemoError(f"no source clip {source}")

        with tempfile.TemporaryDirectory(prefix="ctb_demo_") as tmp:
            work = Path(tmp)
            if args.annotated is None:
                annotated, events = run_pipeline(source, work)
            else:
                annotated, events = args.annotated, args.events
            clip_fps, clip_seconds = clip_info(source)
            seg = segment(first_visit(events), clip_fps, clip_seconds)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            results = []
            for width, fps in LADDER:
                frames, factor = plan(seg, fps)
                candidate = work / f"demo_{width}_{fps}.gif"
                size = render(source, annotated, frames, clip_fps, width, fps, candidate, work)
                results.append((candidate.stat().st_size, width, fps, size, frames, factor,
                                candidate))
                if candidate.stat().st_size <= TARGET_BYTES:
                    break
            nbytes, width, fps, (gw, gh), frames, factor, best = (
                results[-1] if results[-1][0] <= TARGET_BYTES else min(results, key=lambda r: r[0]))
            if nbytes > HARD_LIMIT_BYTES:
                raise DemoError(f"smallest GIF is {nbytes / 1e6:.1f} MB, over the 8 MB limit")
            args.out.write_bytes(best.read_bytes())
    except DemoError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    speed = f"perch {seg.first:.1f}-{seg.last:.1f} s at {factor}x" if factor > 1 else "all at 1x"
    print(f"Wrote {shown(args.out)}: {nbytes / 1e6:.2f} MB, {gw}x{gh}, {fps} fps, "
          f"{len(frames)} frames ({len(frames) / fps:.1f} s); {shown(source)} "
          f"{seg.start:.1f}-{seg.end:.1f} s, {speed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
