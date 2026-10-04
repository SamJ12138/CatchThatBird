"""Download the real-footage corpus: data/samples/corpus/<name>/clip.mp4.

    python scripts/fetch_clips.py                 # every clip in CLIPS
    python scripts/fetch_clips.py car_mynas ...   # only these
    python scripts/fetch_clips.py --list          # names, cases, sources

Each clip is a stock video from a static or handheld camera (see the table in
data/samples/corpus/CREDITS.md). It is downloaded from the CDN link on its
file page, checked against SHA256, and cut: the video stream alone (audio
dropped), copied as it is, or, when the entry has a `trim`, re-encoded from
exactly `start` to `end` (libx264, CRF 18) so that the hand-written ground
truth's seconds are seconds of the output file. Nothing here is committed:
the Pexels License forbids redistributing the files unchanged. What is
committed next to each clip is its ground-truth.json, written by hand from one
frame per second (never from YOLO), and its roi.json.

An existing clip is kept; pass --force to fetch it again. ffmpeg comes from
imageio-ffmpeg (requirements-dev.txt).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:   # run as `python scripts/fetch_clips.py`
    sys.path.insert(0, str(ROOT))

from scripts.fetch_real_clip import FetchError, download, ffmpeg_exe, sha256_of  # noqa: E402
CORPUS = ROOT / "data" / "samples" / "corpus"
ALLOWED_LICENSES = {"Pexels License", "Pixabay Content License", "CC0", "Public domain",
                    "CC BY 4.0", "CC BY 3.0", "CC BY 2.0"}
CASES = {1: "bird on a car", 2: "two or more birds at once", 3: "repeated arrivals",
         4: "bird plus non-bird motion", 5: "low light or backlit", 6: "no bird"}

Trim = Optional[tuple[float, float]]


@dataclass(frozen=True)
class Clip:
    name: str
    page: str            # the file page the license was read on
    url: str             # the rendition downloaded
    sha256: str          # of the download, before cutting
    author: str
    license: str
    cases: tuple[int, ...]
    duration_s: float    # of the output clip
    trim: Trim = None    # (start, end) seconds of the download, or the whole video

    @property
    def out(self) -> Path:
        return CORPUS / self.name / "clip.mp4"


PEXELS = "https://www.pexels.com/video/"
FILES = "https://videos.pexels.com/video-files/"

CLIPS: list[Clip] = [
    Clip("car_mynas", PEXELS + "common-mynas-perched-on-car-roof-35835885/",
         FILES + "35835885/15195564_1080_1920_30fps.mp4",
         "aa2e7657f7218617358f8f16d9b69b759a5955a785cb0c5c75f4d7671912436c",
         "Baba Saleem", "Pexels License", (1, 2), 11.83),
    Clip("car_gull_windshield", PEXELS + "seagull-resting-on-car-windshield-by-the-seaside-29208177/",
         FILES + "29208177/12609740_1920_1080_30fps.mp4",
         "6cfd1f3dbb90dccbe40a429135bc3b1cd986496c40eed4be36062482985f46b1",
         "WeStarMoney Rec", "Pexels License", (1,), 7.94),
    Clip("sparrows_ground", PEXELS + "sparrows-eating-bread-crumbs-on-the-ground-5306060/",
         FILES + "5306060/5306060-hd_1920_1080_30fps.mp4",
         "82e92fd97a1b475816c063c0f76cf28279b5d30c516a35eaddb02b4ff1d84a0f",
         "Magda Ehlers", "Pexels License", (2, 3), 38.10),
    Clip("bird_bath", PEXELS + "birds-drinking-and-bathing-1172590/",
         FILES + "1172590/1172590-hd_1920_1080_30fps.mp4",
         "8f8f26d6673bd6d211cbf3b35723d527a5d085c9b478d8dd61a9e3065fdd110e",
         "David Clausen", "Pexels License", (2, 3), 60.03, (0.0, 60.0)),
    Clip("doves_rain", PEXELS + "birds-perched-on-tree-stems-on-a-rainy-day-4073692/",
         FILES + "4073692/4073692-hd_1920_1080_30fps.mp4",
         "6120809de724f9833c27fb9a5f51ff72e8aa239191a32882d591b34b2f3c643e",
         "martha chinyere", "Pexels License", (2,), 18.08),
    Clip("pigeon_stairs", PEXELS + "city-stairs-with-pedestrian-traffic-and-doves-29430215/",
         FILES + "29430215/12670194_1920_1080_30fps.mp4",
         "f9f75981c0828468fa6979d3929e3d5144ca5857180df830aefc965ee68bec2c",
         "Frank Litschel", "Pexels License", (2, 3, 4, 5), 25.13),
    Clip("silhouette_dusk", PEXELS + "silhouette-of-bird-on-tree-branch-at-dusk-32050158/",
         FILES + "32050158/13662104_1920_1080_50fps.mp4",
         "3d1c3d222d5c4289eadad37345bd3199dbad1cfacfb5881a709bc158ddfef7b8",
         "Dr Photographer", "Pexels License", (5,), 25.36),
    Clip("parked_car_rain", PEXELS + "a-car-parked-in-the-street-while-raining-3922856/",
         FILES + "3922856/3922856-hd_1920_1080_25fps.mp4",
         "2997944b296b5e003775a0fbc352b061341c4962a28025c2182a5c2bde99812d",
         "WeStarMoney Rec", "Pexels License", (5, 6), 10.48, (8.5, 19.0)),
]


def cut(src: Path, dst: Path, trim: Trim) -> None:
    """`src`'s video stream into `dst`, audio dropped: copied whole, or
    re-encoded from exactly trim[0] to trim[1] seconds."""
    cmd = [ffmpeg_exe(), "-v", "error", "-y", "-i", str(src)]
    if trim is None:
        cmd += ["-map", "0:v:0", "-c", "copy", "-an"]
    else:
        start, end = trim
        cmd += ["-ss", f"{start:g}", "-to", f"{end:g}", "-map", "0:v:0", "-an",
                "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]
    result = subprocess.run(cmd + [str(dst)], capture_output=True, text=True)
    if result.returncode != 0:
        raise FetchError(f"ffmpeg failed: {result.stderr.strip()[:300]}")


def fetch(clip: Clip, out: Optional[Path] = None, *,
          download: Optional[Callable[[str, Path], None]] = None,
          cut: Optional[Callable[[Path, Path, Trim], None]] = None,
          force: bool = False) -> Path:
    """Download, verify and cut `clip` into `out` (default clip.out), unless it exists."""
    out = out or clip.out
    if out.exists() and not force:
        return out
    download = download or globals()["download"]   # looked up now, so tests can patch it
    cut = cut or globals()["cut"]
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_name(out.stem + ".download.mp4")
    try:
        download(clip.url, raw)
        got = sha256_of(raw)
        if got != clip.sha256:
            raise FetchError(f"sha256 mismatch for {clip.url}: got {got}, expected {clip.sha256}")
        cut(raw, out, clip.trim)
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    finally:
        raw.unlink(missing_ok=True)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Download the real-footage corpus")
    p.add_argument("names", nargs="*", help="clips to fetch (default: all)")
    p.add_argument("--force", action="store_true", help="download again even if present")
    p.add_argument("--list", action="store_true", help="list the clips and exit")
    p.add_argument("--root", type=Path, default=None,
                   help="write <root>/<name>/clip.mp4 instead of data/samples/corpus")
    args = p.parse_args(argv)
    by_name = {clip.name: clip for clip in CLIPS}
    if args.list:
        for clip in CLIPS:
            cases = ", ".join(CASES[c] for c in clip.cases)
            print(f"{clip.name:22s} {clip.duration_s:5.1f} s  {cases}  ({clip.page})")
        return 0
    unknown = [n for n in args.names if n not in by_name]
    if unknown:
        print(f"error: unknown clip(s) {unknown}; see --list", file=sys.stderr)
        return 2
    failed = 0
    for clip in [by_name[n] for n in args.names] or CLIPS:
        out = (args.root / clip.name / "clip.mp4") if args.root else clip.out
        existed = out.exists() and not args.force
        try:
            fetch(clip, out, force=args.force)
        except FetchError as e:
            print(f"error: {clip.name}: {e}", file=sys.stderr)
            failed += 1
            continue
        state = "already there" if existed else f"{out.stat().st_size / 1e6:.1f} MB, sha256 verified"
        print(f"{clip.name}: {out} ({state})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
