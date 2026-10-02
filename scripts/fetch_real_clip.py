"""Download the real bird clip behind docs/demo-real.gif and the real-clip test.

    python scripts/fetch_real_clip.py            # -> data/samples/real/hummingbird_feeder.mp4

A hummingbird flies in to a feeder, perches about 8 s and flies off, filmed
from a static camera (Pixabay, ZacharyCrespin; source and license in
data/samples/real/CREDITS.md). The 1920x1080 rendition is downloaded, checked
against SHA256, and cut to TRIM with ffmpeg stream copy (no re-encode; the
audio track is dropped). The clip is gitignored, not committed: the Pixabay
Content License does not allow redistributing it unchanged, and it is 10.7 MB.

An existing output file is kept; pass --force to fetch it again. ffmpeg comes
from imageio-ffmpeg (requirements-dev.txt).
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
PAGE = "https://pixabay.com/videos/hummingbird-bird-feeder-landing-110462/"
URL = "https://cdn.pixabay.com/video/2022/03/12/110462-689510229_small.mp4"
AUTHOR = "ZacharyCrespin"
SHA256 = "12096e93a385436ae9a07ed185267b98f212f390653667234d3c969a65008651"
TRIM = (0.0, 16.6)          # seconds kept: the whole clip (empty 3 s, visit, empty 2.6 s)
OUT = ROOT / "data" / "samples" / "real" / "hummingbird_feeder.mp4"
USER_AGENT = "CatchThatBird/fetch_real_clip (github.com/SamJ12138/CatchThatBird)"


class FetchError(Exception):
    """Ends the script with exit code 1 and a one-line message."""


def download(url: str, dest: Path, attempts: int = 4) -> None:
    """GET `url` into `dest`; retries on HTTP 429 and 5xx (2, 4, 8 s)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=60) as r, dest.open("wb") as f:
                while chunk := r.read(1 << 16):
                    f.write(chunk)
            return
        except urllib.error.HTTPError as e:
            if (e.code != 429 and e.code < 500) or attempt == attempts - 1:
                raise FetchError(f"download failed: HTTP {e.code} for {url}") from e
        except urllib.error.URLError as e:
            if attempt == attempts - 1:
                raise FetchError(f"download failed: {e.reason} for {url}") from e
        time.sleep(2 ** (attempt + 1))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as e:
        raise FetchError("imageio-ffmpeg is missing: pip install -r requirements-dev.txt") from e
    return imageio_ffmpeg.get_ffmpeg_exe()


def trim_copy(src: Path, dst: Path, start: float, end: float) -> None:
    """`src` from `start` to `end` seconds into `dst`: video stream copied, audio dropped."""
    cmd = [ffmpeg_exe(), "-v", "error", "-y", "-ss", f"{start:g}", "-to", f"{end:g}",
           "-i", str(src), "-map", "0:v:0", "-c", "copy", "-an", str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FetchError(f"ffmpeg failed: {result.stderr.strip()[:300]}")


def fetch(out: Path = OUT, *, url: str = URL, sha256: str = SHA256, trim_range=TRIM,
          download: Optional[Callable[[str, Path], None]] = None,
          trim: Callable[[Path, Path, float, float], None] = trim_copy,
          force: bool = False) -> Path:
    """Download, verify and trim into `out`, unless it is already there."""
    if out.exists() and not force:
        return out
    download = download or globals()["download"]   # looked up now, so tests can patch it
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_name(out.stem + ".download.mp4")
    try:
        download(url, raw)
        got = sha256_of(raw)
        if got != sha256:
            raise FetchError(f"sha256 mismatch for {url}: got {got}, expected {sha256}")
        trim(raw, out, *trim_range)
    except BaseException:
        out.unlink(missing_ok=True)
        raise
    finally:
        raw.unlink(missing_ok=True)
    return out


def shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Download the real bird clip (Pixabay)")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--force", action="store_true", help="download again even if --out exists")
    args = p.parse_args(argv)
    existed = args.out.exists() and not args.force
    try:
        out = fetch(args.out, force=args.force)
    except FetchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if existed:
        print(f"{shown(out)} already exists (--force to fetch it again)")
    else:
        print(f"Wrote {shown(out)}: {out.stat().st_size / 1e6:.1f} MB, "
              f"{TRIM[0]:g}-{TRIM[1]:g} s of {URL} (sha256 verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
