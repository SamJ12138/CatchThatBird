"""scripts/fetch_real_clip.py: the real bird clip is downloaded, checked against
its sha256 and trimmed, never committed (Pixabay's license forbids
redistributing it as is)."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import cv2
import pytest

from scripts import fetch_real_clip as fetch
from scripts.make_synth_video import make_video
from tests.conftest import ROOT

needs_git = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(), reason="not a git checkout")

PAYLOAD = b"not really an mp4, but the bytes the fake server returns"


def fake_download(payload: bytes, calls: list[str]):
    def download(url: str, dest: Path) -> None:
        calls.append(url)
        dest.write_bytes(payload)
    return download


def copy_trim(src: Path, dst: Path, start: float, end: float) -> None:
    shutil.copyfile(src, dst)


def test_a_good_download_is_trimmed_into_place(tmp_path) -> None:
    out = tmp_path / "real" / "clip.mp4"
    calls: list[str] = []
    got = fetch.fetch(out, url="https://example.invalid/clip.mp4",
                      sha256=hashlib.sha256(PAYLOAD).hexdigest(),
                      download=fake_download(PAYLOAD, calls), trim=copy_trim)
    assert got == out and out.read_bytes() == PAYLOAD
    assert calls == ["https://example.invalid/clip.mp4"]
    assert sorted(p.name for p in out.parent.iterdir()) == ["clip.mp4"]   # no leftovers


def test_a_wrong_checksum_leaves_nothing_behind(tmp_path) -> None:
    out = tmp_path / "real" / "clip.mp4"
    with pytest.raises(fetch.FetchError, match="sha256"):
        fetch.fetch(out, url="u", sha256="0" * 64, download=fake_download(PAYLOAD, []),
                    trim=copy_trim)
    assert not out.exists()
    assert list(out.parent.iterdir()) == []


def test_an_existing_clip_is_not_downloaded_again(tmp_path) -> None:
    out = tmp_path / "clip.mp4"
    out.write_bytes(b"already here")
    calls: list[str] = []
    assert fetch.fetch(out, url="u", sha256="0" * 64, download=fake_download(PAYLOAD, calls),
                       trim=copy_trim) == out
    assert calls == [] and out.read_bytes() == b"already here"


def test_main_reports_a_failure_in_one_line(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(fetch, "download", fake_download(PAYLOAD, []))
    assert fetch.main(["--out", str(tmp_path / "clip.mp4")]) == 1
    err = capsys.readouterr().err.strip()
    assert err.startswith("error:") and "sha256" in err and "\n" not in err


def test_trim_copies_the_video_stream_without_re_encoding(tmp_path) -> None:
    src = make_video(tmp_path / "src.mp4", seconds=4, width=320, height=180, bird=False)
    dst = tmp_path / "trimmed.mp4"
    fetch.trim_copy(src, dst, 0.0, 2.0)
    cap = cv2.VideoCapture(str(dst))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert fps == 30 and 55 <= n <= 65          # 2 s at 30 fps


def test_the_clip_is_full_hd_and_long_enough() -> None:
    start, end = fetch.TRIM
    assert 15 <= end - start <= 60
    assert fetch.URL.startswith("https://cdn.pixabay.com/video/") and fetch.URL.endswith(".mp4")
    assert len(fetch.SHA256) == 64
    assert fetch.OUT == ROOT / "data" / "samples" / "real" / fetch.OUT.name


def test_credits_name_the_source_author_license_and_trim() -> None:
    credits = (fetch.OUT.parent / "CREDITS.md").read_text(encoding="utf-8")
    start, end = fetch.TRIM
    for needle in (fetch.PAGE, fetch.URL, fetch.AUTHOR, "Pixabay Content License",
                   "https://pixabay.com/service/license-summary/", fetch.SHA256,
                   f"{start:g}", f"{end:g}", "python scripts/fetch_real_clip.py"):
        assert needle in credits, needle


@needs_git
def test_the_clip_is_ignored_and_its_credits_are_not() -> None:
    def ignored(path: str) -> bool:
        return subprocess.run(["git", "check-ignore", "-q", "--no-index", path],
                              cwd=ROOT).returncode == 0
    assert ignored(fetch.OUT.relative_to(ROOT).as_posix())
    assert not ignored("data/samples/real/CREDITS.md")
