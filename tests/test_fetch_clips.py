"""scripts/fetch_clips.py and data/samples/corpus/: the real-footage corpus.

Every clip is downloaded from its file page's CDN, checked against its sha256,
cut and stripped of audio; none is committed (the Pexels and Pixabay licenses
forbid redistributing the files as they are). What is committed is the hand-
written ground truth of each clip, its CREDITS row and (Batch B) its ROI."""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import fetch_clips as fc
from scripts.make_synth_video import make_video
from tests.conftest import ROOT

needs_git = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(), reason="not a git checkout")

PAYLOAD = b"bytes the fake CDN returns"
CASES = {1: "bird on a car", 2: "two or more birds at once", 3: "repeated arrivals",
         4: "bird plus non-bird motion", 5: "low light or backlit", 6: "no bird"}


def fake_download(calls: list[str]):
    def download(url: str, dest: Path) -> None:
        calls.append(url)
        dest.write_bytes(PAYLOAD)
    return download


def copy_cut(src: Path, dst: Path, trim) -> None:
    shutil.copyfile(src, dst)


def clip_for_test(**changes) -> fc.Clip:
    base = dict(name="test_clip", page="https://www.pexels.com/video/x-1/",
                url="https://videos.pexels.com/video-files/1/1.mp4",
                sha256=hashlib.sha256(PAYLOAD).hexdigest(), author="someone",
                license="Pexels License", cases=(6,), duration_s=10.0)
    base.update(changes)
    return fc.Clip(**base)


# ------------------------------------------------------------------ entries

def test_six_to_eight_clips_cover_every_case() -> None:
    assert 6 <= len(fc.CLIPS) <= 8
    covered = {case for clip in fc.CLIPS for case in clip.cases}
    assert covered == set(CASES), covered
    assert sum(1 in clip.cases for clip in fc.CLIPS) >= 2      # the brief prefers two on a car


def test_entries_are_well_formed() -> None:
    names = [clip.name for clip in fc.CLIPS]
    assert len(set(names)) == len(names)
    for clip in fc.CLIPS:
        assert re.fullmatch(r"[a-z][a-z0-9_]+", clip.name), clip.name
        assert re.fullmatch(r"[0-9a-f]{64}", clip.sha256), clip.name
        assert clip.page.startswith(("https://www.pexels.com/video/", "https://pixabay.com/videos/",
                                     "https://commons.wikimedia.org/wiki/File:")), clip.page
        assert clip.url.startswith(("https://videos.pexels.com/", "https://cdn.pixabay.com/",
                                    "https://upload.wikimedia.org/")), clip.url
        assert clip.license in fc.ALLOWED_LICENSES, clip.license
        assert set(clip.cases) <= set(CASES)
        if clip.trim is not None:
            start, end = clip.trim
            assert 0 <= start < end
        assert clip.out == ROOT / "data" / "samples" / "corpus" / clip.name / "clip.mp4"


def test_lengths_follow_the_brief() -> None:
    """10 to 90 s, shorter only for the cases the brief allows it (1 and 4)."""
    for clip in fc.CLIPS:
        assert clip.duration_s <= 90, clip.name
        if clip.duration_s < 10:
            assert set(clip.cases) & {1, 4}, clip.name


# ------------------------------------------------------------------ fetch

def test_a_good_download_is_cut_into_place(tmp_path) -> None:
    clip = clip_for_test()
    out = tmp_path / "corpus" / clip.name / "clip.mp4"
    calls: list[str] = []
    assert fc.fetch(clip, out, download=fake_download(calls), cut=copy_cut) == out
    assert out.read_bytes() == PAYLOAD and calls == [clip.url]
    assert sorted(p.name for p in out.parent.iterdir()) == ["clip.mp4"]   # no leftovers


def test_a_wrong_checksum_leaves_nothing_behind(tmp_path) -> None:
    clip = clip_for_test(sha256="0" * 64)
    out = tmp_path / clip.name / "clip.mp4"
    with pytest.raises(fc.FetchError, match="sha256"):
        fc.fetch(clip, out, download=fake_download([]), cut=copy_cut)
    assert list(out.parent.iterdir()) == []


def test_an_existing_clip_is_kept(tmp_path) -> None:
    clip = clip_for_test()
    out = tmp_path / "clip.mp4"
    out.write_bytes(b"already here")
    calls: list[str] = []
    fc.fetch(clip, out, download=fake_download(calls), cut=copy_cut)
    assert calls == [] and out.read_bytes() == b"already here"


def test_main_fetches_the_named_clips_and_reports_failures_in_one_line(
        tmp_path, capsys, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(fc, "download", fake_download(calls))
    monkeypatch.setattr(fc, "cut", copy_cut)
    name = fc.CLIPS[0].name
    assert fc.main(["--root", str(tmp_path), name]) == 1      # PAYLOAD is not the real file
    err = capsys.readouterr().err.strip()
    assert err.startswith(f"error: {name}:") and "sha256" in err and "\n" not in err
    assert calls == [fc.CLIPS[0].url]
    assert fc.main(["--root", str(tmp_path), "no_such_clip"]) == 2


def test_cut_without_trim_copies_the_video_and_drops_audio(tmp_path) -> None:
    src = make_video(tmp_path / "src.mp4", seconds=2, width=320, height=180, bird=False)
    dst = tmp_path / "out.mp4"
    fc.cut(src, dst, None)
    cap = cv2.VideoCapture(str(dst))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 60
    cap.release()


def test_cut_with_trim_is_frame_accurate(tmp_path) -> None:
    """A trimmed clip starts on the frame at `start`, not on the keyframe before
    it, so the ground truth's seconds are seconds of the output file."""
    src = make_video(tmp_path / "src.mp4", seconds=4, width=320, height=180, bird=False)
    dst = tmp_path / "out.mp4"
    fc.cut(src, dst, (1.0, 3.0))
    cap = cv2.VideoCapture(str(dst))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    ok, first = cap.read()
    cap.release()
    assert ok and 59 <= n <= 61
    src_cap = cv2.VideoCapture(str(src))
    frames = [src_cap.read()[1] for _ in range(40)]
    src_cap.release()
    diffs = [float(np.mean(cv2.absdiff(first, f))) for f in frames]
    assert int(np.argmin(diffs)) == 30, diffs


# ------------------------------------------------------------------ committed files

def test_credits_have_a_row_per_clip() -> None:
    credits = (fc.CORPUS / "CREDITS.md").read_text(encoding="utf-8")
    for clip in fc.CLIPS:
        for needle in (clip.name, clip.page, clip.url, clip.author, clip.license, clip.sha256):
            assert needle in credits, (clip.name, needle)
    assert "python scripts/fetch_clips.py" in credits


def load_truth(clip: fc.Clip) -> dict:
    return json.loads((fc.CORPUS / clip.name / "ground-truth.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("clip", fc.CLIPS, ids=lambda c: c.name)
def test_ground_truth_is_complete_and_consistent(clip) -> None:
    """The oracle: written by hand from one frame per second, never by YOLO.
    count_per_second[s] is the number of birds in the frame at second s, and
    every bird (one entry per stay in the frame) is counted from its first to
    its last second."""
    truth = load_truth(clip)
    assert truth["clip"] == clip.name
    assert "YOLO" in truth["method"] and "hand" in truth["method"]
    w, h = truth["width"], truth["height"]
    counts = truth["count_per_second"]
    assert len(counts) == math.floor(truth["duration_s"]) + 1
    assert abs(truth["duration_s"] - clip.duration_s) < 0.5
    birds = truth["birds"]
    for bird in birds:
        assert 0 <= bird["first_s"] <= bird["last_s"] < len(counts), bird
        x, y, bw, bh = bird["box_at_landing"]
        assert 0 <= x and 0 <= y and bw > 0 and bh > 0 and x + bw <= w and y + bh <= h, bird
        assert bird["first_s"] <= bird["landing_s"] <= bird["last_s"], bird
    expected = [sum(b["first_s"] <= s <= b["last_s"] for b in birds) for s in range(len(counts))]
    assert counts == expected
    if 6 in clip.cases:
        assert birds == [] and set(counts) == {0}
    else:
        assert birds
    if 2 in clip.cases:
        assert max(counts) >= 2


@needs_git
def test_clips_are_ignored_and_the_oracle_is_not() -> None:
    def ignored(path: str) -> bool:
        return subprocess.run(["git", "check-ignore", "-q", "--no-index", path],
                              cwd=ROOT).returncode == 0
    for clip in fc.CLIPS:
        rel = clip.out.relative_to(ROOT).as_posix()
        assert ignored(rel), rel
        assert not ignored(f"data/samples/corpus/{clip.name}/ground-truth.json")
    assert not ignored("data/samples/corpus/CREDITS.md")
