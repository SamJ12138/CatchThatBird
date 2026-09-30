"""scripts/make_synth_video.py: the synthetic bird (default) and the blob (--no-bird)."""
from __future__ import annotations

import cv2
import numpy as np

from scripts import make_synth_video as synth
from tests.conftest import ROOT

ROI = (160, 90, 320, 180)  # 640x360 frame


def test_the_bird_asset_and_its_credits_are_committed() -> None:
    sprite = cv2.imread(str(synth.BIRD_IMAGE), cv2.IMREAD_UNCHANGED)
    assert sprite is not None and sprite.shape[2] == 4  # RGBA cutout
    credits = (synth.BIRD_IMAGE.parent / "CREDITS.md").read_text(encoding="utf-8")
    for needle in ("commons.wikimedia.org/wiki/File:House_sparrow_(female)_(50108651573).jpg",
                   "Courtney Celley", "Public domain", "PD-USGov-FWS"):
        assert needle in credits, needle


def test_bird_timeline_enters_perches_and_leaves() -> None:
    sprite_wh = (80, 52)
    pos = lambda t: synth.bird_position(t, ROI, sprite_wh)  # noqa: E731
    assert pos(0.0) is None and pos(2.9) is None           # empty during MOG2 warm-up
    assert pos(19.5) is None                               # gone before the end
    perch = [pos(t / 10) for t in range(40, 170)]              # 4.0 .. 16.9 s
    xs, ys = {p[0] for p in perch}, {p[1] for p in perch}
    assert len(xs) == 1                                       # perched in one place
    assert 0 < max(ys) - min(ys) <= synth.BOB_PX              # bobbing slightly
    # Always fully inside the ROI while visible.
    rx, ry, rw, rh = ROI
    for i in range(0, 200):
        p = pos(i / 10)
        if p is not None:
            assert rx <= p[0] and p[0] + sprite_wh[0] <= rx + rw
            assert ry <= p[1] and p[1] + sprite_wh[1] <= ry + rh


def frames(path, *seconds, fps=30):
    cap = cv2.VideoCapture(str(path))
    out = []
    for s in seconds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * fps))
        ok, img = cap.read()
        assert ok
        out.append(img)
    cap.release()
    return out


def test_bird_clip_shows_the_sprite_inside_the_roi(tmp_path) -> None:
    clip = synth.make_video(tmp_path / "bird.mp4", width=640, height=360, roi=ROI, bird=True)
    empty, perched = frames(clip, 1.0, 7.0)
    diff = cv2.absdiff(perched, empty).max(axis=2) > 40
    ys, xs = np.nonzero(diff)
    assert len(xs) > 500                                     # a bird-sized change
    rx, ry, rw, rh = ROI
    assert rx <= xs.min() and xs.max() < rx + rw and ry <= ys.min() and ys.max() < ry + rh


def test_no_bird_keeps_the_blob(tmp_path) -> None:
    clip = synth.make_video(tmp_path / "blob.mp4", width=640, height=360, roi=ROI, bird=False,
                            seconds=2)
    (first,) = frames(clip, 0.0)
    assert (first.max(axis=2) < 70).sum() > 200              # the dark ellipse is there at once


def test_default_output_name_follows_the_mode() -> None:
    assert synth.default_out(True) == ROOT / "data" / "samples" / "synth_bird.mp4"
    assert synth.default_out(False) == ROOT / "data" / "samples" / "synth_blob.mp4"
