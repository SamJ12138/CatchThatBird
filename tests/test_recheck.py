"""Re-check after a rejected motion crop (R2).

A gated frame whose motion crop YOLO found nothing in opens a re-check
window: the crop's region is classified again on the following frames, and
kept out of the MOG2 update meanwhile, instead of waiting a full cadence for
the next gated frame, by which time a bird that landed and sits still has
been absorbed into the background.

The pipeline is driven like main.run_preview, at 30 fps (captured_wall_time
= T0 + (seq - 1) / 30). The fake predictor only sees birds whose centre is
inside the crop it is given."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

import main
from camera import Frame
from config_schema import DetectionConfig, LoggingConfig, StorageConfig
from detector import Detector
from logger import EventLogger
from tests.conftest import ROOT, read_log, write_test_config
from tests.fakes import FakePredictor

T0 = datetime(2026, 10, 2, 14, 0, 0).timestamp()
W, H = 320, 240
FPS = 30
BIRD = (100, 100, 20, 20)


def image(*blocks: tuple[int, int, int]) -> np.ndarray:
    """Grey frame with dark square blocks (x, y, size)."""
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    for x, y, size in blocks:
        img[y:y + size, x:x + size] = 40
    return img


def frame(seq: int, img: np.ndarray) -> Frame:
    return Frame(image=img, captured_at=0.0, seq=seq, captured_wall_time=T0 + (seq - 1) / FPS)


def run(tmp_path: Path, obs, fake: FakePredictor, images: list[np.ndarray], **cfg):
    """Returns (events, detector). Defaults: gated frames 60, 90, 120 ..."""
    base = dict(motion_warmup_frames=59, process_every_n_frames=30, motion_min_area=50,
                motion_padding_px=10)
    det = Detector(DetectionConfig(**{**base, **cfg}), obs=obs, predictor=fake)
    el = EventLogger(LoggingConfig(events_file=tmp_path / "data" / "events.jsonl",
                                   snapshots_dir=tmp_path / "data" / "snapshots"),
                     StorageConfig(), "recheck1", obs=obs, dedupe_within_seconds=10)
    for seq, img in enumerate(images, start=1):
        f = frame(seq, img)
        el.handle(f, det.process(f, tracks=el.open_tracks(f.captured_wall_time)))
    el.close()
    return read_log(el.events_path) if el.events_path.exists() else [], det


def boxes(seqs, box) -> dict[int, list[tuple[int, int, int, int]]]:
    return {s: [box] for s in seqs}


def yolo_calls(obs) -> dict[int, Counter]:
    """frame_seq -> Counter of crop kinds YOLO ran on."""
    obs.close()
    calls: dict[int, Counter] = defaultdict(Counter)
    for l in read_log(obs.path):
        if l["stage"] == "yolo_infer" and l["event"] == "start":
            calls[l["frame_seq"]][l["context"]["crop"]] += 1
    return calls


def bird_lands_on_90(n: int = 150) -> list[np.ndarray]:
    """Empty until frame 89; from frame 90 a bird that sits perfectly still."""
    return [image()] * 89 + [image((100, 100, 20))] * (n - 89)


# ---------------------------------------------------------------- the visit opens early

def test_a_bird_blurred_on_the_gated_frame_opens_its_visit_within_five_frames(tmp_path, obs) -> None:
    """The brief's case. The bird is there from frame 90, a gated frame, but
    YOLO finds nothing on it (blurred); it would find the bird on 91 onward.
    The visit must open by frame 95, not wait for the gated frame 120. With
    the default re-check on every third frame, it opens on frame 93."""
    fake = FakePredictor(boxes(range(91, 151), BIRD), within_crop=True)

    events, _ = run(tmp_path, obs, fake, bird_lands_on_90())

    (ev,) = events
    assert ev["frame_seq"] == 93
    assert ev["bbox_xywh"] == list(BIRD)
    assert ev["ts"] == datetime.fromtimestamp(T0 + (ev["frame_seq"] - 1) / FPS).astimezone() \
        .isoformat(timespec="milliseconds")
    calls = yolo_calls(obs)
    assert calls[90] == Counter(motion=1)                 # the rejected motion crop
    assert calls[93] == Counter(recheck=1)                # found on a re-check, not at a tick
    assert not calls[91] and not calls[92]                # every third frame of the window


def test_without_the_window_the_same_bird_is_never_logged(tmp_path, obs) -> None:
    """recheck_window_frames 0 is the old behaviour, and R2: by the next
    gated frame (120) the still bird is background, so no visit opens at all."""
    fake = FakePredictor(boxes(range(91, 151), BIRD), within_crop=True)

    events, _ = run(tmp_path, obs, fake, bird_lands_on_90(), recheck_window_frames=0)

    assert events == []
    assert set(yolo_calls(obs)) == {90}


def test_rechecking_every_frame_opens_the_visit_on_the_next_frame(tmp_path, obs) -> None:
    """recheck_every_n_frames 1: two frames earlier, for three times the calls."""
    fake = FakePredictor(boxes(range(91, 151), BIRD), within_crop=True)

    events, _ = run(tmp_path, obs, fake, bird_lands_on_90(), recheck_every_n_frames=1)

    (ev,) = events
    assert ev["frame_seq"] == 91
    assert yolo_calls(obs)[91] == Counter(recheck=1)


def test_the_visit_found_on_a_recheck_is_then_tracked_at_the_cadence(tmp_path, obs) -> None:
    """Once the visit is open the window is over: no more re-check crops, one
    track crop per gated frame, and the still bird stays one visit."""
    fake = FakePredictor(boxes(range(91, 211), BIRD), within_crop=True)

    events, _ = run(tmp_path, obs, fake, bird_lands_on_90(210))

    (ev,) = events
    assert ev["frame_seq"] == 93 and ev["visit_frames"] == 5       # 93, then 120, 150, 180, 210
    calls = yolo_calls(obs)
    assert sorted(calls) == [90, 93, 120, 150, 180, 210]
    assert all(calls[s] == Counter(track=1) for s in (120, 150, 180, 210))


# ---------------------------------------------------------------- the freeze

def test_the_background_does_not_learn_the_rechecked_region(obs) -> None:
    """Gated frames 3, 34, 65. A still object appears on frame 34 and YOLO
    never calls it a bird. With the window (frames 35-64) its region is kept
    out of the MOG2 update, so on frame 65 it is still whole as motion; a
    second object that appeared outside the region on frame 36 has been
    learned meanwhile. Without the window, both are background by frame 65."""
    def motion_area_on_65(window: int) -> float:
        det = Detector(DetectionConfig(motion_warmup_frames=2, process_every_n_frames=31,
                                       motion_min_area=50, motion_padding_px=10,
                                       recheck_window_frames=window),
                       obs=obs, predictor=FakePredictor())
        for seq in range(1, 66):
            blocks = []
            if seq >= 34:
                blocks.append((100, 100, 20))
            if seq >= 36:
                blocks.append((250, 180, 30))   # bigger: it would be the largest contour
            det.process(frame(seq, image(*blocks)))
        return det.last_motion_area

    assert 300 <= motion_area_on_65(window=30) < 500   # the 20x20 object, not the 30x30 one
    assert motion_area_on_65(window=0) < 50            # absorbed


# ---------------------------------------------------------------- bounded cost

def test_a_window_that_finds_nothing_ends_after_recheck_window_frames(tmp_path, obs) -> None:
    """The object is never a bird: the window's 30 frames (91-120), re-checked
    on every third, 10 calls, and no more. The run log records the window's
    start and that it expired."""
    events, _ = run(tmp_path, obs, FakePredictor(within_crop=True), bird_lands_on_90(300))

    assert events == []
    calls = yolo_calls(obs)
    assert sorted(s for s in calls if calls[s]["recheck"]) == list(range(93, 121, 3))
    assert sum(sum(c.values()) for c in calls.values()) <= 11 + 2  # + ticks while it is absorbed
    recheck = [(l["frame_seq"], l["event"], l["context"].get("reason"))
               for l in read_log(obs.path) if l["stage"] == "recheck"]
    assert recheck[:2] == [(90, "start", None), (120, "skip", "expired")]


def test_nothing_moving_means_no_yolo_call(tmp_path, obs) -> None:
    fake = FakePredictor(within_crop=True)

    run(tmp_path, obs, fake, [image()] * 300)

    assert fake.calls == []


def test_steady_non_bird_motion_gets_one_window_not_one_per_tick(tmp_path, obs) -> None:
    """A distractor moves all the time and is never a bird. The first gated
    frame it is rejected on opens a window. The following gated frames
    reject it too and cost one motion crop each, as before. The next window
    can open only after a gated frame on which nothing moved."""
    images = [image()] * 89 + [image(((seq * 7) % 280, 30, 20)) for seq in range(90, 301)]
    images += [image()] * 60 + [image(((seq * 7) % 280, 30, 20)) for seq in range(361, 451)]

    events, _ = run(tmp_path, obs, FakePredictor(within_crop=True), images)

    assert events == []
    calls = yolo_calls(obs)
    rechecked = sorted(s for s in calls if calls[s]["recheck"])
    assert rechecked == list(range(93, 121, 3)) + list(range(393, 421, 3))
    ticks = [150, 180, 210, 240, 270, 300]                # steady motion after the window
    assert all(calls[s] == Counter(motion=1) for s in ticks)
    assert not any(calls[s] for s in range(121, 361) if s not in ticks)
    assert not calls[330] and not calls[360]              # quiet ticks: the motion has stopped
    assert calls[390] == Counter(motion=1)                # motion again: a new window


@pytest.mark.parametrize("bill_x, window", [(132, False), (140, False), (250, True)])
def test_motion_next_to_an_open_visit_opens_no_window(tmp_path, obs, bill_x, window) -> None:
    """R3: a part of the tracked bird outside its box (the hummingbird's bill)
    is motion on gated frames and never a bird on its own. It is not
    re-checked when its crop touches the visit's own crop (x 132), nor when
    it is near enough that a bird found there would join the visit (x 140:
    35 px from the bird's centre, within 2 x its size). Far from the visit
    (x 250) the same motion does open a window: on the landing frame 60,
    since every motion region is classified (the bill is a region of its
    own there), and while that window holds its crop out of the background
    the bill of frame 90 is no motion."""
    fake = FakePredictor(boxes(range(60, 241), BIRD), within_crop=True)
    images = [image()] * 59
    for seq in range(60, 241):   # the "bill" shows on gated frames only, so it is never learned
        images.append(image((100, 100, 20), *([(bill_x, 104, 10)] if seq % 30 == 0 else [])))

    events, _ = run(tmp_path, obs, fake, images)

    (ev,) = events
    assert ev["frame_seq"] == 60 and ev["visit_frames"] == 7
    calls = yolo_calls(obs)
    assert all(calls[s] == Counter(motion=1, track=1) for s in (150, 180, 210, 240))
    if window:
        assert sorted(s for s in calls if calls[s]["recheck"]) == list(range(63, 91, 3))
        assert calls[60] == Counter(motion=2) and calls[90] == Counter(track=1, recheck=1)
    else:
        assert calls[90] == Counter(motion=1, track=1)
        assert set(calls) == {60, 90, 120, 150, 180, 210, 240}


# ---------------------------------------------------------------- config, wiring

def test_window_defaults_and_config_file_keys() -> None:
    cfg = DetectionConfig()
    assert (cfg.recheck_window_frames, cfg.recheck_every_n_frames) == (30, 3)
    on_disk = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["detection"]
    assert on_disk["recheck_window_frames"] == 30 and on_disk["recheck_every_n_frames"] == 3
    with pytest.raises(ValueError):
        DetectionConfig(recheck_every_n_frames=0)
    with pytest.raises(ValueError):
        DetectionConfig(recheck_window_frames=-1)


def test_main_default_config_opens_the_visit_before_the_next_tick(synth_video_15s, tmp_path,
                                                                    fake_predictor) -> None:
    """Through main.py with the default config (gated frames 61, 91, 121) on
    the blob clip, whose blob is motion on every gated frame: the predictor
    finds nothing on frame 61 and a bird from 62 on. The visit opens by
    frame 66; it used to open on frame 91."""
    fake = fake_predictor(boxes(range(62, 451), (300, 150, 30, 20)))
    code = main.main(["--source", str(synth_video_15s), "--headless", "--no-pace",
                      "--config", str(write_test_config(tmp_path)),
                      "--log-dir", str(tmp_path / "logs"),
                      "--roi-file", str(tmp_path / "no_roi.json")], predictor=fake)
    assert code == 0
    (ev,) = read_log(tmp_path / "data" / "events.jsonl")
    assert 62 <= ev["frame_seq"] <= 66
