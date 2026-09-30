"""Track-while-open (P9) and looser visit matching (P10).

The pipeline is driven like main.run_preview: each frame goes through
Detector.process(frame, tracks=events.open_tracks(t)) and then
EventLogger.handle(). Frames are 1 s apart (captured_wall_time) and every
frame is gated (warm-up 2, N 1) unless a test says otherwise. The fake
predictor only sees birds whose centre is inside the crop it is given."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

import main
from camera import Frame
from config_schema import DetectionConfig, LoggingConfig, StorageConfig
from detector import Detector
from logger import EventLogger
from tests.conftest import read_log
from tests.fakes import FakePredictor

T0 = datetime(2026, 9, 30, 14, 0, 0).timestamp()
W, H = 320, 240


def image(*blocks: tuple[int, int, int]) -> np.ndarray:
    """Grey frame with dark square blocks (x, y, size)."""
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    for x, y, size in blocks:
        img[y:y + size, x:x + size] = 40
    return img


def iso(t: float) -> str:
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec="milliseconds")


def run(tmp_path: Path, obs, fake: FakePredictor, images: list[np.ndarray], **cfg):
    """Returns (events, events_written_after_seq, detector)."""
    base = dict(motion_warmup_frames=2, process_every_n_frames=1, motion_min_area=50,
                motion_padding_px=10)
    det = Detector(DetectionConfig(**{**base, **cfg}), obs=obs, predictor=fake)
    el = EventLogger(LoggingConfig(events_file=tmp_path / "data" / "events.jsonl",
                                   snapshots_dir=tmp_path / "data" / "snapshots"),
                     StorageConfig(), "track001", obs=obs, dedupe_within_seconds=10)
    written: dict[int, int] = {}
    for seq, img in enumerate(images, start=1):
        f = Frame(image=img, captured_at=0.0, seq=seq, captured_wall_time=T0 + seq)
        el.handle(f, det.process(f, tracks=el.open_tracks(f.captured_wall_time)))
        written[seq] = len(read_log(el.events_path)) if el.events_path.exists() else 0
    el.close()
    return read_log(el.events_path), written, det


def boxes(seqs, box) -> dict[int, list[tuple[int, int, int, int]]]:
    return {s: [box] for s in seqs}


# ---------------------------------------------------------------- P9

def test_a_still_bird_stays_one_open_visit_while_yolo_confirms_it(tmp_path, obs) -> None:
    """No motion after landing: the last-bbox crop keeps the visit alive while
    the fake confirms the bird (frames 4-20); it closes 10 s after the last
    confirmation, although the bird is still in the picture."""
    bird = (100, 100, 20, 20)
    fake = FakePredictor(boxes(range(4, 21), bird), within_crop=True)
    images = [image()] * 3 + [image((100, 100, 20))] * 28          # frames 1-31

    events, written, _ = run(tmp_path, obs, fake, images)

    (ev,) = events
    assert (ev["frame_seq"], ev["visit_frames"]) == (4, 17)
    assert ev["ts"] == iso(T0 + 4) and ev["last_seen"] == iso(T0 + 20)
    assert written[30] == 0 and written[31] == 1                  # 30 - 20 = 10 s: still open
    assert set(range(5, 32)) <= {c["frame_seq"] for c in fake.calls}  # YOLO ran on the track
    obs.close()
    gate = {l["frame_seq"]: l["context"].get("reason") for l in read_log(obs.path)
            if l["stage"] == "morph_contour" and l["event"] != "start"}
    assert {gate[s] for s in range(5, 31)} == {"no_contours"}      # tracked: never motion
    assert gate[31] is None       # untracked again, the still bird is motion: never learned


def test_the_background_does_not_learn_a_tracked_bird(obs) -> None:
    """While its box is tracked, the bird is kept out of the MOG2 update, so
    once tracking stops the bird is still foreground. Untracked, the same 150
    still frames teach MOG2 the bird and the motion is gone."""
    def motion_area(tracked: bool) -> float:
        fake = FakePredictor()
        det = Detector(DetectionConfig(motion_warmup_frames=2, process_every_n_frames=1,
                                       motion_min_area=50, motion_padding_px=10),
                       obs=obs, predictor=fake)
        tracks = [(100, 100, 20, 20)] if tracked else []
        for seq in range(1, 4):
            det.process(Frame(image(), 0.0, seq, T0 + seq))
        for seq in range(4, 154):
            det.process(Frame(image((100, 100, 20)), 0.0, seq, T0 + seq), tracks=tracks)
        det.process(Frame(image((100, 100, 20)), 0.0, 154, T0 + 154))
        return det.last_motion_area

    assert motion_area(tracked=True) >= 300      # the 20x20 bird is still motion
    assert motion_area(tracked=False) < 50       # absorbed


# ---------------------------------------------------------------- P10

def test_a_hop_of_one_and_a_half_widths_is_still_one_visit(tmp_path, obs) -> None:
    fake = FakePredictor({**boxes(range(4, 11), (100, 100, 20, 20)),
                          **boxes(range(11, 21), (130, 100, 20, 20))}, within_crop=True)
    images = ([image()] * 3 + [image((100, 100, 20))] * 7 + [image((130, 100, 20))] * 10)

    events, _, _ = run(tmp_path, obs, fake, images)

    (ev,) = events
    assert ev["visit_frames"] == 17 and ev["last_seen"] == iso(T0 + 20)


def test_a_second_bird_far_away_is_a_second_visit(tmp_path, obs) -> None:
    a, b = (40, 40, 20, 20), (250, 180, 20, 20)
    fake = FakePredictor({s: [a] + ([b] if s >= 10 else []) for s in range(4, 21)},
                         within_crop=True)
    images = [image()] * 3 + [image((40, 40, 20))] * 6 + [image((40, 40, 20), (250, 180, 20))] * 11

    events, _, _ = run(tmp_path, obs, fake, images)

    by_seq = {e["frame_seq"]: e for e in events}
    assert sorted(by_seq) == [4, 10]
    assert by_seq[4]["bbox_xywh"] == list(a) and by_seq[4]["visit_frames"] == 17
    assert by_seq[10]["bbox_xywh"] == list(b) and by_seq[10]["visit_frames"] == 11


def test_visit_matching_thresholds_come_from_config() -> None:
    cfg = DetectionConfig()
    assert (cfg.visit_iou_threshold, cfg.visit_center_distance) == (0.3, 2.0)


# ---------------------------------------------------------------- YOLO call rate

def yolo_calls_per_seq(obs) -> dict[int, Counter]:
    obs.close()
    calls: dict[int, Counter] = defaultdict(Counter)
    for l in read_log(obs.path):
        if l["stage"] == "yolo_infer" and l["event"] == "start":
            calls[l["frame_seq"]][l["context"]["crop"]] += 1
    return calls


def test_yolo_call_rate_with_and_without_an_open_visit(tmp_path, obs) -> None:
    """Gated frames every 5th (3, 8, 13 ...). A distractor moves along the top
    all the time; the fake never calls it a bird. A still bird lands at frame
    20. Before it: one motion crop per gated frame. With its visit open: the
    motion crop plus the track crop, never more than 2."""
    bird = (200, 170, 30, 30)
    fake = FakePredictor(boxes(range(20, 61), bird), within_crop=True)
    images = []
    for seq in range(1, 61):
        blocks = [((seq * 7) % 250, 10, 20)]
        if seq >= 20:
            blocks.append((200, 170, 30))
        images.append(image(*blocks))

    events, _, _ = run(tmp_path, obs, fake, images, process_every_n_frames=5)
    calls = yolo_calls_per_seq(obs)

    gated = set(range(3, 61, 5))
    assert set(calls) <= gated                                   # YOLO only on gated frames
    before = [s for s in gated if s < 23]
    after = [s for s in gated if s > 23]                         # visit opened on frame 23
    assert all(sum(calls[s].values()) == 1 and calls[s]["motion"] == 1 for s in before)
    assert all(sum(calls[s].values()) <= 2 for s in after)
    assert all(calls[s]["track"] == 1 for s in after)
    (ev,) = events
    assert ev["frame_seq"] == 23 and ev["visit_frames"] == len(after) + 1


def test_main_sends_open_visits_to_the_detector(synth_video_2s, tmp_path, fake_predictor) -> None:
    """Wiring: main.run_preview passes events.open_tracks() to Detector.process."""
    from tests.conftest import write_test_config

    cfg = write_test_config(tmp_path, **{"motion_warmup_frames: 60": "motion_warmup_frames: 2",
                                         "process_every_n_frames: 30": "process_every_n_frames: 1"})
    fake = fake_predictor(boxes(range(10, 61), (300, 150, 30, 20)))
    main.main(["--source", str(synth_video_2s), "--headless", "--no-pace", "--config", str(cfg),
               "--log-dir", str(tmp_path / "logs"), "--roi-file", str(tmp_path / "no_roi.json")],
              predictor=fake)
    (log,) = (tmp_path / "logs").glob("run_*.jsonl")
    crops = Counter(l["context"]["crop"] for l in read_log(log)
                    if l["stage"] == "yolo_infer" and l["event"] == "start")
    assert crops["track"] >= 40
