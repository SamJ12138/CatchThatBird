"""Several birds at once: a motion gate with up to max_motion_regions crops,
Hungarian association of detections to open visits, visit ids, and the
YOLO budget on gated frames.

Driven like main.run_preview: each frame goes through
Detector.process(frame, tracks=events.open_tracks(t)) and EventLogger.handle().
The fake predictor only sees birds whose centre is inside the crop it is
given (within_crop), like YOLO."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from camera import Frame
from config_schema import DetectionConfig, LoggingConfig, StorageConfig
from detector import Detector
from logger import EventLogger
from tests.conftest import read_log
from tests.fakes import FakePredictor

T0 = datetime(2026, 10, 4, 9, 0, 0).timestamp()
W, H = 320, 240


def image(*blocks: tuple[int, int, int]) -> np.ndarray:
    """Grey frame with dark square blocks (x, y, size)."""
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    for x, y, size in blocks:
        img[y:y + size, x:x + size] = 40
    return img


class Pipeline:
    """Detector + EventLogger. `dt` seconds of capture time per frame."""

    def __init__(self, tmp_path: Path, obs, fake: FakePredictor, dt: float = 1.0, **cfg) -> None:
        base = dict(motion_warmup_frames=2, process_every_n_frames=1, motion_min_area=50,
                    motion_padding_px=10)
        self.obs = obs
        self.dt = dt
        self.det = Detector(DetectionConfig(**{**base, **cfg}), obs=obs, predictor=fake)
        self.events = EventLogger(
            LoggingConfig(events_file=tmp_path / "data" / "events.jsonl",
                          snapshots_dir=tmp_path / "data" / "snapshots"),
            StorageConfig(), "multi001", obs=obs, dedupe_within_seconds=10)
        self.seq = 0

    def feed(self, *images: np.ndarray) -> None:
        for img in images:
            self.seq += 1
            f = Frame(image=img, captured_at=0.0, seq=self.seq,
                      captured_wall_time=T0 + self.seq * self.dt)
            self.events.handle(f, self.det.process(
                f, tracks=self.events.open_tracks(f.captured_wall_time)))

    def close(self) -> list[dict]:
        self.events.close()
        self.obs.close()
        path = self.events.events_path
        return read_log(path) if path.exists() else []


def yolo_calls(obs) -> dict[int, Counter]:
    calls: dict[int, Counter] = defaultdict(Counter)
    for l in read_log(obs.path):
        if l["stage"] == "yolo_infer" and l["event"] == "start":
            calls[l["frame_seq"]][l["context"]["crop"]] += 1
    return calls


# ---------------------------------------------------------------- the brief's four cases

@pytest.mark.parametrize("b", [(250, 180, 20, 20),     # far apart: two motion regions
                               (135, 100, 20, 20)])    # 15 px apart: one region, two boxes
def test_two_birds_landing_in_the_same_frame_are_two_visits(tmp_path, obs, b) -> None:
    a = (100, 100, 20, 20)
    fake = FakePredictor({s: [a, b] for s in range(4, 12)}, within_crop=True)
    p = Pipeline(tmp_path, obs, fake)
    p.feed(*[image()] * 3, *[image(a[:2] + (20,), b[:2] + (20,))] * 8)
    events = p.close()

    assert len(events) == 2
    assert {e["frame_seq"] for e in events} == {4}
    assert sorted(tuple(e["bbox_xywh"]) for e in events) == sorted([a, b])
    assert len({e["visit_id"] for e in events}) == 2
    assert all(e["concurrent_max"] == 2 for e in events)
    assert all(e["visit_frames"] == 8 for e in events)


def test_two_birds_crossing_paths_keep_their_visits(tmp_path, obs) -> None:
    """A flies right 40 px a frame along y=100; B walks left 10 px a frame
    along y=130. On frame 7, A's new box is nearer B's last box (36 px) than
    its own (40 px, the edge of the reach): matching each detection to its
    nearest visit hands A's position to B's visit, and A's visit then loses
    its bird. The joint assignment keeps each bird on its own visit."""
    a_x = [40, 80, 120, 160, 200, 240, 280]
    b_x = [200, 190, 180, 170, 160, 150, 140]
    frames = {s: (ax, bx) for s, ax, bx in zip(range(4, 11), a_x, b_x)}
    fake = FakePredictor({s: [(ax, 100, 20, 20), (bx, 130, 20, 20)] for s, (ax, bx) in frames.items()},
                         within_crop=True)
    p = Pipeline(tmp_path, obs, fake)
    p.feed(*[image()] * 3, *[image((ax, 100, 20), (bx, 130, 20)) for ax, bx in frames.values()])

    open_now = {v["first_bbox"][0]: v["bbox_xywh"] for v in p.events.open_visits()}
    events = p.close()

    assert len(events) == 2
    assert open_now == {40: [280, 100, 20, 20], 200: [140, 130, 20, 20]}


def test_a_bird_next_to_a_swaying_branch_is_one_visit(tmp_path, obs) -> None:
    """The branch sways every frame and is the largest motion, but it is a
    separate region from the still bird beside it: both are classified, the
    bird is logged once and the branch never."""
    bird = (210, 100, 20, 20)
    fake = FakePredictor({s: [bird] for s in range(4, 25)}, within_crop=True)
    p = Pipeline(tmp_path, obs, fake, recheck_window_frames=0)
    branch = [(150, 90, 34), (160, 90, 34)]
    p.feed(*[image()] * 3,
           *[image(branch[s % 2], (210, 100, 20)) for s in range(4, 25)])
    events = p.close()

    (ev,) = events
    assert ev["bbox_xywh"] == list(bird) and ev["frame_seq"] == 4
    assert ev["visit_frames"] == 21


def test_the_yolo_budget_defers_crops_and_logs_every_deferral(tmp_path, obs) -> None:
    """30 fps, gated frames 3, 33, 63. Three still birds land on frame 3 and
    are tracked from then on. On frame 33 four blobs start moving: 3 track
    crops + 4 motion regions = 7 crops, budget 5 per second. The tracks go
    first, then the two largest regions; the other two wait, logged, and run
    as the budget refills (5 per second: one every 0.2 s, 6 frames)."""
    birds = [(20, 20, 20, 20), (280, 20, 20, 20), (150, 200, 20, 20)]
    fake = FakePredictor({s: birds for s in range(3, 70)}, within_crop=True)
    p = Pipeline(tmp_path, obs, fake, dt=1 / 30, process_every_n_frames=30,
                 recheck_window_frames=0)
    still = [b[:2] + (20,) for b in birds]
    blob_sizes = (30, 26, 22, 18)                       # largest first
    blob_y = (70, 70, 140, 140)
    blob_x0 = (20, 170, 20, 170)
    imgs = [image()] * 2 + [image(*still)] * 30
    for k in range(30):                                 # frames 33-62: the blobs move
        imgs.append(image(*still, *[(x0 + 3 * (k % 10), y, s)
                                    for x0, y, s in zip(blob_x0, blob_y, blob_sizes)]))
    p.feed(*imgs)
    p.close()

    calls = yolo_calls(obs)
    assert calls[33] == Counter(track=3, motion=2)
    budget = [l for l in read_log(obs.path) if l["stage"] == "yolo_budget"]
    deferred = [l for l in budget if l["event"] == "skip" and l["context"]["reason"] == "deferred"]
    assert [l["frame_seq"] for l in deferred] == [33, 33]
    assert all(l["context"]["crop"] == "motion" for l in deferred)
    ran_late = [l for l in budget if l["event"] == "success"]
    assert [(l["frame_seq"], l["context"]["deferred_from"]) for l in ran_late] == [(39, 33), (45, 33)]
    assert calls[39]["motion"] == 1 and calls[45]["motion"] == 1


# ---------------------------------------------------------------- motion gate

def test_overlapping_padded_contours_are_one_region_and_regions_are_capped(tmp_path, obs) -> None:
    """Six moving blocks: two 15 px apart (padded boxes overlap: one region),
    four others apart. max_motion_regions 4 keeps the four largest regions."""
    fake = FakePredictor()
    p = Pipeline(tmp_path, obs, fake, recheck_window_frames=0)
    blocks = [(20, 20, 30), (65, 20, 30),               # one region (gap 15 < 2 x 10 padding)
              (200, 20, 28), (20, 150, 24), (150, 150, 16), (260, 180, 12)]
    p.feed(*[image()] * 3, image(*blocks), image())
    p.close()

    crops = [l["context"]["crop_xyxy"] for l in read_log(obs.path)
             if l["stage"] == "crop_build" and l["event"] == "success" and l["frame_seq"] == 4]
    assert len(crops) == 4
    assert [10, 10, 105, 60] in crops                     # the merged pair, padded by 10
    assert not any(c[0] >= 250 for c in crops)            # the smallest block is dropped


def test_new_config_keys_and_defaults() -> None:
    cfg = DetectionConfig()
    assert (cfg.max_motion_regions, cfg.max_yolo_calls_per_s) == (4, 5)
    with pytest.raises(ValueError):
        DetectionConfig(max_motion_regions=0)
    with pytest.raises(ValueError):
        DetectionConfig(max_yolo_calls_per_s=0)


# ---------------------------------------------------------------- visit ids, concurrency

def test_visit_ids_count_up_per_run_and_concurrency_is_per_visit(tmp_path, obs) -> None:
    a, b, c = (40, 40, 20, 20), (250, 40, 20, 20), (150, 180, 20, 20)
    seen = {**{s: [a] for s in range(4, 10)}, **{s: [a, b] for s in range(10, 14)},
            **{s: [c] for s in range(40, 44)}}
    fake = FakePredictor(seen, within_crop=True)
    p = Pipeline(tmp_path, obs, fake, recheck_window_frames=0)
    imgs = [image()] * 3 + [image((40, 40, 20))] * 6 + [image((40, 40, 20), (250, 40, 20))] * 4
    imgs += [image()] * 26 + [image((150, 180, 20))] * 4
    p.feed(*imgs)
    events = sorted(p.close(), key=lambda e: e["frame_seq"])

    assert [e["visit_id"] for e in events] == ["multi001-1", "multi001-2", "multi001-3"]
    assert [e["concurrent_max"] for e in events] == [2, 2, 1]


def test_two_visits_opened_on_one_frame_are_both_recovered(tmp_path, obs) -> None:
    """Recovery used to skip a checkpointed visit whose (run_id, frame_seq) was
    already in events.jsonl. Two birds can open visits on the same frame, so
    the visit_id decides; old lines and checkpoints without one still load."""
    events_path = tmp_path / "data" / "events.jsonl"
    events_path.parent.mkdir(parents=True)
    old = {"ts": "2026-10-01T09:00:00.000+00:00", "run_id": "old00001", "frame_seq": 7,
           "class": "bird", "confidence": 0.9, "bbox_xywh": [1, 2, 3, 4], "snapshot_crop": None,
           "snapshot_full": None, "last_seen": "2026-10-01T09:00:05.000+00:00",
           "visit_frames": 5, "truncated": False, "recovered": False}
    first = dict(old, run_id="kill0001", frame_seq=91, visit_id="kill0001-1", concurrent_max=2)
    second = dict(first, visit_id="kill0001-2", bbox_xywh=[50, 2, 3, 4])
    events_path.write_text(json.dumps(old) + "\n" + json.dumps(first) + "\n", encoding="utf-8")
    (events_path.parent / "open_visits.json").write_text(
        json.dumps([first, second, dict(old, frame_seq=8)]), encoding="utf-8")

    EventLogger(LoggingConfig(events_file=events_path, snapshots_dir=tmp_path / "data" / "s"),
                StorageConfig(), "next0001", obs=obs).close()

    lines = read_log(events_path)
    assert [l.get("visit_id") for l in lines] == [None, "kill0001-1", "kill0001-2", None]
    assert [l["recovered"] for l in lines[2:]] == [True, True]


def test_old_lines_without_the_new_fields_load_in_the_scripts(tmp_path) -> None:
    from scripts import failure_report, plot_visits

    path = tmp_path / "events.jsonl"
    old = {"ts": "2026-10-01T09:00:00.000+00:00", "run_id": "old00001", "frame_seq": 7,
           "class": "bird", "confidence": 0.9, "bbox_xywh": [1, 2, 3, 4], "snapshot_crop": None,
           "snapshot_full": None, "last_seen": "2026-10-01T09:00:05.000+00:00",
           "visit_frames": 5, "truncated": False, "recovered": False}
    new = dict(old, ts="2026-10-01T10:00:00.000+00:00", visit_id="new00001-1", concurrent_max=2)
    path.write_text(json.dumps(old) + "\n" + json.dumps(new) + "\n", encoding="utf-8")
    events, bad = plot_visits.load_events(path)
    assert (len(events), bad) == (2, 0)
    assert plot_visits.visits_per_hour(events)[9:11] == [1, 1]

    log = tmp_path / "run_new00001.jsonl"
    persist = {"ts": "2026-10-01T10:00:00.000+00:00", "run_id": "new00001", "stage": "persist",
               "event": "success", "duration_ms": 1.0, "frame_seq": 7, "error_type": None,
               "error_message": None, "context": {"visit_frames": 5}}
    log.write_text(json.dumps(persist) + "\n"
                   + json.dumps(dict(persist, context={"visit_frames": 5, "visit_id": "new00001-1",
                                                       "concurrent_max": 2})) + "\n",
                   encoding="utf-8")
    assert failure_report.main([str(log)]) == 0
