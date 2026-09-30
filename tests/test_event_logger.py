"""Phase 3: EventLogger -- events.jsonl, snapshots, dedupe, daily cap, retention."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

import logger as event_logger_mod
import main
from camera import Frame
from config_schema import LoggingConfig, StorageConfig
from detector import Detection
from logger import EventLogger
from tests.conftest import ROOT, read_log, write_test_config

T0 = datetime(2026, 9, 30, 14, 5, 6, 789000).timestamp()  # local time
EVENT_KEYS = {"ts", "run_id", "frame_seq", "class", "confidence", "bbox_xywh",
              "snapshot_crop", "snapshot_full", "last_seen", "visit_frames"}


def iso(t: float) -> str:
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec="milliseconds")


def cfgs(tmp_path: Path, **storage) -> tuple[LoggingConfig, StorageConfig]:
    logging_cfg = LoggingConfig(events_file=tmp_path / "data" / "events.jsonl",
                                snapshots_dir=tmp_path / "data" / "snapshots")
    return logging_cfg, StorageConfig(**storage)


def make_logger(tmp_path: Path, obs, *, logging_cfg=None, storage=None, **kw) -> EventLogger:
    lc, sc = cfgs(tmp_path)
    return EventLogger(logging_cfg or lc, storage or sc, "run12345", obs=obs,
                       dedupe_within_seconds=10, **kw)


def frame(seq: int, t: float, w: int = 320, h: int = 240) -> Frame:
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    img[:, :, 0] = seq % 255  # make frames distinguishable
    return Frame(image=img, captured_at=0.0, seq=seq, captured_wall_time=t)


def det(f: Frame, bbox=(100, 80, 30, 20), conf: float = 0.9,
        crop=(90, 70, 140, 110)) -> Detection:
    return Detection(class_name="bird", confidence=conf, bbox_xywh=bbox, frame_seq=f.seq,
                     captured_wall_time=f.captured_wall_time, crop_xyxy=crop)


def events(el: EventLogger) -> list[dict]:
    if not el.events_path.exists():
        return []
    return read_log(el.events_path)


# ---------------------------------------------------------------- schema + snapshots

def test_event_schema(tmp_path, obs) -> None:
    el = make_logger(tmp_path, obs)
    f = frame(90, T0)
    el.handle(f, [det(f)])
    el.close()

    (ev,) = events(el)
    assert set(ev) == EVENT_KEYS
    assert ev["ts"] == iso(T0)
    assert ev["last_seen"] == iso(T0)
    assert (ev["run_id"], ev["frame_seq"], ev["class"]) == ("run12345", 90, "bird")
    assert ev["confidence"] == 0.9
    assert ev["bbox_xywh"] == [100, 80, 30, 20]
    assert ev["visit_frames"] == 1
    assert ev["snapshot_crop"] and ev["snapshot_full"]


def test_snapshot_full_is_null_when_disabled(tmp_path, obs) -> None:
    lc, sc = cfgs(tmp_path)
    lc = lc.model_copy(update={"save_full_frame": False})
    el = make_logger(tmp_path, obs, logging_cfg=lc)
    f = frame(90, T0)
    el.handle(f, [det(f)])
    el.close()

    (ev,) = events(el)
    assert ev["snapshot_full"] is None
    assert len(list(el.snapshots_dir.iterdir())) == 1


def test_snapshots_are_the_padded_crop_and_the_full_frame(tmp_path, obs) -> None:
    el = make_logger(tmp_path, obs)
    f = frame(90, T0)
    el.handle(f, [det(f, crop=(90, 70, 140, 110))])
    el.close()

    (ev,) = events(el)
    crop_path = tmp_path / ev["snapshot_crop"]
    full_path = tmp_path / ev["snapshot_full"]
    assert crop_path.parent == el.snapshots_dir and full_path.parent == el.snapshots_dir
    assert "20260930T140506.789" in crop_path.name and "seq000090" in crop_path.name
    assert cv2.imread(str(crop_path)).shape == (40, 50, 3)
    assert cv2.imread(str(full_path)).shape == (240, 320, 3)


def test_snapshots_use_snapshot_quality(tmp_path, obs, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(event_logger_mod.cv2, "imwrite",
                        lambda path, img, params=None: calls.append((path, params)) or True)
    lc, sc = cfgs(tmp_path)
    el = make_logger(tmp_path, obs, logging_cfg=lc.model_copy(update={"snapshot_quality": 77}))
    f = frame(90, T0)
    el.handle(f, [det(f)])
    el.close()

    assert len(calls) == 2
    assert all(params == [cv2.IMWRITE_JPEG_QUALITY, 77] for _, params in calls)
    assert all(path.endswith(".jpg") for path, _ in calls)


def test_failed_snapshot_write_is_logged_and_event_kept(tmp_path, obs, monkeypatch) -> None:
    monkeypatch.setattr(event_logger_mod.cv2, "imwrite", lambda *a, **k: False)
    el = make_logger(tmp_path, obs)
    f = frame(90, T0)
    el.handle(f, [det(f)])
    el.close()

    (ev,) = events(el)
    assert ev["snapshot_crop"] is None and ev["snapshot_full"] is None
    obs.close()
    fails = [l for l in read_log(obs.path) if l["stage"] == "persist" and l["event"] == "fail"]
    assert len(fails) == 2


# ---------------------------------------------------------------- dedupe

def test_same_bird_within_window_is_one_visit(tmp_path, obs) -> None:
    el = make_logger(tmp_path, obs)
    for seq, dt in [(90, 0), (120, 3), (150, 6)]:
        f = frame(seq, T0 + dt)
        el.handle(f, [det(f, bbox=(100 + dt, 80, 30, 20))])  # small drift, IoU > 0.5
    el.close()

    (ev,) = events(el)
    assert ev["frame_seq"] == 90
    assert ev["visit_frames"] == 3
    assert ev["last_seen"] == iso(T0 + 6)
    assert len(list(el.snapshots_dir.iterdir())) == 2  # snapshots only for the first frame


def test_visit_closes_after_dedupe_window(tmp_path, obs) -> None:
    el = make_logger(tmp_path, obs)
    f1, f2 = frame(90, T0), frame(500, T0 + 10.5)
    el.handle(f1, [det(f1)])
    el.handle(f2, [det(f2)])
    written_before_close = len(events(el))
    el.close()

    assert written_before_close == 1  # the first visit was flushed when it expired
    assert [e["frame_seq"] for e in events(el)] == [90, 500]


def test_low_iou_is_a_new_event(tmp_path, obs) -> None:
    el = make_logger(tmp_path, obs)
    f1, f2 = frame(90, T0), frame(120, T0 + 1)
    el.handle(f1, [det(f1, bbox=(100, 80, 30, 20))])
    el.handle(f2, [det(f2, bbox=(118, 80, 30, 20))])  # IoU = 12*20 / (2*600 - 240) = 0.25
    el.close()
    assert len(events(el)) == 2


def test_two_birds_far_apart_in_the_same_second_are_two_events(tmp_path, obs) -> None:
    el = make_logger(tmp_path, obs)
    f = frame(90, T0)
    el.handle(f, [det(f, bbox=(10, 10, 30, 20), crop=(0, 0, 50, 40)),
                  det(f, bbox=(250, 200, 30, 20), crop=(240, 190, 290, 230))])
    f2 = frame(91, T0 + 0.5)
    el.handle(f2, [det(f2, bbox=(10, 10, 30, 20)), det(f2, bbox=(250, 200, 30, 20))])
    el.close()

    evs = events(el)
    assert len(evs) == 2
    assert sorted(e["visit_frames"] for e in evs) == [2, 2]
    assert len({e["snapshot_crop"] for e in evs}) == 2  # no file-name collision


def test_dedupe_merges_are_logged(tmp_path, obs) -> None:
    el = make_logger(tmp_path, obs)
    for seq, dt in [(90, 0), (120, 1)]:
        f = frame(seq, T0 + dt)
        el.handle(f, [det(f)])
    el.close()
    obs.close()
    persist = [l for l in read_log(obs.path) if l["stage"] == "persist"]
    assert [(l["event"], l["context"].get("reason")) for l in persist] == [
        ("skip", "dedupe"), ("success", None)]


# ---------------------------------------------------------------- daily cap

def test_daily_cap_drops_event_n_plus_1(tmp_path, obs) -> None:
    lc, _ = cfgs(tmp_path)
    el = make_logger(tmp_path, obs, storage=StorageConfig(max_events_per_day=2))
    for i in range(3):
        f = frame(90 + i, T0 + i * 20)  # 20 s apart: three separate visits
        el.handle(f, [det(f)])
    el.close()

    assert [e["frame_seq"] for e in events(el)] == [90, 91]
    obs.close()
    capped = [l for l in read_log(obs.path) if l["context"].get("reason") == "daily_cap"]
    assert len(capped) == 1 and capped[0]["stage"] == "persist" and capped[0]["event"] == "skip"


def test_daily_cap_counts_events_already_on_disk(tmp_path, obs) -> None:
    lc, _ = cfgs(tmp_path)
    lc.events_file.parent.mkdir(parents=True)
    old = {k: None for k in EVENT_KEYS}
    lines = [dict(old, ts=iso(T0 - 3600)), dict(old, ts=iso(T0 - 7200)),
             dict(old, ts=iso(T0 - 86400 * 2))]  # two today, one two days ago
    lc.events_file.write_text("".join(json.dumps(l) + "\n" for l in lines))

    el = make_logger(tmp_path, obs, storage=StorageConfig(max_events_per_day=3))
    for i in range(2):
        f = frame(90 + i, T0 + i * 20)
        el.handle(f, [det(f)])
    el.close()

    assert len(events(el)) == 4  # 3 old + only 1 new (cap 3 per day)


# ---------------------------------------------------------------- retention

def test_retention_deletes_old_snapshots_never_events(tmp_path, obs) -> None:
    lc, _ = cfgs(tmp_path)
    lc.snapshots_dir.mkdir(parents=True)
    old_file = lc.snapshots_dir / "old_crop.jpg"
    new_file = lc.snapshots_dir / "new_crop.jpg"
    other = lc.snapshots_dir / "notes.txt"
    for p in (old_file, new_file, other):
        p.write_bytes(b"x")
    forty_days_ago = time.time() - 40 * 86400
    os.utime(old_file, (forty_days_ago, forty_days_ago))
    os.utime(other, (forty_days_ago, forty_days_ago))
    old_event = {k: None for k in EVENT_KEYS} | {"ts": iso(forty_days_ago)}
    lc.events_file.write_text(json.dumps(old_event) + "\n")
    events_before = lc.events_file.read_bytes()

    el = make_logger(tmp_path, obs, storage=StorageConfig(retention_days=30))
    el.close()

    assert not old_file.exists()
    assert new_file.exists()
    assert other.exists()  # only image files are swept
    assert lc.events_file.read_bytes() == events_before


# ---------------------------------------------------------------- paths

def test_relative_paths_resolve_against_project_root_not_cwd(tmp_path, obs, monkeypatch) -> None:
    project = tmp_path / "project"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    el = EventLogger(LoggingConfig(), StorageConfig(), "run12345", obs=obs,
                     dedupe_within_seconds=10, root=project)
    f = frame(90, T0)
    el.handle(f, [det(f)])
    el.close()

    assert el.events_path == project / "data" / "events.jsonl"
    assert el.events_path.exists()
    (ev,) = events(el)
    assert ev["snapshot_crop"].startswith("data/snapshots/")
    assert list(elsewhere.iterdir()) == []


def test_default_root_is_the_project_directory(obs, tmp_path) -> None:
    lc = LoggingConfig(events_file=Path("data/events.jsonl"), snapshots_dir=tmp_path / "s")
    el = EventLogger(lc, StorageConfig(), "r", obs=obs, dedupe_within_seconds=10)
    assert el.events_path == ROOT / "data" / "events.jsonl"
    el.close()


def test_manual_snapshot_path_is_under_snapshots_dir(tmp_path, obs) -> None:
    el = make_logger(tmp_path, obs)
    path = el.manual_snapshot_path(frame(42, T0))
    assert path.parent == el.snapshots_dir
    assert path.name.startswith("manual_") and "seq000042" in path.name
    el.close()


# ---------------------------------------------------------------- integration

def test_integration_two_events_from_synth_video(synth_video_15s, tmp_path, fake_predictor) -> None:
    """Fake bird on frames 90, 120, 150 (same box) and 400 (different box)."""
    box_a, box_b = (300, 150, 30, 20), (100, 60, 30, 20)
    fake = fake_predictor({90: [box_a], 120: [box_a], 150: [box_a], 400: [box_b]})
    data = tmp_path / "data"
    # N=10 so frame 400 is a gated frame (with N=30 the gates are 90, 120, ... 390, 420).
    cfg = write_test_config(tmp_path, **{"process_every_n_frames: 30": "process_every_n_frames: 10"})

    code = main.main(["--source", str(synth_video_15s), "--headless", "--no-pace",
                      "--config", str(cfg), "--log-dir", str(tmp_path / "logs"),
                      "--roi-file", str(tmp_path / "no_roi.json")], predictor=fake)

    assert code == 0
    evs = read_log(data / "events.jsonl")
    assert len(evs) == 2
    first, second = sorted(evs, key=lambda e: e["frame_seq"])
    assert (first["frame_seq"], first["visit_frames"], first["bbox_xywh"]) == (90, 3, list(box_a))
    assert (second["frame_seq"], second["visit_frames"], second["bbox_xywh"]) == (400, 1, list(box_b))
    for ev in evs:
        assert set(ev) == EVENT_KEYS
        assert Path(ev["snapshot_crop"]).exists() and Path(ev["snapshot_full"]).exists()
    (log,) = (tmp_path / "logs").glob("run_*.jsonl")
    persist = [l for l in read_log(log) if l["stage"] == "persist"]
    assert [l["event"] for l in persist].count("success") == 2
    assert [l["context"].get("reason") for l in persist].count("dedupe") == 2
