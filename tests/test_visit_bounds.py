"""Bounded visits (max_visit_seconds) and crash recovery (open_visits.json).

Frames are driven straight into EventLogger with explicit capture times, as
Detector output would arrive: one confirmation of the same box per second."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from camera import Frame
from config_schema import DetectionConfig, LoggingConfig, StorageConfig
from detector import Detection
from logger import EventLogger
from tests.conftest import read_log

T0 = datetime(2026, 9, 30, 6, 0, 0).timestamp()
BOX = (100, 80, 30, 20)


def iso(t: float) -> str:
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec="milliseconds")


def make(tmp_path: Path, obs, **kw) -> EventLogger:
    lc = LoggingConfig(events_file=tmp_path / "data" / "events.jsonl",
                       snapshots_dir=tmp_path / "data" / "snapshots")
    return EventLogger(lc, StorageConfig(), kw.pop("run_id", "bound001"), obs=obs,
                       dedupe_within_seconds=10, **kw)


def confirm(el: EventLogger, seq: int, t: float) -> None:
    img = np.full((240, 320, 3), 128, dtype=np.uint8)
    f = Frame(image=img, captured_at=0.0, seq=seq, captured_wall_time=t)
    el.handle(f, [Detection("bird", 0.9, BOX, seq, t, crop_xyxy=(90, 70, 140, 110))])


def events(el: EventLogger) -> list[dict]:
    return read_log(el.events_path) if el.events_path.exists() else []


# ---------------------------------------------------------------- max_visit_seconds

def test_max_visit_seconds_defaults_to_600() -> None:
    assert DetectionConfig().max_visit_seconds == 600


def test_a_bird_confirmed_forever_is_cut_every_max_visit_seconds(tmp_path, obs) -> None:
    """Confirmed every second for 35 s with max_visit_seconds=10: a line every
    10 s of clip time, truncated except the last (closed at the end of the run)."""
    el = make(tmp_path, obs, max_visit_seconds=10)
    for i in range(35):
        confirm(el, 100 + i, T0 + i)
    el.close()

    evs = events(el)
    assert [e["ts"] for e in evs] == [iso(T0), iso(T0 + 10), iso(T0 + 20), iso(T0 + 30)]
    assert [e["last_seen"] for e in evs] == [iso(T0 + 9), iso(T0 + 19), iso(T0 + 29), iso(T0 + 34)]
    assert [e["truncated"] for e in evs] == [True, True, True, False]
    assert [e["visit_frames"] for e in evs] == [10, 10, 10, 5]
    assert not any(e["recovered"] for e in evs)


def test_every_line_carries_truncated_and_recovered(tmp_path, obs) -> None:
    el = make(tmp_path, obs)
    confirm(el, 1, T0)
    el.close()
    (ev,) = events(el)
    assert (ev["truncated"], ev["recovered"]) == (False, False)


# ---------------------------------------------------------------- checkpoint + recovery

def test_open_visits_are_checkpointed(tmp_path, obs) -> None:
    """Written when a visit opens or closes, and every 60 s of clip time."""
    el = make(tmp_path, obs)
    sidecar = el.events_path.parent / "open_visits.json"
    confirm(el, 1, T0)
    assert [v["last_seen"] for v in json.loads(sidecar.read_text())] == [iso(T0)]  # on open
    for i in range(1, 59):
        confirm(el, 1 + i, T0 + i)
    assert [v["last_seen"] for v in json.loads(sidecar.read_text())] == [iso(T0)]  # not yet
    for i in range(59, 62):
        confirm(el, 1 + i, T0 + i)
    (visit,) = json.loads(sidecar.read_text())
    assert visit["last_seen"] == iso(T0 + 60) and visit["visit_frames"] == 61   # 60 s later
    el.close()
    assert not sidecar.exists()                     # a clean close leaves no sidecar


def test_a_hard_kill_is_recovered_on_the_next_start(tmp_path, obs) -> None:
    el = make(tmp_path, obs)
    for i in range(130):
        confirm(el, 1 + i, T0 + i)
    # Hard kill: no close(); the open visit was never written.
    assert events(el) == []

    restarted = make(tmp_path, obs, run_id="bound002")
    (ev,) = events(restarted)
    assert ev["recovered"] is True and ev["truncated"] is False
    assert ev["run_id"] == "bound001" and ev["ts"] == iso(T0)
    assert ev["last_seen"] == iso(T0 + 120)          # the last 60 s checkpoint
    assert not (restarted.events_path.parent / "open_visits.json").exists()
    restarted.close()
    assert len(events(restarted)) == 1               # recovered once, not again at close


def test_a_visit_already_written_is_not_recovered_twice(tmp_path, obs) -> None:
    """A kill between appending the line and updating the sidecar leaves the
    visit in both; recovery skips it (same run_id and frame_seq)."""
    el = make(tmp_path, obs)
    confirm(el, 1, T0)
    sidecar = el.events_path.parent / "open_visits.json"
    stale = sidecar.read_text()
    confirm(el, 500, T0 + 30)                       # first visit expired and written
    sidecar.write_text(stale)                       # as if the kill came before the rewrite

    restarted = make(tmp_path, obs, run_id="bound002")
    assert [e["frame_seq"] for e in events(restarted)] == [1]
    restarted.close()


def test_recovered_lines_count_toward_the_daily_cap(tmp_path, obs) -> None:
    el = make(tmp_path, obs)
    confirm(el, 1, T0)
    restarted = EventLogger(
        LoggingConfig(events_file=tmp_path / "data" / "events.jsonl",
                      snapshots_dir=tmp_path / "data" / "snapshots"),
        StorageConfig(max_events_per_day=1), "bound002", obs=obs, dedupe_within_seconds=10)
    confirm(restarted, 2, T0 + 100)                 # a new visit the same day: over the cap
    restarted.close()
    assert [e["recovered"] for e in events(restarted)] == [True]
