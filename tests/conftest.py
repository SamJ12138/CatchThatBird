from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from loguru import logger

from obs import ObsLogger
from scripts.make_synth_video import make_video
from tests.fakes import FakePredictor

ROOT = Path(__file__).resolve().parent.parent  # pyproject.toml puts it on sys.path


WALL_TIME_LIMIT_S = 5.0   # per test, setup + call + teardown (the `slow` test is exempt)
SLEEP_LIMIT_S = 0.1       # longest real time.sleep() a single call may make

# nodeid -> {"setup": s, "call": s, "teardown": s}, filled as reports arrive.
_durations: dict[str, dict[str, float]] = defaultdict(dict)

# One-time environment costs paid in pytest_sessionstart, outside every
# test's budget: matplotlib builds its font cache on first import (20 s on
# a fresh GitHub Windows runner).
WARMED: list[str] = []


def pytest_sessionstart(session: pytest.Session) -> None:
    try:
        import matplotlib.font_manager  # noqa: F401
    except ImportError:  # dev dependency; the plot tests fail on their own without it
        return
    WARMED.append("matplotlib.font_manager")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if "slow" not in report.keywords:
        _durations[report.nodeid][report.when] = report.duration


def over_budget(durations: dict[str, dict[str, float]],
                      limit: float) -> list[tuple[str, float]]:
    """(nodeid, total seconds) for every test whose phases add up to more than `limit`."""
    return sorted((nodeid, round(sum(phases.values()), 2))
                  for nodeid, phases in durations.items() if sum(phases.values()) > limit)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    # Unit tests must never import ultralytics in-process; the one slow test
    # runs the real model in a subprocess.
    if "ultralytics" in sys.modules:
        print("\nERROR: ultralytics was imported in the test process", file=sys.stderr)
        session.exitstatus = 1
    # Session-level wall-time guard: timing tests run on fake clocks, so a
    # test that takes this long is waiting on real time somewhere.
    for nodeid, total in over_budget(_durations, WALL_TIME_LIMIT_S):
        print(f"\nERROR: {nodeid} took {total:.2f}s (limit {WALL_TIME_LIMIT_S:g}s)",
              file=sys.stderr)
        session.exitstatus = 1


def guard_sleep(real_sleep: Callable[[float], None], limit: float,
                violations: list[float]) -> Callable[[float], None]:
    """time.sleep stand-in that records every call longer than `limit`."""
    def sleep(seconds: float) -> None:
        if seconds > limit:
            violations.append(seconds)
        real_sleep(seconds)
    return sleep


@pytest.fixture(autouse=True)
def no_long_real_sleep(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Fails a test (any thread) that calls time.sleep() for more than
    SLEEP_LIMIT_S. Delays under test go through a FakeClock instead."""
    if request.node.get_closest_marker("slow"):
        yield
        return
    violations: list[float] = []
    monkeypatch.setattr(time, "sleep", guard_sleep(time.sleep, SLEEP_LIMIT_S, violations))
    yield
    assert not violations, f"real time.sleep() longer than {SLEEP_LIMIT_S}s: {violations}"


# ---------------------------------------------------------------- config isolation

def write_test_config(directory: Path, **replacements: str) -> Path:
    """config.yaml copy whose events/snapshots live under `directory`/data."""
    text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    data = (directory / "data").as_posix()
    text = text.replace("events_file: data/events.jsonl", f'events_file: "{data}/events.jsonl"')
    text = text.replace("snapshots_dir: data/snapshots", f'snapshots_dir: "{data}/snapshots"')
    for old, new in replacements.items():
        assert old in text, old
        text = text.replace(old, new)
    path = directory / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No test writes events or snapshots into the real project data/."""
    import main

    path = write_test_config(tmp_path)
    monkeypatch.setattr(main, "CONFIG_FILE", path)
    return path


# ---------------------------------------------------------------- videos

@pytest.fixture
def synth_video_2s(tmp_path: Path) -> Path:
    return make_video(tmp_path / "synth_2s.mp4", seconds=2, width=640, height=360)


@pytest.fixture
def synth_video_2s_1080p(tmp_path: Path) -> Path:
    return make_video(tmp_path / "synth_2s_1080p.mp4", seconds=2, width=1920, height=1080)


@pytest.fixture(scope="session")
def synth_video_15s(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("video") / "synth_15s.mp4"
    return make_video(out, seconds=15, width=640, height=360)


# ---------------------------------------------------------------- fakes

@pytest.fixture
def fake_predictor() -> Callable[..., FakePredictor]:
    """Factory: fake_predictor({90: [(x, y, w, h)]}, conf=0.8, ...)."""
    return FakePredictor


# ---------------------------------------------------------------- logging

@pytest.fixture
def obs(tmp_path: Path):
    o = ObsLogger("test0001", tmp_path / "logs")
    yield o
    o.close()


def read_log(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture
def log_messages():
    """Captured loguru records as (level_name, message)."""
    records: list[tuple[str, str]] = []
    sink_id = logger.add(lambda m: records.append((m.record["level"].name, m.record["message"])),
                         level="DEBUG")
    yield records
    try:
        logger.remove(sink_id)
    except ValueError:  # main.configure_logger() already removed it
        pass


# ---------------------------------------------------------------- frames

def grey_frame(w: int = 320, h: int = 240) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


def frame_with_block(x: int, y: int, size: int = 20, w: int = 320, h: int = 240) -> np.ndarray:
    img = grey_frame(w, h)
    img[y:y + size, x:x + size] = 40
    return img
