from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from obs import ObsLogger  # noqa: E402
from scripts.make_synth_video import make_video  # noqa: E402
from tests.fakes import FakePredictor  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: loads the real yolov8n.pt (one test only)")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    # Unit tests must never import ultralytics in-process; the one slow test
    # runs the real model in a subprocess.
    if "ultralytics" in sys.modules:
        print("\nERROR: ultralytics was imported in the test process", file=sys.stderr)
        session.exitstatus = 1


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
