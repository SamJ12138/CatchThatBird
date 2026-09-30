"""ROI load / save / mismatch handling (obs #7, #9, #10, #11)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import main
from tests.conftest import read_log

SAVED_1080P = {"x": 710, "y": 426, "w": 626, "h": 471, "frame_width": 1920, "frame_height": 1080}


def frame(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


@pytest.fixture
def dialog(monkeypatch):
    """Replace the OpenCV ROI dialog. Set dialog.selection before calling."""
    class Dialog:
        selection = (10, 20, 30, 40)
        calls = 0

    def select_roi(_window, _image, **_kw):
        Dialog.calls += 1
        return Dialog.selection

    monkeypatch.setattr(main.cv2, "selectROI", select_roi)
    for fn in ("namedWindow", "resizeWindow", "destroyWindow"):
        monkeypatch.setattr(main.cv2, fn, lambda *a, **k: None)
    return Dialog


def write_roi(path: Path, data) -> Path:
    path.write_text(data if isinstance(data, str) else json.dumps(data))
    return path


def load(path: Path, img: np.ndarray, obs, *, headless: bool, force: bool = False):
    return main.load_or_select_roi(img, path, force, headless=headless, obs=obs)


# ---------------------------------------------------------------- load / save

def test_load_exact_match(tmp_path, obs, dialog) -> None:
    path = write_roi(tmp_path / "roi.json", SAVED_1080P)
    assert load(path, frame(1920, 1080), obs, headless=True) == (710, 426, 626, 471)
    assert dialog.calls == 0


def test_dialog_selection_is_saved_with_frame_size(tmp_path, obs, dialog) -> None:
    path = tmp_path / "roi.json"
    assert load(path, frame(640, 360), obs, headless=False) == (10, 20, 30, 40)
    saved = json.loads(path.read_text())
    assert saved == {"x": 10, "y": 20, "w": 30, "h": 40, "frame_width": 640, "frame_height": 360}


def test_headless_without_roi_file_uses_whole_frame(tmp_path, obs, dialog) -> None:
    assert load(tmp_path / "missing.json", frame(640, 360), obs, headless=True) is None
    assert dialog.calls == 0


def test_headless_falls_back_to_the_example_roi_with_a_warning(tmp_path, obs, dialog,
                                                               log_messages) -> None:
    """No roi.json yet (a fresh clone): headless runs use roi.example.json
    next to it, rescaled to the frame, and say so. Nothing is written."""
    write_roi(tmp_path / "roi.example.json", SAVED_1080P)
    path = tmp_path / "roi.json"

    roi = load(path, frame(640, 360), obs, headless=True)

    assert roi == (237, 142, 209, 157)
    assert not path.exists()
    assert dialog.calls == 0
    assert any(level == "WARNING" and "roi.example.json" in m for level, m in log_messages)
    obs.close()
    end = [l for l in read_log(obs.path) if l["stage"] == "roi_load" and l["event"] != "start"][-1]
    assert end["event"] == "success" and end["context"]["source"] == "example_rescaled"


def test_interactive_without_roi_file_opens_the_dialog_even_with_an_example(tmp_path, obs,
                                                                            dialog) -> None:
    write_roi(tmp_path / "roi.example.json", SAVED_1080P)
    path = tmp_path / "roi.json"
    assert load(path, frame(640, 360), obs, headless=False) == (10, 20, 30, 40)
    assert dialog.calls == 1 and path.exists()


def test_the_repo_ships_an_example_roi_next_to_the_default_roi_file() -> None:
    assert main.ROI_FILE.name == "roi.json"
    example = main.example_roi_path(main.ROI_FILE)
    assert example == main.ROI_FILE.with_name("roi.example.json")
    data = json.loads(example.read_text())
    assert {"x", "y", "w", "h", "frame_width", "frame_height"} <= set(data)


# ---------------------------------------------------------------- mismatch

@pytest.mark.parametrize("headless", [True, False])
def test_same_aspect_ratio_is_rescaled_with_warning(tmp_path, obs, dialog, log_messages, headless) -> None:
    """obs #7/#9: a 16:9 ROI saved at 1080p applies to a 640x360 stream."""
    path = write_roi(tmp_path / "roi.json", SAVED_1080P)

    roi = load(path, frame(640, 360), obs, headless=headless)

    assert roi == (237, 142, 209, 157)  # 710/3, 426/3, 626/3, 471/3 rounded
    assert dialog.calls == 0
    assert any(level == "WARNING" and "rescal" in m.lower() for level, m in log_messages)
    assert json.loads(path.read_text()) == SAVED_1080P  # the saved file is not rewritten


def test_different_aspect_ratio_headless_exits_1(tmp_path, obs, dialog) -> None:
    path = write_roi(tmp_path / "roi.json", SAVED_1080P)

    with pytest.raises(main.ExitError) as exc:
        load(path, frame(640, 480), obs, headless=True)

    assert exc.value.code == 1
    assert "640x480" in str(exc.value)
    assert dialog.calls == 0


def test_different_aspect_ratio_interactive_reselects(tmp_path, obs, dialog) -> None:
    path = write_roi(tmp_path / "roi.json", SAVED_1080P)

    assert load(path, frame(640, 480), obs, headless=False) == (10, 20, 30, 40)
    assert dialog.calls == 1


# ---------------------------------------------------------------- corrupt file

def test_corrupt_roi_headless_exits_1(tmp_path, obs, dialog) -> None:
    """obs #10: a corrupt roi.json no longer silently widens the watch area."""
    path = write_roi(tmp_path / "roi.json", "{not json")

    with pytest.raises(main.ExitError) as exc:
        load(path, frame(640, 360), obs, headless=True)

    assert exc.value.code == 1
    obs.close()
    fails = [l for l in read_log(obs.path) if l["stage"] == "roi_load" and l["event"] == "fail"]
    assert fails and fails[0]["error_type"] == "parse"


@pytest.mark.parametrize("content", ["{not json", '{"x": 1}', '{"x": null, "y": 0, "w": 1, "h": 1, '
                                     '"frame_width": 640, "frame_height": 360}'])
def test_corrupt_roi_interactive_reselects(tmp_path, obs, dialog, content) -> None:
    path = write_roi(tmp_path / "roi.json", content)
    assert load(path, frame(640, 360), obs, headless=False) == (10, 20, 30, 40)
    assert dialog.calls == 1


def test_unexpected_error_while_loading_roi_is_not_swallowed(tmp_path, obs, dialog, monkeypatch) -> None:
    """obs #10: the except is narrowed to parse/IO errors."""
    path = write_roi(tmp_path / "roi.json", SAVED_1080P)

    def boom(_text):
        raise RuntimeError("boom")

    monkeypatch.setattr(main.json, "loads", boom)
    with pytest.raises(RuntimeError, match="boom"):
        load(path, frame(1920, 1080), obs, headless=False)


# ---------------------------------------------------------------- whole frame

def test_explicit_whole_frame_choice_is_saved_and_reused(tmp_path, obs, dialog) -> None:
    """obs #11: cancelling the dialog is remembered instead of re-prompting."""
    path = tmp_path / "roi.json"
    dialog.selection = (0, 0, 0, 0)

    assert load(path, frame(640, 360), obs, headless=False) is None
    assert json.loads(path.read_text())["whole_frame"] is True

    assert load(path, frame(640, 360), obs, headless=False) is None
    assert load(path, frame(1920, 1080), obs, headless=True) is None  # any resolution
    assert dialog.calls == 1


def test_select_roi_flag_overrides_saved_whole_frame(tmp_path, obs, dialog) -> None:
    path = write_roi(tmp_path / "roi.json", {"whole_frame": True, "frame_width": 640, "frame_height": 360})
    assert load(path, frame(640, 360), obs, headless=False, force=True) == (10, 20, 30, 40)
