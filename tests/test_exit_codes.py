"""Exit codes and error surfaces, end to end in a subprocess (obs #2, #3, #4, #12,
config errors). Fakes are installed by tests/main_with_fakes.py."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import main
from tests.conftest import ROOT, read_log, write_test_config

WRAPPER = ROOT / "tests" / "main_with_fakes.py"


def run_proc(script: Path, args: list[str], tmp_path: Path, fakes: dict | None = None,
             timeout: float = 30) -> tuple[subprocess.CompletedProcess, list[dict]]:
    log_dir = tmp_path / "logs"
    env = {**os.environ, "CTB_FAKES": json.dumps(fakes or {})}
    if "--config" not in args:
        args = [*args, "--config", str(write_test_config(tmp_path))]
    proc = subprocess.run(
        [sys.executable, str(script), *args, "--log-dir", str(log_dir),
         "--roi-file", str(tmp_path / "no_roi.json")],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=timeout,
        stdin=subprocess.DEVNULL,
    )
    logs = sorted(log_dir.glob("run_*.jsonl"))
    return proc, (read_log(logs[0]) if logs else [])


def terminal(lines: list[dict], stage: str) -> dict:
    ends = [l for l in lines if l["stage"] == stage and l["event"] != "start"]
    assert ends, f"no terminal line for {stage}"
    return ends[-1]


def ts(line: dict) -> float:
    return datetime.fromisoformat(line["ts"]).timestamp()


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def error_lines(stderr: str) -> list[str]:
    return [l for l in ANSI.sub("", stderr).splitlines() if "| ERROR" in l]


# ---------------------------------------------------------------- obs #3

def test_no_first_frame_within_timeout_exits_1(tmp_path) -> None:
    proc, lines = run_proc(
        WRAPPER, ["--source", "fake.mp4", "--headless", "--no-pace", "--first-frame-timeout", "1"],
        tmp_path, {"capture": {"block": True}},
    )
    assert proc.returncode == 1, proc.stderr[-1500:]
    assert "Traceback" not in proc.stderr
    assert any("No frames" in l for l in error_lines(proc.stderr))
    wait = terminal(lines, "first_frame_wait")
    assert (wait["event"], wait["error_type"]) == ("fail", "timeout")


def test_source_that_ends_before_any_frame_exits_1_without_waiting(tmp_path) -> None:
    proc, lines = run_proc(
        WRAPPER, ["--source", "empty.mp4", "--headless", "--no-pace"],
        tmp_path, {"capture": {"frames": 0}},
    )
    assert proc.returncode == 1, proc.stderr[-1500:]
    assert terminal(lines, "first_frame_wait")["duration_ms"] < 1000  # did not sit out 5 s


# ---------------------------------------------------------------- obs #2

def test_grabber_thread_death_exits_nonzero_within_3s(tmp_path) -> None:
    proc, lines = run_proc(
        WRAPPER, ["--source", "fake.mp4", "--headless", "--no-pace"],
        tmp_path, {"capture": {"raise_after": 90}}, timeout=20,
    )
    assert proc.returncode != 0
    died = [l for l in lines if l["context"].get("thread_died")]
    assert died, "grabber death was not logged"
    run_end = terminal(lines, "run")
    assert ts(run_end) - ts(died[0]) < 3.0
    assert any("fake capture exploded" in l for l in error_lines(proc.stderr))


# ---------------------------------------------------------------- obs #4

def test_checklist_eof_exits_2_with_hint(tmp_path) -> None:
    proc, lines = run_proc(
        WRAPPER, ["--headless"], tmp_path, {"devices": [[0, "DJI Osmo Pocket 3"]]},
    )
    assert proc.returncode == 2, proc.stderr[-1500:]
    assert "Traceback" not in proc.stderr
    (msg,) = error_lines(proc.stderr)
    assert "--yes" in msg
    assert terminal(lines, "checklist")["event"] == "fail"


def test_yes_skips_checklist(tmp_path) -> None:
    proc, lines = run_proc(
        WRAPPER, ["--headless", "--yes"], tmp_path,
        {"devices": [[0, "DJI Osmo Pocket 3"]], "capture": {"raise_after": 10}},
    )
    assert "Pre-flight checklist" not in proc.stdout
    assert not any(l["stage"] == "checklist" for l in lines)
    assert terminal(lines, "capture_open")["event"] == "success"


# ---------------------------------------------------------------- bad config

GOOD_CONFIG = (ROOT / "config.yaml").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "content, needle",
    [
        ("camera: [unclosed\n", "YAML"),
        (GOOD_CONFIG.replace("fps: 30", "fps: fast"), "camera.fps"),
        (GOOD_CONFIG.replace("fps: 30", "fps: 30\n  frame_rate: 30"), "camera.frame_rate"),
    ],
    ids=["yaml-syntax", "pydantic-type", "pydantic-extra-key"],
)
def test_bad_config_exits_1_with_one_line_message(tmp_path, content: str, needle: str) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(content, encoding="utf-8")

    proc, lines = run_proc(ROOT / "main.py",
                           ["--config", str(cfg), "--source", "x.mp4", "--headless"], tmp_path)

    assert proc.returncode == 1
    assert "Traceback" not in proc.stdout + proc.stderr
    (msg,) = error_lines(proc.stderr)
    assert needle in msg and str(cfg.name) in msg
    load = terminal(lines, "config_load")
    assert (load["event"], load["error_type"]) == ("fail", "parse")


def test_missing_config_exits_1(tmp_path) -> None:
    proc, _ = run_proc(ROOT / "main.py",
                       ["--config", str(tmp_path / "nope.yaml"), "--source", "x.mp4", "--headless"],
                       tmp_path)
    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr
    assert len(error_lines(proc.stderr)) == 1


# ---------------------------------------------------------------- obs #12

def test_snapshot_write_failure_is_reported(tmp_path, obs, log_messages, monkeypatch) -> None:
    monkeypatch.setattr(main.cv2, "imwrite", lambda *a, **k: False)

    ok = main.save_snapshot(tmp_path / "snap.jpg", np.zeros((8, 8, 3), np.uint8), obs, frame_seq=7)

    assert ok is False
    assert not any("Saved" in m for _, m in log_messages)
    assert any(level == "ERROR" and "snap.jpg" in m for level, m in log_messages)
    obs.close()
    fail = terminal([l for l in read_log(obs.path)], "snapshot_save")
    assert (fail["event"], fail["frame_seq"]) == ("fail", 7)


def test_snapshot_write_success_is_reported(tmp_path, obs, log_messages) -> None:
    path = tmp_path / "snap.jpg"
    assert main.save_snapshot(path, np.zeros((8, 8, 3), np.uint8), obs, frame_seq=3) is True
    assert path.exists()
    assert any("Saved" in m for _, m in log_messages)
