"""Open visits are flushed to events.jsonl on SIGINT, SIGTERM (POSIX),
Ctrl-Break (Windows) and a normal exit. End to end in a subprocess: a fake
bird opens a visit, and the signal arrives while it is still open (the
dedupe window is 10 s, so the visit cannot have expired on its own)."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import ROOT, read_log, write_test_config

WRAPPER = ROOT / "tests" / "main_with_fakes.py"
BOX = [100, 110, 20, 20]

if sys.platform == "win32":
    # Windows has no catchable SIGTERM: os.kill(pid, SIGTERM) is TerminateProcess,
    # a hard kill. Ctrl-C cannot be sent to a single child either; Ctrl-Break
    # (SIGBREAK in the child) is the console signal a process group can receive.
    SIGNALS = [pytest.param(signal.CTRL_BREAK_EVENT, id="CTRL_BREAK")]
    POPEN_FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP
else:
    SIGNALS = [pytest.param(signal.SIGINT, id="SIGINT"), pytest.param(signal.SIGTERM, id="SIGTERM")]
    POPEN_FLAGS = 0


def start(tmp_path: Path) -> subprocess.Popen:
    cfg = write_test_config(tmp_path, **{"motion_warmup_frames: 60": "motion_warmup_frames: 2",
                                         "process_every_n_frames: 30": "process_every_n_frames: 1"})
    # An endless fake source with a fake bird on frames 5..40: one visit, open.
    fakes = {"capture": {}, "predictor": {"boxes_by_seq": {str(s): [BOX] for s in range(5, 41)}}}
    return subprocess.Popen(
        [sys.executable, str(WRAPPER), "--source", "endless.mp4", "--headless", "--no-pace",
         "--config", str(cfg), "--log-dir", str(tmp_path / "logs"),
         "--roi-file", str(tmp_path / "no_roi.json")],
        cwd=tmp_path, env={**os.environ, "CTB_FAKES": json.dumps(fakes)},
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, creationflags=POPEN_FLAGS,
    )


def wait_for_detection(proc: subprocess.Popen) -> list[str]:
    seen: list[str] = []
    for line in proc.stderr:            # blocks on the pipe; no polling sleep
        seen.append(line)
        if "BIRD detected" in line:
            return seen
    raise AssertionError("process ended before a detection:\n" + "".join(seen[-30:]))


@pytest.mark.parametrize("sig", SIGNALS)
def test_open_visit_is_flushed_on_signal(tmp_path, sig) -> None:
    proc = start(tmp_path)
    try:
        wait_for_detection(proc)
        events_file = tmp_path / "data" / "events.jsonl"
        assert not events_file.exists()  # the visit is open, nothing written yet
        os.kill(proc.pid, sig)
        _, stderr = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == 0, stderr[-2000:]
    (ev,) = read_log(events_file)
    assert ev["frame_seq"] >= 5 and ev["bbox_xywh"] == BOX
    assert ev["visit_frames"] >= 1
    (log,) = (tmp_path / "logs").glob("run_*.jsonl")
    run_end = [l for l in read_log(log) if l["stage"] == "run" and l["event"] != "start"]
    assert run_end[-1]["event"] == "skip" and run_end[-1]["context"]["signal"]


def test_open_visit_is_flushed_at_end_of_file(synth_video_2s, tmp_path, fake_predictor) -> None:
    """Normal exit: the source ends while the visit is still open."""
    import main

    cfg = write_test_config(tmp_path, **{"motion_warmup_frames: 60": "motion_warmup_frames: 2",
                                         "process_every_n_frames: 30": "process_every_n_frames: 1"})
    fake = fake_predictor({seq: [tuple(BOX)] for seq in range(50, 61)})
    code = main.main(["--source", str(synth_video_2s), "--headless", "--no-pace",
                      "--config", str(cfg), "--log-dir", str(tmp_path / "logs"),
                      "--roi-file", str(tmp_path / "no_roi.json")], predictor=fake)

    assert code == 0
    (ev,) = read_log(tmp_path / "data" / "events.jsonl")
    assert ev["visit_frames"] >= 1


def test_signal_handlers_are_restored_after_main(synth_video_2s, tmp_path, fake_predictor) -> None:
    """main() installs its handlers for the run only (tests call it in-process)."""
    import main

    before = {s: signal.getsignal(s) for s in main.SHUTDOWN_SIGNALS}
    main.main(["--source", str(synth_video_2s), "--headless", "--no-pace",
               "--log-dir", str(tmp_path / "logs"), "--roi-file", str(tmp_path / "no_roi.json")],
              predictor=fake_predictor())
    assert {s: signal.getsignal(s) for s in main.SHUTDOWN_SIGNALS} == before
