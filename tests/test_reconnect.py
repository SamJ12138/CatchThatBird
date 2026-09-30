"""Camera reconnect with backoff (obs #1) and stop() with a stuck reader (obs #8).
Timings are scaled down through constructor arguments; the defaults are
checked separately."""
from __future__ import annotations

import threading
import time

import pytest

import camera
from camera import FrameGrabber
from tests.conftest import read_log
from tests.fakes import FakeDevice


def grabber(obs, **kw) -> FrameGrabber:
    return FrameGrabber(0, 320, 240, 30, obs=obs, **kw)


def wait_for(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def device(monkeypatch):
    def install(**kw) -> FakeDevice:
        dev = FakeDevice(**kw)
        monkeypatch.setattr(camera.cv2, "VideoCapture", dev)
        return dev
    return install


def test_default_reconnect_policy() -> None:
    """2 s of failures, then backoff 1, 2, 4 ... capped at 30 s."""
    assert FrameGrabber.RECONNECT_AFTER_S == 2.0
    g_delays = [camera.backoff_delay(k) for k in range(8)]
    assert g_delays == [1, 2, 4, 8, 16, 30, 30, 30]


def test_stalled_camera_is_closed_and_reopened(obs, device) -> None:
    dev = device(fail_reads=8)  # ~0.4 s of failed reads at 50 ms each
    g = grabber(obs, reconnect_after_s=0.2, backoff_s=(0.1, 0.2, 0.4))
    with g:
        assert wait_for(lambda: g.read_latest() is not None), "no frame after reconnect"

    assert len(dev.open_attempts) == 2          # initial open + one reconnect
    assert dev.captures[0].released             # the stalled handle was closed
    obs.close()
    lines = read_log(obs.path)
    reconnect = [l for l in lines if l["stage"] == "capture_reconnect" and l["event"] != "start"]
    assert [l["event"] for l in reconnect] == ["success"]
    assert reconnect[0]["context"]["attempts"] == 1


def test_reopen_backoff_grows_and_is_capped(obs, device) -> None:
    dev = device(fail_reads=5)
    g = grabber(obs, reconnect_after_s=0.2, backoff_s=(0.1, 0.2, 0.4))
    with g:
        dev.fail_opens = 4                      # the reopens fail, not the first open
        assert wait_for(lambda: g.read_latest() is not None, timeout=15)

    attempts = dev.open_attempts[1:]            # skip the initial open
    assert len(attempts) == 5                   # 4 failed reopens, then success
    gaps = [b - a for a, b in zip(attempts, attempts[1:])]
    for gap, expected in zip(gaps, [0.2, 0.4, 0.4, 0.4]):  # 0.1 before the first
        assert expected * 0.9 <= gap < expected + 0.3, gaps


def test_backoff_resets_after_a_good_frame(obs, device) -> None:
    dev = device(fail_reads=5)
    g = grabber(obs, reconnect_after_s=0.2, backoff_s=(0.1, 0.2, 0.4))
    with g:
        assert wait_for(lambda: g.read_latest() is not None)
        dev.fail_reads = 5                      # stalls again later
        seq = g.read_latest().seq
        assert wait_for(lambda: len(dev.open_attempts) == 3)
        assert wait_for(lambda: g.read_latest().seq > seq + 1)

    gap_after_stall = dev.open_attempts[2] - dev.open_attempts[1]
    assert gap_after_stall > 0.2                # ~0.25 stall + 0.1 delay: first step again


def test_stop_during_backoff_returns_promptly(obs, device) -> None:
    dev = device(fail_reads=100)
    g = grabber(obs, reconnect_after_s=0.1, backoff_s=(5.0,))
    g.start()
    dev.fail_opens = 100
    time.sleep(0.4)                             # now waiting 5 s before a reopen
    t0 = time.monotonic()
    g.stop()
    assert time.monotonic() - t0 < 1.0


def test_file_source_treats_a_failed_read_as_end_of_file(obs, device) -> None:
    dev = device(fail_reads=3)
    g = FrameGrabber(0, 320, 240, 30, obs=obs, source="clip.mp4", pace=False,
                     reconnect_after_s=0.0)
    with g:
        assert wait_for(lambda: g.finished)
        time.sleep(0.2)
    assert len(dev.open_attempts) == 1          # no reconnect for files
    obs.close()
    assert not any(l["stage"] == "capture_reconnect" for l in read_log(obs.path))


def test_stop_with_stuck_reader_logs_and_keeps_handle(obs, device, log_messages) -> None:
    """obs #8: never release a capture another thread is still reading from."""
    dev = device(block_reads=True)
    g = grabber(obs, join_timeout_s=0.2)
    g.start()
    assert wait_for(lambda: dev.captures and dev.captures[0].reads == 0)
    time.sleep(0.1)

    g.stop()

    assert dev.captures[0].released is False
    assert any(level == "WARNING" and "still" in m for level, m in log_messages)
    obs.close()
    fails = [l for l in read_log(obs.path) if l["stage"] == "capture_close" and l["event"] == "fail"]
    assert fails and fails[0]["error_type"] == "timeout"
    assert any(t.name == "FrameGrabber" for t in threading.enumerate())  # daemon, left behind
