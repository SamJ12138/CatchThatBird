"""Camera reconnect with backoff (obs #1) and stop() with a stuck reader (obs #8).
Every delay runs on a FakeClock: the real policy (2 s stall, 1, 2, 4 ... 30 s
backoff) is exercised in fake seconds, and no test sleeps for real."""
from __future__ import annotations

import threading
import time

import pytest

import camera
from camera import FrameGrabber
from tests.conftest import read_log
from tests.fakes import FakeClock, FakeDevice

STALL_READS = 45  # > 2 s of failed reads at 50 ms each, whatever the float rounding


def grabber(obs, clock: FakeClock, **kw) -> FrameGrabber:
    return FrameGrabber(0, 320, 240, 30, obs=obs, clock=clock.monotonic, sleep=clock.sleep, **kw)


def wait_for(predicate, timeout: float = 5.0) -> bool:
    """Real-time poll for another thread's progress (not a delay under test)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return False


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def device(monkeypatch, clock):
    def install(**kw) -> FakeDevice:
        dev = FakeDevice(clock=clock.monotonic, **kw)
        monkeypatch.setattr(camera.cv2, "VideoCapture", dev)
        return dev
    return install


def test_default_reconnect_policy() -> None:
    """2 s of failures, then backoff 1, 2, 4 ... capped at 30 s."""
    assert FrameGrabber.RECONNECT_AFTER_S == 2.0
    g_delays = [camera.backoff_delay(k) for k in range(8)]
    assert g_delays == [1, 2, 4, 8, 16, 30, 30, 30]


def test_defaults_are_the_real_clock_and_sleep(obs) -> None:
    g = FrameGrabber(0, 320, 240, 30, obs=obs)
    assert g._clock is time.monotonic
    assert g._sleep is time.sleep


def test_stalled_camera_is_closed_and_reopened(obs, clock, device) -> None:
    dev = device(fail_reads=STALL_READS)
    g = grabber(obs, clock)
    with g:
        assert wait_for(lambda: g.read_latest() is not None), "no frame after reconnect"

    assert len(dev.open_attempts) == 2          # initial open + one reconnect
    first = dev.captures[0]
    assert first.released                       # the stalled handle was closed
    assert first.released_at - dev.open_attempts[0] == pytest.approx(2.0, abs=0.051)
    assert dev.open_attempts[1] - first.released_at == pytest.approx(1.0)  # first backoff step
    obs.close()
    lines = read_log(obs.path)
    reconnect = [l for l in lines if l["stage"] == "capture_reconnect" and l["event"] != "start"]
    assert [l["event"] for l in reconnect] == ["success"]
    assert reconnect[0]["context"]["attempts"] == 1


def test_reopen_backoff_grows_and_is_capped(obs, clock, device) -> None:
    dev = device(fail_reads=STALL_READS, fail_reopens=7)
    g = grabber(obs, clock)
    with g:
        assert wait_for(lambda: g.read_latest() is not None)

    reopens = dev.open_attempts[1:]             # skip the initial open
    assert len(reopens) == 8                    # 7 failed reopens, then success
    assert reopens[0] - dev.captures[0].released_at == pytest.approx(1.0)
    gaps = [b - a for a, b in zip(reopens, reopens[1:])]
    assert gaps == pytest.approx([2, 4, 8, 16, 30, 30, 30])


def test_backoff_resets_after_a_good_frame(obs, clock, device) -> None:
    dev = device(fail_reads=STALL_READS, fail_reopens=2)   # first stall: delays 1, 2, 4
    g = grabber(obs, clock)
    with g:
        assert wait_for(lambda: g.read_latest() is not None)
        dev.fail_reads = STALL_READS                         # stalls again later
        assert wait_for(lambda: len(dev.open_attempts) == 5)

    opened = [c for c in dev.captures if c.reads > 0]      # initial, 3rd reopen, 4th reopen
    second_stall = opened[1]
    # Without the reset the next delay would be 8 s (step 4); it is step 1 again.
    assert dev.open_attempts[4] - second_stall.released_at == pytest.approx(1.0)


def test_stop_during_backoff_returns_promptly(obs, clock, device) -> None:
    dev = device(fail_reads=10_000, fail_reopens=10_000)
    g = grabber(obs, clock, backoff_s=(30.0,))
    g.start()
    assert wait_for(lambda: len(dev.open_attempts) >= 3)  # inside the 30 s backoff loop
    t0 = time.monotonic()
    g.stop()
    assert time.monotonic() - t0 < 1.0
    # Every wait is cut into <= STOP_POLL_S slices with a stop check between
    # them, so with the real clock stop() interrupts a 30 s backoff within 0.1 s.
    assert max(clock.slept) <= FrameGrabber.STOP_POLL_S == 0.1
    obs.close()
    ends = [l for l in read_log(obs.path) if l["stage"] == "capture_reconnect" and l["event"] != "start"]
    assert ends[-1]["event"] == "skip" and ends[-1]["context"]["reason"] == "stopped"


def test_file_source_treats_a_failed_read_as_end_of_file(obs, clock, device) -> None:
    dev = device(fail_reads=3)
    g = FrameGrabber(0, 320, 240, 30, obs=obs, source="clip.mp4", pace=False,
                     reconnect_after_s=0.0, clock=clock.monotonic, sleep=clock.sleep)
    with g:
        assert wait_for(lambda: g.finished)
    assert len(dev.open_attempts) == 1          # no reconnect for files
    obs.close()
    assert not any(l["stage"] == "capture_reconnect" for l in read_log(obs.path))


def test_stop_with_stuck_reader_logs_and_keeps_handle(obs, clock, device, log_messages) -> None:
    """obs #8: never release a capture another thread is still reading from."""
    dev = device(block_reads=True)
    g = grabber(obs, clock, join_timeout_s=0.05)
    g.start()
    assert wait_for(lambda: dev.captures and dev.captures[0].in_read.is_set())

    g.stop()

    assert dev.captures[0].released is False
    assert any(level == "WARNING" and "still" in m for level, m in log_messages)
    obs.close()
    fails = [l for l in read_log(obs.path) if l["stage"] == "capture_close" and l["event"] == "fail"]
    assert fails and fails[0]["error_type"] == "timeout"
    assert any(t.name == "FrameGrabber" for t in threading.enumerate())  # daemon, left behind
