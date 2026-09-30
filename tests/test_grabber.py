from __future__ import annotations

import time

import pytest

from camera import FrameGrabber
from tests.fakes import FakeClock


def consume_all(grabber: FrameGrabber, timeout: float = 20.0) -> list[int]:
    """Main-loop stand-in: take every new frame, ack it, stop at end of file."""
    seen: list[int] = []
    last = 0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = grabber.read_latest()
        if frame is not None and frame.seq != last:
            seen.append(frame.seq)
            last = frame.seq
            grabber.ack(frame.seq)
        elif grabber.finished and grabber.read_latest().seq == last:
            return seen
        else:
            time.sleep(0.001)
    raise AssertionError(f"source did not finish within {timeout}s; saw {len(seen)} frames")


def test_no_pace_delivers_every_frame_in_order(synth_video_2s, obs) -> None:
    grabber = FrameGrabber(0, 640, 360, 30, obs=obs, source=str(synth_video_2s), pace=False)
    with grabber:
        seen = consume_all(grabber)
    assert seen == list(range(1, 61))


def test_wait_for_frame_after_returns_on_the_next_frame(synth_video_2s, obs) -> None:
    grabber = FrameGrabber(0, 640, 360, 30, obs=obs, source=str(synth_video_2s), pace=False)
    with grabber:
        grabber.wait_for_frame_after(0, timeout=5)
        first = grabber.read_latest()
        assert first is not None and first.seq == 1
        grabber.ack(1)
        t0 = time.monotonic()
        grabber.wait_for_frame_after(1, timeout=5)
        assert grabber.read_latest().seq == 2
        assert time.monotonic() - t0 < 1.0     # woken by the publish, not the timeout
        seen = consume_all(grabber)
    assert seen[-1] == 60
    t0 = time.monotonic()
    grabber.wait_for_frame_after(60, timeout=5)  # finished: returns at once
    assert time.monotonic() - t0 < 1.0


def test_paced_source_runs_at_file_rate(synth_video_2s, obs) -> None:
    clock = FakeClock()
    t0 = clock.monotonic()
    grabber = FrameGrabber(0, 640, 360, 30, obs=obs, source=str(synth_video_2s),
                           clock=clock.monotonic, sleep=clock.sleep)
    with grabber:
        consume_all(grabber)
    # 60 frames at 30 fps, in fake time: one frame interval after each frame.
    assert clock.monotonic() - t0 == pytest.approx(2.0, abs=1 / 30)
    assert all(s <= 1 / 30 + 1e-9 for s in clock.slept)


def test_file_frames_carry_media_time(synth_video_2s, obs) -> None:
    """File sources: captured_wall_time = time at the first read + (seq - 1) / fps,
    so --no-pace runs keep the clip's own timeline (visit durations, the
    dedupe window). Cameras keep the wall clock."""
    grabber = FrameGrabber(0, 640, 360, 30, obs=obs, source=str(synth_video_2s), pace=False)
    times: dict[int, float] = {}
    with grabber:
        last = 0
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            frame = grabber.read_latest()
            if frame is not None and frame.seq != last:
                times[frame.seq] = frame.captured_wall_time
                last = frame.seq
                grabber.ack(frame.seq)
            elif grabber.finished and grabber.read_latest().seq == last:
                break
    assert len(times) == 60
    for seq, t in times.items():
        assert t - times[1] == pytest.approx((seq - 1) / 30, abs=1e-6)
