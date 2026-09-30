from __future__ import annotations

import time

from camera import FrameGrabber


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


def test_paced_source_runs_at_file_rate(synth_video_2s, obs) -> None:
    grabber = FrameGrabber(0, 640, 360, 30, obs=obs, source=str(synth_video_2s))
    t0 = time.monotonic()
    with grabber:
        consume_all(grabber)
    assert time.monotonic() - t0 >= 1.8  # 60 frames at 30 fps
