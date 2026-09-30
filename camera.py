from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
from loguru import logger

from obs import ObsLogger, describe


# Heuristic name fragments. Matching is case-insensitive.
_POCKET_KEYWORDS = ("dji", "osmo", "pocket")
_BUILTIN_KEYWORDS = (
    "integrated", "built-in", "built in", "internal",
    "lenovo", "hp ", "dell", "surface", "thinkpad", "ideapad",
)


def list_devices(obs: ObsLogger) -> list[tuple[int, str]]:
    """Enumerate video capture devices as [(opencv_index, name), ...].

    On Windows we use pygrabber, whose enumeration order matches OpenCV's
    CAP_DSHOW. If pygrabber is unavailable, we fall back to probing indexes
    0-5 with no name (the user has to identify by elimination)."""
    with obs.span("device_list") as sp:
        try:
            from pygrabber.dshow_graph import FilterGraph  # type: ignore
            graph = FilterGraph()
            names = graph.get_input_devices()
            sp.success({"enumerator": "pygrabber", "devices": names})
            return list(enumerate(names))
        except Exception as e:
            logger.debug(f"pygrabber unavailable ({e}); probing indexes 0-5")
            obs.emit(
                "device_list", "fail", error_type="external_api",
                error_message=describe(e), context={"fallback": "probe_indexes_0_5"},
            )
            found: list[tuple[int, str]] = []
            for idx in range(6):
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                try:
                    if cap.isOpened():
                        found.append((idx, f"<device {idx}>"))
                finally:
                    cap.release()
            sp.success({"enumerator": "probe", "devices": [n for _, n in found]})
            return found


def classify_device(name: str) -> str:
    """Return 'pocket', 'builtin', or 'unknown' based on the device name."""
    lower = name.lower()
    if any(k in lower for k in _POCKET_KEYWORDS):
        return "pocket"
    if any(k in lower for k in _BUILTIN_KEYWORDS):
        return "builtin"
    return "unknown"


def find_pocket_index(devices: list[tuple[int, str]]) -> Optional[int]:
    """Return the index of the first device that looks like a Pocket 3."""
    for idx, name in devices:
        if classify_device(name) == "pocket":
            return idx
    return None


BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)  # camera reopen delays; the last one repeats


def backoff_delay(attempt: int, schedule: tuple[float, ...] = BACKOFF_S) -> float:
    """Delay before reopen attempt `attempt` (0-based): 1, 2, 4 ... capped at 30 s."""
    return schedule[min(attempt, len(schedule) - 1)]


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    captured_at: float  # time.monotonic() taken right after cap.read() returned
    seq: int
    captured_wall_time: float  # time.time() taken at the same moment (for event timestamps)


class FrameGrabber:
    """Threaded UVC capture. Producer always overwrites the latest-frame slot;
    consumer peeks the slot so display never blocks and detection can skip
    re-processing the same seq.

    With `source` set, frames come from a video file instead of a camera. The
    file is read at its own frame rate (like a camera would deliver it), and
    Frame.captured_wall_time is media time (first read + (seq - 1) / fps), and
    `finished` becomes True once the file runs out. With `pace=False` a file
    is read as fast as the consumer takes frames: the producer waits for
    `ack(seq)` before reading the next one, so no frame is skipped.

    Camera only: after `reconnect_after_s` of consecutive failed reads the
    capture is released and reopened, waiting backoff_delay(k) before attempt
    k; the schedule restarts after the next good frame.

    `clock` (default time.monotonic) and `sleep` (default time.sleep) are the
    only time sources for pacing, the stall timer and the backoff, so tests
    can run the real policy on a fake clock. Long waits are cut into
    STOP_POLL_S slices with a stop() check between them."""

    RECONNECT_AFTER_S = 2.0
    JOIN_TIMEOUT_S = 2.0
    STOP_POLL_S = 0.1   # longest single sleep during a backoff wait
    RETRY_READ_S = 0.05

    def __init__(
        self,
        device_index: int,
        width: int,
        height: int,
        fps: int,
        *,
        obs: ObsLogger,
        source: Optional[str] = None,
        pace: bool = True,
        reconnect_after_s: float = RECONNECT_AFTER_S,
        backoff_s: tuple[float, ...] = BACKOFF_S,
        join_timeout_s: float = JOIN_TIMEOUT_S,
        clock: Optional[Callable[[], float]] = None,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        # Resolved here, not as default arguments, so a patched time.sleep
        # (the test suite's real-sleep guard) is seen.
        self._clock = clock if clock is not None else time.monotonic
        self._sleep = sleep if sleep is not None else time.sleep
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps
        self._obs = obs
        self._source = source
        self._pace = pace
        self._reconnect_after_s = reconnect_after_s
        self._backoff_s = backoff_s
        self._join_timeout_s = join_timeout_s
        self._reopen_attempt = 0  # index into the backoff schedule
        self._frame_interval = 1.0 / fps
        self._media_t0: Optional[float] = None  # file sources: wall time of the first frame
        self._finished = threading.Event()
        self._cap: Optional[cv2.VideoCapture] = None
        self._latest: Optional[Frame] = None
        self._lock = threading.Lock()
        self._published = threading.Condition(self._lock)  # new frame, end of file, or death
        self._acked = threading.Condition(threading.Lock())
        self._acked_seq = 0
        self._error: Optional[BaseException] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._seq = 0
        self._frames_captured = 0
        self._read_failures = 0
        self._read_counter = obs.counter(
            "capture_read",
            context_fn=lambda: {"failures_total": self._read_failures},
        )

    def _open_capture(self) -> cv2.VideoCapture:
        if self._source is not None:
            return self._open_file()
        with self._obs.span(
            "capture_open",
            context={"device_index": self._device_index},
            error_type="hardware",
        ) as sp:
            tried: list[str] = []
            for backend, name in [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")]:
                cap = cv2.VideoCapture(self._device_index, backend)
                if cap.isOpened():
                    logger.info(f"Opened camera index {self._device_index} via {name}")
                    cap = self._configure(cap)
                    sp.success({"backend": name, "backends_failed": tried,
                                **self._negotiated})
                    return cap
                cap.release()
                tried.append(name)
                self._obs.emit(
                    "capture_open", "fail", error_type="hardware",
                    error_message=f"backend {name} did not open device",
                    context={"backend": name},
                )
            raise RuntimeError(
                f"Failed to open camera index {self._device_index} "
                f"(tried backends: {', '.join(tried)}). "
                "Confirm the Pocket 3 is in Webcam mode (USB-C connected, screen "
                "shows Webcam) and no other app is holding the camera."
            )

    def _open_file(self) -> cv2.VideoCapture:
        with self._obs.span(
            "capture_open", context={"source": self._source}, error_type="input_invalid"
        ) as sp:
            cap = cv2.VideoCapture(self._source)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"Failed to open video file {self._source}")
            file_fps = cap.get(cv2.CAP_PROP_FPS)
            if file_fps > 0:
                self._frame_interval = 1.0 / file_fps
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(
                f"Opened video file {self._source}: {w}x{h} @ {file_fps:.1f}fps, "
                f"{n} frames"
            )
            sp.success({"width": w, "height": h, "fps": file_fps, "frame_count": n})
            return cap

    def _configure(self, cap: cv2.VideoCapture) -> cv2.VideoCapture:
        # MJPG over USB gives the best 1080p30 headroom on most UVC pipelines.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        # Drivers may ignore this, but ask for the shortest internal queue.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            f"Camera negotiated {actual_w}x{actual_h} @ {actual_fps:.1f}fps "
            f"(requested {self._width}x{self._height}@{self._fps})"
        )
        self._negotiated = {
            "negotiated": [actual_w, actual_h, actual_fps],
            "requested": [self._width, self._height, self._fps],
            "negotiated_matches_request": (actual_w, actual_h) == (self._width, self._height),
        }
        return cap

    def start(self) -> None:
        self._cap = self._open_capture()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="FrameGrabber", daemon=True
        )
        self._thread.start()
        logger.info("FrameGrabber thread started")

    def _run(self) -> None:
        assert self._cap is not None
        next_due = self._clock()
        stalled_since: Optional[float] = None
        try:
            while not self._stop_event.is_set():
                if self._cap is None and not self._reconnect():
                    return  # stopped while waiting to reopen
                t0 = time.perf_counter()
                ok, image = self._cap.read()
                captured_at = self._clock()
                captured_wall_time = time.time()
                read_ms = (time.perf_counter() - t0) * 1000.0
                if not ok or image is None:
                    if self._source is not None:
                        logger.info(
                            f"End of video file after {self._frames_captured} frames"
                        )
                        self._obs.emit(
                            "capture_read", "skip", frame_seq=self._seq,
                            context={"reason": "end_of_file",
                                     "frames_captured": self._frames_captured},
                        )
                        self._finished.set()
                        with self._published:
                            self._published.notify_all()
                        return
                    self._read_failures += 1
                    self._read_counter.record("fail", read_ms)
                    self._obs.emit(
                        "capture_read", "fail", duration_ms=read_ms,
                        error_type="hardware",
                        error_message="cap.read() returned no frame",
                        context={"failures_total": self._read_failures},
                    )
                    if self._read_failures == 1 or self._read_failures % 30 == 0:
                        logger.warning(
                            f"cap.read() failed (total={self._read_failures})"
                        )
                    if stalled_since is None:
                        stalled_since = captured_at
                    elif captured_at - stalled_since >= self._reconnect_after_s:
                        logger.warning(
                            f"Camera stalled for {captured_at - stalled_since:.1f}s; "
                            "closing it to reconnect"
                        )
                        self._release_for_reconnect(captured_at - stalled_since)
                        stalled_since = None
                        continue
                    self._sleep(self.RETRY_READ_S)
                    continue
                stalled_since = None
                self._reopen_attempt = 0
                self._seq += 1
                self._frames_captured += 1
                if self._source is not None:
                    # Media time: the clip's own timeline, anchored at the first
                    # read, so --no-pace runs keep real durations.
                    if self._media_t0 is None:
                        self._media_t0 = captured_wall_time
                    captured_wall_time = self._media_t0 + (self._seq - 1) * self._frame_interval
                frame = Frame(image=image, captured_at=captured_at, seq=self._seq,
                              captured_wall_time=captured_wall_time)
                with self._published:
                    self._latest = frame
                    self._published.notify_all()
                self._read_counter.record("success", read_ms, frame_seq=self._seq)
                if self._source is not None and self._pace:
                    # Deliver file frames at the file's rate, as a camera would.
                    next_due += self._frame_interval
                    delay = next_due - self._clock()
                    if delay > 0:
                        self._sleep(delay)
                elif self._source is not None:
                    # Unpaced: hand frames over one at a time, never overwrite.
                    with self._acked:
                        while (self._acked_seq < self._seq
                               and not self._stop_event.is_set()):
                            self._acked.wait(0.05)
        except Exception as e:
            # The thread still dies; `error` lets the consumer notice instead
            # of waiting forever for a frame that will never come.
            self._error = e
            with self._published:
                self._published.notify_all()
            self._obs.emit(
                "capture_read", "fail", frame_seq=self._seq, error_type="unknown",
                error_message=describe(e), context={"thread_died": True},
            )
            raise

    def _release_for_reconnect(self, stalled_s: float) -> None:
        assert self._cap is not None
        self._cap.release()
        self._cap = None
        self._reconnect_span = self._obs.span(
            "capture_reconnect", error_type="hardware",
            context={"stalled_s": round(stalled_s, 3), "failures_total": self._read_failures},
        )

    def _reconnect(self) -> bool:
        """Reopen the camera, waiting backoff_delay(k) before each attempt.
        Returns False if stop() was called while waiting."""
        sp = self._reconnect_span
        attempts = 0
        while True:
            delay = backoff_delay(self._reopen_attempt, self._backoff_s)
            self._reopen_attempt += 1
            logger.info(f"Reopening camera in {delay:g}s (attempt {self._reopen_attempt})")
            if self._wait(delay):
                sp.skip("stopped", context={"attempts": attempts})
                return False
            attempts += 1
            try:
                self._cap = self._open_capture()
            except RuntimeError as e:
                logger.warning(f"Reconnect attempt {attempts} failed: {e}")
                continue
            logger.info(f"Camera reconnected after {attempts} attempt(s)")
            sp.success({"attempts": attempts})
            return True

    def _wait(self, seconds: float) -> bool:
        """Wait `seconds` on the injected clock, in STOP_POLL_S slices.
        Returns True as soon as stop() has been called."""
        end = self._clock() + seconds
        while not self._stop_event.is_set():
            remaining = end - self._clock()
            if remaining <= 0:
                return False
            self._sleep(min(remaining, self.STOP_POLL_S))
        return True

    @property
    def finished(self) -> bool:
        """True once a file source has been read to the end."""
        return self._finished.is_set()

    @property
    def error(self) -> Optional[BaseException]:
        """The exception that killed the capture thread, if it died."""
        return self._error

    def ack(self, seq: int) -> None:
        """Consumer is done with frame `seq`. Only an unpaced file source waits
        on this; for cameras and paced files it just records the seq."""
        with self._acked:
            self._acked_seq = max(self._acked_seq, seq)
            self._acked.notify_all()

    def wait_for_frame_after(self, seq: int, timeout: float) -> None:
        """Block until a frame newer than `seq` is published, the source ends,
        or the capture thread dies; at most `timeout` real seconds. The
        consumer side of the unpaced hand-off (pace=False), where the next
        frame is only a decode away and a fixed idle sleep would dominate."""
        with self._published:
            self._published.wait_for(
                lambda: (self._latest is not None and self._latest.seq > seq)
                or self._finished.is_set() or self._error is not None,
                timeout,
            )

    def read_latest(self) -> Optional[Frame]:
        """Peek the latest frame. Returns None until the first frame arrives."""
        with self._lock:
            return self._latest

    def stop(self) -> None:
        sp = self._obs.span("capture_close")
        self._stop_event.set()
        reader_stuck = False
        if self._thread is not None:
            self._thread.join(timeout=self._join_timeout_s)
            if self._thread.is_alive():
                # Releasing a capture that another thread is still reading
                # from can crash inside the driver: leave it to process exit.
                reader_stuck = True
                logger.warning(
                    f"Capture thread still inside read() after {self._join_timeout_s:g}s; "
                    "leaving the device handle open until the process exits"
                )
                self._obs.emit(
                    "capture_close", "fail", error_type="timeout",
                    error_message=f"FrameGrabber thread still alive after "
                                  f"{self._join_timeout_s:g}s join; capture not released",
                )
            self._thread = None
        if self._cap is not None and not reader_stuck:
            self._cap.release()
            self._cap = None
        logger.info(
            "FrameGrabber stopped "
            f"(captured={self._frames_captured}, "
            f"failures={self._read_failures})"
        )
        self._read_counter.flush()
        sp.success({"captured": self._frames_captured, "failures": self._read_failures})

    @property
    def stats(self) -> dict[str, int]:
        return {
            "captured": self._frames_captured,
            "failures": self._read_failures,
        }

    def __enter__(self) -> "FrameGrabber":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()
