from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

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


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    captured_at: float  # time.monotonic() taken right after cap.read() returned
    seq: int


class FrameGrabber:
    """Threaded UVC capture. Producer always overwrites the latest-frame slot;
    consumer peeks the slot so display never blocks and detection can skip
    re-processing the same seq.

    With `source` set, frames come from a video file instead of a camera. The
    file is read at its own frame rate (like a camera would deliver it), and
    `finished` becomes True once the file runs out. With `pace=False` a file
    is read as fast as the consumer takes frames: the producer waits for
    `ack(seq)` before reading the next one, so no frame is skipped."""

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
    ) -> None:
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps
        self._obs = obs
        self._source = source
        self._pace = pace
        self._frame_interval = 1.0 / fps
        self._finished = threading.Event()
        self._cap: Optional[cv2.VideoCapture] = None
        self._latest: Optional[Frame] = None
        self._lock = threading.Lock()
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
        next_due = time.monotonic()
        try:
            while not self._stop_event.is_set():
                t0 = time.perf_counter()
                ok, image = self._cap.read()
                captured_at = time.monotonic()
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
                    time.sleep(0.05)
                    continue
                self._seq += 1
                self._frames_captured += 1
                frame = Frame(image=image, captured_at=captured_at, seq=self._seq)
                with self._lock:
                    self._latest = frame
                self._read_counter.record("success", read_ms, frame_seq=self._seq)
                if self._source is not None and self._pace:
                    # Deliver file frames at the file's rate, as a camera would.
                    next_due += self._frame_interval
                    delay = next_due - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
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
            self._obs.emit(
                "capture_read", "fail", frame_seq=self._seq, error_type="unknown",
                error_message=describe(e), context={"thread_died": True},
            )
            raise

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

    def read_latest(self) -> Optional[Frame]:
        """Peek the latest frame. Returns None until the first frame arrives."""
        with self._lock:
            return self._latest

    def stop(self) -> None:
        sp = self._obs.span("capture_close")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                self._obs.emit(
                    "capture_close", "fail", error_type="timeout",
                    error_message="FrameGrabber thread still alive after 2s join; "
                                  "releasing capture anyway",
                )
            self._thread = None
        if self._cap is not None:
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
