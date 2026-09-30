from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from loguru import logger


# Heuristic name fragments. Matching is case-insensitive.
_POCKET_KEYWORDS = ("dji", "osmo", "pocket")
_BUILTIN_KEYWORDS = (
    "integrated", "built-in", "built in", "internal",
    "lenovo", "hp ", "dell", "surface", "thinkpad", "ideapad",
)


def list_devices() -> list[tuple[int, str]]:
    """Enumerate video capture devices as [(opencv_index, name), ...].

    On Windows we use pygrabber, whose enumeration order matches OpenCV's
    CAP_DSHOW. If pygrabber is unavailable, we fall back to probing indexes
    0-5 with no name (the user has to identify by elimination)."""
    try:
        from pygrabber.dshow_graph import FilterGraph  # type: ignore
        graph = FilterGraph()
        names = graph.get_input_devices()
        return list(enumerate(names))
    except Exception as e:
        logger.debug(f"pygrabber unavailable ({e}); probing indexes 0-5")
        found: list[tuple[int, str]] = []
        for idx in range(6):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            try:
                if cap.isOpened():
                    found.append((idx, f"<device {idx}>"))
            finally:
                cap.release()
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
    re-processing the same seq."""

    def __init__(
        self,
        device_index: int,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._latest: Optional[Frame] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._seq = 0
        self._frames_captured = 0
        self._read_failures = 0

    def _open_capture(self) -> cv2.VideoCapture:
        tried: list[str] = []
        for backend, name in [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")]:
            cap = cv2.VideoCapture(self._device_index, backend)
            if cap.isOpened():
                logger.info(f"Opened camera index {self._device_index} via {name}")
                return self._configure(cap)
            cap.release()
            tried.append(name)
        raise RuntimeError(
            f"Failed to open camera index {self._device_index} "
            f"(tried backends: {', '.join(tried)}). "
            "Confirm the Pocket 3 is in Webcam mode (USB-C connected, screen "
            "shows Webcam) and no other app is holding the camera."
        )

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
        while not self._stop_event.is_set():
            ok, image = self._cap.read()
            captured_at = time.monotonic()
            if not ok or image is None:
                self._read_failures += 1
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

    def read_latest(self) -> Optional[Frame]:
        """Peek the latest frame. Returns None until the first frame arrives."""
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info(
            "FrameGrabber stopped "
            f"(captured={self._frames_captured}, "
            f"failures={self._read_failures})"
        )

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
