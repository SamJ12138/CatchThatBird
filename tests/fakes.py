"""Test doubles. Nothing here touches a camera or imports ultralytics."""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

import cv2
import numpy as np

BIRD = 14  # COCO class id for "bird"


class FakeClock:
    """Deterministic time for FrameGrabber / main.run_preview (their `clock`
    and `sleep` arguments). No real time passes.

    monotonic() returns fake seconds. sleep(s):
    - instant (default): advances fake time by s and returns at once.
    - lockstep=True: a two-actor discrete-event clock. Only the driver thread
      (the one that created the clock) moves time, and only while the other
      thread is idle, i.e. blocked in its own sleep() with its wake-up still
      in the future. Any other thread's sleep blocks until the driver has
      moved time past its wake-up point. A paced producer thread then runs
      only while the consumer idles and never falls behind or races ahead,
      so frame hand-off is deterministic. If no other thread is asleep (none
      started yet, or it has exited), the driver waits at most `settle` real
      seconds before moving time anyway. A blocked sleep that sees no
      progress for `stall_timeout` real seconds raises instead of hanging.
    `slept` records every sleep request (any thread).
    """

    def __init__(self, start: float = 1000.0, *, lockstep: bool = False,
                 settle: float = 0.05, stall_timeout: float = 5.0) -> None:
        self._now = start
        self._cond = threading.Condition()
        self._lockstep = lockstep
        self._driver = threading.current_thread()
        self._settle = settle
        self._stall_timeout = stall_timeout
        self._sleepers: dict[threading.Thread, float] = {}  # thread -> wake-up time
        self.slept: list[float] = []

    def monotonic(self) -> float:
        with self._cond:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._cond:
            self._now += max(0.0, seconds)
            self._cond.notify_all()

    def sleep(self, seconds: float) -> None:
        seconds = max(0.0, seconds)
        with self._cond:
            self.slept.append(seconds)
        if not self._lockstep:
            self.advance(seconds)
            time.sleep(0)  # yield the GIL so other threads see the new time
            return
        me = threading.current_thread()
        with self._cond:
            if me is self._driver:
                # Wait until the other thread is idle: asleep, and not already due.
                self._cond.wait_for(
                    lambda: self._sleepers and all(w > self._now for w in self._sleepers.values()),
                    timeout=self._settle,
                )
                self._now += seconds
                self._cond.notify_all()
                return
            wake = self._now + seconds
            self._sleepers[me] = wake
            self._cond.notify_all()
            try:
                if not self._cond.wait_for(lambda: self._now >= wake, timeout=self._stall_timeout):
                    raise RuntimeError("FakeClock: no thread advanced time (lockstep stall)")
            finally:
                del self._sleepers[me]
                self._cond.notify_all()


class FakeDevice:
    """A physical camera shared by every FakeCapture opened on it, so a
    reconnect sees the device come back. Use as the cv2.VideoCapture factory.

    fail_reads: the next N read() calls (on any capture) return False
    fail_opens: the next N open attempts fail on every backend
    fail_reopens: the N open attempts after the first one fail
    clock: timestamps open attempts and releases (default time.monotonic)
    """

    def __init__(self, *, fail_reads: int = 0, fail_opens: int = 0, fail_reopens: int = 0,
                 block_reads: bool = False,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.fail_reads = fail_reads
        self.fail_opens = fail_opens
        self.fail_reopens = fail_reopens
        self.block_reads = block_reads
        self.clock = clock
        self.open_attempts: list[float] = []  # clock time of each attempt
        self.captures: list["FakeCapture"] = []
        self._attempt_fails = False

    def __call__(self, index: Any = 0, backend: Optional[int] = None) -> "FakeCapture":
        if backend in (None, cv2.CAP_DSHOW):  # first backend tried = a new attempt
            self.open_attempts.append(self.clock())
            self._attempt_fails = self.fail_opens > 0
            if self._attempt_fails:
                self.fail_opens -= 1
            elif len(self.open_attempts) > 1 and self.fail_reopens > 0:
                self._attempt_fails = True
                self.fail_reopens -= 1
        cap = FakeCapture(device=self, block=self.block_reads)
        if self._attempt_fails:
            cap.released = True  # isOpened() -> False
        self.captures.append(cap)
        return cap


class FakeCapture:
    """cv2.VideoCapture stand-in (camera or file).

    frames: successful reads before read() returns False (None = endless)
    raise_after: read number raise_after + 1 raises RuntimeError
    block: read() never returns (a hung driver; `in_read` is set while inside)
    device: a FakeDevice whose fail_reads budget this capture consumes
    A dark block moves across a grey frame so MOG2 sees motion.
    """

    def __init__(
        self,
        *_args: Any,
        frames: Optional[int] = None,
        raise_after: Optional[int] = None,
        block: bool = False,
        width: int = 320,
        height: int = 240,
        fps: float = 30.0,
        device: Optional[FakeDevice] = None,
    ) -> None:
        self.frames = frames
        self.raise_after = raise_after
        self.block = block
        self.width = width
        self.height = height
        self.fps = fps
        self.device = device
        self.reads = 0
        self.released = False
        self.released_at: Optional[float] = None
        self.in_read = threading.Event()

    def isOpened(self) -> bool:
        return not self.released

    def get(self, prop: int) -> float:
        return {
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_WIDTH: self.width,
            cv2.CAP_PROP_FRAME_HEIGHT: self.height,
            cv2.CAP_PROP_FRAME_COUNT: self.frames or 0,
        }.get(prop, 0.0)

    def set(self, _prop: int, _value: float) -> bool:
        return True

    def read(self):
        if self.block:
            self.in_read.set()
            threading.Event().wait()  # hung driver: never returns (daemon thread)
        self.reads += 1
        if self.device is not None and self.device.fail_reads > 0:
            self.device.fail_reads -= 1
            return False, None
        if self.raise_after is not None and self.reads > self.raise_after:
            raise RuntimeError(f"fake capture exploded on read {self.reads}")
        if self.frames is not None and self.reads > self.frames:
            return False, None
        img = np.full((self.height, self.width, 3), 128, dtype=np.uint8)
        x = (self.reads * 3) % max(1, self.width - 20)
        img[self.height // 2:self.height // 2 + 20, x:x + 20] = 40
        return True, img

    def release(self) -> None:
        self.released = True
        if self.device is not None:
            self.released_at = self.device.clock()


class FakePredictor:
    """Stands in for detector.YoloPredictor.

    boxes_by_seq: {frame_seq: [(x, y, w, h), ...]} in FULL-FRAME pixels.
        The fake converts them to crop coordinates (as YOLO would report them)
        using the crop origin, so the Detector maps them back to the same box.
    crop_boxes: [(x1, y1, x2, y2), ...] in CROP pixels, returned on every call
        (for testing the crop -> full-frame mapping itself).
    Any other frame_seq returns no boxes.
    within_crop: like real YOLO, only return boxes whose centre lies inside
        the crop it was given (default False: every call on that frame
        returns every box, whatever the crop).
    """

    def __init__(
        self,
        boxes_by_seq: Optional[dict[int, list[tuple[int, int, int, int]]]] = None,
        *,
        crop_boxes: Optional[list[tuple[float, float, float, float]]] = None,
        conf: float = 0.9,
        cls_id: int = BIRD,
        load_delay: float = 0.0,
        predict_delay: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
        within_crop: bool = False,
    ) -> None:
        self.within_crop = within_crop
        self.boxes_by_seq = {
            int(k): [tuple(b) for b in v] for k, v in (boxes_by_seq or {}).items()
        }
        self.crop_boxes = [tuple(b) for b in crop_boxes] if crop_boxes else []
        self.conf = conf
        self.cls_id = cls_id
        self.load_delay = load_delay
        self.predict_delay = predict_delay
        self.sleep = sleep  # pass FakeClock.sleep to "load" in fake time
        self.names = {0: "person", BIRD: "bird"}
        self.device = "fake"
        self.last_meta: dict[str, Any] = {}
        self.calls: list[dict[str, Any]] = []
        self.loaded = False

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> "FakePredictor":
        return cls(
            {int(k): v for k, v in spec.get("boxes_by_seq", {}).items()},
            crop_boxes=spec.get("crop_boxes"),
            conf=spec.get("conf", 0.9),
            load_delay=spec.get("load_delay", 0.0),
            predict_delay=spec.get("predict_delay", 0.0),
        )

    def load(self) -> None:
        if self.load_delay:
            self.sleep(self.load_delay)
        self.loaded = True

    def predict(self, crop, *, conf, classes, frame_seq, origin):
        from detector import RawBox

        if self.predict_delay:
            self.sleep(self.predict_delay)
        self.calls.append(
            {"frame_seq": frame_seq, "origin": tuple(origin), "crop_shape": crop.shape[:2]}
        )
        ox, oy = origin
        ch, cw = crop.shape[:2]
        out = [RawBox(self.cls_id, self.conf, *b) for b in self.crop_boxes]
        for x, y, w, h in self.boxes_by_seq.get(frame_seq, []):
            cx, cy = x + w / 2, y + h / 2
            if self.within_crop and not (ox <= cx < ox + cw and oy <= cy < oy + ch):
                continue
            out.append(RawBox(self.cls_id, self.conf, x - ox, y - oy, x - ox + w, y - oy + h))
        self.last_meta = {"n_results": 1, "n_results_without_boxes": 0}
        return out
