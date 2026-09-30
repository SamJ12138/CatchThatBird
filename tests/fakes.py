"""Test doubles. Nothing here touches a camera or imports ultralytics."""
from __future__ import annotations

import time
from typing import Any, Optional

import cv2
import numpy as np

BIRD = 14  # COCO class id for "bird"


class FakeDevice:
    """A physical camera shared by every FakeCapture opened on it, so a
    reconnect sees the device come back. Use as the cv2.VideoCapture factory.

    fail_reads: the next N read() calls (on any capture) return False
    fail_opens: the next N open attempts fail on every backend
    """

    def __init__(self, *, fail_reads: int = 0, fail_opens: int = 0,
                 block_reads: bool = False) -> None:
        self.fail_reads = fail_reads
        self.fail_opens = fail_opens
        self.block_reads = block_reads
        self.open_attempts: list[float] = []  # monotonic time of each attempt
        self.captures: list["FakeCapture"] = []
        self._attempt_fails = False

    def __call__(self, index: Any = 0, backend: Optional[int] = None) -> "FakeCapture":
        if backend in (None, cv2.CAP_DSHOW):  # first backend tried = a new attempt
            self.open_attempts.append(time.monotonic())
            self._attempt_fails = self.fail_opens > 0
            if self._attempt_fails:
                self.fail_opens -= 1
        cap = FakeCapture(device=self, block=self.block_reads)
        if self._attempt_fails:
            cap.released = True  # isOpened() -> False
        self.captures.append(cap)
        return cap


class FakeCapture:
    """cv2.VideoCapture stand-in (camera or file).

    frames: successful reads before read() returns False (None = endless)
    raise_after: read number raise_after + 1 raises RuntimeError
    block: read() never returns (a hung driver)
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
            time.sleep(3600)
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


class FakePredictor:
    """Stands in for detector.YoloPredictor.

    boxes_by_seq: {frame_seq: [(x, y, w, h), ...]} in FULL-FRAME pixels.
        The fake converts them to crop coordinates (as YOLO would report them)
        using the crop origin, so the Detector maps them back to the same box.
    crop_boxes: [(x1, y1, x2, y2), ...] in CROP pixels, returned on every call
        (for testing the crop -> full-frame mapping itself).
    Any other frame_seq returns no boxes.
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
    ) -> None:
        self.boxes_by_seq = {
            int(k): [tuple(b) for b in v] for k, v in (boxes_by_seq or {}).items()
        }
        self.crop_boxes = [tuple(b) for b in crop_boxes] if crop_boxes else []
        self.conf = conf
        self.cls_id = cls_id
        self.load_delay = load_delay
        self.predict_delay = predict_delay
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
        time.sleep(self.load_delay)
        self.loaded = True

    def predict(self, crop, *, conf, classes, frame_seq, origin):
        from detector import RawBox

        if self.predict_delay:
            time.sleep(self.predict_delay)
        self.calls.append(
            {"frame_seq": frame_seq, "origin": tuple(origin), "crop_shape": crop.shape[:2]}
        )
        ox, oy = origin
        out = [RawBox(self.cls_id, self.conf, *b) for b in self.crop_boxes]
        for x, y, w, h in self.boxes_by_seq.get(frame_seq, []):
            out.append(RawBox(self.cls_id, self.conf, x - ox, y - oy, x - ox + w, y - oy + h))
        self.last_meta = {"n_results": 1, "n_results_without_boxes": 0}
        return out
