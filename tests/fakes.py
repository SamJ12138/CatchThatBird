"""Test doubles. Nothing here touches a camera or imports ultralytics."""
from __future__ import annotations

import time
from typing import Any, Optional

BIRD = 14  # COCO class id for "bird"


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
