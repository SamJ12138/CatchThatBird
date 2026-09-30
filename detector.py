from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Protocol, Sequence

import cv2
import numpy as np
from loguru import logger

from camera import Frame
from config_schema import DetectionConfig
from obs import ObsLogger, describe


ROI = tuple[int, int, int, int]  # (x, y, w, h) in full-frame coords


def iou(a: ROI, b: ROI) -> float:
    """Intersection over union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _padded_xyxy(box: ROI, pad: int, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """(x, y, w, h) grown by `pad` on every side and clamped to the frame."""
    x, y, w, h = box
    return (max(0, x - pad), max(0, y - pad), min(shape[1], x + w + pad), min(shape[0], y + h + pad))


def _contains(xyxy: tuple[int, int, int, int], box: ROI) -> bool:
    x0, y0, x1, y1 = xyxy
    x, y, w, h = box
    return x0 <= x and y0 <= y and x + w <= x1 and y + h <= y1


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox_xywh: tuple[int, int, int, int]   # full-frame coordinates
    frame_seq: int
    captured_wall_time: float              # Frame.captured_wall_time (capture moment)
    crop_xyxy: tuple[int, int, int, int] = (0, 0, 0, 0)  # padded predictor crop, full-frame


def _dedupe(detections: list["Detection"], threshold: float = 0.5) -> list["Detection"]:
    """One detection per bird when the motion and a track crop both saw it:
    same class and IoU >= threshold -> keep the more confident one."""
    kept: list[Detection] = []
    for d in sorted(detections, key=lambda d: -d.confidence):
        if not any(k.class_name == d.class_name and iou(k.bbox_xywh, d.bbox_xywh) >= threshold
                   for k in kept):
            kept.append(d)
    return kept


class RawBox(NamedTuple):
    """One predictor box, in CROP pixel coordinates."""
    cls_id: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


class Predictor(Protocol):
    names: dict[int, str]  # class id -> class name, valid after load()
    device: str            # valid after load()
    last_meta: dict[str, Any]

    def load(self) -> None: ...

    def predict(
        self,
        crop: np.ndarray,
        *,
        conf: float,
        classes: list[int],
        frame_seq: int,
        origin: tuple[int, int],
    ) -> list[RawBox]:
        """Boxes in crop coordinates. `frame_seq` and `origin` (crop top-left
        in the full frame) are context for test fakes; YOLO ignores them."""
        ...


class YoloPredictor:
    """The real predictor: ultralytics YOLO. torch/ultralytics are imported in
    load(), so importing this module stays cheap (and test-safe)."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.names: dict[int, str] = {}
        self.device = "cpu"
        self.last_meta: dict[str, Any] = {}
        self._model: Any = None

    def load(self) -> None:
        import torch
        from ultralytics import YOLO

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"YOLO device=cuda ({gpu_name})")
        else:
            logger.warning(
                "YOLO device=cpu -- torch was not built with CUDA. "
                "It will still work; if you wanted GPU, reinstall torch with a "
                "CUDA wheel (see requirements.txt)."
            )
        logger.info(f"Loading YOLO model '{self.model_path}' (auto-downloads on first run)")
        self._model = YOLO(self.model_path)
        self.names = dict(self._model.names)

    def predict(
        self,
        crop: np.ndarray,
        *,
        conf: float,
        classes: list[int],
        frame_seq: int,
        origin: tuple[int, int],
    ) -> list[RawBox]:
        results = self._model.predict(
            crop, device=self.device, verbose=False, conf=conf, classes=classes,
        )
        boxes: list[RawBox] = []
        without_boxes = 0
        for r in results:
            if r.boxes is None:
                without_boxes += 1
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                boxes.append(RawBox(int(box.cls[0]), float(box.conf[0]), x1, y1, x2, y2))
        self.last_meta = {"n_results": len(results), "n_results_without_boxes": without_boxes}
        return boxes


class Detector:
    """Two-stage gating:
      every frame  -> MOG2 background subtraction on ROI region (or full frame)
      every N-th   -> motion contour gate -> crop+pad -> predictor -> class/conf filter

    The predictor defaults to YoloPredictor(config.yolo_model); tests inject a fake.

    Restricting MOG2 to the ROI region (rather than full frame + mask) gives
    us two wins at once: false motion outside the ROI never enters the pipeline,
    and the per-frame cost drops with the ROI's area.
    """

    def __init__(
        self,
        config: DetectionConfig,
        roi: Optional[ROI] = None,
        *,
        obs: ObsLogger,
        predictor: Optional[Predictor] = None,
    ) -> None:
        self._obs = obs
        sp = obs.span("detector_init", context={"model": config.yolo_model})
        self._config = config
        self._roi = roi
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=config.motion_threshold,
            detectShadows=False,
        )
        # Morphology kernel for noise cleanup on the gated frame's mask.
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self._predictor: Predictor = (
            predictor if predictor is not None else YoloPredictor(config.yolo_model)
        )
        try:
            self._predictor.load()
        except Exception as e:
            sp.fail("external_api", describe(e))
            raise
        self._device = self._predictor.device
        try:
            self._target_class_ids = self._resolve_class_ids()
        except ValueError as e:
            sp.fail("input_invalid", describe(e))
            raise

        self._log_roi()
        logger.info(
            f"Detector ready: target_classes={config.yolo_target_classes} "
            f"(ids={self._target_class_ids}), "
            f"conf>={config.yolo_confidence_threshold}, "
            f"motion_min_area={config.motion_min_area}, "
            f"every_n_frames={config.process_every_n_frames}"
        )

        self._frame_count = 0
        self._warmup_logged = False
        self._background: Optional[np.ndarray] = None  # MOG2 background, refreshed per gated frame
        self.last_motion_area = 0.0  # largest contour on the last gated frame (for tests)
        self._motion_gates_opened = 0
        self._yolo_invocations = 0
        self._detections_total = 0

        self._mog2_counter = obs.counter("mog2_apply")
        self._gate_counter = obs.counter("gate_check")
        sp.success({
            "device": self._device,
            "cuda_available": self._device == "cuda",
            "target_class_ids": self._target_class_ids,
            "roi": list(roi) if roi is not None else None,
        })

    def _log_roi(self) -> None:
        roi = self._roi
        if roi is not None:
            logger.info(
                f"Detector ROI active: x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]} "
                f"(MOG2 input area {roi[2] * roi[3]} px)"
            )
        else:
            logger.info("Detector ROI: whole frame")

    def set_roi(self, roi: Optional[ROI]) -> None:
        """Set the ROI after construction (the model loads before capture
        starts; the ROI needs the first frame). Only before the first frame."""
        if self._frame_count:
            raise RuntimeError("set_roi() after frames were processed")
        self._roi = roi
        self._log_roi()

    def _resolve_class_ids(self) -> list[int]:
        names: dict[int, str] = self._predictor.names
        name_to_id = {n: i for i, n in names.items()}
        ids: list[int] = []
        unknown: list[str] = []
        for cn in self._config.yolo_target_classes:
            if cn in name_to_id:
                ids.append(name_to_id[cn])
            else:
                unknown.append(cn)
        if unknown:
            sample = list(names.values())[:15]
            raise ValueError(
                f"Target classes not in YOLO model: {unknown}. "
                f"Example available classes: {sample}"
            )
        return ids

    @property
    def roi(self) -> Optional[ROI]:
        return self._roi

    def process(self, frame: Frame, tracks: Sequence[ROI] = ()) -> list[Detection]:
        """Always updates MOG2 (on the ROI region if set). On every Nth frame
        after warm-up (a gated frame) YOLO runs on:
          - the padded box around the largest motion contour, if it is at
            least motion_min_area ("motion" crop), and
          - the padded last box of every open visit in `tracks`, whether or
            not MOG2 saw motion there ("track" crop, P9), unless the motion
            crop already contains that box.
        `tracks` (full-frame x, y, w, h) are also kept out of the MOG2 update:
        their padded boxes are replaced with the model's own background image
        before apply(), so a bird that sits still is never learned into the
        background. Returns [] when nothing qualifies."""
        if self._roi is not None:
            rx, ry, rw, rh = self._roi
            region = frame.image[ry:ry + rh, rx:rx + rw]
        else:
            rx, ry, rw, rh = 0, 0, frame.image.shape[1], frame.image.shape[0]
            region = frame.image
        pad = self._config.motion_padding_px

        seq = frame.seq
        n_next = self._frame_count + 1
        warmup, every = self._config.motion_warmup_frames, self._config.process_every_n_frames
        gated_next = n_next > warmup and (n_next - warmup - 1) % every == 0
        if self._frame_count and gated_next and not tracks:
            # Once a second, while no visit is open, and BEFORE this frame is
            # applied: the bird-free background that masks tracked boxes. It
            # is not refreshed while a visit is open, because the frames just
            # before the visit opened already taught the model part of the bird.
            self._background = self._mog2.getBackgroundImage()
        update = region
        if tracks and self._background is not None:
            update = region.copy()
            for x, y, w, h in tracks:
                x0, y0 = max(0, x - rx - pad), max(0, y - ry - pad)
                x1, y1 = min(rw, x - rx + w + pad), min(rh, y - ry + h + pad)
                if x1 > x0 and y1 > y0:
                    update[y0:y1, x0:x1] = self._background[y0:y1, x0:x1]
        t0 = time.perf_counter()
        try:
            fg_mask = self._mog2.apply(update)
        except Exception as e:
            self._obs.emit(
                "mog2_apply", "fail", frame_seq=seq, error_type="unknown",
                error_message=describe(e), context={"region_shape": list(region.shape)},
            )
            raise
        self._mog2_counter.record(
            "success", (time.perf_counter() - t0) * 1000.0, frame_seq=seq
        )
        self._frame_count += 1

        if self._frame_count <= self._config.motion_warmup_frames:
            if self._frame_count == 1:
                self._obs.emit(
                    "gate_check", "skip", frame_seq=seq,
                    context={"reason": "mog2_warmup_begin",
                             "warmup_frames": self._config.motion_warmup_frames},
                )
            self._gate_counter.record("skip", frame_seq=seq, reason="warmup")
            return []
        if not self._warmup_logged:
            logger.info(
                f"MOG2 warmup complete ({self._config.motion_warmup_frames} "
                "frames). Detection pipeline active."
            )
            self._obs.emit(
                "gate_check", "success", frame_seq=seq,
                context={"reason": "mog2_warmup_complete"},
            )
            self._warmup_logged = True

        # Cadence counts from the end of warm-up: the first frame after it is
        # gated, then every Nth (warm-up 60, N 30 -> 61, 91, 121 ...).
        since_warmup = self._frame_count - self._config.motion_warmup_frames - 1
        if since_warmup % self._config.process_every_n_frames != 0:
            self._gate_counter.record("skip", frame_seq=seq, reason="cadence")
            return []
        self._gate_counter.record("success", frame_seq=seq)

        crops: list[tuple[str, tuple[int, int, int, int]]] = []
        motion_xyxy = self._motion_crop(frame, fg_mask, (rx, ry), pad)
        if motion_xyxy is not None:
            crops.append(("motion", motion_xyxy))
        for box in tracks:
            if motion_xyxy is not None and _contains(motion_xyxy, box):
                continue  # the motion crop already shows this bird
            crops.append(("track", _padded_xyxy(box, pad, frame.image.shape)))

        detections: list[Detection] = []
        for kind, xyxy in crops:
            detections += self._classify(frame, kind, xyxy)
        detections = _dedupe(detections)
        self._detections_total += len(detections)
        return detections

    def _motion_crop(
        self, frame: Frame, fg_mask: np.ndarray, offset: tuple[int, int], pad: int,
    ) -> Optional[tuple[int, int, int, int]]:
        """Padded full-frame crop (x0, y0, x1, y1) around the largest motion
        contour, or None if there is none of at least motion_min_area."""
        seq = frame.seq
        self.last_motion_area = 0.0
        with self._obs.span("morph_contour", frame_seq=seq) as sp:
            # Morphology cleanup -- run only on the gated frame so cost stays 1/sec.
            fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._morph_kernel)
            fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_CLOSE, self._morph_kernel)

            contours, _ = cv2.findContours(
                fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                sp.skip("no_contours")
                return None
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            self.last_motion_area = area
            sp.context.update(n_contours=len(contours), largest_area=area)
            if area < self._config.motion_min_area:
                sp.skip("area_below_min",
                        context={"motion_min_area": self._config.motion_min_area})
                return None
        self._motion_gates_opened += 1
        x, y, w, h = cv2.boundingRect(largest)
        # Contour coords are relative to the ROI region; translate to full frame.
        return _padded_xyxy((offset[0] + x, offset[1] + y, w, h), pad, frame.image.shape)

    def _classify(
        self, frame: Frame, kind: str, xyxy: tuple[int, int, int, int],
    ) -> list[Detection]:
        """Run the predictor on one crop; boxes mapped back to full-frame coords."""
        seq = frame.seq
        x0, y0, x1, y1 = xyxy
        with self._obs.span("crop_build", frame_seq=seq, context={"crop": kind}) as sp:
            crop = frame.image[y0:y1, x0:x1]
            sp.context.update(crop_xyxy=[x0, y0, x1, y1])
            if crop.size == 0:
                sp.skip("empty_crop", error_type="input_invalid",
                        error_message="padded crop has zero size")
                return []

        self._yolo_invocations += 1
        with self._obs.span(
            "yolo_infer", frame_seq=seq, error_type="external_api",
            context={"crop": kind, "crop_wh": [x1 - x0, y1 - y0], "device": self._device},
        ) as sp:
            boxes = self._predictor.predict(
                crop,
                conf=self._config.yolo_confidence_threshold,
                classes=self._target_class_ids,
                frame_seq=seq,
                origin=(x0, y0),
            )
            sp.context.update(n_raw_boxes=len(boxes), **self._predictor.last_meta)

        with self._obs.span("detection_map", frame_seq=seq, context={"crop": kind}) as sp:
            detections: list[Detection] = []
            dropped_class = dropped_conf = 0
            for box in boxes:
                if box.cls_id not in self._target_class_ids:
                    dropped_class += 1
                    continue
                if box.conf < self._config.yolo_confidence_threshold:
                    dropped_conf += 1
                    continue
                detections.append(
                    Detection(
                        class_name=self._predictor.names[box.cls_id],
                        confidence=box.conf,
                        bbox_xywh=(
                            int(x0 + box.x1),
                            int(y0 + box.y1),
                            int(box.x2 - box.x1),
                            int(box.y2 - box.y1),
                        ),
                        frame_seq=frame.seq,
                        captured_wall_time=frame.captured_wall_time,
                        crop_xyxy=(x0, y0, x1, y1),
                    )
                )
            sp.context.update(dropped_class=dropped_class, dropped_conf=dropped_conf)
            if not detections:
                sp.skip("yolo_empty")
            else:
                sp.success({
                    "n_detections": len(detections),
                    "detections": [
                        {"class": d.class_name, "conf": round(d.confidence, 3),
                         "bbox_xywh": list(d.bbox_xywh)}
                        for d in detections
                    ],
                })
        return detections

    @property
    def stats(self) -> dict[str, int]:
        return {
            "frames_processed": self._frame_count,
            "motion_gates_opened": self._motion_gates_opened,
            "yolo_invocations": self._yolo_invocations,
            "detections_total": self._detections_total,
        }
