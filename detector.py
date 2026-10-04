from __future__ import annotations

import math
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


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Two (x0, y0, x1, y1) rectangles share at least one pixel."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _center_within(xyxy: tuple[int, int, int, int], box: ROI, reach: float) -> bool:
    """The crop's centre is within reach x max(w, h) of the box's centre."""
    x, y, w, h = box
    dx = (xyxy[0] + xyxy[2]) / 2 - (x + w / 2)
    dy = (xyxy[1] + xyxy[3]) / 2 - (y + h / 2)
    return math.hypot(dx, dy) <= reach * max(w, h)


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox_xywh: tuple[int, int, int, int]   # full-frame coordinates
    frame_seq: int
    captured_wall_time: float              # Frame.captured_wall_time (capture moment)
    crop_xyxy: tuple[int, int, int, int] = (0, 0, 0, 0)  # padded predictor crop, full-frame


@dataclass
class _Recheck:
    """A motion crop YOLO found nothing in, classified again on the following
    frames and kept out of the MOG2 update meanwhile (R2)."""
    xyxy: tuple[int, int, int, int]  # the rejected crop, full-frame
    started_seq: int
    frames: int = 0                  # frames processed since the rejected one
    yolo_calls: int = 0


@dataclass
class _Deferred:
    """A gated frame's crop the YOLO budget did not allow yet."""
    kind: str                        # "track" or "motion"
    xyxy: tuple[int, int, int, int]
    from_seq: int


XYXY = tuple[int, int, int, int]


def merge_regions(boxes: list[tuple[XYXY, float]]) -> list[tuple[XYXY, float]]:
    """(padded box, contour area) pairs -> regions: boxes that share a pixel are
    merged into their union, repeatedly, with their areas summed. Largest
    area first."""
    regions = list(boxes)
    merged = True
    while merged:
        merged = False
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                (a, area_a), (b, area_b) = regions[i], regions[j]
                if _intersects(a, b):
                    regions[i] = ((min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])),
                                  area_a + area_b)
                    del regions[j]
                    merged = True
                    break
            if merged:
                break
    return sorted(regions, key=lambda r: -r[1])


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
      after a motion crop the predictor found nothing in -> that crop again on
                      the next recheck_window_frames frames (the re-check window)

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
            f"every_n_frames={config.process_every_n_frames}, "
            f"max_motion_regions={config.max_motion_regions}, "
            f"max_yolo_calls_per_s={config.max_yolo_calls_per_s}"
        )

        self._frame_count = 0
        self._warmup_logged = False
        self._background: Optional[np.ndarray] = None  # MOG2 background, refreshed per gated frame
        self._recheck: Optional[_Recheck] = None       # the open re-check window, if any
        self._motion_burst = False   # motion that was no bird, on every gated frame since a quiet one
        self.last_motion_area = 0.0  # largest contour on the last gated frame (for tests)
        # YOLO budget for the crops gated frames start: a token bucket in
        # capture time, max_yolo_calls_per_s tokens, refilled at that rate.
        self._tokens = float(config.max_yolo_calls_per_s)
        self._tokens_at: Optional[float] = None
        self._deferred: list[_Deferred] = []
        self._motion_gates_opened = 0
        self._yolo_invocations = 0
        self._detections_total = 0
        self._recheck_windows = 0

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
          - the padded last box of every open visit in `tracks`, whether or
            not MOG2 saw motion there ("track" crop, P9), unless a motion
            region already contains that box, and
          - up to max_motion_regions motion regions, largest first: every
            contour of at least motion_min_area, padded, merged with the
            others whose padded boxes it shares a pixel with ("motion" crops).
        Those crops share a budget of max_yolo_calls_per_s (a token bucket in
        capture time), track crops first: the ones past it are logged
        ("yolo_budget" skip "deferred") and classified on the following
        frames as the budget refills, until the next gated frame replaces
        them.
        `tracks` (full-frame x, y, w, h) are also kept out of the MOG2 update:
        their padded boxes are replaced with the model's own background image
        before apply(), so a bird that sits still is never learned into the
        background.

        The largest motion region in which YOLO found nothing opens a re-check
        window (R2): for recheck_window_frames frames that crop is kept out of the
        MOG2 update the same way, and on every recheck_every_n_frames-th of
        them YOLO runs on it again ("recheck" crop), gated or not. A bird
        that was blurred in flight on the gated frame is then found as soon
        as it can be, before the background has learned it. The window ends
        when a bird is found or its frames are used up. Returns [] when
        nothing qualifies."""
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
        recheck = self._recheck
        frozen = [_padded_xyxy(box, pad, frame.image.shape) for box in tracks]
        if recheck is not None:
            frozen.append(recheck.xyxy)
        update = region
        if frozen and self._background is not None:
            update = region.copy()
            for fx0, fy0, fx1, fy1 in frozen:
                x0, y0 = max(0, fx0 - rx), max(0, fy0 - ry)
                x1, y1 = min(rw, fx1 - rx), min(rh, fy1 - ry)
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
        tick = since_warmup % self._config.process_every_n_frames == 0
        recheck_due = False
        if recheck is not None:
            recheck.frames += 1
            recheck_due = recheck.frames % self._config.recheck_every_n_frames == 0
        if tick:
            self._gate_counter.record("success", frame_seq=seq)
        else:
            self._gate_counter.record("skip", frame_seq=seq,
                                      reason="recheck" if recheck_due else "cadence")

        self._refill(frame.captured_wall_time)
        # (kind, crop, frame the crop was deferred from or None)
        crops: list[tuple[str, XYXY, Optional[int]]] = []
        regions: list[XYXY] = []
        if tick:
            self._expire_deferred(seq)
            regions = self._motion_regions(frame, fg_mask, (rx, ry), pad)
            # Open-visit re-checks first, then motion regions by area; past the
            # budget, the rest wait for it to refill (_run_deferred).
            wanted: list[tuple[str, XYXY]] = [
                ("track", _padded_xyxy(box, pad, frame.image.shape)) for box in tracks
                if not any(_contains(r, box) for r in regions)]  # else a region shows it
            wanted += [("motion", r) for r in regions]
            for kind, xyxy in wanted:
                if self._take_token():
                    crops.append((kind, xyxy, None))
                else:
                    self._defer(kind, xyxy, seq)
        else:
            crops += self._run_deferred(seq)
        if recheck is not None and recheck_due:
            crops.append(("recheck", recheck.xyxy, None))   # bounded by its window, not budgeted
            recheck.yolo_calls += 1

        detections: list[Detection] = []
        found_in: set[str] = set()
        rejected: list[XYXY] = []   # this gated frame's motion regions YOLO found nothing in
        for kind, xyxy, deferred_from in crops:
            found = self._classify(frame, kind, xyxy)
            if found:
                found_in.add(kind)
            elif kind == "motion" and deferred_from is None:
                rejected.append(xyxy)
            detections += found
        detections = _dedupe(detections)
        self._detections_total += len(detections)

        if recheck is not None:
            self._end_recheck(recheck, seq, found=bool(found_in & {"motion", "recheck"}))
        # A rejected region that belongs to an open visit, or lies by a bird
        # found on this frame (it opens a visit now), is not worth a window.
        birds = list(tracks) + [d.bbox_xywh for d in detections]
        bird_crops = frozen[:len(tracks)] + [_padded_xyxy(b, pad, frame.image.shape)
                                             for b in birds[len(tracks):]]
        rejected = [r for r in rejected if not self._belongs_to_a_visit(r, birds, bird_crops)]
        # One window per burst of motion. Steady motion that is no bird is
        # rejected on every gated frame; it must not hold a window open, and
        # its region frozen, for as long as it lasts. The next window can
        # open only after a gated frame on which nothing moved.
        if tick and recheck is not None:
            self._motion_burst = True
        elif tick and not regions:
            self._motion_burst = False
        elif tick and rejected:
            # Motion that was no bird, and no bird's either: the largest such region.
            if not self._motion_burst and self._config.recheck_window_frames > 0:
                self._recheck = _Recheck(rejected[0], seq)
                self._recheck_windows += 1
                self._obs.emit(
                    "recheck", "start", frame_seq=seq,
                    context={"crop_xyxy": list(rejected[0]),
                             "window_frames": self._config.recheck_window_frames,
                             "every_n_frames": self._config.recheck_every_n_frames},
                )
            self._motion_burst = True
        return detections

    # ------------------------------------------------------------ YOLO budget

    def _refill(self, t: float) -> None:
        cap = float(self._config.max_yolo_calls_per_s)
        if self._tokens_at is not None and t > self._tokens_at:
            self._tokens = min(cap, self._tokens + (t - self._tokens_at) * cap)
        self._tokens_at = t

    def _take_token(self) -> bool:
        if self._tokens >= 1 - 1e-6:     # capture times are float seconds
            self._tokens -= 1
            return True
        return False

    def _defer(self, kind: str, xyxy: XYXY, seq: int) -> None:
        self._deferred.append(_Deferred(kind, xyxy, seq))
        self._obs.emit("yolo_budget", "skip", frame_seq=seq, context={
            "reason": "deferred", "crop": kind, "crop_xyxy": list(xyxy),
            "max_yolo_calls_per_s": self._config.max_yolo_calls_per_s})

    def _run_deferred(self, seq: int) -> list[tuple[str, XYXY, Optional[int]]]:
        """Deferred crops, oldest first, as far as the budget has refilled.
        They are classified on this frame, at the crop of the gated frame."""
        out = []
        while self._deferred and self._take_token():
            d = self._deferred.pop(0)
            self._obs.emit("yolo_budget", "success", frame_seq=seq, context={
                "crop": d.kind, "crop_xyxy": list(d.xyxy), "deferred_from": d.from_seq})
            out.append((d.kind, d.xyxy, d.from_seq))
        return out

    def _expire_deferred(self, seq: int) -> None:
        """A new gated frame replaces the crops still waiting from the last one."""
        for d in self._deferred:
            self._obs.emit("yolo_budget", "skip", frame_seq=seq, context={
                "reason": "expired", "crop": d.kind, "crop_xyxy": list(d.xyxy),
                "deferred_from": d.from_seq})
        self._deferred = []

    def _belongs_to_a_visit(
        self, motion_xyxy: tuple[int, int, int, int], tracks: Sequence[ROI],
        track_crops: Sequence[tuple[int, int, int, int]],
    ) -> bool:
        """A rejected motion crop that a window is not worth opening for: it
        touches an open visit's track crop (a part of that bird outside its
        box, R3), or lies so near the visit that a bird found in it would
        join that visit (visit_center_distance, P10) and not open one."""
        return any(
            _intersects(motion_xyxy, crop)
            or _center_within(motion_xyxy, box, self._config.visit_center_distance)
            for box, crop in zip(tracks, track_crops)
        )

    def _end_recheck(self, recheck: _Recheck, seq: int, *, found: bool) -> None:
        """Close the window if a bird was found (its visit takes over as a
        track) or its frames are used up."""
        if not found and recheck.frames < self._config.recheck_window_frames:
            return
        self._recheck = None
        context = {"started_frame_seq": recheck.started_seq, "frames": recheck.frames,
                   "yolo_calls": recheck.yolo_calls}
        if found:
            self._obs.emit("recheck", "success", frame_seq=seq, context=context)
        else:
            self._obs.emit("recheck", "skip", frame_seq=seq,
                           context={"reason": "expired", **context})

    def _motion_regions(
        self, frame: Frame, fg_mask: np.ndarray, offset: tuple[int, int], pad: int,
    ) -> list[XYXY]:
        """Padded full-frame crops (x0, y0, x1, y1) around motion: every
        contour of at least motion_min_area, padded by motion_padding_px;
        crops that share a pixel are merged into one; at most
        max_motion_regions, largest contour area first. [] if none."""
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
                return []
            areas = [cv2.contourArea(c) for c in contours]
            self.last_motion_area = max(areas)
            sp.context.update(n_contours=len(contours), largest_area=self.last_motion_area)
            # Contour coords are relative to the ROI region; translate to full frame.
            boxes = []
            for c, area in zip(contours, areas):
                if area >= self._config.motion_min_area:
                    x, y, w, h = cv2.boundingRect(c)
                    boxes.append((_padded_xyxy((offset[0] + x, offset[1] + y, w, h), pad,
                                               frame.image.shape), area))
            if not boxes:
                sp.skip("area_below_min",
                        context={"motion_min_area": self._config.motion_min_area})
                return []
            regions = merge_regions(boxes)
            cap = self._config.max_motion_regions
            sp.context.update(n_regions=len(regions), n_regions_dropped=max(0, len(regions) - cap))
        self._motion_gates_opened += 1
        return [xyxy for xyxy, _ in regions[:cap]]

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
            "recheck_windows": self._recheck_windows,
        }
