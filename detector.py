from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import torch
from loguru import logger
from ultralytics import YOLO

from camera import Frame
from config_schema import DetectionConfig
from obs import ObsLogger, describe


ROI = tuple[int, int, int, int]  # (x, y, w, h) in full-frame coords


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox_xywh: tuple[int, int, int, int]   # full-frame coordinates
    frame_seq: int
    captured_wall_time: float              # time.time() at detection moment


class Detector:
    """Two-stage gating:
      every frame  -> MOG2 background subtraction on ROI region (or full frame)
      every N-th   -> motion contour gate -> crop+pad -> YOLOv8n -> class/conf filter

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

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"YOLO device=cuda ({gpu_name})")
        else:
            logger.warning(
                "YOLO device=cpu -- torch was not built with CUDA. "
                "It will still work; if you wanted GPU, reinstall torch with a "
                "CUDA wheel (see requirements.txt)."
            )

        logger.info(f"Loading YOLO model '{config.yolo_model}' (auto-downloads on first run)")
        try:
            self._model = YOLO(config.yolo_model)
        except Exception as e:
            sp.fail("external_api", describe(e))
            raise
        try:
            self._target_class_ids = self._resolve_class_ids()
        except ValueError as e:
            sp.fail("input_invalid", describe(e))
            raise

        if roi is not None:
            logger.info(
                f"Detector ROI active: x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]} "
                f"(MOG2 input area {roi[2] * roi[3]} px)"
            )
        else:
            logger.info("Detector ROI: whole frame")
        logger.info(
            f"Detector ready: target_classes={config.yolo_target_classes} "
            f"(ids={self._target_class_ids}), "
            f"conf>={config.yolo_confidence_threshold}, "
            f"motion_min_area={config.motion_min_area}, "
            f"every_n_frames={config.process_every_n_frames}"
        )

        self._frame_count = 0
        self._warmup_logged = False
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

    def _resolve_class_ids(self) -> list[int]:
        names: dict[int, str] = self._model.names
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

    def process(self, frame: Frame) -> list[Detection]:
        """Always updates MOG2 (on ROI region if set). Runs the motion gate +
        YOLO only every N frames after warmup. Returns [] when no qualifying
        detection is found."""
        if self._roi is not None:
            rx, ry, rw, rh = self._roi
            region = frame.image[ry:ry + rh, rx:rx + rw]
        else:
            rx, ry, rw, rh = 0, 0, frame.image.shape[1], frame.image.shape[0]
            region = frame.image

        seq = frame.seq
        t0 = time.perf_counter()
        try:
            fg_mask = self._mog2.apply(region)
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

        if self._frame_count < self._config.motion_warmup_frames:
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

        if self._frame_count % self._config.process_every_n_frames != 0:
            self._gate_counter.record("skip", frame_seq=seq, reason="cadence")
            return []
        self._gate_counter.record("success", frame_seq=seq)

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
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            sp.context.update(n_contours=len(contours), largest_area=area)
            if area < self._config.motion_min_area:
                sp.skip("area_below_min",
                        context={"motion_min_area": self._config.motion_min_area})
                return []
        self._motion_gates_opened += 1

        with self._obs.span("crop_build", frame_seq=seq) as sp:
            H_full, W_full = frame.image.shape[:2]
            x, y, w, h = cv2.boundingRect(largest)
            # Contour coords are relative to the ROI region; translate to full frame.
            fx = rx + x
            fy = ry + y
            pad = self._config.motion_padding_px
            x0 = max(0, fx - pad)
            y0 = max(0, fy - pad)
            x1 = min(W_full, fx + w + pad)
            y1 = min(H_full, fy + h + pad)
            crop = frame.image[y0:y1, x0:x1]
            sp.context.update(motion_bbox_xywh=[fx, fy, w, h], crop_xyxy=[x0, y0, x1, y1])
            if crop.size == 0:
                sp.skip("empty_crop", error_type="input_invalid",
                        error_message="padded crop has zero size")
                return []

        self._yolo_invocations += 1
        with self._obs.span(
            "yolo_infer", frame_seq=seq, error_type="external_api",
            context={"crop_wh": [x1 - x0, y1 - y0], "device": self._device},
        ) as sp:
            results = self._model.predict(
                crop,
                device=self._device,
                verbose=False,
                conf=self._config.yolo_confidence_threshold,
                classes=self._target_class_ids,
            )
            sp.context.update(
                n_results=len(results),
                n_raw_boxes=sum(len(r.boxes) for r in results if r.boxes is not None),
                n_results_without_boxes=sum(1 for r in results if r.boxes is None),
            )

        with self._obs.span("detection_map", frame_seq=seq) as sp:
            wall = time.time()
            detections: list[Detection] = []
            dropped_class = dropped_conf = 0
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in self._target_class_ids:
                        dropped_class += 1
                        continue
                    conf = float(box.conf[0])
                    if conf < self._config.yolo_confidence_threshold:
                        dropped_conf += 1
                        continue
                    bx1, by1, bx2, by2 = (float(v) for v in box.xyxy[0].tolist())
                    detections.append(
                        Detection(
                            class_name=self._model.names[cls_id],
                            confidence=conf,
                            bbox_xywh=(
                                int(x0 + bx1),
                                int(y0 + by1),
                                int(bx2 - bx1),
                                int(by2 - by1),
                            ),
                            frame_seq=frame.seq,
                            captured_wall_time=wall,
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

        self._detections_total += len(detections)
        return detections

    @property
    def stats(self) -> dict[str, int]:
        return {
            "frames_processed": self._frame_count,
            "motion_gates_opened": self._motion_gates_opened,
            "yolo_invocations": self._yolo_invocations,
            "detections_total": self._detections_total,
        }
