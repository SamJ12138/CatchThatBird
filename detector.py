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
    ) -> None:
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
        self._model = YOLO(config.yolo_model)
        self._target_class_ids = self._resolve_class_ids()

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

        fg_mask = self._mog2.apply(region)
        self._frame_count += 1

        if self._frame_count < self._config.motion_warmup_frames:
            return []
        if not self._warmup_logged:
            logger.info(
                f"MOG2 warmup complete ({self._config.motion_warmup_frames} "
                "frames). Detection pipeline active."
            )
            self._warmup_logged = True

        if self._frame_count % self._config.process_every_n_frames != 0:
            return []

        # Morphology cleanup -- run only on the gated frame so cost stays 1/sec.
        fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._morph_kernel)
        fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_CLOSE, self._morph_kernel)

        contours, _ = cv2.findContours(
            fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < self._config.motion_min_area:
            return []
        self._motion_gates_opened += 1

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
        if crop.size == 0:
            return []

        self._yolo_invocations += 1
        results = self._model.predict(
            crop,
            device=self._device,
            verbose=False,
            conf=self._config.yolo_confidence_threshold,
            classes=self._target_class_ids,
        )

        wall = time.time()
        detections: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self._target_class_ids:
                    continue
                conf = float(box.conf[0])
                if conf < self._config.yolo_confidence_threshold:
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
