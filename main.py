from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from camera import (
    Frame,
    FrameGrabber,
    classify_device,
    find_pocket_index,
    list_devices,
)
from config_schema import AppConfig, load_config


ROI_FILE = Path(__file__).resolve().parent / "data" / "roi.json"
ROI_SELECT_WINDOW = "Select ROI -- drag a rectangle around the car. Enter/Space = confirm, C = whole frame"


CHECKLIST = """
==================================================================
  DJI Osmo Pocket 3 -- Pre-flight checklist
==================================================================
Before pressing Enter, confirm on the Pocket 3:

  [ ] 1. Gimbal mode set to LOCK
  [ ] 2. ActiveTrack DISABLED
  [ ] 3. Focus mode = MF (manual), focused on the car
  [ ] 4. Pinch to 4x digital zoom (~80mm equivalent)
  [ ] 5. USB-C connected to PC, mode = Webcam
  [ ] 6. No other app is using the camera
  [ ] 7. The device index shown above points at the Pocket 3
         (re-run with --list-devices, or override with --device N)

After Enter, an ROI window will pop up so you can draw a rectangle
around the car (ignores trees / sky). The selection is saved to
data/roi.json and reused. Pass --select-roi to redo it.

A live preview window will then open. Controls:
    q / Esc   quit
    s         save the current frame to ./snap_NNN.jpg

The HUD shows display FPS, pipeline latency, and detection counts.

Press Enter to start, or Ctrl-C to abort.
==================================================================
"""

WINDOW_NAME = "CatchThatBird -- preview"


def configure_logger() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <7}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )


def wait_for_enter() -> None:
    sys.stdout.write(CHECKLIST)
    sys.stdout.flush()
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        logger.info("Aborted before camera start")
        sys.exit(0)


def print_device_list(
    selected_index: int, devices: list[tuple[int, str]]
) -> None:
    if not devices:
        logger.warning(
            "No video capture devices detected. Is the Pocket 3 plugged in "
            "and set to Webcam mode?"
        )
        return
    print()
    print("Available video capture devices:")
    for idx, name in devices:
        kind = classify_device(name)
        hint = ""
        if kind == "pocket":
            hint = "  <- looks like a DJI Pocket"
        elif kind == "builtin":
            hint = "  <- looks like a built-in webcam"
        marker = "  [SELECTED]" if idx == selected_index else ""
        print(f"  [{idx}] {name}{hint}{marker}")
    print()


def resolve_device_selection(
    config: AppConfig, devices: list[tuple[int, str]]
) -> str:
    """Sanity-check the configured device index. Returns the device name
    (or '<unknown>') and logs a warning if it looks like the wrong device."""
    selected_name = next(
        (n for i, n in devices if i == config.camera.device_index), None
    )
    if selected_name is None:
        pocket_idx = find_pocket_index(devices)
        if pocket_idx is not None:
            logger.warning(
                f"Configured device_index={config.camera.device_index} is not "
                f"in the device list; a DJI-like device is at index "
                f"{pocket_idx}. Pass --device {pocket_idx} or update config.yaml."
            )
        else:
            logger.warning(
                f"Configured device_index={config.camera.device_index} is not "
                "in the device list and no DJI-like device was found."
            )
        return "<unknown>"

    kind = classify_device(selected_name)
    if kind == "pocket":
        logger.info(
            f"Selected device [{config.camera.device_index}] "
            f"'{selected_name}' (looks like a DJI Pocket)"
        )
    elif kind == "builtin":
        pocket_idx = find_pocket_index(devices)
        suggestion = (
            f" The Pocket 3 appears to be at index {pocket_idx} -- "
            f"pass --device {pocket_idx} or update config.yaml."
            if pocket_idx is not None
            else ""
        )
        logger.warning(
            f"Selected device [{config.camera.device_index}] "
            f"'{selected_name}' looks like a built-in webcam, not the "
            f"Pocket 3.{suggestion}"
        )
    else:
        logger.info(
            f"Selected device [{config.camera.device_index}] '{selected_name}'"
        )
    return selected_name


def _put_text(img, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)


def _wait_for_first_frame(
    grabber: FrameGrabber, timeout: float = 5.0
) -> Optional[Frame]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = grabber.read_latest()
        if frame is not None:
            return frame
        time.sleep(0.02)
    return None


def load_or_select_roi(
    sample_image: np.ndarray,
    roi_path: Path,
    force_select: bool,
) -> Optional[tuple[int, int, int, int]]:
    """Returns (x, y, w, h) in full-frame coords, or None for whole-frame.
    Reads from roi_path if a compatible saved ROI exists; otherwise prompts
    the user via cv2.selectROI and persists the result."""
    H, W = sample_image.shape[:2]

    if roi_path.exists() and not force_select:
        try:
            data = json.loads(roi_path.read_text())
            if (
                data.get("frame_width") == W
                and data.get("frame_height") == H
            ):
                roi = (
                    int(data["x"]),
                    int(data["y"]),
                    int(data["w"]),
                    int(data["h"]),
                )
                logger.info(
                    f"Loaded ROI from {roi_path}: "
                    f"x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]} "
                    "(pass --select-roi to redo)"
                )
                return roi
            logger.warning(
                f"Saved ROI was for "
                f"{data.get('frame_width')}x{data.get('frame_height')}, "
                f"current frame is {W}x{H}. Re-selecting."
            )
        except Exception as e:
            logger.warning(
                f"Failed to load ROI from {roi_path}: {e}. Re-selecting."
            )

    print()
    print("Draw a rectangle around the area to monitor (the car). "
          "Enter/Space confirms, C cancels (= use whole frame).")
    print()
    cv2.namedWindow(ROI_SELECT_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ROI_SELECT_WINDOW, 1280, 720)
    selection = cv2.selectROI(
        ROI_SELECT_WINDOW, sample_image,
        showCrosshair=False, fromCenter=False,
    )
    cv2.destroyWindow(ROI_SELECT_WINDOW)

    x, y, w, h = (int(v) for v in selection)
    if w == 0 or h == 0:
        logger.info("No ROI selected -- using whole frame")
        return None

    roi_path.parent.mkdir(parents=True, exist_ok=True)
    roi_path.write_text(json.dumps(
        {"x": x, "y": y, "w": w, "h": h, "frame_width": W, "frame_height": H},
        indent=2,
    ))
    logger.info(
        f"Saved ROI to {roi_path}: x={x} y={y} w={w} h={h}"
    )
    return (x, y, w, h)


def run_preview(config: AppConfig, *, force_select_roi: bool) -> None:
    # Lazy import: pulling ultralytics costs ~1-3s, skip it for --list-devices.
    from detector import Detection, Detector

    recent_detections: list[tuple[float, Detection]] = []
    detection_overlay_ttl = 1.5  # seconds to keep a box visible after detection

    with FrameGrabber(
        device_index=config.camera.device_index,
        width=config.camera.width,
        height=config.camera.height,
        fps=config.camera.fps,
    ) as grabber:
        first_frame = _wait_for_first_frame(grabber, timeout=5.0)
        if first_frame is None:
            logger.error("No frames received within 5s -- aborting before ROI selection")
            return

        roi = load_or_select_roi(first_frame.image, ROI_FILE, force_select=force_select_roi)
        detector = Detector(config.detection, roi=roi)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

        last_seq = 0  # 0 == none seen yet; producer seqs start at 1
        last_new_frame_time = time.monotonic()
        display_fps_ema = 0.0
        latency_ms_ema = 0.0
        ema_alpha = 0.1
        snap_count = 0
        skipped_total = 0  # producer frames the display never showed

        while True:
            frame = grabber.read_latest()
            if frame is None:
                # Producer hiccup mid-run; service window and try again
                if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                    return
                continue

            is_new = frame.seq != last_seq
            if not is_new:
                # No fresh producer frame -- skip redraw, just service window events.
                # This is the main fix for skipped: we stop wasting copy+imshow
                # cycles on frames the display already showed.
                key = cv2.waitKey(5) & 0xFF
                if key in (ord("q"), 27):
                    return
                continue

            # ---- New-frame path: full pipeline ----
            now = time.monotonic()
            latency_ms = (now - frame.captured_at) * 1000.0

            if last_seq > 0:
                gap = frame.seq - last_seq - 1
                if gap > 0:
                    skipped_total += gap
            dt = now - last_new_frame_time
            last_new_frame_time = now
            if dt > 0:
                instant_fps = 1.0 / dt
                display_fps_ema = (
                    instant_fps if display_fps_ema == 0
                    else (1 - ema_alpha) * display_fps_ema + ema_alpha * instant_fps
                )
            latency_ms_ema = (
                latency_ms if latency_ms_ema == 0
                else (1 - ema_alpha) * latency_ms_ema + ema_alpha * latency_ms
            )
            last_seq = frame.seq

            for det in detector.process(frame):
                logger.info(
                    f"{det.class_name.upper()} detected "
                    f"(conf={det.confidence:.2f}, "
                    f"bbox={det.bbox_xywh[0]},{det.bbox_xywh[1]},"
                    f"{det.bbox_xywh[2]},{det.bbox_xywh[3]})"
                )
                recent_detections.append((now, det))

            recent_detections = [
                (t, d) for t, d in recent_detections
                if now - t < detection_overlay_ttl
            ]

            display = frame.image.copy()

            if detector.roi is not None:
                rx, ry, rw, rh = detector.roi
                cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh),
                              (0, 165, 255), 2)
                _put_text(display, "ROI", (rx + 8, ry + 24), (0, 165, 255))

            for _, det in recent_detections:
                bx, by, bw, bh = det.bbox_xywh
                cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                label = f"{det.class_name} {det.confidence:.2f}"
                label_y = max(20, by - 8)
                cv2.putText(display, label, (bx, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(display, label, (bx, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            _put_text(
                display,
                f"seq={frame.seq:>6}  fps={display_fps_ema:5.1f}  "
                f"pipe_lat~{latency_ms_ema:5.1f} ms",
                (16, 36),
                (0, 255, 0),
            )
            stats = grabber.stats
            dstats = detector.stats
            _put_text(
                display,
                f"captured={stats['captured']}  skipped={skipped_total}  "
                f"read_fail={stats['failures']}",
                (16, 64),
                (200, 200, 200),
            )
            _put_text(
                display,
                f"motion={dstats['motion_gates_opened']}  "
                f"yolo={dstats['yolo_invocations']}  "
                f"dets={dstats['detections_total']}",
                (16, 92),
                (200, 200, 200),
            )

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                logger.info("Quit requested via keyboard")
                return
            if key == ord("s"):
                snap_path = Path(f"snap_{snap_count:03d}.jpg")
                cv2.imwrite(str(snap_path), frame.image)
                logger.info(f"Saved {snap_path.resolve()}")
                snap_count += 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CatchThatBird -- bird detector on Pocket 3 webcam feed"
    )
    p.add_argument(
        "--list-devices",
        action="store_true",
        help="List available capture devices and exit.",
    )
    p.add_argument(
        "--device",
        type=int,
        default=None,
        metavar="N",
        help="Override camera.device_index from config.yaml.",
    )
    p.add_argument(
        "--select-roi",
        action="store_true",
        help="Re-run the ROI selection dialog even if data/roi.json exists.",
    )
    return p.parse_args()


def main() -> int:
    configure_logger()
    args = parse_args()

    config_path = Path(__file__).resolve().parent / "config.yaml"
    if not config_path.exists():
        logger.error(f"Missing config at {config_path}")
        return 1
    config = load_config(config_path)

    if args.device is not None:
        logger.info(
            f"Overriding device_index {config.camera.device_index} "
            f"-> {args.device} via --device"
        )
        config.camera.device_index = args.device

    devices = list_devices()
    print_device_list(config.camera.device_index, devices)

    if args.list_devices:
        return 0

    resolve_device_selection(config, devices)
    logger.info(
        f"Camera config: index={config.camera.device_index} "
        f"{config.camera.width}x{config.camera.height}@{config.camera.fps}"
    )
    wait_for_enter()
    try:
        run_preview(config, force_select_roi=args.select_roi)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.exception("Preview crashed")
        return 1
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
