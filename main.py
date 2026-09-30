from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import yaml
from loguru import logger
from pydantic import ValidationError

from camera import (
    Frame,
    FrameGrabber,
    classify_device,
    find_pocket_index,
    list_devices,
)
from config_schema import AppConfig, load_config
from obs import ObsLogger, Span, describe, new_run_id

if TYPE_CHECKING:
    from detector import Predictor


CONFIG_FILE = Path(__file__).resolve().parent / "config.yaml"
ROI_FILE = Path(__file__).resolve().parent / "data" / "roi.json"
LOG_DIR = Path(__file__).resolve().parent / "logs"
ASPECT_TOLERANCE = 0.01  # relative; 1920x1080 vs 1280x720 match, 640x480 does not


class ExitError(Exception):
    """A startup/run condition that ends the process with `code` and a
    one-line message (no traceback)."""

    def __init__(self, message: str, code: int = 1, error_type: str = "input_invalid") -> None:
        super().__init__(message)
        self.code = code
        self.error_type = error_type
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
    s         save the current frame to data/snapshots/manual_*.jpg

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


def wait_for_enter(obs: ObsLogger) -> None:
    sp = obs.span("checklist")
    sys.stdout.write(CHECKLIST)
    sys.stdout.flush()
    try:
        input()
    except EOFError:
        message = ("stdin closed before the pre-flight checklist was confirmed; "
                   "pass --yes to skip the checklist in unattended runs")
        sp.fail("input_invalid", message)
        raise ExitError(message, code=2)
    except KeyboardInterrupt:
        logger.info("Aborted before camera start")
        sp.skip("keyboard_interrupt")
        sys.exit(0)
    sp.success()


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
    config: AppConfig, devices: list[tuple[int, str]], obs: ObsLogger
) -> str:
    """Sanity-check the configured device index. Returns the device name
    (or '<unknown>') and logs a warning if it looks like the wrong device."""
    sp = obs.span("device_select", context={"device_index": config.camera.device_index})
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
        sp.fail("input_invalid", "configured device_index not in device list; continuing",
                {"pocket_index": pocket_idx})
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
        sp.fail("input_invalid", "selected device looks like a built-in webcam; continuing",
                {"name": selected_name, "pocket_index": pocket_idx})
    else:
        pocket_idx = find_pocket_index(devices)
        suggestion = (
            f" A DJI-like device is at index {pocket_idx} -- pass --device {pocket_idx}."
            if pocket_idx is not None
            else ""
        )
        logger.warning(
            f"Selected device [{config.camera.device_index}] '{selected_name}' "
            f"is not recognised as the Pocket 3; continuing anyway.{suggestion}"
        )
        sp.fail("input_invalid", "selected device not recognised as a Pocket 3; continuing",
                {"name": selected_name, "pocket_index": pocket_idx})
    sp.success({"name": selected_name, "kind": kind})
    return selected_name


def _put_text(img, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)


def _wait_for_first_frame(
    grabber: FrameGrabber, timeout: float = 5.0
) -> Optional[Frame]:
    """None if nothing arrives within `timeout`, or as soon as the source ends
    or the capture thread dies without producing a frame."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = grabber.read_latest()
        if frame is not None:
            return frame
        if grabber.finished or grabber.error is not None:
            return grabber.read_latest()
        time.sleep(0.02)
    return None


def save_snapshot(path: Path, image: np.ndarray, obs: ObsLogger, *, frame_seq: int) -> bool:
    """Write `image` to `path`; returns False (and says so) if OpenCV could not."""
    sp = obs.span("snapshot_save", frame_seq=frame_seq, context={"path": str(path)})
    if cv2.imwrite(str(path), image):
        sp.success()
        logger.info(f"Saved {path}")
        return True
    sp.fail("unknown", "cv2.imwrite returned False")
    logger.error(f"Failed to write snapshot {path} (cv2.imwrite returned False)")
    return False


def load_or_select_roi(
    sample_image: np.ndarray,
    roi_path: Path,
    force_select: bool,
    *,
    headless: bool,
    obs: ObsLogger,
) -> Optional[tuple[int, int, int, int]]:
    """Returns (x, y, w, h) in full-frame coords, or None for whole-frame.
    Reads from roi_path if a compatible saved ROI exists; otherwise prompts
    the user via cv2.selectROI and persists the result. Headless never
    prompts: no compatible saved ROI means whole frame."""
    with obs.span("roi_load", context={"path": str(roi_path)}) as sp:
        return _load_or_select_roi(sample_image, roi_path, force_select, headless, obs, sp)


def _load_or_select_roi(
    sample_image: np.ndarray,
    roi_path: Path,
    force_select: bool,
    headless: bool,
    obs: ObsLogger,
    sp: Span,
) -> Optional[tuple[int, int, int, int]]:
    H, W = sample_image.shape[:2]
    sp.context["frame_wh"] = [W, H]

    if headless and force_select:
        logger.warning("--select-roi ignored in --headless mode (no dialog)")
        force_select = False

    if roi_path.exists() and not force_select:
        try:
            data = json.loads(roi_path.read_text())
            saved_w, saved_h = int(data["frame_width"]), int(data["frame_height"])
            if saved_w <= 0 or saved_h <= 0:
                raise ValueError(f"frame size {saved_w}x{saved_h} is not positive")
            if data.get("whole_frame") is True:
                logger.info(f"ROI file {roi_path} says whole frame (pass --select-roi to redo)")
                sp.success({"source": "file_whole_frame", "roi": None})
                return None
            roi = (int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"]))
        except (OSError, ValueError, KeyError, TypeError) as e:
            logger.warning(
                f"Failed to load ROI from {roi_path}: {e}. Re-selecting."
            )
            obs.emit("roi_load", "fail", error_type="parse", error_message=describe(e))
            if headless:
                raise ExitError(
                    f"Cannot read ROI file {roi_path} ({e}); fix or delete it, or run "
                    "without --headless to select a new ROI"
                ) from e
        else:
            if (saved_w, saved_h) == (W, H):
                logger.info(
                    f"Loaded ROI from {roi_path}: "
                    f"x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]} "
                    "(pass --select-roi to redo)"
                )
                sp.success({"source": "file", "roi": list(roi)})
                return roi
            obs.emit(
                "roi_load", "fail", error_type="input_invalid",
                error_message="saved ROI resolution does not match frame",
                context={"saved_wh": [saved_w, saved_h], "frame_wh": [W, H]},
            )
            if abs((saved_w / saved_h) / (W / H) - 1) <= ASPECT_TOLERANCE:
                roi = _rescale_roi(roi, (saved_w, saved_h), (W, H))
                logger.warning(
                    f"Saved ROI was for {saved_w}x{saved_h}, current frame is "
                    f"{W}x{H} (same aspect ratio): rescaled to "
                    f"x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}"
                )
                sp.success({"source": "file_rescaled", "roi": list(roi),
                            "saved_wh": [saved_w, saved_h]})
                return roi
            logger.warning(
                f"Saved ROI was for {saved_w}x{saved_h}, current frame is {W}x{H} "
                "(different aspect ratio). Re-selecting."
            )
            if headless:
                raise ExitError(
                    f"Saved ROI in {roi_path} is for {saved_w}x{saved_h} but frames are "
                    f"{W}x{H} (different aspect ratio); run without --headless and "
                    "pass --select-roi"
                )

    if headless:
        logger.info("Headless: no saved ROI -- using whole frame")
        sp.skip("headless_whole_frame")
        return None

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
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    if w == 0 or h == 0:
        roi_path.write_text(json.dumps(
            {"whole_frame": True, "frame_width": W, "frame_height": H}, indent=2,
        ))
        logger.info(f"No ROI selected -- using whole frame (saved to {roi_path})")
        sp.skip("selection_empty", error_type="input_invalid",
                error_message="cv2.selectROI returned zero width/height; using whole frame",
                context={"saved": True})
        return None

    roi_path.write_text(json.dumps(
        {"x": x, "y": y, "w": w, "h": h, "frame_width": W, "frame_height": H},
        indent=2,
    ))
    logger.info(
        f"Saved ROI to {roi_path}: x={x} y={y} w={w} h={h}"
    )
    sp.success({"source": "dialog", "roi": [x, y, w, h]})
    return (x, y, w, h)


def _rescale_roi(
    roi: tuple[int, int, int, int],
    saved_wh: tuple[int, int],
    frame_wh: tuple[int, int],
) -> tuple[int, int, int, int]:
    sx = frame_wh[0] / saved_wh[0]
    sy = frame_wh[1] / saved_wh[1]
    x, y = round(roi[0] * sx), round(roi[1] * sy)
    w = min(round(roi[2] * sx), frame_wh[0] - x)
    h = min(round(roi[3] * sy), frame_wh[1] - y)
    return (x, y, w, h)


def run_preview(
    config: AppConfig,
    *,
    force_select_roi: bool,
    obs: ObsLogger,
    source: Optional[str] = None,
    headless: bool = False,
    pace: bool = True,
    roi_file: Path = ROI_FILE,
    predictor: Optional[Predictor] = None,
    first_frame_timeout: float = 5.0,
) -> None:
    # Lazy import: pulling ultralytics costs ~1-3s, skip it for --list-devices.
    from detector import Detection, Detector
    from logger import EventLogger

    recent_detections: list[tuple[float, Detection]] = []
    detection_overlay_ttl = 1.5  # seconds to keep a box visible after detection

    # Load the model BEFORE capture starts so no frames are lost while it
    # loads; the ROI (which needs the first frame) is set afterwards.
    detector = Detector(config.detection, obs=obs, predictor=predictor)
    # Also before capture: the retention sweep runs in the constructor.
    events = EventLogger(
        config.logging, config.storage, obs.run_id, obs=obs,
        dedupe_within_seconds=config.detection.dedupe_within_seconds,
    )

    with events, FrameGrabber(
        device_index=config.camera.device_index,
        width=config.camera.width,
        height=config.camera.height,
        fps=config.camera.fps,
        obs=obs,
        source=source,
        pace=pace,
    ) as grabber:
        sp = obs.span("first_frame_wait", context={"timeout_s": first_frame_timeout})
        first_frame = _wait_for_first_frame(grabber, timeout=first_frame_timeout)
        if first_frame is None:
            if grabber.error is not None:
                reason = f"capture thread died before the first frame: {describe(grabber.error)}"
            elif grabber.finished:
                reason = f"source {source} ended before producing any frame"
            else:
                reason = f"no frames received within {first_frame_timeout:g}s"
            sp.fail("timeout", reason)
            raise ExitError(f"No frames: {reason}", error_type="timeout")
        sp.success({"frame_seq": first_frame.seq})

        roi = load_or_select_roi(
            first_frame.image, roi_file, force_select=force_select_roi,
            headless=headless, obs=obs,
        )
        detector.set_roi(roi)

        if not headless:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 1280, 720)

        last_seq = 0  # 0 == none seen yet; producer seqs start at 1
        last_new_frame_time = time.monotonic()
        display_fps_ema = 0.0
        latency_ms_ema = 0.0
        ema_alpha = 0.1
        skipped_total = 0  # producer frames the display never showed
        render_counter = obs.counter(
            "render",
            context_fn=lambda: {
                "skipped_total": skipped_total,
                "display_fps_ema": round(display_fps_ema, 2),
                "latency_ms_ema": round(latency_ms_ema, 2),
            },
        )

        while True:
            frame = grabber.read_latest()
            assert frame is not None  # the first frame arrived before the loop

            is_new = frame.seq != last_seq
            if not is_new:
                # A file source has ended once no newer frame can arrive.
                if grabber.finished and grabber.read_latest().seq == last_seq:
                    logger.info("Video source finished")
                    return
                if grabber.error is not None:
                    raise ExitError(f"Capture thread died: {describe(grabber.error)}",
                                    error_type="unknown")
                if headless:
                    time.sleep(0.005)
                    continue
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

            detections = detector.process(frame)
            for det in detections:
                logger.info(
                    f"{det.class_name.upper()} detected "
                    f"(conf={det.confidence:.2f}, "
                    f"bbox={det.bbox_xywh[0]},{det.bbox_xywh[1]},"
                    f"{det.bbox_xywh[2]},{det.bbox_xywh[3]})"
                )
                recent_detections.append((now, det))
            events.handle(frame, detections)
            grabber.ack(frame.seq)

            recent_detections = [
                (t, d) for t, d in recent_detections
                if now - t < detection_overlay_ttl
            ]

            if headless:
                render_counter.record("skip", frame_seq=frame.seq, reason="headless")
                continue

            render_t0 = time.perf_counter()
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

            try:
                cv2.imshow(WINDOW_NAME, display)
            except Exception as e:
                obs.emit("render", "fail", frame_seq=frame.seq, error_type="unknown",
                         error_message=describe(e))
                raise
            key = cv2.waitKey(1) & 0xFF
            render_counter.record(
                "success", (time.perf_counter() - render_t0) * 1000.0, frame_seq=frame.seq
            )
            if key in (ord("q"), 27):
                logger.info("Quit requested via keyboard")
                return
            if key == ord("s"):
                save_snapshot(events.manual_snapshot_path(frame), frame.image, obs,
                              frame_seq=frame.seq)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
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
    p.add_argument(
        "--source",
        default=None,
        metavar="PATH",
        help="Read frames from a video file instead of the camera. Skips the "
             "device listing and the Pocket 3 checklist; exits 0 at end of file.",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="No windows: skip the preview and the ROI dialog (uses "
             "data/roi.json only if it matches the frame size, else whole frame).",
    )
    p.add_argument(
        "--no-pace",
        action="store_true",
        help="With --source: read the file as fast as the pipeline consumes it "
             "(no real-time pacing, no dropped frames). For tests.",
    )
    p.add_argument(
        "--roi-file",
        type=Path,
        default=ROI_FILE,
        metavar="PATH",
        help=f"ROI file to load/save (default: {ROI_FILE.relative_to(ROI_FILE.parents[1])}).",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=LOG_DIR,
        metavar="DIR",
        help="Directory for the run_<id>.jsonl structured log (default: logs/).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=CONFIG_FILE,
        metavar="PATH",
        help="Config file (default: config.yaml next to main.py).",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the Pocket 3 pre-flight checklist prompt (unattended runs).",
    )
    p.add_argument(
        "--first-frame-timeout",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Exit 1 if the source produces no frame within this time (default: 5).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None, *, predictor: Optional[Predictor] = None) -> int:
    """`argv` and `predictor` are seams for tests (a fake predictor avoids YOLO)."""
    configure_logger()
    args = parse_args(argv)

    obs = ObsLogger(new_run_id(), args.log_dir)
    logger.info(f"Run {obs.run_id}: structured log -> {obs.path}")
    try:
        with obs.span("run", context={"argv": sys.argv[1:] if argv is None else argv}) as run:
            try:
                code = _main(args, obs, run, predictor)
            except ExitError as e:
                logger.error(str(e))
                run.fail(e.error_type, str(e), {"exit_code": e.code})
                return e.code
            if code != 0:
                run.fail("unknown", f"exit code {code}", {"exit_code": code})
            run.success({"exit_code": code})
            return code
    finally:
        obs.close()


def load_app_config(path: Path) -> AppConfig:
    """load_config() with every failure turned into a one-line ExitError."""
    if not path.exists():
        raise ExitError(f"Missing config file {path}", error_type="input_invalid")
    try:
        return load_config(path)
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        problem = getattr(e, "problem", None) or str(e).splitlines()[0]
        raise ExitError(f"Invalid YAML in config {path}{where}: {problem}",
                        error_type="parse") from e
    except ValidationError as e:
        errors = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        raise ExitError(f"Invalid config {path}: {e.error_count()} error(s): {errors}",
                        error_type="parse") from e


def _main(
    args: argparse.Namespace,
    obs: ObsLogger,
    run: Span,
    predictor: Optional[Predictor] = None,
) -> int:
    with obs.span("config_load", error_type="parse", context={"path": str(args.config)}) as sp:
        try:
            config = load_app_config(args.config)
        except ExitError as e:
            sp.fail(e.error_type, str(e))
            raise

    if args.device is not None:
        logger.info(
            f"Overriding device_index {config.camera.device_index} "
            f"-> {args.device} via --device"
        )
        config.camera.device_index = args.device

    if args.source is None or args.list_devices:
        devices = list_devices(obs)
        print_device_list(config.camera.device_index, devices)

        if args.list_devices:
            return 0

        resolve_device_selection(config, devices, obs)
        logger.info(
            f"Camera config: index={config.camera.device_index} "
            f"{config.camera.width}x{config.camera.height}@{config.camera.fps}"
        )
        if not args.yes:
            wait_for_enter(obs)
    else:
        logger.info(
            f"Source: video file {args.source} "
            "(device listing and Pocket 3 checklist skipped)"
        )
    try:
        run_preview(
            config, force_select_roi=args.select_roi, obs=obs,
            source=args.source, headless=args.headless, pace=not args.no_pace,
            roi_file=args.roi_file, predictor=predictor,
            first_frame_timeout=args.first_frame_timeout,
        )
    except ExitError:
        raise  # one-line message + exit code, handled in main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        run.skip("keyboard_interrupt", context={"exit_code": 0})
    except Exception as e:
        logger.exception("Preview crashed")
        run.fail("unknown", describe(e), {"exit_code": 1})
        return 1
    finally:
        if not args.headless:  # no windows exist; avoids GUI init on a displayless CI box
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
