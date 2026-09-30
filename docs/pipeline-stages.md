# Pipeline stages

This lists every stage a run goes through, in order, as the code does it today. The stage names match the `stage` field in `logs/run_<run_id>.jsonl` (see `obs.py`). Line numbers refer to the commit that added this file.

**How each stage is logged:**
- **Per-event stages** emit a `start` line and one terminal line: `success`, `fail`, or `skip`.
- **Per-frame stages** (marked ⏱) run at the camera rate. They emit individual lines only for `fail` or abnormal `skip`, plus one summary line every 300 frames (`context.summary = true`, with counts and p50/p95/max ms).
- **Swallowed errors:** when a stage catches a sub-error and carries on, it emits a `fail` line for that sub-error, then its normal terminal line.

---

## A. Run setup (once per run)

Order since phase 2 batch 3: `detector_init` (model load) runs **before** `capture_open`, and the ROI from `roi_load` is applied with `Detector.set_roi()` after `first_frame_wait`. The table keeps the original numbering.

| # | Stage | Owner | Input | Output | Failure modes seen in the code |
|---|---|---|---|---|---|
| 0 | `run` | `main.main` (main.py:522) | argv | exit code | Crash in `run_preview` becomes exit 1. `KeyboardInterrupt` becomes exit 0. An uncaught exception from config load propagates with a traceback |
| 1 | `config_load` | `main._main` (main.py:540), `config_schema.load_config` | `config.yaml` | `AppConfig` | Missing file → exit 1 (`input_invalid`). YAML error or pydantic `ValidationError` (unknown key, bad type) → exit 1 with a one-line message (`parse`). Path from `--config` |
| 2 | `device_list` | `camera.list_devices` (camera.py:23) | – | `[(index, name)]` | pygrabber import/COM failure is **swallowed** and falls back to probing indexes 0–5 with no names (`external_api`). Skipped when `--source` is given |
| 3 | `device_select` | `main.resolve_device_selection` (main.py:115) | config index, device list | device name | Index not in the list, or looks like a built-in webcam → warning only, run continues (`input_invalid`). Skipped with `--source` |
| 4 | `checklist` | `main.wait_for_enter` (main.py:76) | stdin Enter | – | EOF on stdin → exit 2 with a `--yes` hint (`input_invalid`). Ctrl‑C → `sys.exit(0)` (skip). Skipped with `--source` or `--yes` |
| 5 | `capture_open` | `FrameGrabber._open_capture` / `_open_file` / `_configure` (camera.py:119–191) | device index or file path | opened `cv2.VideoCapture` | DSHOW fails → **silently** tries MSMF (`hardware`). Both fail → `RuntimeError`. File won't open → `RuntimeError` (`input_invalid`). The driver may negotiate a different resolution/fps than requested (info log only; recorded in `context.negotiated_matches_request`) |
| 6 | `first_frame_wait` | `main._wait_for_first_frame` (main.py:175), called at main.py:314 | grabber | first `Frame` | No frame within `--first-frame-timeout` (default 5 s), or the source ends / the capture thread dies first → exit 1 (`timeout`) |
| 7 | `roi_load` | `main.load_or_select_roi` (main.py:187) | first frame, `data/roi.json` | `(x,y,w,h)` or `None` (whole frame) | Unparseable JSON or missing keys → warning, re-select (`parse`). Saved frame size ≠ current → re-select (`input_invalid`). `selectROI` returns 0×0 (cancel) → whole frame, nothing saved (`input_invalid` skip). Headless with no match → whole frame (skip). `write_text` error propagates |
| 8 | `detector_init` | `Detector.__init__` (detector.py:39) | `DetectionConfig`, ROI | ready `Detector` | No CUDA → **silent** CPU fallback (warning; `context.device`). YOLO weights load/download failure → exception (`external_api`). Target class not in the model → `ValueError` (`input_invalid`) |

## B. One frame's journey

The work is split across two threads. The grabber thread runs `capture_read` continuously. The main thread picks up the newest frame and runs everything else on it.

| # | Stage | Owner | Input | Output | Failure modes seen in the code |
|---|---|---|---|---|---|
| 9 ⏱ | `capture_read` | `FrameGrabber._run` (camera.py:202), grabber thread | open capture | `Frame(image, captured_at, seq)` written into the single latest-frame slot | `cap.read()` returns False → count, sleep 50 ms and retry (`hardware`, one line per failure); after `reconnect_after_s` (2 s) of consecutive failures → `capture_reconnect`. Exception in the loop → thread dies and sets `grabber.error`; the main loop exits 1. File source: first failed read = end of file → `finished` set, thread exits (skip `end_of_file`) |
| – | *(frame pickup)* | `run_preview` loop head (main.py:347–373) | latest-frame slot | new `Frame`, or nothing | Not a logged stage: this is the seq-dedupe spin that waits for a newer frame. Producer frames that are overwritten before pickup are counted in `skipped_total` (the `render` summary) |
| 10 ⏱ | `mog2_apply` | `Detector.process` (detector.py:145–157) | frame image (ROI slice or full) | foreground mask | `cv2.error` on a bad/empty ROI slice would propagate (`unknown`). Runs on **every picked-up frame**, not every captured one |
| 11 ⏱ | `gate_check` | `Detector.process` (detector.py:159–182) | frame counter | continue / `[]` | Not an error path. `frame_count <= motion_warmup_frames` → skip `warmup` (frames 1–60 by default; was 1–59 before phase 2). `frame_count % N != 0` → skip `cadence`. The count is of picked-up frames, not camera frames |
| 12 | `morph_contour` | `Detector.process` (detector.py:184–201) | foreground mask | largest contour + area | No contours → skip `no_contours`. Largest area < `motion_min_area` → skip `area_below_min`. Only the **largest** contour moves on |
| 13 | `crop_build` | `Detector.process` (detector.py:204–220) | contour bbox, ROI offset, frame | padded crop `(x0,y0,x1,y1)` | Zero-size crop → skip (`input_invalid`) |
| 14 | `yolo_infer` | `Detector.process` → `YOLO.predict` (detector.py:222–238) | crop | ultralytics `Results` | Any ultralytics/torch exception propagates → the run crashes with exit 1 (`external_api`). Empty results are normal, see the next stage |
| 15 | `detection_map` | `Detector.process` (detector.py:240–285) | `Results`, crop origin | `[Detection]` in full-frame coords | `r.boxes is None` → skipped silently. Class/confidence re-filtered (redundant with `predict`). No boxes → skip `yolo_empty` (not an error). `captured_wall_time` is copied from `Frame.captured_wall_time` (capture moment; was taken after inference before phase 2) |
| 16 ⏱ | `render` | `run_preview` (main.py:411–468) | frame, ROI, recent detections | preview window, HUD | `imshow` exception propagates (`unknown`). In `--headless`, every frame is a skip `headless` |
| 16a | `snapshot_save` | `run_preview` `s` key (main.py:472–482) | current frame | `./snap_NNN.jpg` | `save_snapshot()`: `cv2.imwrite` returning False → error logged, no "Saved" line (`unknown`). Never happens in headless |

## C. Run teardown

| # | Stage | Owner | Input | Output | Failure modes seen in the code |
|---|---|---|---|---|---|
| 9a | `capture_reconnect` | `FrameGrabber._release_for_reconnect` / `_reconnect` (camera only) | stalled capture | reopened capture | Releases the capture, then waits `backoff_delay(k)` = 1, 2, 4, 8, 16, 30, 30 … s before each `capture_open` attempt; restarts at 1 s after the next good frame. `skip` `stopped` if `stop()` is called while waiting. Files never reconnect: their first failed read is end of file |
| 17 | `capture_close` | `FrameGrabber.stop` | grabber | released capture | Thread still alive after the join (`join_timeout_s`, 2 s) → warning, `fail` (`timeout`), and the capture is **not** released (a release during `read()` can crash the driver) |

## D. Persistence (Phase 3, `logger.py`)

| # | Stage | Owner | Input | Output | Failure modes seen in the code |
|---|---|---|---|---|---|
| 8a | `retention` | `EventLogger.__init__` → `sweep_retention` (runs before capture) | `snapshots_dir` | deleted count | `unlink` OSError → `fail` (`hardware`) for that file, sweep continues. Only `.jpg/.jpeg/.png` are touched; `events.jsonl` never is |
| 15a | `persist` | `EventLogger.handle` (called for every processed frame, right after `detector.process`) | `Frame`, `[Detection]` | open visits in memory; snapshots at visit open | Detection matching an open visit → `skip` `dedupe`. New visit over `max_events_per_day` → `skip` `daily_cap` (dropped). Snapshot write fails → `fail` (`hardware`), event kept with a null path |
| 15b | `persist` | `EventLogger._write_event` (visit expired, or `close()` at shutdown) | closed visit | one line in `events.jsonl` | `success` line per event. A hard kill loses visits that are still open (at most `dedupe_within_seconds` after the bird's last sighting, or longer for a long-perching bird) |
