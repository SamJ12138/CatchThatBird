# Observations

These come from the observability pass on 2026-09-30. Line numbers refer to the commit that added this file.

In that pass no pre-existing behaviour was changed; every item was **logged** only (see `docs/pipeline-stages.md`). Phase 2 (test-driven fixes) adds the **Status** column: the fixing commit, a planned batch, or "deferred" with the reason. Line numbers in the "Where" column still refer to `bb06fc6`.

## Swallowed or unbounded errors

| # | Where | What happens today | error_type logged | What I would change | Status |
|---|---|---|---|---|---|
| 1 | camera.py:202–236 (`FrameGrabber._run`, camera mode) | When `cap.read()` returns False, the loop counts the failure, sleeps 50 ms, and retries **forever**. It never reopens the device and never gives up. An unplugged or sleeping Pocket 3 gives a live-looking run that captures nothing. The JSONL now gets about 20 `capture_read` fail lines per second while this lasts | `hardware` | After N consecutive failures (≈2 s), release and re-run `_open_capture` with backoff (Phase 5). Surface "stalled" on the HUD | planned: batch 6 (camera reconnect) |
| 2 | camera.py:249–255 (`_run` exception) → main.py:358–373 | Any exception inside the grabber loop kills the daemon thread. The main loop never sees a new `seq`, and `finished` is only set at end of file, so the process **hangs forever** redrawing nothing. Also true for `--source` | `unknown` (`context.thread_died`) | Set a "dead" flag in the grabber; the main loop exits non-zero when it is set | fixed in batch 2 `ae6be3e` (grabber exposes `error`; the main loop exits 1 with a one-line message, <3 s) |
| 3 | main.py:314–319 (`first_frame_wait`) | No frame within 5 s logs an error and `return`s, so the **process exits 0**. A scheduler would read this as a successful run | `timeout` | Return a status from `run_preview` and exit 1 | fixed in batch 2 `ae6be3e` (exit 1; also exits at once if the source ends or the thread dies first; `--first-frame-timeout`) |
| 4 | main.py:76–88 (`wait_for_enter`) | EOF on stdin (no terminal, piped, scheduled task) → `sys.exit(0)` before the camera opens. The run looks successful but did nothing. `--headless` without `--source` still blocks on this prompt | `input_invalid` | Add a `--yes` / skip-checklist flag for unattended runs, and exit non-zero on EOF | fixed in batch 2 `ae6be3e` (EOF → exit 2 with a hint; `--yes` skips the checklist) |
| 5 | camera.py:23–53 (`list_devices`) | A broad `except Exception` around pygrabber falls back to index probing with placeholder names. It is logged at DEBUG only, so it is invisible on the INFO console | `external_api` | Log at WARNING, and narrow the except to `ImportError` / COM errors | deferred: not in the phase-2 brief; logging level only, no data impact |
| 6 | camera.py:119–148 (`_open_capture`) | DSHOW failure silently falls through to MSMF. The console never says DSHOW failed | `hardware` (one line per backend) | Log the failed backend at WARNING | deferred: not in the phase-2 brief; logging level only, no data impact |
| 7 | camera.py:171–191 (`_configure`) | If the driver negotiates a resolution other than the one requested, there is only an INFO line. That mismatch then makes the saved ROI mismatch (#9) and forces a re-select | none (recorded as `context.negotiated_matches_request`) | Warn on mismatch. Rescale the ROI (#9) | fixed in batch 1 `1513b40` (ROI now rescales on same aspect; the negotiation line itself stays INFO) |
| 8 | camera.py:267–287 (`stop`) | `join(timeout=2.0)` can expire and nothing checks it; the capture is released while `read()` may still be running on the other thread | `timeout` | Check `is_alive()`, and skip `release()` (or wait longer) if the reader is still inside `read()` | planned: batch 6 |
| 9 | main.py:218–245 (`roi_load`, size mismatch) | A saved ROI for a different frame size is discarded and the user must re-select. In `--headless` this becomes a **silent whole-frame run**, which is what happened in the requested 1280×720 run: the 1920×1080 ROI was dropped | `input_invalid` | Store the ROI normalised (0–1), or rescale when the aspect ratio matches | fixed in batch 1 `1513b40` (same aspect → rescale + warning; different aspect → headless exit 1, interactive re-select) |
| 10 | main.py:249–253 (`roi_load`, parse) | A broad `except Exception` on `json.loads` / `data["x"]`: a corrupt `roi.json` logs a warning and re-selects (headless: whole frame) | `parse` | Narrow it to `(OSError, ValueError, KeyError)`. In headless, fail rather than silently widen the watch area | fixed in batch 1 `1513b40` (except narrowed to OSError/ValueError/KeyError/TypeError; headless exit 1) |
| 11 | main.py:273–277 (`roi_load`, `selectROI` 0×0) | Esc, C, or an empty drag all mean "whole frame". Nothing is saved, so the next run prompts again | `input_invalid` (skip) | Accept explicitly (e.g. save `{"whole_frame": true}`) or re-prompt | fixed in batch 1 `1513b40` (0×0 selection saved as `{"whole_frame": true}` and reused) |
| 12 | main.py:472–482 (`snapshot_save`) | `cv2.imwrite` returning False was ignored, and "Saved <path>" is still logged | `unknown` | Branch the console message on the return value | fixed in batch 2 `ae6be3e` (`save_snapshot()` logs an error and no "Saved" line when imwrite fails) |
| 13 | main.py:115–168 (`device_select`) | Index not in the device list, or looks like a built-in webcam → a warning, and the run proceeds anyway. The heuristic also misses this PC's own webcam: `USB2.0 HD UVC WebCam` classifies as `unknown`, so no warning fires (seen in run `febc0231`) | `input_invalid` | Refuse to start on a non-`pocket` device unless `--device` was given explicitly | fixed in batch 1 `1513b40` (decision: warn, do not refuse — unknown devices now warn) |
| 14 | detector.py:58–67 (`detector_init`) | No CUDA → silent CPU fallback (WARNING on the console). This machine's torch is `2.12.0+cpu` | none (recorded as `context.device`) | Fine as is; keep it visible in `context` | no change needed |
| 15 | detector.py:245 (`detection_map`) | `r.boxes is None` → `continue` with no trace. It was never hit in either test run | counted in `yolo_infer.context.n_results_without_boxes` | Fine as is | no change needed (`r.boxes is None` now counted inside `YoloPredictor.predict`) |
| 16 | main.py:349–356 (loop head, `frame is None`) | This retry loop has no bound, but it cannot be reached: `run_preview` returns earlier (F3) when there is no first frame | – | Delete the branch or assert | fixed in batch 2 `ae6be3e` (branch deleted; the not-new/EOF path is covered by a paced-run test) |
| 17 | main.py:578–584 (`_main` handlers) | `KeyboardInterrupt` → exit 0; any other exception → traceback + exit 1. Bounded, not swallowed | `run` terminal line | Fine as is | no change needed |

Not swallowed, but uncaught with no friendly message: a `config.yaml` YAML or validation error propagates from main.py:546 as a raw traceback (`config_load` fail, `parse`). **Status:** fixed in batch 2 `ae6be3e` (YAML and pydantic errors → exit 1 with one line naming the file and field; `--config PATH`).

## Other findings from running the code

Numbering: `#N` is a row of the table above, `FN` a finding below. The phase-2 brief's "obs #5 (warm-up)" is F5.

The runs used `data/samples/synth_blob.mp4` (1280×720, run `401cd60f`, the requested run) and `synth_blob_1080p.mp4` (1920×1080, run `a38671d5`, a supplementary run so the ROI path would be exercised). Both were generated by `scripts/make_synth_video.py`. The machine was on CPU torch.

F1. **MOG2 is the dominant per-second CPU cost, not YOLO.**
   - On the whole 1280×720 frame (ROI dropped, #9), `mog2_apply` took p50 ≈14 ms and p95 ≈23 ms on every frame. That is about 420 ms of main-thread work per second, against about 62 ms per second for YOLO.
   - With the 1080p ROI (≈295k px) MOG2 fell to p50 ≈3.9 ms. That is still about 115 ms/s against YOLO's ≈65 ms/s.
   - A whole-frame 1080p run would likely be roughly 2× the 720p cost. That run was not tested.
   - **Status:** deferred: performance, not in the phase-2 brief. The ROI rescale (#9) already cuts it 3–4× when a saved ROI exists
F2. **The detection cadence drifts, as predicted, but only slightly here.**
   - The grabber read 600 frames. The main loop picked up 546 of them (720p, whole frame) and 575 (1080p, ROI).
   - Excluding startup (F3), the steady-state loss was 24 of 570 frames (≈4%) at 720p and 21 of 596 (≈3.5%) at 1080p with the ROI.
   - YOLO ran 17 and 18 times in about 18 s after warm-up, so ≈0.95 Hz rather than 1 Hz.
   - Headless skips rendering. With `imshow` on a 1080p window, the drop will be larger. That was not measured.
   - **Status:** deferred: inherent to the latest-frame-slot design (by intent). `--no-pace` gives lossless, deterministic runs for tests
F3. **Frames are dropped at startup for as long as detector init takes.** The grabber starts before `roi_load` and `detector_init`.
   - Run `401cd60f`: a cold YOLO load took 987 ms. The first frame reaching MOG2 was seq 31, so frames 2–30 were never processed, and warm-up completed at seq 90 rather than seq 60.
   - Run `a38671d5`: a warm load took 103 ms, and the first frame processed was seq 5.
   - With a camera, the time the ROI dialog is open is lost the same way.
   - **Status:** fixed in batch 3 `ea01151` (Detector and the model are built before `FrameGrabber.start()`; the ROI is applied with `Detector.set_roi()` after the first frame. Test: paced run with a 1 s model load → first processed seq ≤ 2, was 31). Also: `Frame.captured_wall_time` is taken at capture and copied into `Detection` (was `time.time()` after inference)
F4. **YOLO's first call is cold.** It took 275 ms, against a steady p50 of ≈60–65 ms and p95 ≈100–112 ms on CPU for a ≈130×120 crop.
   - **Status:** deferred: not in the brief; a dummy inference in `YoloPredictor.load()` would move the cost to startup
F5. **The warm-up is off by one.** `frame_count < motion_warmup_frames` skips 59 frames, not 60 (detector.py:159). This is harmless.
   - **Status:** fixed in batch 1 `1513b40` (`<` → `<=`; with the default 60/30 config the first gated frame is now 90, not 60)
F6. **YOLO never labelled the synthetic blob a bird** (`detection_map` skip `yolo_empty` ×17 / ×18). This was expected, so the persistence path was not exercised because no detection existed. It has not been built yet anyway.
   - **Status:** n/a (expected). Persistence (Phase 3 `EventLogger`, batch 4 `d12b264`) is exercised with a fake predictor: synth video + fake birds on frames 90/120/150/400 → exactly 2 events

## Implementation notes (deviations from the brief)

- **run_id plumbing:** `main()` creates the run_id and one `ObsLogger(run_id, logs/)`, and passes that object (which carries `.run_id`) into `FrameGrabber(obs=…)` and `Detector(obs=…)`. There is no global. I passed the object rather than the bare id so that a single writer, with one lock, serves both threads; two handles appending to one file on Windows can interleave lines.
- **Per-frame throttling:** `render` also runs at 30 fps, so it gets the same 300-frame summary treatment as `capture_read`, `mog2_apply` and `gate_check`. For `gate_check`, the routine skips (`warmup`, `cadence`) happen 29 frames out of 30. They are counted in the summary rather than written one line each; only the warm-up begin and complete transitions get their own lines. Every `capture_read` failure still gets its own line, as specified (see item 1 for the resulting rate).
- **`--source` pacing:** the file is delivered at its own fps, from `CAP_PROP_FPS`. Unpaced, the grabber would decode hundreds of fps and the main loop would drop most frames, which would not behave like the camera. End of file is the first failed `read()`.
- **`--headless` + `--select-roi`:** `--select-roi` is ignored with a warning, since there is no dialog.
- **Snapshots from the `s` key** (PROJECT_REPORT §8.7): now written to `snapshots_dir` (root-relative) as `manual_<ts>_seq<N>.jpg` instead of `./snap_NNN.jpg` in the CWD. Fixed in batch 4 `d12b264`.
- **`.gitignore`:** `logs/` was added before the baseline commit; `data/samples/` (generated clips) was added in the feature commit.
