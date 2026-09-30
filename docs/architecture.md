# Architecture

CatchThatBird watches a fixed camera view (a parked car) and writes one line per bird visit to `data/events.jsonl`, with snapshots. It is meant to produce a time series for later analysis. It is not an alarm, and it records no video.

This document covers the module graph, the threading model, the two-stage detection design, and the measured cost profile. The stage-by-stage failure modes are in [pipeline-stages.md](pipeline-stages.md), the output format is in [events-schema.md](events-schema.md), and the history of findings is in [observations.md](observations.md).

## 1. Module graph

```
                        config.yaml
                            |  yaml.safe_load + pydantic (extra keys rejected)
                            v
                    +------------------+
                    | config_schema.py |  AppConfig {camera, detection, logging, storage}
                    +------------------+  PROJECT_ROOT, resolve_path()
                      ^      ^       ^
                      |      |       |
   +-----------+      |      |       |        +--------+
   |  main.py  |------+      |       +--------| obs.py |  per-run JSONL log, spans,
   |  CLI, ROI,|             |                +--------+  300-frame summary counters
   |  preview, |---> camera.py (FrameGrabber, Frame, device listing)      ^
   |  signals  |---> detector.py (lazy import) ---- imports Frame --------+ every module
   +-----------+---> logger.py   (lazy import) ---- imports Frame, Detection   emits to it
         |
         +--> data/roi.json (or data/roi.example.json), logs/run_<id>.jsonl

   scripts/make_synth_video.py   synthetic clip for --source runs: a real bird photo, or a blob
   scripts/failure_report.py     summarises logs/run_<id>.jsonl (stdlib only)
   tests/fakes.py                FakeCapture, FakeDevice, FakePredictor, FakeClock
```

| Module | Owns | Depends on |
|---|---|---|
| `main.py` | argument parsing, pre-flight checklist, device sanity check, ROI load/select, the frame loop, HUD, shutdown signals, exit codes | everything below |
| `camera.py` | `FrameGrabber` (capture thread, reconnect with backoff, file pacing), `Frame`, `list_devices()` | OpenCV, `obs` |
| `detector.py` | `Detector` (MOG2 gate + predictor), `Predictor` protocol, `YoloPredictor` | `camera.Frame`, `config_schema`, `obs`; torch and ultralytics are imported inside `YoloPredictor.load()` |
| `logger.py` | `EventLogger`: visits, dedupe, snapshots, daily cap, retention | `camera`, `detector.Detection`, `config_schema`, `obs` |
| `config_schema.py` | pydantic models, path resolution | pydantic, PyYAML |
| `obs.py` | structured run log (`start` / `success` / `fail` / `skip`) | stdlib |

`detector` and `logger` are imported inside `run_preview()`, so `--list-devices` never loads torch. The detector reaches YOLO only through the `Predictor` protocol: tests inject `FakePredictor`, and the test process never imports ultralytics (a session hook in `tests/conftest.py` enforces this).

## 2. Threading model

Two threads share one frame slot.

```
 grabber thread (FrameGrabber._run)            main thread (run_preview loop)
 ----------------------------------            ------------------------------
 cap.read()                                    frame = grabber.read_latest()
 Frame(image, captured_at, seq, wall time)     same seq as last time? -> idle
 lock; slot = frame; notify                    detector.process(frame)      \ held against
 camera: loop immediately                      events.handle(frame, dets)   / shutdown signals
 file, paced: sleep until the next frame is due grabber.ack(seq)
 file, --no-pace: wait for ack(seq)            draw HUD, imshow, waitKey (unless --headless)
```

- **Latest-frame slot, not a queue.** The producer overwrites the slot, so a slow consumer skips frames and never builds a backlog. Skipped frames are counted (`skipped_total` in the `render` summary).
- **Unpaced hand-off (`--no-pace`, files only).** The producer waits for the consumer's `ack(seq)` before reading on, and the consumer waits on the grabber's publish condition. Every frame is processed exactly once, in order. Tests and the quickstart use this mode.
- **Time.** `FrameGrabber` and `run_preview` take `clock` and `sleep` arguments (defaults `time.monotonic` / `time.sleep`). Pacing, the read-retry delay, the 2 s stall timer, the reconnect backoff (1, 2, 4 … 30 s), the first-frame timeout and the headless idle loop all use them. So the tests run the real policy on `tests/fakes.FakeClock`, and no test waits in real time.
- **Reconnect** runs on the grabber thread. Backoff waits are cut into 0.1 s slices with a `stop()` check between them.
- **Shutdown.** SIGINT, SIGTERM and SIGBREAK raise `Shutdown` in the main thread, but never while a frame is inside `detector.process` / `events.handle`: there it is held until the frame is done. The `with` blocks then stop the grabber (join with a 2 s timeout; a capture still inside `read()` is left open rather than released under the reader) and close the `EventLogger`, which writes every open visit.
- **Detection runs on the main thread.** On CPU a YOLO call blocks the preview for its duration once a second (§4).

## 3. Two-stage detection

Birds are rare and small, and most of the time nothing moves. So a cheap per-frame motion model decides whether the expensive classifier runs at all.

```
every picked-up frame
  region = frame[ROI]  (or the whole frame)
  mask = MOG2.apply(region)                 keeps the background model current
  frame_count <= motion_warmup_frames?  -> skip "warmup"
  (frame_count - warmup - 1) % N != 0?  -> skip "cadence"   (gated: 61, 91, 121 ...)
gated frame (about 1 per second at 30 fps, N = 30)
  morphology open + close (5x5 ellipse)
  largest external contour; area < motion_min_area?  -> skip
  bounding box -> full-frame coords -> pad by motion_padding_px, clamp
  YOLOv8n on the padded crop, target classes only, conf >= threshold
  boxes -> full-frame coords -> [Detection]
EventLogger.handle(frame, detections)       every processed frame, so visits expire on time
  match an open visit (same class, IoU >= 0.5, within dedupe_within_seconds)? extend it
  else open a visit: crop + full-frame snapshot now, line written when it closes
```

The design choices behind this:

- **MOG2 runs on the ROI sub-image, not a masked full frame.** Motion outside the ROI (trees, sky, passers-by) never enters the pipeline, and the per-frame cost scales with the ROI's area (§4).
- **MOG2 sees every frame; the rest runs every Nth.** A background model fed a thinned sequence adapts worse. Morphology, contours and YOLO are only paid on gated frames.
- **YOLO sees a padded crop, not the frame.** A small bird fills more of the network's input. The crop is also the snapshot saved with the event.
- **Only the largest contour goes on.** Two birds far apart in the ROI produce one crop per gated frame. This is a known limitation.
- **The capture time, not the inference time, is the event time.** `Frame.captured_wall_time` is taken right after `cap.read()` and carried into `Detection` and `events.jsonl`.

## 4. Cost profile

Every figure below comes from a run log in [runs/](runs/). Those files are copies of `logs/run_<id>.jsonl` with the absolute project and temp directories replaced by `.` and `<tmp>`. All runs were done on one Windows 11 laptop with CPU-only torch 2.12.0 (`detector_init` context `device: cpu`). Each processed a 600-frame, 30 fps synthetic clip from `scripts/make_synth_video.py` with `main.py --source <clip> --headless`, paced at the file's frame rate like a camera, with the real `yolov8n.pt`. YOLO never labels the synthetic blob a bird, so these runs measure the pipeline, not detection quality.

Per-frame stages log a p50 and a p95 per 300-frame window. Two windows per run give the ranges shown.

### Current code (2026-09-30, after the phase-3 fixes)

| run | clip | MOG2 input | `mog2_apply` p50 / p95 ms | frames reaching MOG2 | YOLO calls, rate | YOLO first call ms | YOLO p50 / p95 ms (warm) |
|---|---|---|---|---|---|---|---|
| run_id `74a7ba12` | 1920×1080 | whole frame, 2.07 Mpx | 18.9–19.0 / 22.8–24.3 | 589 / 600 | 18, 0.98 /s | 103.5 | 42.1 / 47.3 |
| run_id `196c1173` | 1920×1080 | ROI 626×471, 0.29 Mpx | 2.75–2.79 / 3.66–3.69 | 597 / 600 | 18, 1.00 /s | 100.4 | 38.8 / 44.0 |
| run_id `3b7d24cd` | 1280×720 | whole frame, 0.92 Mpx | 7.9–8.0 / 10.3–10.5 | 598 / 600 | 18, 1.00 /s | 104.6 | 39.7 / 43.3 |
| run_id `937040f3` | 1280×720 | ROI 417×314, 0.13 Mpx | 1.33–1.57 / 2.13–2.17 | 598 / 600 | 18, 1.00 /s | 98.0 | 41.1 / 47.3 |

Same runs, other stages:

- `detector_init` (lazy torch + ultralytics import and weight load, before capture starts): 2233–2257 ms in all four.
- `capture_read` (decode, grabber thread) p50: 3.0–3.7 ms at 1080p (run_id `74a7ba12`), 1.4 ms at 720p (run_id `3b7d24cd`).
- `morph_contour` p50 on a gated frame: 2.88 ms on the whole 1080p frame (run_id `74a7ba12`), 0.42 ms on the 1080p ROI (run_id `196c1173`).

### Earlier session (code at `7633218`, the observability pass)

| run | clip | MOG2 input | `mog2_apply` p50 / p95 ms | frames reaching MOG2 | YOLO calls, rate | YOLO first call ms | YOLO p50 / p95 ms (warm) |
|---|---|---|---|---|---|---|---|
| run_id `401cd60f` | 1280×720 | whole frame | 12.5–15.5 / 20.9–22.7 | 546 / 600 | 17, 0.97 /s | 275.0 | 61.8 / 68.3 |
| run_id `a38671d5` | 1920×1080 | ROI 626×471 | 3.85–3.92 / 6.67–6.75 | 575 / 600 | 18, 0.97 /s | 266.1 | 64.8 / 70.7 |

These are the figures behind the round numbers used elsewhere in this project: MOG2 about 14 ms/frame on a full frame against about 4 ms on the ROI, and YOLO about 60 ms per call at about 1 Hz. The same laptop was 1.4–2.0× slower in that session than in the current one: 720p whole-frame MOG2 was 12.5–15.5 ms against 7.9–8.0 ms, 1080p ROI MOG2 was 3.85–3.92 ms against 2.75–2.79 ms, and warm YOLO was 61.8–64.8 ms against 38.8–39.7 ms. The ratios between stages did not change. In that session fewer frames reached MOG2 because the model was still loaded after capture had started (obs F3, fixed in `3ba2301`).

### What the numbers mean

- **MOG2 costs roughly 9–10 ms per megapixel of input** in the current session: 18.9 ms / 2.07 Mpx, 7.9 ms / 0.92 Mpx, 2.75 ms / 0.29 Mpx and 1.33 ms / 0.13 Mpx, which is 8.6 to 10.2 ms per Mpx. The ROI is therefore the main cost control. The default ROI is 14% of a 1080p frame, and it cuts per-frame MOG2 time about 7× (run_id `74a7ba12` against run_id `196c1173`).
- **Per second of 30 fps video, on the main thread:** MOG2 costs 30 × 18.9 ≈ 570 ms on the whole 1080p frame, but 30 × 2.75 ≈ 83 ms on the ROI. YOLO adds one call per second, 39–42 ms in the current session (62–65 ms in the earlier one). With an ROI, detection therefore uses roughly an eighth of one core. Without one, MOG2 alone takes more than half.
- **YOLO's first call is cold:** 98–105 ms now (2.4–2.7× a warm call), 266–275 ms before (4.1–4.4×). It is paid once per run.
- **At 1 Hz, YOLO is not the bottleneck on CPU;** per-frame MOG2 on an unrestricted frame is. A GPU speeds up only the YOLO part.

To reproduce a row, regenerate the blob clip these runs used (`python scripts/make_synth_video.py --no-bird [--width 1920 --height 1080 --out data/samples/synth_blob_1080p.mp4]`; without `--no-bird` the script now makes the bird clip), then run `main.py --source <clip> --headless` with `--roi-file` pointing either at `data/roi.example.json` or at a file holding `{"whole_frame": true, "frame_width": W, "frame_height": H}`. Read the result with `python scripts/failure_report.py --latest`.

## 5. Startup order

1. Parse arguments and open the run log (`logs/run_<id>.jsonl`).
2. Load `config.yaml` (any error: exit 1 with a one-line message).
3. Camera only: list devices, check the selected one, show the Pocket 3 checklist (skipped with `--yes`; stdin EOF: exit 2).
4. Build the `Detector` and load the model (§4: about 2.2 s on CPU), then the `EventLogger` (retention sweep). Both run before capture starts, so no frame is lost while they load.
5. Start the grabber and wait for the first frame (`--first-frame-timeout`, default 5 s; exit 1 if none arrives).
6. Load the ROI: `data/roi.json`, rescaled if only the resolution changed. Headless without it, `data/roi.example.json` (with a warning). Otherwise the selection dialog, whose result is saved.
7. Run the frame loop until end of file, `q` / `Esc`, a shutdown signal, or an error.

## 6. Known limitations

- **Capture backends are Windows-first.** Device names come from DirectShow (pygrabber), and the capture tries DSHOW and then MSMF. On Linux and macOS, `--source` files work, but live capture and device naming are untested.
- **Only the largest contour** in the ROI is classified on a gated frame.
- **CPU by default.** CUDA works if a CUDA build of torch is installed (see `requirements.txt`), but only YOLO benefits.
- **Detection shares the UI thread,** so the preview stalls for one YOLO call per second.
- **A hard kill loses the open visit** (see [events-schema.md](events-schema.md)).
