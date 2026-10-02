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
   scripts/make_demo_gif.py      docs/demo.gif, docs/demo-real.gif from main.py --annotate-out
   scripts/fetch_real_clip.py    downloads the real bird clip (Pixabay) into data/samples/real/
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
 file, --no-pace: wait for ack(seq)            draw HUD, imshow, waitKey (unless --headless);
                                               --annotate-out: write the same image to a file
```

- **Latest-frame slot, not a queue.** The producer overwrites the slot, so a slow consumer skips frames and never builds a backlog. Skipped frames are counted (`skipped_total` in the `render` summary).
- **Unpaced hand-off (`--no-pace`, files only).** The producer waits for the consumer's `ack(seq)` before reading on, and the consumer waits on the grabber's publish condition. Every frame is processed exactly once, in order. Tests and the quickstart use this mode.
- **Time.** `FrameGrabber` and `run_preview` take `clock` and `sleep` arguments (defaults `time.monotonic` / `time.sleep`). Pacing, the read-retry delay, the 2 s stall timer, the reconnect backoff (1, 2, 4 … 30 s), the first-frame timeout and the headless idle loop all use them. So the tests run the real policy on `tests/fakes.FakeClock`, and no test waits in real time.
- **Reconnect** runs on the grabber thread. Backoff waits are cut into 0.1 s slices with a `stop()` check between them.
- **Shutdown.** SIGINT, SIGTERM and SIGBREAK raise `Shutdown` in the main thread, but never while a frame is inside `detector.process` / `events.handle`: there it is held until the frame is done. The `with` blocks then stop the grabber (join with a 2 s timeout; a capture still inside `read()` is left open rather than released under the reader) and close the `EventLogger`, which writes every open visit.
- **Detection runs on the main thread.** On CPU a YOLO call blocks the preview for its duration once a second (§4).

## 3. Two-stage detection

Birds are rare and small, and most of the time nothing moves. So a cheap per-frame motion model decides whether the expensive classifier runs at all, until a visit is open. From then on, the bird is re-checked where it was last seen, whether or not it moves.

```
every picked-up frame
  region = frame[ROI]  (or the whole frame)
  each open visit's padded last box := cached background image
  mask = MOG2.apply(region)                 keeps the background model current,
                                            without ever learning a tracked bird
  frame_count <= motion_warmup_frames?  -> skip "warmup"
  (frame_count - warmup - 1) % N != 0?  -> skip "cadence"   (gated: 61, 91, 121 ...)
gated frame (about 1 per second at 30 fps, N = 30)
  morphology open + close (5x5 ellipse)
  largest external contour >= motion_min_area?  -> "motion" crop
    (bounding box -> full-frame coords -> pad by motion_padding_px, clamp)
  each open visit's last box, padded the same way -> "track" crop
    (motion or not; skipped if the motion crop already contains the box)
  YOLOv8n on each crop, target classes only, conf >= threshold
  boxes -> full-frame coords; duplicates across crops (IoU >= 0.5) -> one
EventLogger.handle(frame, detections)       every processed frame, so visits expire on time
  joins an open visit (same class, within dedupe_within_seconds, and IoU >= 0.3
    with its last box or centre within 2 x max(w, h) of it)? extend it
  else open a visit: crop + full-frame snapshot now, line written when it closes
```

The design choices behind this:

- **MOG2 runs on the ROI sub-image, not a masked full frame.** Motion outside the ROI (trees, sky, passers-by) never enters the pipeline, and the per-frame cost scales with the ROI's area (§4).
- **MOG2 sees every frame; the rest runs every Nth.** A background model fed a thinned sequence adapts worse. Morphology, contours and YOLO are only paid on gated frames.
- **YOLO sees a padded crop, not the frame.** A small bird fills more of the network's input. The crop is also the snapshot saved with the event.
- **Only the largest contour goes on.** Two birds that arrive far apart in the same second produce one motion crop. Once the first has an open visit, it is masked out of the motion model, and the second is the motion on a later gated frame if it moves.
- **Open visits are tracked past the motion gate (P9).** A bird that holds still is learned into a MOG2 background in about a second and stops being motion. So while a visit is open, its last box gets its own YOLO crop on every gated frame, and the visit closes only after YOLO has not confirmed the bird for `dedupe_within_seconds`.
- **Tracked birds are kept out of the MOG2 update.** OpenCV's MOG2 has one global learning rate (`apply(learningRate=...)`), not a per-pixel one. Zeroing it while a visit is open would freeze adaptation to light over the whole ROI for the length of the visit. Instead, each open visit's padded box is replaced, in the image passed to `apply()`, by the model's own background image. The model therefore sees background there and never learns the bird, while it goes on adapting everywhere else.
  - **When the background image is taken.** It is refreshed on gated frames while no visit is open, before `apply()`. Taken after `apply()`, or while a visit is open, it already holds part of the bird from the frames between its arrival and the gated frame that opened the visit. A test caught this: the bird leaked into the mask.
- **Visit matching is loose on purpose (P10).** A bird that hops more than half its length between gated frames, or a motion crop that cuts it off (a truncated box), has an IoU below 0.5 with its last box. Visits therefore join on IoU >= 0.3 OR a centre within 2 x max(w, h). Both thresholds are in `config.yaml`.
- **The capture time, not the inference time, is the event time.** `Frame.captured_wall_time` is taken right after `cap.read()` and carried into `Detection` and `events.jsonl`. For a video file it is media time: the wall time of the first read plus (seq - 1) / fps. A `--no-pace` run then keeps the clip's own timeline, for visit durations and for the dedupe window.

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

### With a visit open (after P9, current code)

These runs are the default bird clip (`synth_bird.mp4`, 1280×720, example ROI) and the blob clip, both paced at 30 fps like a camera, with the real model. Calls per second are counted per gated frame, one per second of video.

| run | clip | gated frames with no visit open | YOLO calls / s | gated frames with a visit open | YOLO calls / s (max per gated frame) | YOLO p50 ms, motion / track crop |
|---|---|---|---|---|---|---|
| run_id `64dd9bfc` | blob, never a bird | 18 | 1.00 (18 calls; motion every second) | 0 | – | 63.1 / – |
| run_id `2f100fc6` | bird: lands just before 4 s, perches, leaves at ~17 s | 3 | 0.33 (1 call; nothing moved before the landing) | 15 | 1.07 (16 calls; max 2) | 118.9 (2 calls, 1 cold) / 60.4 (15 calls) |

- With no visit open, the call rate is what it was: one call per gated frame that has motion.
- With one visit open, each gated frame adds one track crop, and the motion crop still runs when something else moves. The maximum is 2 calls per gated frame, and n + 1 with n open visits.
- The track crop keeps being classified until `dedupe_within_seconds` after the bird's last confirmation. After the bird left at ~17 s, the last gated frames ran track crops that found nothing.
- A track crop (the 103×67 bird padded by 50 px, about 203×167) cost the same as a motion crop in this session: about 60 ms. MOG2 p50 stayed 1.8–1.9 ms on the ROI. Masking a box costs one copy of the ROI per frame while a visit is open, and the background image costs about 1 ms per gated frame.
- Run_id `2f100fc6` logged one visit: `visit_frames` 14, `last_seen` 13.3 s after `ts`. Before this change, the same clip gave `visit_frames` 1 and `last_seen == ts`.
- On real footage (a hummingbird at a feeder, 1920x1080, ROI 0.99 Mpx), run_id `493402ed` held one visit through the whole perch at 1.84 YOLO calls/s, 2 per gated frame: the bird's bill, outside its box, kept passing the motion gate. MOG2 p50 was 12.0–12.5 ms. All four real-clip runs are in [observations.md](observations.md), "Real-clip findings".

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
- **Only the largest motion contour** in the ROI is classified on a gated frame (open visits are classified separately).
- **Two birds within 2 x a bird's size of each other merge into one visit.**
- **Anything YOLO keeps calling a bird is one long visit,** written as truncated every `max_visit_seconds` (600 s).
- **CPU by default.** CUDA works if a CUDA build of torch is installed (see `requirements.txt`), but only YOLO benefits.
- **Detection shares the UI thread,** so the preview stalls for one YOLO call per second.
- **A hard kill loses at most the last 60 s of an open visit.** It is recovered from `open_visits.json` at the next start (see [events-schema.md](events-schema.md)).
