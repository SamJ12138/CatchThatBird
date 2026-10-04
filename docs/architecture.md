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
   scripts/make_demo_gif.py      docs/demo.gif, docs/demo-real.gif, docs/demo-multi.gif from main.py --annotate-out
   scripts/fetch_real_clip.py    downloads the real bird clip (Pixabay) into data/samples/real/
   scripts/fetch_clips.py        downloads the eight-clip real-footage corpus into data/samples/corpus/
   scripts/corpus_eval.py        runs main.py on the corpus and scores it against hand-written ground truth
   scripts/visit_report.py       visits per hour, median length, most birds at once, busiest hour
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

Birds are rare and small, and most of the time nothing moves. So a cheap per-frame motion model decides whether the expensive classifier runs at all, until a visit is open. From then on, the bird is re-checked where it was last seen, whether or not it moves. Motion that the classifier rejects is looked at again for about a second, in case it was a bird still in flight.

```
every picked-up frame
  region = frame[ROI]  (or the whole frame)
  each open visit's padded last box := cached background image
  the re-check window's crop, if one is open := cached background image
  mask = MOG2.apply(region)                 keeps the background model current,
                                            without ever learning a tracked bird
  frame_count <= motion_warmup_frames?  -> skip "warmup"
  re-check window open, and this is one of its every-nth frames?
                                        -> YOLOv8n on the window's crop ("recheck" crop)
                                           bird found, or recheck_window_frames used up: window closed
  (frame_count - warmup - 1) % N != 0?  -> skip "cadence"   (gated: 61, 91, 121 ...)
gated frame (about 1 per second at 30 fps, N = 30)
  morphology open + close (5x5 ellipse)
  every external contour >= motion_min_area -> its bounding box, full-frame coords,
    padded by motion_padding_px and clamped; padded boxes that share a pixel merged
    (repeatedly); the max_motion_regions (4) with the largest contour area -> "motion" crops
  each open visit's last box, padded the same way -> "track" crop
    (motion or not; skipped if a motion region already contains the box)
  YOLO budget: track crops first, then motion crops by area, while the token bucket
    (max_yolo_calls_per_s, refilled in capture time) has tokens; the rest are logged as
    deferred and run on the next frames as tokens refill, until the next gated frame
  YOLOv8n on each crop, target classes only, conf >= threshold
  boxes -> full-frame coords; duplicates across crops (IoU >= 0.5) -> one
  nothing found in a motion region, no rejected motion since the last gated frame
    without motion, and the region is not part of an open visit or by a bird found
    on this frame?                      -> open a re-check window on the largest such region
EventLogger.handle(frame, detections)       every processed frame, so visits expire on time
  cost of each (detection, open visit) pair: (1 - IoU) + centre distance / reach,
    reach = 2 x max(w, h) of the visit's last box; allowed only for the same class,
    within dedupe_within_seconds, and IoU >= 0.3 or centre within reach
  Hungarian assignment, one detection per visit: matched -> extend that visit
  unmatched and mostly inside a box handled on this frame -> duplicate of that bird
  unmatched otherwise -> open a visit (visit_id <run_id>-<n>): crop + full-frame snapshot
    now, line written when it closes; concurrent_max tracked while it is open
```

The design choices behind this:

- **MOG2 runs on the ROI sub-image, not a masked full frame.** Motion outside the ROI (trees, sky, passers-by) never enters the pipeline, and the per-frame cost scales with the ROI's area (§4).
- **MOG2 sees every frame; the rest runs every Nth.** A background model fed a thinned sequence adapts worse. Morphology, contours and YOLO are only paid on gated frames.
- **YOLO sees a padded crop, not the frame.** A small bird fills more of the network's input. The crop is also the snapshot saved with the event.
- **Every motion region goes on, up to four (multi-bird, `ebd40c4`).** Until then only the largest contour was classified: two birds that arrived in the same second produced one crop, and a still bird beside a larger moving branch was never looked at. Contours whose padded boxes overlap are merged first, so one bird broken into several contours is one crop. The cap bounds the cost on a busy frame; the YOLO budget bounds it per second.
- **Visits are matched jointly.** Matching each detection to its nearest open visit hands a fast bird's new position to a neighbour's visit when it lands nearer the neighbour's last box than its own; that visit then loses its bird and a second one opens (tested in `tests/test_multi_bird.py`). The detections of a frame are therefore assigned to the visits open before it with the Hungarian method (`scipy.optimize.linear_sum_assignment`, installed with ultralytics), at a cost of (1 - IoU) plus the centre distance over the reach. The old thresholds stay as the gate. A box mostly inside another box that joined or opened a visit on the same frame is the same bird (R5's head beside its body).
- **The YOLO budget counts what gated frames start.** Track and motion crops share `max_yolo_calls_per_s` (5) as a token bucket in capture time, so the limit is per second whatever the cadence. The re-check window's crops are not counted: they are bounded by the window, and throttling them would undo R2.
- **Open visits are tracked past the motion gate (P9).** A bird that holds still is learned into a MOG2 background in about a second and stops being motion. So while a visit is open, its last box gets its own YOLO crop on every gated frame, and the visit closes only after YOLO has not confirmed the bird for `dedupe_within_seconds`.
- **Tracked birds are kept out of the MOG2 update.** OpenCV's MOG2 has one global learning rate (`apply(learningRate=...)`), not a per-pixel one. Zeroing it while a visit is open would freeze adaptation to light over the whole ROI for the length of the visit. Instead, each open visit's padded box is replaced, in the image passed to `apply()`, by the model's own background image. The model therefore sees background there and never learns the bird, while it goes on adapting everywhere else.
  - **When the background image is taken.** It is refreshed on gated frames while no visit is open, before `apply()`. Taken after `apply()`, or while a visit is open, it already holds part of the bird from the frames between its arrival and the gated frame that opened the visit. A test caught this: the bird leaked into the mask.
- **Rejected motion is re-checked for a second (R2, `2cfe133`).** A visit opens only when YOLO finds the bird in a motion crop. A gated frame can catch the bird in flight, blurred, and by the next one a bird that landed and holds still is background, with no visit open to keep it out of the model. So a motion crop in which YOLO finds nothing opens a re-check window on that crop:
  - **Frozen.** For `recheck_window_frames` frames (30) the crop is replaced by the cached background image before `apply()`, exactly as an open visit's box is. What is in it stays foreground.
  - **Re-checked.** On every `recheck_every_n_frames`-th of those frames (3) YOLO runs on the same crop again, gated frame or not. Every third and not every frame, because on a laptop CPU a YOLO call plus MOG2 takes longer than a frame interval: at 1 the loop drops one frame in three for the length of the window (§4).
  - **Closed** when a bird is found (its visit then protects it as a track) or when the frames are used up. After that the crop is learned as usual, so a still object that is no bird is absorbed.
  - **One window per burst of motion.** Something that moves all the time and is no bird is rejected on every gated frame. If each rejection opened a window, YOLO would run on every frame for as long as the motion lasts, and the crop would stay frozen: after a change of light over the whole ROI, the model could never adapt. So after a rejected motion crop, or a window, the next window can open only once a gated frame has seen no motion at all.
  - **Not for motion that belongs to an open visit.** No window opens for a crop that touches a visit's track crop (a part of the bird outside its box, such as the hummingbird's bill), or whose centre is within `visit_center_distance` × max(w, h) of the visit's box: a bird found there would join that visit, not open one.
  - **The crop is fixed.** The window re-checks where the motion was on the gated frame. It does not follow the bird.
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
- With one visit open, each gated frame adds one track crop, and the motion crop still runs when something else moves. The maximum was 2 calls per gated frame, and n + 1 with n open visits (plus the re-check crop while a window is open, below). Since multi-bird tracking it is n + (motion regions, at most 4), capped by the budget at 5 per second; the corpus measurements are in [corpus-baseline.md](corpus-baseline.md).
- The track crop keeps being classified until `dedupe_within_seconds` after the bird's last confirmation. After the bird left at ~17 s, the last gated frames ran track crops that found nothing.
- A track crop (the 103×67 bird padded by 50 px, about 203×167) cost the same as a motion crop in this session: about 60 ms. MOG2 p50 stayed 1.8–1.9 ms on the ROI. Masking a box costs one copy of the ROI per frame while a visit is open, and the background image costs about 1 ms per gated frame.
- Run_id `2f100fc6` logged one visit: `visit_frames` 14, `last_seen` 13.3 s after `ts`. Before this change, the same clip gave `visit_frames` 1 and `last_seen == ts`.
- On real footage (a hummingbird at a feeder, 1920x1080, ROI 0.99 Mpx), run_id `493402ed` held one visit through the whole perch at 1.84 YOLO calls/s, 2 per gated frame: the bird's bill, outside its box, kept passing the motion gate. MOG2 p50 was 12.0–12.5 ms. All four real-clip runs are in [observations.md](observations.md), "Real-clip findings".

### The re-check window (R2, `2cfe133`)

All runs here are paced at 30 fps like a camera, with the real model on the same laptop. "Window off" is `recheck_window_frames: 0`, which is the code path before the fix.

| run | clip | window | YOLO calls in the run | inside the window | outside it |
|---|---|---|---|---|---|
| run_id `2fefc154` | blob, never a bird, motion on every gated frame | off | 18 | – | 1.00 /s |
| run_id `29e8f0eb` | blob | on, `recheck_every_n_frames: 1` (every frame) | 48: 18 motion + 30 re-check | 31 calls in 1.52 s: 20.4 /s. The 30 frames took 1.53 s of the clip (frames 61 to 107) | 1.00 /s |
| run_id `41a55d98` | blob | on, every third frame (default) | 28: 18 motion + 10 re-check | 11 calls in 1.06 s: 10.4 /s. The 30 frames took 1.10 s of the clip (61 to 94) | 1.00 /s |
| run_id `184c5229` | real clip, before the fix | – | 23 | – | 0 calls before the visit (1 gated frame, nothing moving); 23 with it open |
| run_id `123b6cc8`, run_id `6e73d3cf` | real clip, default config (every frame in the first run, every third in the second) | on, never opened | 23 | – | the same: 0, then 23 |
| run_id `8cc6c764` | real clip, warm-up 65 so that a gated frame catches the bird blurred | on, every frame, opened on frame 97 | 24 | 2 re-check calls (frames 101, 103: found) | 0 before the bird, 1 rejected motion crop on frame 97, 21 with the visit open |
| run_id `29bcb284` | the same | on, every third frame (default), opened on frame 97 | 24 | 2 re-check calls (frames 102, 106: found) | the same: 0, 1, then 21 |

- **Nothing moving, no visit open: no YOLO call,** as before. A window needs a gated frame with motion that YOLO rejects.
- **Inside a window, at the default of every third frame, the loop keeps up:** 10 calls/s, the window's 30 frames last 1.1 s, and all but a few frames reach MOG2. Its cost is bounded by the frame count: at most 10 re-check calls per window.
- **At every frame the CPU cannot keep up:** 20 calls/s here, not 30. A call (about 45 ms) plus MOG2 is longer than a frame interval, so the paced loop gets to about two frames in three, the window's 30 frames last about 1.5 s, and it costs up to 30 calls. That is why the default is 3. The price on the real clip is a visit that opens about 0.1 s later: 3.47–3.50 s against 3.40 s paced, 3.57 s against 3.43 s in the forced unpaced case (`docs/observations.md`, "The R2 fix").
- **Steady motion costs one window, then the old rate.** The blob moves for the whole clip: one window after the first gated frame, then one motion crop per gated frame, 1.00 /s, as in run_id `64dd9bfc`.
- **A window that finds its bird is short.** On the real clip the bird was found on the second re-check (run_id `8cc6c764`, run_id `29bcb284`), or after 1 to 5 calls in the forced runs (1 to 13 at every frame).
- **MOG2 during a window** pays the same copy of the ROI per frame as with a visit open.
- **The preview** stalls for one YOLO call on every third frame during a window, not one per second.

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
- **At most `max_motion_regions` (4) motion regions** are classified on a gated frame, and at most `max_yolo_calls_per_s` (5) gated-frame crops a second.
- **Close birds can still share a visit, and one bird can be two.** When YOLO misses one of two neighbours on a frame, the other's box can take its visit; one blurred bird boxed twice, side by side, is two visits. Corpus numbers: [corpus-baseline.md](corpus-baseline.md).
- **Every YOLO call skips frames on 1080p footage.** A call (about 58 ms) is longer than a frame interval at 30 fps; a gated frame with 5 calls skips about 8 frames (observations M3).
- **Anything YOLO keeps calling a bird is one long visit,** written as truncated every `max_visit_seconds` (600 s).
- **CPU by default.** CUDA works if a CUDA build of torch is installed (see `requirements.txt`), but only YOLO benefits.
- **Detection shares the UI thread,** so the preview stalls for one YOLO call per second, and for one on every third frame during a re-check window (about a second).
- **A visit can start up to one cadence late.** The motion gate looks once a second. A bird that enters just after a gated frame is first looked at on the next one. The re-check window (§3) only covers the second after a gated frame on which YOLO rejected motion.
- **A hard kill loses at most the last 60 s of an open visit.** It is recovered from `open_visits.json` at the next start (see [events-schema.md](events-schema.md)).
