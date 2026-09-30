# CatchThatBird — Project Structure Report

*Generated 2026-09-30 from the source tree at `CatchThatBird`. All files last modified 2026-06-03.*

---

## 1. What the project is

CatchThatBird records when birds visit the owner's parked car. A **DJI Osmo Pocket 3** runs in USB‑C Webcam (UVC) mode, pointed at the car. A Python pipeline reads the video, uses cheap background subtraction to spot motion inside a user‑drawn region of interest (ROI), and runs **YOLOv8n** on the moving region to confirm that it is a bird.

The goal is a **passive time-series log** of bird visits for later analysis. It is not a security or alerting system. The planned outputs are `data/events.jsonl` (one event per line) plus snapshot JPEGs, and later a Streamlit dashboard to review them.

### Current status at a glance

| Phase | Scope | State |
|---|---|---|
| 1 | Camera skeleton: device discovery, threaded capture, live preview + HUD | **Done** |
| 2 | Detection: MOG2 motion gate → YOLOv8n bird classifier, ROI selection | **Done** (detections are logged to the console and drawn on screen only) |
| 3 | Event logger: `events.jsonl`, snapshots, IoU/time dedupe | **Not started.** Its config keys exist but no code reads them |
| 4 | Streamlit review dashboard | Not started |
| 5 | Reliability: camera auto-reconnect, disk guards, retention | Not started |
| 6 | (Optional) visit-pattern analysis | Not started |

`requirements.txt` says so directly: *"Phases 3‑6 will add: streamlit, plotly, scikit-learn, scipy."*

---

## 2. Directory layout

```
CatchThatBird/
├── main.py              447 lines  CLI entry point, pre-flight checklist, ROI selection, preview/HUD loop
├── camera.py            187 lines  UVC device enumeration + threaded FrameGrabber
├── detector.py          213 lines  MOG2 motion gate + YOLOv8n bird detection
├── config_schema.py      61 lines  pydantic v2 models + YAML loader
├── config.yaml                     runtime configuration (camera / detection / logging / storage)
├── requirements.txt                pip dependencies (+ CUDA torch install notes)
├── .gitignore                      excludes venv, generated data, *.pt weights, IDE dirs
├── yolov8n.pt           6.5 MB     YOLOv8-nano COCO weights (auto-downloaded by ultralytics)
├── data/
│   ├── roi.json                    saved ROI rectangle (persisted by main.py)
│   └── snapshots/                  empty; reserved for Phase 3
├── .venv/               1.3 GB     Python 3.14.2 virtualenv
└── __pycache__/                    compiled .pyc for the 4 modules (cpython-314)
```

**908 lines of Python in total.** There are no tests, README, DEVLOG, or CLAUDE.md. The folder is **not a git repository**, even though it has a `.gitignore`.

---

## 3. Architecture

### 3.1 Module dependency graph

```
                 config.yaml
                     │ yaml.safe_load
                     ▼
            ┌──────────────────┐
            │ config_schema.py │  AppConfig{Camera,Detection,Logging,Storage}Config
            └────────┬─────────┘
                     │
        ┌────────────┼──────────────────────────┐
        ▼            ▼                          ▼
  ┌──────────┐  ┌───────────┐  imports Frame  ┌─────────────┐
  │ main.py  │─▶│ camera.py │◀────────────────│ detector.py │
  │ (entry)  │  └───────────┘                 └─────────────┘
  │          │──── lazy import (inside run_preview) ──▶ detector.py
  └──────────┘
        │ reads/writes
        ▼
   data/roi.json
```

- `camera.py` depends only on OpenCV, numpy, and loguru (plus `pygrabber` on Windows, imported lazily).
- `detector.py` depends on `camera.Frame`, `config_schema.DetectionConfig`, `torch`, and `ultralytics`.
- `main.py` imports `detector` **lazily**, inside `run_preview()`, so `--list-devices` skips the 1–3 s ultralytics/torch import.

### 3.2 Runtime flow

```
python main.py [--list-devices] [--device N] [--select-roi]
 │
 ├─ configure_logger()                 loguru → stderr, INFO, coloured format
 ├─ parse_args()
 ├─ load_config(config.yaml)           pydantic validation (extra keys forbidden)
 ├─ --device N ? → override camera.device_index
 ├─ list_devices()                     pygrabber DirectShow names, or probe idx 0-5
 ├─ print_device_list()                tags each device "DJI Pocket" / "built-in webcam"
 ├─ --list-devices ? → exit 0
 ├─ resolve_device_selection()         warns if the chosen index looks like a laptop cam
 ├─ wait_for_enter()                   prints 7-item Pocket 3 checklist, blocks on Enter
 └─ run_preview()
      ├─ FrameGrabber.start()          background thread: cap.read() → latest-frame slot
      ├─ _wait_for_first_frame(5 s)
      ├─ load_or_select_roi()          reuse data/roi.json if the resolution matches, else cv2.selectROI
      ├─ Detector(config.detection, roi)
      └─ loop:
           frame = grabber.read_latest()
           same seq as last time? → waitKey(5), continue      (no redundant redraws)
           update display-FPS / latency EMAs, skipped count
           detector.process(frame) → [Detection] → log + keep on screen for 1.5 s
           draw ROI box, detection boxes, 3-line HUD → imshow
           keys: q/Esc quit · s save full frame to ./snap_NNN.jpg
```

### 3.3 Threading and timing model

| Thread | Work | Rate |
|---|---|---|
| `FrameGrabber` (daemon) | `cap.read()`; wraps each image in a `Frame(image, captured_at=monotonic, seq)`; overwrites a single slot under a `threading.Lock` | Camera rate (up to 30 fps) |
| Main thread | Peeks the slot; for each **new** `seq`: MOG2 update, then every Nth frame morphology + contours + YOLO; draws the HUD; `imshow` | Up to camera rate; YOLO about 1 Hz |

The design uses a **single latest-frame slot** rather than a queue. The consumer never falls behind: when it is slow, it skips frames (counted in `skipped=` on the HUD) instead of building a backlog. `pipe_lat~` is the time from `cap.read()` returning to the frame being processed on the main thread.

---

## 4. Components in detail

### 4.1 `config_schema.py` — typed configuration

Five pydantic v2 `BaseModel`s. Each sets `ConfigDict(extra="forbid")`, so a mistyped key in `config.yaml` fails at startup instead of being silently ignored.

| Model | Fields (default) |
|---|---|
| `CameraConfig` | `device_index` 0, `width` 1920, `height` 1080, `fps` 30 |
| `DetectionConfig` | `process_every_n_frames` 30, `motion_min_area` 200, `motion_threshold` 16, `motion_padding_px` 50, `motion_warmup_frames` 60, `yolo_model` "yolov8n.pt", `yolo_target_classes` ["bird"], `yolo_confidence_threshold` 0.35, `dedupe_within_seconds` 10 |
| `LoggingConfig` | `events_file` data/events.jsonl, `snapshots_dir` data/snapshots, `snapshot_format` "jpeg", `snapshot_quality` 90, `save_full_frame` True |
| `StorageConfig` | `max_events_per_day` 500, `retention_days` 30 |
| `AppConfig` | Requires all four sections above (none can be omitted) |

`load_config(path)` → `yaml.safe_load` → `AppConfig.model_validate`.

### 4.2 `config.yaml` — runtime values

Its values match the schema defaults, with comments giving the reasoning behind each:

- **Camera:** 1080p30 at device index 0.
- **Detection:** YOLO runs on 1 of every 30 frames ("30fps grab, process 1 fps"). `motion_min_area: 200` is set low because birds are small. The confidence threshold of 0.35 tolerates distant birds. MOG2 warms up for 60 frames (~2 s). YOLO sees 50 px of padding around the motion box for context. `dedupe_within_seconds: 10` is meant to collapse a perching bird into one event.
- **Logging / storage:** JSONL events and JPEG snapshots at quality 90, keeping both the crop and the full frame. There is a disk-fill guard of 500 events/day and a 30-day retention period.

> ⚠️ The `logging` and `storage` sections and `detection.dedupe_within_seconds` are **validated but never read** by any code yet. They are placeholders for Phases 3 and 5.

### 4.3 `camera.py` — device discovery and capture

**Device discovery**
- `list_devices()` uses `pygrabber.dshow_graph.FilterGraph().get_input_devices()` to get real DirectShow device names. Its enumeration order matches OpenCV's `CAP_DSHOW` indices, so names map straight to indices. If pygrabber is missing or fails, it falls back to probing indices 0–5 with `cv2.VideoCapture` and returns placeholder names `<device N>`.
- `classify_device(name)` matches keywords case-insensitively:
  - `pocket` for `dji`, `osmo`, or `pocket`
  - `builtin` for `integrated`, `built-in`, `internal`, or laptop brands (`lenovo`, `hp `, `dell`, `surface`, `thinkpad`, `ideapad`)
  - `unknown` for everything else
- `find_pocket_index(devices)` returns the first device classified as `pocket`.

**`Frame`** is a frozen dataclass: `image: np.ndarray`, `captured_at: float` (`time.monotonic()` right after `read()` returns), `seq: int` (starts at 1).

**`FrameGrabber`** is a context manager (`__enter__` → `start`, `__exit__` → `stop`).
- `_open_capture()` tries **DirectShow first, then Media Foundation**. If both fail, it raises a `RuntimeError` that tells the user to check Webcam mode and whether another app holds the camera.
- `_configure()` requests the **MJPG** FourCC (for 1080p30 headroom over USB), the configured width/height/fps, and `BUFFERSIZE=1`. It then logs the resolution and fps the driver actually negotiated.
- `_run()` is the producer loop. When a read fails, it increments a counter, logs a warning on the 1st failure and every 30th after that, and sleeps 50 ms. When a read succeeds, it bumps `seq` and publishes the frame under the lock.
- `read_latest()` returns the current slot (`None` until the first frame).
- `stop()` sets the stop event, joins the thread (2 s timeout), releases the capture, and logs totals.
- `stats` returns `{"captured", "failures"}`.

### 4.4 `detector.py` — two-stage bird detection

**`Detection`** is a frozen dataclass: `class_name`, `confidence`, `bbox_xywh` (full-frame pixels), `frame_seq`, `captured_wall_time` (`time.time()`).

**`Detector(config, roi=None)`**

Setup:
1. Creates `cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=motion_threshold, detectShadows=False)` and a 5×5 elliptical morphology kernel.
2. Picks the torch device: `cuda` if available, otherwise `cpu`. On CPU it logs a warning that points to the CUDA install note in `requirements.txt`.
3. Loads `YOLO(config.yolo_model)`. Ultralytics downloads the weights on first run, which is where `yolov8n.pt` in the project root came from.
4. `_resolve_class_ids()` maps the configured class names (e.g. `bird`) to COCO class ids. It raises `ValueError` if a name is unknown and lists sample valid names.

`process(frame)` is called once per new frame:

```
region = frame[ROI] (or whole frame)
fg_mask = MOG2.apply(region)                       ← every frame (keeps the background model current)
frame_count < warmup?            → []
frame_count % N != 0?            → []              ← gate: only every Nth frame continues
morph OPEN then CLOSE (5×5 ellipse)                ← noise cleanup, gated frames only
findContours(RETR_EXTERNAL) → largest contour
area < motion_min_area?          → []
bbox → translate ROI→full-frame → pad by motion_padding_px, clamp to frame
YOLO.predict(crop, conf=…, classes=[bird ids], device=…)
map each box back to full-frame coords → [Detection]
```

Design notes:
- Running MOG2 **on the ROI sub-image** rather than on a masked full frame means motion outside the ROI (trees, sky) never enters the pipeline. It also makes the per-frame cost scale with ROI area: the saved ROI is 626×471 ≈ 295k px, about 14% of a 1080p frame.
- Only the gated frame pays for morphology and YOLO, so the expensive work runs about once per second.
- YOLO runs on a **padded crop**, not the full frame, so small birds take up more of the network's input.
- `stats` returns `frames_processed`, `motion_gates_opened`, `yolo_invocations`, and `detections_total`; these appear on the HUD.

### 4.5 `main.py` — entry point, UX, and preview loop

| Function | Role |
|---|---|
| `configure_logger()` | Replaces loguru's default sink with a coloured stderr sink at INFO level |
| `parse_args()` | `--list-devices`, `--device N`, `--select-roi` |
| `print_device_list()` | Prints `[idx] name`, a Pocket/built-in hint, and a `[SELECTED]` marker |
| `resolve_device_selection()` | Checks the configured index. Warns and suggests `--device <pocket idx>` if the index is missing or looks like a built-in webcam |
| `wait_for_enter()` | Prints the `CHECKLIST` banner and blocks on `input()`; exits cleanly on Ctrl‑C/EOF |
| `_wait_for_first_frame()` | Polls the grabber every 20 ms, for up to 5 s |
| `load_or_select_roi()` | Loads `data/roi.json` if its `frame_width`/`frame_height` match the live frame. Otherwise opens `cv2.selectROI` in a 1280×720 window. A zero-size selection means "whole frame". The result is saved with the frame size so a resolution change forces re-selection |
| `_put_text()` | Draws outlined HUD text (black 4 px stroke under a coloured 1 px fill) |
| `run_preview()` | Main loop (§3.2): new-frame detection, EMA stats (α = 0.1), overlays, keyboard handling |
| `main()` | Wires everything together. Returns exit code 0/1; always calls `cv2.destroyAllWindows()` |

**Pre-flight checklist** (printed before the camera opens):
1. Gimbal mode = LOCK
2. ActiveTrack disabled
3. Focus = MF, focused on the car
4. 4× digital zoom (~80 mm equivalent)
5. USB‑C connected, mode = Webcam
6. No other app using the camera
7. The selected device index is the Pocket 3

**Overlays:**
- Orange ROI rectangle
- Green detection boxes with `class conf` labels, kept for 1.5 s
- Three HUD lines:
  - `seq / fps / pipe_lat`
  - `captured / skipped / read_fail`
  - `motion / yolo / dets`

---

## 5. Data artifacts

| Path | Producer | Contents | Status |
|---|---|---|---|
| `data/roi.json` | `load_or_select_roi` | `{"x":710,"y":426,"w":626,"h":471,"frame_width":1920,"frame_height":1080}` | Present (saved 2026-06-03) |
| `./snap_NNN.jpg` | `s` key in preview | Raw full frame | Written to the **current working directory**, not `data/` |
| `data/events.jsonl` | Phase 3 logger | One JSON event per line | Not implemented; file does not exist |
| `data/snapshots/` | Phase 3 logger | Crop + optional full frame | Empty directory |
| `yolov8n.pt` | ultralytics auto-download | COCO YOLOv8-nano weights | Present, 6.5 MB |

`.gitignore` excludes `data/events.jsonl`, `data/snapshots/`, `snap_*.jpg`, `*.pt`, the venvs, `.env`, and IDE folders.

---

## 6. Dependencies and environment

**Declared (`requirements.txt`)**

| Package | Purpose |
|---|---|
| `opencv-python>=4.10` | UVC capture, MOG2, morphology, contours, GUI windows, ROI selector |
| `numpy>=1.26` | Frame arrays |
| `pydantic>=2.6` | Config validation |
| `pyyaml>=6.0` | Config parsing |
| `loguru>=0.7` | Logging |
| `pygrabber>=0.2` (win32 only) | DirectShow device names |
| `ultralytics>=8.1` | YOLOv8 (pulls in torch) |

The file also says to install a **CUDA torch wheel first** (`--index-url …/whl/cu124`) if GPU is wanted, and notes that CPU is fine for YOLOv8n at 1 fps.

**Installed in `.venv`** (Python **3.14.2**, from `C:\Python314`):

| Package | Version |
|---|---|
| loguru | 0.7.3 |
| numpy | 2.4.6 |
| opencv-python | 4.13.0.92 |
| pydantic | 2.13.4 |
| pygrabber | 0.2 |
| PyYAML | 6.0.3 |
| torch | **2.12.0+cpu** (`cuda.is_available() == False`) |
| torchvision | 0.27.0 |
| ultralytics | 8.4.60 |

As installed, YOLO therefore runs **on CPU**, and the detector logs its CPU warning at startup.

**Platform:** Windows-first. The code relies on the DirectShow/MSMF backends and pygrabber. On other operating systems, device naming falls back to index probing, and both capture backends it tries are Windows-only.

---

## 7. How to run

```powershell
cd "CatchThatBird"
.\.venv\Scripts\activate
python main.py --list-devices          # find the Pocket 3's index
python main.py --device 1              # run with a specific camera (or set camera.device_index)
python main.py --select-roi            # redraw the ROI around the car
```

Preview keys: `q`/`Esc` quit, `s` save a snapshot.

---

## 8. Observations, gaps, and risks

These come from reading the code. None was confirmed by running it against the camera.

1. **Nothing is persisted yet.** Detections go only to the console and the preview overlay. `events.jsonl`, snapshots, dedupe, the daily cap, and retention are all configured but not implemented (Phase 3/5). Running the app today produces no analyzable data.
2. **Detection cadence depends on the display loop, not the camera clock.** `Detector.process` is called only for frames the main loop actually picks up. `process_every_n_frames` and `motion_warmup_frames` therefore count *displayed* frames. If rendering or YOLO slows the loop below 30 fps, YOLO runs less often than 1 Hz, and MOG2 learns from a thinned frame sequence. The HUD's `skipped=` counter shows how many frames were missed.
3. **YOLO runs synchronously on the UI thread.** On the installed CPU-only torch, each inference stalls the preview (and MOG2 updates) for its duration once per second. A GPU wheel or a separate detection worker would remove the stall.
4. **Only the largest motion contour is examined.** Two birds far apart inside the ROI produce one YOLO crop, so the smaller or quieter one can be missed on that tick.
5. **The detection timestamp is taken after inference.** `captured_wall_time` is `time.time()` after YOLO returns, not when the frame was captured (`Frame.captured_at` is monotonic and has no wall-clock equivalent). The error is small at 1 Hz but is worth fixing before events are logged.
6. **No camera reconnect.** If the Pocket 3 disconnects or sleeps, `FrameGrabber._run` retries `read()` forever every 50 ms, logging every 30th failure, and never reopens the device. This is planned for Phase 5.
7. **Path anchoring is inconsistent.** `config.yaml` and `roi.json` resolve relative to the script's folder. `snap_NNN.jpg` and the future `logging.*` paths are relative to the **current working directory**, so running from another folder would scatter output.
8. **`torch` is imported directly but not listed** in `requirements.txt`. It arrives transitively through ultralytics. This works, but pinning it explicitly (with the CUDA note) would make the dependency visible.
9. **Redundant filtering:** `predict(conf=…, classes=…)` already filters by class and confidence, and the loop re-checks both. This is harmless.
10. **Hard-coded tuning:** MOG2 `history=500`, the 5×5 kernel, the 1.5 s overlay TTL, and the 5 s first-frame timeout are constants rather than config values.
11. **No version control, tests, or README.** The folder has a `.gitignore` but no `.git`. Pure functions such as `classify_device`, ROI load/validate, and the ROI→full-frame coordinate mapping in `Detector.process` would be easy to unit-test.
12. **The Python version has drifted** from the originally stated 3.11+ to a 3.14 venv. It works with the installed wheels; pinning it matters only if the environment is ever rebuilt.

---

## 9. Suggested next step (Phase 3 outline)

Add a `logger.py` (per the original plan) with an `EventLogger` built from `LoggingConfig` and `StorageConfig`:

- **Input:** `list[Detection]` + `Frame`.
- **Dedupe:** suppress a detection when its IoU with the last event's bbox is ≥ a threshold and it falls within `dedupe_within_seconds`, so a perching bird counts as one visit (optionally extend that visit's `last_seen`).
- **Write:** append one JSON line (ISO timestamp, confidence, bbox, frame seq, snapshot paths). Save the crop, and the full frame when `save_full_frame` is set, at `snapshot_quality`.
- **Guards:** apply the `max_events_per_day` cap and a startup sweep that deletes snapshots older than `retention_days`.
- **Hook point:** call it in `run_preview()` right after `detector.process(frame)`. Ideally move detection and logging off the UI thread at the same time (see §8.2–8.3).
