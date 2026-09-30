# events.jsonl schema

`logger.EventLogger` appends one JSON object per line to `logging.events_file` (default `data/events.jsonl`). Each line is one **visit**: one line for a bird that stays, hops or is seen only partly, however many detections it produces. A detection joins an open visit of the same class if its box has IoU ≥ `detection.visit_iou_threshold` (0.3) with the visit's latest box, or if its centre is within `detection.visit_center_distance` (2.0) × max(w, h) of that box's centre.

## Fields

| Field | Type | Meaning |
|---|---|---|
| `ts` | string | When the visit started. This is the capture time of the first detection's frame, as local ISO 8601 with milliseconds and offset: `2026-09-30T14:05:06.789+02:00`. For a video file (`--source`) it is media time: when the run started reading, plus the frame's position in the clip |
| `run_id` | string | 8 hex characters. The same id names the structured run log `logs/run_<run_id>.jsonl` |
| `frame_seq` | int | Capture sequence number of that first frame (starts at 1 each run) |
| `class` | string | Detected class name, from `detection.yolo_target_classes` (default `bird`) |
| `confidence` | float | Detector confidence of the first detection, rounded to 4 places |
| `bbox_xywh` | [int, int, int, int] | First detection's box in full-frame pixels: x, y, width, height |
| `snapshot_crop` | string or null | The padded crop YOLO saw, as a path relative to the snapshots directory's **parent**: `snapshots/<name>.jpg`. `null` if the write failed |
| `snapshot_full` | string or null | The full frame, same path convention. `null` when `logging.save_full_frame` is false or the write failed |
| `last_seen` | string | Capture time of the visit's last matching detection (same format as `ts`) |
| `visit_frames` | int | Number of gated frames with a matching detection, ≥ 1 |

Lines are written when a visit **closes**, so they appear in close order. Sort by `ts` for start order.

## Snapshot files

Names are `<local time>_seq<frame_seq, 6 digits>_d<detection index>_{crop,full}.jpg`, for example `20260930T140506.789_seq000091_d0_crop.jpg`. Snapshots taken with the preview's `s` key are `manual_<local time>_seq<N>.jpg` and are not referenced from `events.jsonl`.

Paths are stored relative to the snapshots directory's parent. With the default config, resolve them against the directory that holds `events.jsonl` (`data/`). This means a data directory can be moved or copied to another machine as a whole. Relative config paths themselves resolve against `--data-root` (default: the project root), never the current directory.

## When a visit is written

A visit closes when no matching detection has been seen for `detection.dedupe_within_seconds` (10 s by default, checked on every frame), or when the run ends. While it is open, YOLO re-checks its last box on every gated frame even if nothing moves, so a bird that sits still keeps confirming it (`last_seen` and `visit_frames` grow). At the end of a run, `EventLogger.close()` writes every visit still open. That happens on:

- a normal exit (end of a `--source` file, `q` / `Esc` in the preview)
- an error that ends the run (exit code 1)
- Ctrl-C / SIGINT, SIGTERM (POSIX), and Ctrl-Break / SIGBREAK (Windows)

A signal that arrives while a frame is being detected and persisted is held until that frame is done, so its visit is complete when it is written.

**Only a hard kill loses the open visits:** SIGKILL, a power loss, or on Windows `TerminateProcess`. That last one covers `taskkill /F`, Task Manager's End task, and `os.kill(pid, signal.SIGTERM)` from another process. What is lost is the visit in progress: at most the last `dedupe_within_seconds` after the bird was last seen, or longer for a bird that is still perching. Closed visits are already on disk. Each line is appended and the file closed right away.

## Limits

- `storage.max_events_per_day` (500): the first detection of a new visit beyond the cap is dropped. This is logged as `persist` skip `daily_cap` in the run log. Events already in the file for today count toward the cap.
- `storage.retention_days` (30): at startup, image files in the snapshots directory older than this are deleted. `events.jsonl` is never truncated, so old lines can point at deleted snapshots.
