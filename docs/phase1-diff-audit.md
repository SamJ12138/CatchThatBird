# Phase 1 diff audit (`git diff 4cf95da 7633218`)

Scope: `camera.py`, `detector.py`, `main.py`. There are 30 hunks: 9 in camera.py, 5 in detector.py, and 16 in main.py.

The allowed categories are:
- **(a)** an obs.py import
- **(b)** an obs log call (including spans, counters, and timing done only to feed them)
- **(c)** run_id plumbing (the `obs` object and parameters that carry it)
- **(d)** the `--source` / `--headless` additions

Most hunks mix (b)/(c) with re-indentation. Re-indentation happens where a block moved under a `with obs.span(...)`; `git diff -w` shows those blocks are otherwise unchanged. Everything below is **outside (a)–(d)**, or is (d) code that could also affect the camera path.

## Hunks outside the allowed categories

| # | File / hunk | What changed | Behaviour differs? |
|---|---|---|---|
| 1 | main.py `@@ -174,11 +188,32 @@` (`load_or_select_roi`) | **Structural refactor.** The function is split into a thin wrapper that opens the `roi_load` span and a private `_load_or_select_roi(…, sp)` that holds the old body. This avoided re-indenting the whole body under the span | No. Same arguments in, same return values, same file writes, same console lines |
| 2 | main.py `@@ -406,11 +523,27 @@` and `@@ -419,24 +552,35 @@` (`main`) | **Structural refactor.** `main()` now builds the `ObsLogger`, opens the `run` span, and calls `_main(args, obs, run)`, which holds the old body. `obs.close()` sits in a `finally` | **Yes, slightly, at startup.** (1) `logs/run_<id>.jsonl` is created on every invocation, including `--list-devices` and a missing-config exit. If `logs/` cannot be created, the program now fails where it used to work. (2) A new console line: `Run <id>: structured log -> <path>`. Exit codes, the config/device/checklist order, and exception propagation are all unchanged |
| 3 | main.py `@@ -419,24 +552,35 @@` (`if args.source is None or args.list_devices:`) | **Control-flow restructure** to skip device listing/checklist for `--source` | No for the camera path: with `--source` absent the order is identical. `--list-devices` with `--source` still lists and exits 0 |
| 4 | camera.py `@@ -91,22 +111,62 @@` (`_open_capture`) | `return self._configure(cap)` became `cap = self._configure(cap); sp.success(...); return cap` | No |
| 5 | camera.py `@@ -136,22 +201,63 @@` (`_run`) | The whole loop body is wrapped in `try: … except Exception: emit; raise` | No. The exception still propagates and the thread still dies with the same traceback from `threading.excepthook` |
| 6 | camera.py `@@ -136,22 +201,63 @@` (`_run`, pacing) | **(d), but not just plumbing:** file frames are paced to the file's fps with `time.sleep` | No for the camera path, which never enters the `if self._source is not None` branches. **Kept:** it is required for `--source` to deliver frames the way a camera does. Without it the reader decodes several hundred fps into the single latest-frame slot, and the consumer drops most frames |
| 7 | main.py `@@ -268,17 +335,35 @@` (loop head) | **(d):** `grabber.finished` end-of-file exit, and `time.sleep` in place of `cv2.waitKey` when headless | No for the camera path: `finished` is never set for a camera, and `headless` defaults to False |
| 8 | main.py `@@ -366,14 +456,28 @@` (`s` key) | `cv2.imwrite(...)` became `written = cv2.imwrite(...)` feeding a span | No. The "Saved" line is still printed unconditionally (obs #12) |
| 9 | main.py `@@ -71,14 +73,20 @@` (`wait_for_enter`) | `except (EOFError, KeyboardInterrupt):` became `... as e:` with an `isinstance` branch that only picks which obs line to write | No. Both still `sys.exit(0)` |
| 10 | detector.py `@@ -119,86 +141,145 @@` (`process`) | `seq = frame.seq`, counters, and `dropped_class` / `dropped_conf` tallies added inside the existing filter loop | No. The same boxes are kept/dropped and the same `Detection`s returned. The extra work is O(boxes) |
| 11 | camera.py / detector.py / main.py, per-frame obs | Per-frame counter updates take a `threading.Lock`; summaries write and flush a file line every 300 frames | Timing only (well under 0.1 ms per frame, measured indirectly: `capture_read` p50 was 1.8 ms including decode). No logic change |

## Verdict

Nothing needs reverting. The only behaviour change outside (a)–(d) is hunk 2: the log file is created at startup, which can fail if `logs/` is not writable. That change is inherent to the logging requirement and is kept. Hunk 6 (pacing) is the one (d) addition with real logic, and it is kept because `--source` needs it to behave like the camera.

## Batch 0 addition: `--no-pace`

`--no-pace` (file sources only) turns pacing off. The grabber then reads **as fast as the pipeline consumes**: after publishing a frame, it waits for the main loop's `grabber.ack(seq)` before reading the next one.

A purely unpaced reader (no hand-off) would be faster still. But with a single latest-frame slot it drops most frames, and which ones it drops varies from run to run, so tests could not rely on a frame's `seq`. With the hand-off, every frame is processed in order and `frame_count == seq` inside the detector. The default (paced) and camera paths are unchanged: for them `ack()` only records the seq.
