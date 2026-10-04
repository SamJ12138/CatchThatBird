# Corpus baseline

How the pipeline does on eight real clips, scored against ground truth written by hand. The clips, their sources and licenses are in [data/samples/corpus/CREDITS.md](../data/samples/corpus/CREDITS.md); each clip's ground truth is `data/samples/corpus/<name>/ground-truth.json`, its region of interest `roi.json` next to it.

## How it was measured

- **Runs.** Each clip, three times: `python scripts/corpus_eval.py --runs 3 --keep-logs docs/runs/corpus`. One run is `main.py --source <clip> --roi-file <clip dir>/roi.json --headless --yes` with the default `config.yaml`, paced at the clip's frame rate like a camera (no `--no-pace`), one run at a time, on the i9-13900H laptop with CPU-only torch. Nothing was tuned for any clip. The run logs and the events of every run are in [runs/corpus/](runs/corpus/) (`run_<run_id>.jsonl`, `events_<run_id>.jsonl`); `python scripts/corpus_eval.py --rescore docs/runs/corpus` scores them again.
- **Machine load.** Other programs were running on the laptop during the runs (a local web API and another project's test harness, idle most of the time). The CPU load sampled every 5 s is given with each table. Frame skips depend on load.
- **The ground truth** was written from one frame per second (frame round(s x fps)), never from YOLO: for each bird, one entry per stay in the frame with its first and last second, the second and box at landing, coarse positions when it moves (or when the camera pans), and the number of birds in each second. A blurred shape that is probably a bird in flight is marked uncertain.
- **ROI.** One rectangle per clip around the surface of interest (the roof, the hood, the bath with its chains, the branch, the steps, the car), with room above it for a standing bird, drawn once by hand on the first frame; the ground fills the frame in `sparrows_ground`, so that clip uses the whole frame.

## How a run is scored

`scripts/corpus_eval.py` compares a run's `events.jsonl` with the clip's ground truth:

- **When and where a visit is.** From `(frame_seq - 1) / fps` into the clip to that plus `last_seen - ts`, at its first box (`bbox_xywh`).
- **Which bird a visit belongs to.** A bird whose stay contains the visit's start, with one second of slack either side (the ground truth samples once a second), and whose box at that second overlaps the visit's box or has its centre within max(w, h) of it. The nearest, if there are several. A visit that belongs to no bird is a **false visit**.
- **Missed, merged, split.** A bird with no visit of its own is **merged** if the visit of another bird was open during its stay with that visit's box centre within the logger's reach of it (2 x max(w, h) of the visit's box: the logger would have folded its detections into that visit), else **missed**. A bird with n visits adds n - 1 to **split visits**. An uncertain bird counts neither way.
- **visit_frames** of each visit, from `events.jsonl`.
- **Max YOLO calls/s** is the most `yolo_infer` calls in any one second of the clip, from the run log, and **frames skipped** the producer frames the paced loop never processed (the run log's last `render` summary).

Counts below are the range over the three runs (one number when all three agree).

## Baseline (before multi-bird tracking)

Code at `2b48b9a` (one motion crop per gated frame: the largest contour; each detection joins the open visit it overlaps most, or the nearest within reach). 24 runs on 2026-10-04, CPU load median 34 %, max 62 % (108 samples).

| clip | case | visits logged / real | missed | split | merged | false | visit_frames per visit (run 1; run 2; run 3) | max YOLO calls/s | frames skipped / in clip | run_ids |
|---|---|---|---|---|---|---|---|---|---|---|
| `car_mynas` | bird on a car, two birds | 1-2 / 2 | 0 | 0-1 | 1 | 0 | 3,1; 5; 5 | 5-6 | 129-141 / 355 | `25059cef`, `10957e06`, `48591085` |
| `car_gull_windshield` | bird on a car | 1 / 1 | 0 | 0 | 0 | 0 | 4; 2; 3 | 2-6 | 67-92 / 238 | `ef751f50`, `cf1e9b89`, `8f806be9` |
| `sparrows_ground` | two birds, arrivals | 3-5 / 2 | 0 | 1-3 | 0-1 | 0 | 12,6,1; 16,8,1; 12,3,1,1,2 | 3-7 | 422-448 / 1143 | `f446113a`, `81ad2e00`, `1a0dcfc8` |
| `bird_bath` | up to 3 birds, repeated arrivals | 3-5 / 6 | 0-1 | 0 | 1-2 | 0 | 4,16,16; 4,13,3,2; 3,10,17,1,2 | 6-9 | 241-273 / 1799 | `e57ccf13`, `002afddb`, `4c755188` |
| `doves_rain` | two birds, one leaves | 1 / 2 | 0-1 | 0 | 0-1 | 0 | 8; 5; 4 | 6 | 53-56 / 542 | `7c27922a`, `17a7ebb2`, `bf7cbf16` |
| `pigeon_stairs` | bird plus people, backlit | 1 / 2 | 0 | 0 | 1 | 0 | 3; 17; 2 | 6-7 | 70-79 / 753 | `c7450656`, `3d48f312`, `08e40cff` |
| `silhouette_dusk` | low light, backlit | 1 / 1 | 0 | 0 | 0 | 0 | 21; 23; 23 | 6-8 | 189-220 / 1268 | `f5c83ad5`, `e0febaf5`, `1d1b579e` |
| `parked_car_rain` | no bird | 0 / 0 | 0 | 0 | 0 | 0 | none | 8 | 7-11 / 262 | `61e6fc2d`, `05954f36`, `0e17ea58` |

What the numbers show:

- **No false visits, and no bird on a car missed.** Not one visit in 24 runs belongs to no bird, including the rainy car at dusk, and both car clips log their bird in every run.
- **Two birds close together are one visit.** The two mynas touch: one visit in all three runs (one run split it into two, both on the near bird). The second pigeon stands beside the first for two seconds and is folded into its visit every time; so are birds at the bath that perch near another's box. That is the limitation the README already states: a detection joins an open visit if its centre is within 2 x the visit's box size.
- **A bird that walks far is split.** The male sparrow walks across the ground; its visit loses it and a new one opens: 1-3 extra visits per run.
- **A bird that is there when the run starts is found late.** The motion gate learns the first two seconds as background (the warm-up), so a bird that sits still from the first frame is no motion until it moves. Visit opened, seconds after the bird's first second in the clip (median over the runs): mynas 4.9, pigeon 6.8, doves 9.1, silhouette 5.4. Of the two bath birds there from the first frame, one is logged at 11.2 s in every run (when it hops to the corner of the rim), the other never gets a visit of its own. A bird that arrives during the run is found within a second: the sparrows at 0.3-0.6 s.
- **The backlit pigeon is hard to keep.** Its one visit was confirmed on 3, 17 and 2 gated frames of the clip's 25 seconds: in two runs YOLO stopped finding it after the visit opened, and it was never logged again (`missed` counts a bird with no visit, not a visit that loses its bird).
- **YOLO rate.** At most 5-9 calls in one second. Over 5, the second holds a re-check window's calls (every third frame for 30 frames, about 10 per window) on top of a gated frame's.
- **Frames skipped.** The paced loop skips frames whenever a frame takes longer than the frame interval. At 1920x1080 with a large ROI or the whole frame (`sparrows_ground`: whole frame, 37-39 % skipped; `silhouette_dusk`: 50 fps, 15-17 %), MOG2 alone is near the frame interval and every YOLO call drops frames. The handheld clips skip more (mynas 36-40 %): the moving background is motion on every gated frame, which keeps re-check windows and large crops coming.

The slow tests in `tests/test_slow_corpus.py` run each clip once and fail if missed, split, merged or false visits are more than 1 above the most the reference runs show for that clip, or if any second has more than 12 YOLO calls (`data/samples/corpus/baseline.json`). The reference was this table until multi-bird tracking; it is now the table below.

## After multi-bird tracking

Code at `ebd40c4`: every motion region classified (up to 4), detections matched to open visits jointly (Hungarian method), visit ids, a budget of 5 YOLO calls per second for the crops gated frames start. Same clips, ROIs, config defaults and scoring; 24 runs on 2026-10-04, CPU load median 18 %, max 43 % (105 samples). Run logs and events in [runs/corpus-multi/](runs/corpus-multi/). `data/samples/corpus/baseline.json` now holds these runs as the slow tests' reference (and the runs above under `before`).

| clip | visits logged / real | missed | split | merged | false | visit_frames per visit (run 1; run 2; run 3) | max YOLO calls/s | frames skipped / in clip | run_ids |
|---|---|---|---|---|---|---|---|---|---|
| `car_mynas` | 2 / 2 | 0 | 0-1 | 0-1 | 0 | 7,5; 7,5; 7,4 | 6 | 106-113 / 355 | `6da0ba42`, `d504c0e2`, `08a203ed` |
| `car_gull_windshield` | 1 / 1 | 0 | 0 | 0 | 0 | 4; 4; 4 | 5 | 59-85 / 238 | `361e5a38`, `5120eb80`, `90db6853` |
| `sparrows_ground` | 4 / 2 | 0 | 2 | 0 | 0 | 19,4,9,1; 13,3,17,1; 12,5,3,14 | 5-6 | 393-409 / 1143 | `b6287290`, `70329a84`, `5207bd51` |
| `bird_bath` | 5-8 / 6 | 0 | 0-3 | 1 | 0 | 8,17,7,3,2,6,2; 10,4,15,5,8,2,7,2; 10,6,13,12,1 | 8 | 296-350 / 1799 | `653b4177`, `c6251709`, `c3186178` |
| `doves_rain` | 1-2 / 2 | 0 | 0 | 1 | 0-1 | 6; 11,1; 10 | 6-7 | 84-92 / 542 | `723fe8ea`, `abbcda9b`, `3c0c431c` |
| `pigeon_stairs` | 1-2 / 2 | 0 | 0 | 0-1 | 0 | 16,1; 16; 16,2 | 7 | 122-139 / 753 | `4d1fd59e`, `3dc6bde0`, `9fec1631` |
| `silhouette_dusk` | 1 / 1 | 0 | 0 | 0 | 0 | 25; 25; 25 | 7 | 263-276 / 1268 | `e9d03661`, `40f1a83d`, `0a6986fd` |
| `parked_car_rain` | 0 / 0 | 0 | 0 | 0 | 0 | none | 9 | 16 / 262 | `0b5e55dd`, `5d344d47`, `571c825e` |

### Before and after

Ranges over the three runs; before at `2b48b9a`, after at `ebd40c4`.

| clip | missed | split | merged | false | first visit opens, s after the bird is first in frame | frames skipped |
|---|---|---|---|---|---|---|
| `car_mynas` | 0 → 0 | 0-1 → 0-1 | 1 → 0-1 | 0 → 0 | 4.6-5.0 → 2.1-2.6 | 129-141 → 106-113 |
| `car_gull_windshield` | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 2.3-5.0 → 2.1-2.2 | 67-92 → 59-85 |
| `sparrows_ground` | 0 → 0 | 1-3 → 2 | 0-1 → 0 | 0 → 0 | 0.3-0.6 → 0.0-0.4 | 422-448 → 393-409 |
| `bird_bath` | 0-1 → 0 | **0 → 0-3** | 1-2 → 1 | 0 → 0 | 11.2 → 2.0 | **241-273 → 296-350** |
| `doves_rain` | 0-1 → 0 | 0 → 0 | 0-1 → 1 | **0 → 0-1** | 7.9-9.1 → 2.6-4.7 | **53-56 → 84-92** |
| `pigeon_stairs` | 0 → 0 | 0 → 0 | 1 → 0-1 | 0 → 0 | 6.8 → 3.7-6.1 | **70-79 → 122-139** |
| `silhouette_dusk` | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 5.2-6.9 → 3.9 | **189-220 → 263-276** |
| `parked_car_rain` | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | no bird | 7-11 → 16 |

What changed:

- **Totals over each set of 24 runs:** missed 4 → 0, merged 12 → 9, split 7 → 13, false 0 → 1.
- **Better.** No bird missed in any run. Two birds close together are now two visits more often: the mynas in all three runs (they touch; before, one visit every time), the second pigeon in two of three. A bird that is there from the first frame is found sooner: a median 3.3 s after its first frame instead of 6.1 s over the six clips with one (2.0-6.1 s, was 2.3-11.2 s; table above): with every region classified, its small movements are enough. The bath's pale bird, which never had a visit of its own before, is logged at 2.0 s in every run.
- **Worse, and not tuned away** (`docs/observations.md`, "Multi-bird findings", M1-M3):
  - **`bird_bath` splits more: 0 → 0-3.** The out-of-focus goldfinch at the top of the chain is sometimes boxed by YOLO as two boxes side by side. They do not overlap, so they are two birds to the association, and two visits; before, the centre rule folded them into one (M1).
  - **`doves_rain` merges in every run: 0-1 → 1.** Before, the large dove's visit opened at 8.9-10.1 s, after the small dove had flown off at 8 s, and the small dove was merged (one run) or missed (two). Now the small dove's visit opens at 2.6-4.7 s; when it flies off, the large dove's detections join that visit (its centre is 322 px from the small dove's last box, inside the reach of 416 px), so the large dove never has a visit of its own. A visit can pass from one bird to a neighbour that stays (M1).
  - **`doves_rain` has a false visit in one run** (`abbcda9b`, a 0.57 box on a branch at 10.7 s). With every region classified, YOLO sees crops it never saw before, and a branch scored above the threshold once (M2).
  - **More frames skipped on five clips**, by up to 8 points of the clip's frames (`pigeon_stairs` 9-10 % → 16-18 %): more YOLO calls per gated frame (M3, below).
- **Unchanged.** `sparrows_ground` still splits the walking male (2 extra visits per run): it walks more than the visit's reach between gated frames. No false visit on the rainy car.
- **A caveat on scoring.** A visit is scored by its first box. In two `car_mynas` runs the first visit opened on a box of the far bird's lower body and later carried the near bird; both visits then count for the far bird (split 1, merged 1), although each bird had a visit.

### YOLO budget and frame drops

- **The budget binds on busy frames.** It deferred 63 crops over the 24 runs (bath 39, sparrows 9, mynas 7, pigeon 5, doves 3); each ran a few frames later, as the budget refilled, and none expired. The most YOLO calls on one frame was 5 (6 on the bath, with a re-check). Before, it was 2-4.
- **Every YOLO call drops frames here.** A call took 57.6 ms (p50, 1661 calls; p95 73.2 ms) and MOG2 20.6 ms per frame (median of the window p50s) on these 1080p regions, against a 33 ms frame interval. A frame with one call is late by about a frame; one with 5 calls (about 290 ms) skips about 8. The loop skipped 3-40 % of a clip's frames before the change and 6-36 % after.
- **No default was changed.** A lower budget would shorten the bursts but defer more crops; no budget of 1 or more keeps this loop from skipping frames while a call takes longer than a frame interval. Reported in observations M3.
