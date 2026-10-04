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

The slow tests in `tests/test_slow_corpus.py` run each clip once and fail if missed, split, merged or false visits are more than 1 above the most this table shows for that clip, or if any second has more than 12 YOLO calls (`data/samples/corpus/baseline.json`).
