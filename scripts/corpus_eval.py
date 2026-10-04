"""Measure the pipeline on the real-footage corpus against its ground truth.

    python scripts/corpus_eval.py                         # every clip, 3 paced runs each
    python scripts/corpus_eval.py bird_bath --runs 1
    python scripts/corpus_eval.py --json out.json --keep-logs docs/runs/corpus

Each run is `main.py --source <clip> --roi-file <clip dir>/roi.json --headless
--yes` with the default config.yaml, paced at the clip's frame rate like a
camera (no --no-pace), in a temporary data root, one run at a time. Its
events.jsonl is scored against data/samples/corpus/<name>/ground-truth.json
(written by hand, never by YOLO; scripts/fetch_clips.py) and its run log gives
the YOLO rate and the frames the loop skipped.

Scoring, per clip (docs/corpus-baseline.md says the same in prose):
- A visit's clip time is (frame_seq - 1) / fps to that plus last_seen - ts.
- A visit belongs to the bird whose stay contains the visit's start (one second
  of slack either side: the ground truth samples once a second) and whose box
  at that second (its `positions`, else its landing box) overlaps the visit's
  first box or has its centre within max(w, h) of the bird box's centre;
  the nearest such bird if there are several. A visit that belongs to no bird
  is false.
- real: birds in the ground truth, without the "uncertain" ones. A real bird
  with no visit is merged if a visit of another bird was open during its stay
  with its box's centre within visit_center_distance x max(w, h) of the bird
  (the logger's own reach), else missed. A bird with n > 1 visits adds n - 1
  to split. Visits on an uncertain bird are neither false nor counted for it.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:   # run as `python scripts/corpus_eval.py`
    sys.path.insert(0, str(ROOT))

from scripts.fetch_clips import CLIPS, CORPUS  # noqa: E402

Box = list  # [x, y, w, h]
METRICS = ("real", "logged", "missed", "split", "merged", "false")


def load_events(path: Path) -> list[dict[str, Any]]:
    """events.jsonl lines; lines from before the multi-bird fields load the same."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_truth(name: str) -> dict[str, Any]:
    return json.loads((CORPUS / name / "ground-truth.json").read_text(encoding="utf-8"))


def _centre(b: Box) -> tuple[float, float]:
    return b[0] + b[2] / 2, b[1] + b[3] / 2


def _dist(a: Box, b: Box) -> float:
    (ax, ay), (bx, by) = _centre(a), _centre(b)
    return math.hypot(ax - bx, ay - by)


def _overlaps(a: Box, b: Box) -> bool:
    return (min(a[0] + a[2], b[0] + b[2]) > max(a[0], b[0])
            and min(a[1] + a[3], b[1] + b[3]) > max(a[1], b[1]))


def bird_box(bird: dict[str, Any], t: float) -> Box:
    """Where the bird is at clip second t (clamped to its stay)."""
    k = min(max(int(math.floor(t)), bird["first_s"]), bird["last_s"])
    for seg in bird.get("positions", []):
        if seg["from_s"] <= k <= seg["to_s"]:
            return seg["box"]
    return bird["box_at_landing"]


def _visit_times(event: dict[str, Any], fps: float) -> tuple[float, float]:
    start = (event["frame_seq"] - 1) / fps
    stay = (datetime.fromisoformat(event["last_seen"]) - datetime.fromisoformat(event["ts"])).total_seconds()
    return start, start + stay


def evaluate(events: list[dict[str, Any]], truth: dict[str, Any],
             center_reach: float = 2.0) -> dict[str, Any]:
    fps = truth["fps"]
    birds = truth["birds"]
    visits = []
    for ev in events:
        start, end = _visit_times(ev, fps)
        box = ev["bbox_xywh"]
        best, best_d = None, None
        for bird in birds:
            if not bird["first_s"] - 1 <= start <= bird["last_s"] + 1:
                continue
            bb = bird_box(bird, start)
            d = _dist(box, bb)
            if not (_overlaps(box, bb) or d <= max(bb[2], bb[3])):
                continue
            if best_d is None or d < best_d:
                best, best_d = bird, d
        visits.append({"start": start, "end": end, "box": box, "bird": best,
                       "frames": ev.get("visit_frames", 1)})

    real = [b for b in birds if not b.get("uncertain")]
    missed = merged = split = 0
    delays: list[float] = []
    for bird in real:
        own = sorted((v for v in visits if v["bird"] is bird), key=lambda v: v["start"])
        if own:
            split += len(own) - 1
            delays.append(round(own[0]["start"] - bird["first_s"], 2))
            continue
        absorbed = any(
            v["bird"] is not None and v["start"] - 1 <= s <= v["end"] + 1
            and _dist(v["box"], bird_box(bird, s)) <= center_reach * max(v["box"][2], v["box"][3])
            for v in visits for s in range(bird["first_s"], bird["last_s"] + 1))
        if absorbed:
            merged += 1
        else:
            missed += 1
    on_uncertain = sum(1 for v in visits if v["bird"] is not None and v["bird"].get("uncertain"))
    return {
        "real": len(real), "logged": len(visits), "missed": missed, "split": split,
        "merged": merged, "false": sum(1 for v in visits if v["bird"] is None),
        "on_uncertain": on_uncertain,
        "visit_frames": [v["frames"] for v in sorted(visits, key=lambda v: v["start"])],
        "open_delay_s": delays,
    }


def yolo_rate(log_lines: list[dict[str, Any]], fps: float) -> dict[str, int]:
    """YOLO calls in the run, and the most in any one second of the clip."""
    per_second: dict[int, int] = {}
    calls = 0
    for line in log_lines:
        if line.get("stage") != "yolo_infer" or line.get("event") not in ("success", "fail"):
            continue
        calls += 1
        sec = int((line["frame_seq"] - 1) // fps)
        per_second[sec] = per_second.get(sec, 0) + 1
    return {"calls": calls, "max_per_s": max(per_second.values(), default=0)}


def frames_seen(log_lines: list[dict[str, Any]]) -> dict[str, int]:
    """Frames the main loop processed, and the producer frames it skipped
    (from the render summaries; the last one carries the run's total)."""
    summaries = [l["context"] for l in log_lines
                 if l.get("stage") == "render" and (l.get("context") or {}).get("summary")]
    return {"processed": sum(s["records"] for s in summaries),
            "skipped": summaries[-1]["skipped_total"] if summaries else 0}


def run_clip(name: str, workdir: Path, *, keep_log: Optional[Path] = None,
             config: Path = ROOT / "config.yaml", timeout: float = 600) -> dict[str, Any]:
    """One paced run of main.py on clip `name`, scored. Returns the metrics,
    the run_id, the YOLO rate and the frame counts."""
    clip_dir = CORPUS / name
    workdir.mkdir(parents=True, exist_ok=True)
    log_dir = workdir / "logs"
    cmd = [sys.executable, str(ROOT / "main.py"), "--source", str(clip_dir / "clip.mp4"),
           "--roi-file", str(clip_dir / "roi.json"), "--headless", "--yes",
           "--config", str(config), "--data-root", str(workdir), "--log-dir", str(log_dir)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"main.py failed on {name} (exit {proc.returncode}): {proc.stderr[-1500:]}")
    (log,) = log_dir.glob("run_*.jsonl")
    events_path = workdir / "data" / "events.jsonl"
    lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    result = score(name, lines, load_events(events_path))
    if keep_log is not None:
        keep_log.mkdir(parents=True, exist_ok=True)
        text = log.read_text(encoding="utf-8")
        for path, shown in ((str(ROOT), "."), (str(workdir), "<tmp>")):
            for variant in (path, path.replace("\\", "\\\\"), path.replace("\\", "/")):
                text = text.replace(variant, shown)
        (keep_log / log.name).write_text(text, encoding="utf-8")
        events_text = events_path.read_text(encoding="utf-8") if events_path.exists() else ""
        (keep_log / f"events_{result['run_id']}.jsonl").write_text(events_text, encoding="utf-8")
    return result


def score(name: str, log_lines: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    """Metrics of one run of clip `name` from its run log and its events."""
    truth = load_truth(name)
    result = evaluate(events, truth)
    rate = yolo_rate(log_lines, truth["fps"])
    frames = frames_seen(log_lines)
    result.update(run_id=log_lines[0]["run_id"], yolo_calls=rate["calls"],
                  max_yolo_per_s=rate["max_per_s"], frames_processed=frames["processed"],
                  frames_skipped=frames["skipped"])
    return result


def rescore(directory: Path) -> dict[str, list[dict[str, Any]]]:
    """Score again every kept run in `directory` (run_<id>.jsonl next to
    events_<id>.jsonl, as --keep-logs writes them), against the current
    ground truth. The clip is read from the run's --source argument."""
    results: dict[str, list[dict[str, Any]]] = {}
    for log in sorted(directory.glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime):
        lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
        argv = lines[0]["context"]["argv"]
        name = Path(argv[argv.index("--source") + 1]).parent.name
        events = load_events(directory / f"events_{lines[0]['run_id']}.jsonl")
        results.setdefault(name, []).append(score(name, lines, events))
    return results


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """min-max of each count over the runs, as shown in the doc's table."""
    out = {k: (min(r[k] for r in runs), max(r[k] for r in runs)) for k in
           METRICS + ("max_yolo_per_s", "frames_skipped")}
    out["visit_frames"] = [r["visit_frames"] for r in runs]
    out["run_ids"] = [r["run_id"] for r in runs]
    delays = [d for r in runs for d in r["open_delay_s"]]
    out["open_delay_median_s"] = round(statistics.median(delays), 2) if delays else None
    return out


def _span(pair: tuple[int, int]) -> str:
    lo, hi = pair
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


def print_table(results: dict[str, list[dict[str, Any]]]) -> None:
    print()
    print("| clip | visits logged / real | missed | split | merged | false | visit_frames | "
          "max YOLO calls/s | frames skipped | run_ids |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, runs in results.items():
        s = summarise(runs)
        frames = "; ".join(",".join(map(str, vf)) or "-" for vf in s["visit_frames"])
        print(f"| {name} | {_span(s['logged'])} / {s['real'][0]} | {_span(s['missed'])} | "
              f"{_span(s['split'])} | {_span(s['merged'])} | {_span(s['false'])} | {frames} | "
              f"{_span(s['max_yolo_per_s'])} | {_span(s['frames_skipped'])} | "
              f"{', '.join(s['run_ids'])} |")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Score the pipeline on the real-footage corpus")
    p.add_argument("names", nargs="*", help="clips (default: all)")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--json", type=Path, help="write every run's result here")
    p.add_argument("--keep-logs", type=Path, help="copy each run log here, paths scrubbed")
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--rescore", type=Path, metavar="DIR",
                   help="score the runs kept in DIR again (no new runs)")
    args = p.parse_args(argv)
    if args.rescore:
        results = rescore(args.rescore)
        print_table(results)
        if args.json:
            args.json.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
        return 0
    names = args.names or [c.name for c in CLIPS]
    missing = [n for n in names if not (CORPUS / n / "clip.mp4").exists()]
    if missing:
        print(f"error: not fetched: {missing} (python scripts/fetch_clips.py)", file=sys.stderr)
        return 1
    results: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        results[name] = []
        for i in range(args.runs):
            work = Path(tempfile.mkdtemp(prefix=f"ctb_{name}_"))
            try:
                r = run_clip(name, work, keep_log=args.keep_logs, config=args.config)
            finally:
                shutil.rmtree(work, ignore_errors=True)
            results[name].append(r)
            print(f"{name} run {i + 1}: {r['run_id']} " + " ".join(f"{k}={r[k]}" for k in METRICS)
                  + f" visit_frames={r['visit_frames']} max_yolo/s={r['max_yolo_per_s']}"
                  + f" skipped={r['frames_skipped']}", flush=True)
    print_table(results)
    if args.json:
        args.json.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
