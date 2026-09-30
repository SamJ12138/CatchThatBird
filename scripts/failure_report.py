"""Summarise CatchThatBird run logs (logs/run_<run_id>.jsonl). Stdlib only.

    python scripts/failure_report.py                 # every logs/*.jsonl
    python scripts/failure_report.py --latest        # newest log only
    python scripts/failure_report.py logs/run_ab12cd34.jsonl [more.jsonl ...]

Prints:
  1. stage x error_type -> count (every fail/skip line that carries an error_type)
  2. stage x skip reason -> count
  3. per stage: success / fail / skip totals and p50 / p95 duration_ms

Per-frame stages (capture_read, mog2_apply, gate_check, render) are logged as
300-frame summary lines (context.summary = true). For those stages the totals
come from the summaries, and p50/p95 are approximate: the median of the
window p50s and the max of the window p95s (marked "~").
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ERROR_TYPES = ["input_invalid", "external_api", "parse", "timeout", "hardware", "unknown"]
TERMINAL = ("success", "fail", "skip")


def pct(vals: list[float], q: int) -> float:
    vals = sorted(vals)
    if len(vals) == 1:
        return vals[0]
    return statistics.quantiles(vals, n=100, method="inclusive")[q - 1]


def fmt_ms(v: float | None, approx: bool = False) -> str:
    if v is None:
        return "-"
    return f"{'~' if approx else ''}{v:.2f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    line = "  ".join
    out = [line(h.ljust(w) if i == 0 else h.rjust(w) for i, (h, w) in enumerate(zip(headers, widths)))]
    out.append(line("-" * w for w in widths))
    for r in rows:
        out.append(line(str(c).ljust(w) if i == 0 else str(c).rjust(w)
                        for i, (c, w) in enumerate(zip(r, widths))))
    return "\n".join(out)


def load(paths: list[Path]) -> tuple[list[dict], int]:
    records: list[dict] = []
    bad = 0
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for n, raw in enumerate(f, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    records.append(json.loads(raw))
                except json.JSONDecodeError:
                    bad += 1
                    print(f"warning: {path.name}:{n} is not valid JSON, skipped", file=sys.stderr)
    return records, bad


def main() -> int:
    p = argparse.ArgumentParser(description="Failure report for CatchThatBird run logs")
    p.add_argument("paths", nargs="*", type=Path, help="log files (default: logs/*.jsonl)")
    p.add_argument("--latest", action="store_true", help="only the newest logs/*.jsonl")
    args = p.parse_args()

    paths = args.paths or sorted((ROOT / "logs").glob("*.jsonl"), key=lambda q: q.stat().st_mtime)
    if args.latest and paths:
        paths = paths[-1:]
    if not paths:
        print("No log files found.")
        return 1

    records, bad = load(paths)
    runs = sorted({r.get("run_id") for r in records})
    print(f"Files: {', '.join(q.name for q in paths)}")
    print(f"Runs: {len(runs)}  Lines: {len(records)}  Unparseable lines: {bad}")
    for run_id in runs:
        lines = [r for r in records if r.get("run_id") == run_id]
        end = next((r for r in reversed(lines) if r["stage"] == "run" and r["event"] in TERMINAL), None)
        verdict = "no terminal run line (crashed or still running)"
        if end is not None:
            verdict = (f"run {end['event']}, exit_code={end['context'].get('exit_code')}, "
                       f"{end['duration_ms'] / 1000:.1f}s")
        print(f"  {run_id}: {lines[0]['ts']} .. {lines[-1]['ts']}  {verdict}")
    print()

    stage_order: list[str] = []
    errors: dict[str, Counter] = defaultdict(Counter)
    reasons: dict[str, Counter] = defaultdict(Counter)
    events: dict[str, Counter] = defaultdict(Counter)          # individual lines
    summary_totals: dict[str, Counter] = defaultdict(Counter)  # from summary lines
    durations: dict[str, list[float]] = defaultdict(list)
    win_p50: dict[str, list[float]] = defaultdict(list)
    win_p95: dict[str, list[float]] = defaultdict(list)

    for r in records:
        stage, event, ctx = r["stage"], r["event"], r.get("context") or {}
        if stage not in stage_order:
            stage_order.append(stage)
        if ctx.get("summary"):
            for k in TERMINAL:
                summary_totals[stage][k] += ctx.get(k, 0)
            for reason, n in (ctx.get("reasons") or {}).items():
                reasons[stage][reason] += n
            if "p50_ms" in ctx:
                win_p50[stage].append(ctx["p50_ms"])
                win_p95[stage].append(ctx["p95_ms"])
            continue
        events[stage][event] += 1
        if r.get("error_type"):
            errors[stage][r["error_type"]] += 1
        if event == "skip" and ctx.get("reason"):
            reasons[stage][ctx["reason"]] += 1
        if event in TERMINAL and r.get("duration_ms") is not None:
            durations[stage].append(r["duration_ms"])

    # 1. stage x error_type
    print("== Errors: stage x error_type (fail/skip lines carrying an error_type) ==")
    err_stages = [s for s in stage_order if errors[s]]
    if err_stages:
        rows = [[s, *[errors[s][e] or "." for e in ERROR_TYPES], sum(errors[s].values())]
                for s in err_stages]
        rows.append(["TOTAL", *[sum(errors[s][e] for s in err_stages) or "." for e in ERROR_TYPES],
                     sum(sum(errors[s].values()) for s in err_stages)])
        print(table(["stage", *ERROR_TYPES, "total"], rows))
    else:
        print("(none)")
    print()

    # 2. skip reasons
    print("== Skips: stage x reason ==")
    reason_rows = [[s, reason, n] for s in stage_order for reason, n in reasons[s].most_common()]
    print(table(["stage", "reason", "count"], reason_rows) if reason_rows else "(none)")
    print()

    # 3. per-stage totals + latency
    print("== Stages: totals and duration_ms ==")
    rows = []
    grand = Counter()
    for s in stage_order:
        if summary_totals[s]:
            tot = summary_totals[s]
            name = f"{s} *"
            p50 = statistics.median(win_p50[s]) if win_p50[s] else None
            p95 = max(win_p95[s]) if win_p95[s] else None
            approx = True
        else:
            tot = Counter({k: events[s][k] for k in TERMINAL})
            name = s
            d = durations[s]
            p50 = pct(d, 50) if d else None
            p95 = pct(d, 95) if d else None
            approx = False
        grand.update(tot)
        rows.append([name, tot["success"], tot["fail"], tot["skip"],
                     fmt_ms(p50, approx), fmt_ms(p95, approx)])
    rows.append(["TOTAL", grand["success"], grand["fail"], grand["skip"], "", ""])
    print(table(["stage", "success", "fail", "skip", "p50_ms", "p95_ms"], rows))
    print("\n* per-frame stage: totals from 300-frame summary lines; ~ = approximate "
          "percentile (median of window p50s / max of window p95s).")
    print("  A stage that swallows a sub-error logs a fail line and then its normal "
          "terminal line, so fail + success can exceed the number of stage runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
