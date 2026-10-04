"""Summarise bird visits from events.jsonl.

    python scripts/visit_report.py                       # data/events.jsonl
    python scripts/visit_report.py path/to/events.jsonl

Prints visits per hour of day (the local clock in each `ts`, all days
summed, as scripts/plot_visits.py counts them), the median visit length
(last_seen - ts), the most birds there at once and the busiest hour.
"Most birds at once" is the most visits whose [ts, last_seen] overlap; lines
written since visit ids exist also carry concurrent_max, the most visits
open at once while each was (open visits include ones waiting out the
dedupe window), and its maximum is printed next to it. Lines without those
fields load the same. Stdlib only.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:   # run as `python scripts/visit_report.py`
    sys.path.insert(0, str(ROOT))

from scripts.plot_visits import load_events, span, visits_per_hour  # noqa: E402


def _length_s(event: dict[str, Any]) -> float:
    return (datetime.fromisoformat(event["last_seen"]) - datetime.fromisoformat(event["ts"])).total_seconds()


def max_overlap(events: list[dict[str, Any]]) -> int:
    """Most visits whose [ts, last_seen] intervals overlap at one moment."""
    points = []
    for e in events:
        points.append((datetime.fromisoformat(e["ts"]).timestamp(), 1))
        points.append((datetime.fromisoformat(e["last_seen"]).timestamp(), -1))
    best = now = 0
    for _, step in sorted(points, key=lambda p: (p[0], -p[1])):   # a start before an end at one instant
        now += step
        best = max(best, now)
    return best


def summarise(events: list[dict[str, Any]]) -> dict[str, Any]:
    per_hour = visits_per_hour(events)
    busiest = max(range(24), key=lambda h: (per_hour[h], -h))
    fields = [e["concurrent_max"] for e in events if isinstance(e.get("concurrent_max"), int)]
    return {
        "per_hour": per_hour,
        "median_length_s": round(statistics.median(_length_s(e) for e in events), 1),
        "max_overlap": max_overlap(events),
        "max_concurrent_field": max(fields) if fields else None,
        "busiest_hour": (busiest, per_hour[busiest]),
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Summarise bird visits from events.jsonl")
    p.add_argument("events", nargs="?", type=Path, default=ROOT / "data" / "events.jsonl",
                   help="events.jsonl (default: data/events.jsonl)")
    args = p.parse_args(argv)
    if not args.events.exists():
        print(f"No visits: {args.events} does not exist")
        return 1
    events, bad = load_events(args.events)
    if bad:
        print(f"warning: skipped {bad} malformed line(s) in {args.events}", file=sys.stderr)
    if not events:
        print(f"No visits in {args.events}")
        return 1
    s = summarise(events)
    print(f"{len(events)} visits, {span(events)}")
    print()
    print("Visits per hour (local time of ts, all days):")
    print("  hour  visits")
    for hour, n in enumerate(s["per_hour"]):
        print(f"  {hour:02d}:00 {n:3d}")
    print()
    print(f"Median visit length: {s['median_length_s']:.1f} s")
    field = s["max_concurrent_field"]
    print(f"Max concurrent birds: {s['max_overlap']} (visits overlapping in time)"
          + (f"; concurrent_max on the lines: {field}" if field is not None else ""))
    hour, n = s["busiest_hour"]
    print(f"Busiest hour: {hour:02d}:00-{(hour + 1) % 24:02d}:00 ({n} visits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
