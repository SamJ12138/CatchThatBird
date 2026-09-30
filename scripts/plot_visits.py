"""Plot bird visits per hour of day from events.jsonl.

    python scripts/plot_visits.py                                   # data/events.jsonl -> docs/visits.png
    python scripts/plot_visits.py path/to/events.jsonl --out visits.png --title "Week 1"

Each events.jsonl line is one visit. It is counted in the hour of its `ts`
(the visit start), read in the timestamp's own UTC offset, i.e. the local
clock of the machine that recorded it. All days are summed into 24 bars.
Lines that are not valid events are skipped and counted on stderr.
matplotlib is a dev dependency (requirements-dev.txt).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def load_events(path: Path) -> tuple[list[dict[str, Any]], int]:
    """(events with a parseable `ts`, number of lines skipped)."""
    events: list[dict[str, Any]] = []
    bad = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            datetime.fromisoformat(event["ts"])
        except (ValueError, KeyError, TypeError):
            bad += 1
            continue
        events.append(event)
    return events, bad


def visits_per_hour(events: list[dict[str, Any]]) -> list[int]:
    counts = [0] * 24
    for event in events:
        counts[datetime.fromisoformat(event["ts"]).hour] += 1
    return counts


def span(events: list[dict[str, Any]]) -> str:
    days = sorted({datetime.fromisoformat(e["ts"]).date() for e in events})
    return f"{days[0]}" if len(days) == 1 else f"{days[0]} to {days[-1]} ({len(days)} days with visits)"


def plot(counts: list[int], out: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=120)
    ax.bar(range(24), counts, width=0.8, color="#4a6f8a")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], fontsize=8)
    ax.set_xlim(-0.6, 23.6)
    ax.set_xlabel("hour of day (local time of the recording)")
    ax.set_ylabel("visits")
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Plot bird visits per hour of day")
    p.add_argument("events", nargs="?", type=Path, default=ROOT / "data" / "events.jsonl",
                   help="events.jsonl (default: data/events.jsonl)")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "visits.png",
                   help="PNG to write (default: docs/visits.png)")
    p.add_argument("--title", default=None, help="plot title (default: date range and count)")
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
    counts = visits_per_hour(events)
    title = args.title or f"Bird visits per hour of day, {span(events)}"
    plot(counts, args.out, f"{title}: {len(events)} visits")
    print(f"{len(events)} visits, {span(events)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
