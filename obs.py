"""Structured run log: one JSON object per line in logs/run_<run_id>.jsonl.

Additive to the loguru console output. One ObsLogger per run, created in
main() and passed to FrameGrabber / Detector through their constructors.

Line schema (every field present on every line, null when not applicable):
    ts, run_id, stage, event, duration_ms, frame_seq,
    error_type, error_message, context

event:      start | success | fail | skip
error_type: input_invalid | external_api | parse | timeout | hardware | unknown

A stage that swallows a sub-error and carries on (e.g. DSHOW fails, MSMF
works) emits a `fail` line for the sub-error followed by its normal terminal
line. Per-frame stages log through a WindowCounter instead: individual lines
only for fail/abnormal skip, plus one summary line (context.summary=true)
every `every` records.
"""
from __future__ import annotations

import json
import statistics
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

EVENTS = ("start", "success", "fail", "skip")
ERROR_TYPES = (
    "input_invalid", "external_api", "parse", "timeout", "hardware", "unknown",
)


def new_run_id() -> str:
    return uuid.uuid4().hex[:8]


def describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _percentile(sorted_vals: list[float], q: float) -> float:
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    return statistics.quantiles(sorted_vals, n=100, method="inclusive")[int(q) - 1]


class ObsLogger:
    def __init__(self, run_id: str, log_dir: Path) -> None:
        self.run_id = run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"run_{run_id}.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")
        # Grabber thread and main thread both write; one lock keeps lines whole.
        self._lock = threading.Lock()
        self._counters: list[WindowCounter] = []

    def emit(
        self,
        stage: str,
        event: str,
        *,
        duration_ms: Optional[float] = None,
        frame_seq: Optional[int] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        if event not in EVENTS:
            raise ValueError(f"bad event {event!r}")
        if error_type is not None and error_type not in ERROR_TYPES:
            raise ValueError(f"bad error_type {error_type!r}")
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "stage": stage,
            "event": event,
            "duration_ms": None if duration_ms is None else round(duration_ms, 3),
            "frame_seq": frame_seq,
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
        }
        line = json.dumps(record, default=str)
        with self._lock:
            if self._fh.closed:
                return
            self._fh.write(line + "\n")
            self._fh.flush()

    def span(
        self,
        stage: str,
        *,
        frame_seq: Optional[int] = None,
        context: Optional[dict[str, Any]] = None,
        error_type: str = "unknown",
    ) -> "Span":
        """Emit `start` now; the returned Span emits the terminal line.
        `error_type` classifies an exception escaping a `with` block."""
        return Span(self, stage, frame_seq, context, error_type)

    def counter(
        self,
        stage: str,
        every: int = 300,
        context_fn: Optional[Callable[[], dict[str, Any]]] = None,
    ) -> "WindowCounter":
        c = WindowCounter(self, stage, every, context_fn)
        self._counters.append(c)
        return c

    def close(self) -> None:
        for c in self._counters:
            c.flush()
        with self._lock:
            self._fh.close()


class Span:
    """start -> exactly one of success / fail / skip.

    Call success()/fail()/skip() explicitly (e.g. before an early return), or
    let the `with` block end: normal exit -> success, exception -> fail
    (re-raised, never swallowed), KeyboardInterrupt -> skip, SystemExit(0)
    -> success."""

    def __init__(
        self,
        obs: ObsLogger,
        stage: str,
        frame_seq: Optional[int],
        context: Optional[dict[str, Any]],
        error_type: str,
    ) -> None:
        self._obs = obs
        self.stage = stage
        self.frame_seq = frame_seq
        self.context: dict[str, Any] = dict(context or {})
        self._error_type = error_type
        self._t0 = time.perf_counter()
        self.ended = False
        obs.emit(stage, "start", frame_seq=frame_seq, context=self.context)

    def _end(
        self,
        event: str,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        if self.ended:
            return
        self.ended = True
        if context:
            self.context.update(context)
        self._obs.emit(
            self.stage,
            event,
            duration_ms=(time.perf_counter() - self._t0) * 1000.0,
            frame_seq=self.frame_seq,
            error_type=error_type,
            error_message=error_message,
            context=self.context,
        )

    def success(self, context: Optional[dict[str, Any]] = None) -> None:
        self._end("success", context=context)

    def skip(
        self,
        reason: str,
        *,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        self._end("skip", error_type, error_message, {"reason": reason, **(context or {})})

    def fail(
        self,
        error_type: str,
        error_message: str,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        self._end("fail", error_type, error_message, context)

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type, exc, _tb) -> bool:
        if exc is None:
            self.success()
        elif isinstance(exc, SystemExit):
            if exc.code in (0, None):
                self.success({"exit_code": 0})
            else:
                self.fail("unknown", f"SystemExit({exc.code})")
        elif isinstance(exc, KeyboardInterrupt):
            self.skip("keyboard_interrupt")
        else:
            self.fail(self._error_type, describe(exc))
        return False


class WindowCounter:
    """Aggregates a high-rate stage into one summary line per `every` records."""

    def __init__(
        self,
        obs: ObsLogger,
        stage: str,
        every: int,
        context_fn: Optional[Callable[[], dict[str, Any]]],
    ) -> None:
        self._obs = obs
        self._stage = stage
        self._every = every
        self._context_fn = context_fn
        self._lock = threading.Lock()
        self._reset()

    def _reset(self) -> None:
        self._n = 0
        self._outcomes: Counter[str] = Counter()
        self._reasons: Counter[str] = Counter()
        self._durations: list[float] = []
        self._first_seq: Optional[int] = None
        self._last_seq: Optional[int] = None

    def record(
        self,
        outcome: str,
        duration_ms: Optional[float] = None,
        frame_seq: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._n += 1
            self._outcomes[outcome] += 1
            if reason is not None:
                self._reasons[reason] += 1
            if duration_ms is not None:
                self._durations.append(duration_ms)
            if frame_seq is not None:
                if self._first_seq is None:
                    self._first_seq = frame_seq
                self._last_seq = frame_seq
            full = self._n >= self._every
        if full:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            if self._n == 0:
                return
            durs = sorted(self._durations)
            ctx: dict[str, Any] = {
                "summary": True,
                "records": self._n,
                "success": self._outcomes["success"],
                "fail": self._outcomes["fail"],
                "skip": self._outcomes["skip"],
                "seq_range": [self._first_seq, self._last_seq],
            }
            if self._reasons:
                ctx["reasons"] = dict(self._reasons)
            if durs:
                ctx["p50_ms"] = round(_percentile(durs, 50), 3)
                ctx["p95_ms"] = round(_percentile(durs, 95), 3)
                ctx["max_ms"] = round(durs[-1], 3)
            if self._context_fn is not None:
                ctx.update(self._context_fn())
            mean = statistics.fmean(durs) if durs else None
            last_seq = self._last_seq
            self._reset()
        self._obs.emit(
            self._stage, "success", duration_ms=mean, frame_seq=last_seq, context=ctx
        )
