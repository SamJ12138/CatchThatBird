"""Phase 3: persist bird visits to events.jsonl + snapshots.

One JSON line per *visit*. A visit opens on a detection that matches no open
visit, and absorbs later detections of the same class within
`dedupe_within_seconds` of its last sighting whose box has IoU >= 0.3 with the
visit's most recent box, or whose centre is within 2 x max(w, h) of that box's
centre (both configurable; the centre rule keeps a hop or a truncated box in
the same visit). The detections of one frame are matched to the visits open
before it jointly, one to one, at the lowest total cost of (1 - IoU) plus the
centre distance over that reach (Hungarian method), so two birds near each
other keep their own visits; a box mostly inside another box that joined or
opened a visit on the same frame is the same bird (a duplicate). Each visit
gets a visit_id (<run_id>-<n>) and concurrent_max, the most visits open at
once while it was. While a visit is open, main.py passes open_tracks() to the
Detector, which runs YOLO on each open visit's last box every gated frame, so
a bird that sits still keeps confirming its visit. Open visits live in memory and are written when they expire (checked
on every handled frame) or when the logger closes, so `last_seen` and
`visit_frames` are final when the line is written. Lines therefore appear in
visit-close order; sort by `ts` to get visit-open order.

Snapshots (the padded predictor crop, plus the full frame if enabled) are
written when the visit opens. On startup, image files in `snapshots_dir`
older than `retention_days` are deleted; events.jsonl is never truncated.
Relative config paths resolve against `root` (main.py's --data-root, default
the project root), never the CWD. Snapshot paths in events.jsonl are stored
relative to snapshots_dir's parent ("snapshots/<name>.jpg"), so a data
directory can be moved as a whole.

Open visits are written by close(), which main.py reaches on a normal exit,
an error, Ctrl-C / SIGINT, SIGTERM and Ctrl-Break. A visit that reaches
`max_visit_seconds` is written with "truncated": true and a new one opens on
the next confirmation. Open visits are checkpointed to open_visits.json next
to events.jsonl whenever one opens or closes and every `checkpoint_seconds`
(60) of capture time; the next start writes them with "recovered": true. So a
hard kill (SIGKILL, TerminateProcess, power loss) loses at most the last 60 s
of an open visit. Schema: docs/events-schema.md.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
from loguru import logger

from camera import Frame
from config_schema import PROJECT_ROOT, LoggingConfig, StorageConfig, resolve_path
from detector import Detection, iou
from obs import ObsLogger, describe

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
BBox = tuple[int, int, int, int]


def center_distance(a: BBox, b: BBox) -> float:
    return math.hypot((a[0] + a[2] / 2) - (b[0] + b[2] / 2), (a[1] + a[3] / 2) - (b[1] + b[3] / 2))


def _local(t: float) -> datetime:
    return datetime.fromtimestamp(t).astimezone()


def iso(t: float) -> str:
    return _local(t).isoformat(timespec="milliseconds")


def stamp(t: float) -> str:
    """File-name timestamp, local time: 20260930T140506.789"""
    dt = _local(t)
    return dt.strftime("%Y%m%dT%H%M%S") + f".{dt.microsecond // 1000:03d}"


@dataclass
class _Visit:
    event: dict[str, Any]
    class_name: str
    last_bbox: BBox
    last_seen: float
    last_seq: int
    visit_frames: int = 1
    started: float = 0.0  # capture time of the first detection
    concurrent_max: int = 1  # most visits open at once while this one was


def _overlap_of_smaller(a: BBox, b: BBox) -> float:
    """Intersection area over the smaller box's area."""
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    smaller = min(a[2] * a[3], b[2] * b[3])
    return iw * ih / smaller if smaller > 0 else 0.0


DUPLICATE_OVERLAP = 0.5  # a box mostly inside another box of the same frame is the same bird


class EventLogger:
    def __init__(
        self,
        logging_cfg: LoggingConfig,
        storage_cfg: StorageConfig,
        run_id: str,
        *,
        obs: ObsLogger,
        dedupe_within_seconds: float = 10.0,
        iou_threshold: float = 0.3,
        center_distance: float = 2.0,
        max_visit_seconds: float = 600.0,
        checkpoint_seconds: float = 60.0,
        root: Path = PROJECT_ROOT,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._run_id = run_id
        self._obs = obs
        self._root = root
        self.events_path = resolve_path(logging_cfg.events_file, root)
        self.snapshots_dir = resolve_path(logging_cfg.snapshots_dir, root)

        fmt = logging_cfg.snapshot_format.lower()
        if fmt in ("jpeg", "jpg"):
            self._ext = ".jpg"
            self._write_params = [cv2.IMWRITE_JPEG_QUALITY, int(logging_cfg.snapshot_quality)]
        elif fmt == "png":
            self._ext = ".png"
            self._write_params = []
        else:
            raise ValueError(f"snapshot_format must be jpeg or png, not {fmt!r}")
        self._save_full_frame = logging_cfg.save_full_frame
        self._max_per_day = storage_cfg.max_events_per_day
        self._retention_days = storage_cfg.retention_days
        self._window = float(dedupe_within_seconds)
        self._iou_threshold = iou_threshold
        self._center_distance = center_distance
        self._max_visit = float(max_visit_seconds)
        self._checkpoint_every = float(checkpoint_seconds)
        self.sidecar_path = self.events_path.parent / "open_visits.json"
        self._last_checkpoint: Optional[float] = None

        self._open: list[_Visit] = []
        self._visit_counter = 0  # visit_id = <run_id>-<counter>, from 1 each run
        self._capped_days: set[date] = set()
        self._closed = False

        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._recover()  # before counting, so recovered lines count toward the daily cap
        self._day_counts = self._count_existing_events()
        self.sweep_retention(now())

    # ------------------------------------------------------------ public

    def handle(self, frame: Frame, detections: list[Detection]) -> list[dict[str, Any]]:
        """Feed every processed frame (even with no detections) so visits
        expire on time. Returns the events opened on this frame."""
        t = frame.captured_wall_time
        n_before = len(self._open)
        closed = self._expire(t)
        opened: list[dict[str, Any]] = []
        assigned = self._associate(detections, t)
        # Boxes that join or open a visit on this frame: the assigned ones, and
        # each one that opens a visit, as it does.
        used: list[BBox] = [detections[i].bbox_xywh for i in assigned]
        for index, det in enumerate(detections):
            visit = assigned.get(index)
            if visit is None and any(_overlap_of_smaller(det.bbox_xywh, u) >= DUPLICATE_OVERLAP
                                     for u in used):
                # A second box of a bird already handled on this frame (its
                # head next to its whole body, R5): neither a visit nor its box.
                self._obs.emit("persist", "skip", frame_seq=frame.seq,
                               context={"reason": "duplicate", "bbox_xywh": list(det.bbox_xywh)})
                continue
            if visit is None:
                used.append(det.bbox_xywh)
            if visit is not None:
                if visit.last_seq != frame.seq:
                    visit.visit_frames += 1
                visit.last_seen = t
                visit.last_seq = frame.seq
                visit.last_bbox = det.bbox_xywh
                self._obs.emit(
                    "persist", "skip", frame_seq=frame.seq,
                    context={"reason": "dedupe", "visit_frame_seq": visit.event["frame_seq"],
                             "visit_id": visit.event["visit_id"],
                             "visit_frames": visit.visit_frames},
                )
                continue
            day = _local(t).date()
            if self._day_counts[day] >= self._max_per_day:
                if day not in self._capped_days:
                    self._capped_days.add(day)
                    logger.warning(
                        f"Daily event cap reached ({self._max_per_day} on {day}); "
                        "further new visits today are dropped"
                    )
                self._obs.emit(
                    "persist", "skip", frame_seq=frame.seq,
                    context={"reason": "daily_cap", "day": str(day),
                             "max_events_per_day": self._max_per_day},
                )
                continue
            self._day_counts[day] += 1
            event = self._open_visit(frame, det, index)
            opened.append(event)
        changed = bool(closed or opened) or len(self._open) != n_before
        if changed or (self._last_checkpoint is not None
                       and t - self._last_checkpoint >= self._checkpoint_every):
            self._checkpoint(t)
        return opened

    def open_tracks(self, t: float) -> list[BBox]:
        """Last box of every visit still open at time `t` (for the Detector's
        track crops)."""
        return [v.last_bbox for v in self._open if t - v.last_seen <= self._window]

    def open_visits(self) -> list[dict[str, Any]]:
        """The open visits: visit_id, first box, last box, as they are now."""
        return [{"visit_id": v.event["visit_id"], "first_bbox": v.event["bbox_xywh"],
                 "bbox_xywh": list(v.last_bbox), "last_seen": iso(v.last_seen)} for v in self._open]

    def close(self) -> None:
        """Write every still-open visit. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        for visit in self._open:
            self._write_event(visit)
        self._open.clear()
        self._checkpoint(None)  # nothing open: removes the sidecar

    def manual_snapshot_path(self, frame: Frame) -> Path:
        """Where the preview's `s` key saves a frame."""
        return self.snapshots_dir / (
            f"manual_{stamp(frame.captured_wall_time)}_seq{frame.seq:06d}{self._ext}"
        )

    def sweep_retention(self, now: float) -> int:
        """Delete image files in snapshots_dir older than retention_days."""
        sp = self._obs.span("retention", context={"dir": str(self.snapshots_dir),
                                                  "retention_days": self._retention_days})
        cutoff = now - self._retention_days * 86400
        deleted = 0
        for path in self.snapshots_dir.iterdir():
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    deleted += 1
            except OSError as e:
                self._obs.emit("retention", "fail", error_type="hardware",
                               error_message=describe(e), context={"path": str(path)})
        if deleted:
            logger.info(f"Retention: deleted {deleted} snapshot(s) older than "
                        f"{self._retention_days} days")
        sp.success({"deleted": deleted})
        return deleted

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------ internals

    def _count_existing_events(self) -> Counter[date]:
        counts: Counter[date] = Counter()
        if not self.events_path.exists():
            return counts
        with self.events_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    counts[datetime.fromisoformat(json.loads(line)["ts"]).astimezone().date()] += 1
                except (ValueError, KeyError, TypeError):
                    continue  # a malformed line never blocks logging
        return counts

    def _cost(self, det: Detection, visit: _Visit, t: float) -> Optional[float]:
        """Cost of `det` joining `visit`: (1 - IoU) plus the centre distance
        over the visit's reach (center_distance x max(w, h) of its last box),
        or None when the pair fails the acceptance gate: another class, past
        the dedupe window, or IoU < threshold with the centre out of reach."""
        if visit.class_name != det.class_name or t - visit.last_seen > self._window:
            return None
        overlap = iou(visit.last_bbox, det.bbox_xywh)
        dist = center_distance(visit.last_bbox, det.bbox_xywh)
        reach = self._center_distance * max(visit.last_bbox[2], visit.last_bbox[3])
        if overlap < self._iou_threshold and dist > reach:
            return None
        return (1.0 - overlap) + (dist / reach if reach > 0 else 0.0)

    def _associate(self, detections: list[Detection], t: float) -> dict[int, _Visit]:
        """Detections of one frame against the visits open before it, one
        visit per detection and one detection per visit, at the lowest total
        cost (Hungarian method). Pairs that fail the gate are never chosen.
        Returns {detection index: visit}; the others open visits (or are
        duplicates of a box that joined one)."""
        if not detections or not self._open:
            return {}
        costs = [[self._cost(d, v, t) for v in self._open] for d in detections]
        if all(c is None for row in costs for c in row):
            return {}
        from scipy.optimize import linear_sum_assignment   # scipy comes with ultralytics

        forbidden = 1e6
        matrix = np.array([[forbidden if c is None else c for c in row] for row in costs])
        rows, cols = linear_sum_assignment(matrix)
        return {int(i): self._open[int(j)] for i, j in zip(rows, cols) if costs[i][j] is not None}

    def _expire(self, t: float) -> int:
        """Write visits unconfirmed for longer than the window, and cut visits
        that have lasted max_visit_seconds (truncated). Returns how many."""
        still_open = []
        for visit in self._open:
            if t - visit.last_seen > self._window:
                self._write_event(visit)
            elif t - visit.started >= self._max_visit:
                self._write_event(visit, truncated=True)
            else:
                still_open.append(visit)
        closed = len(self._open) - len(still_open)
        self._open = still_open
        return closed

    def _snapshot_event(self, visit: _Visit, **flags: bool) -> dict[str, Any]:
        return dict(visit.event, last_seen=iso(visit.last_seen), visit_frames=visit.visit_frames,
                    truncated=flags.get("truncated", False), recovered=flags.get("recovered", False),
                    concurrent_max=visit.concurrent_max)

    def _checkpoint(self, t: Optional[float]) -> None:
        """Rewrite open_visits.json with every open visit (atomic replace,
        fsync), or remove it when nothing is open."""
        self._last_checkpoint = t
        try:
            if not self._open:
                self.sidecar_path.unlink(missing_ok=True)
                return
            tmp = self.sidecar_path.with_name(self.sidecar_path.name + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump([self._snapshot_event(v) for v in self._open], f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.sidecar_path)
        except OSError as e:
            logger.error(f"Could not checkpoint open visits to {self.sidecar_path}: {e}")
            self._obs.emit("checkpoint", "fail", error_type="hardware", error_message=describe(e),
                           context={"path": str(self.sidecar_path)})

    def _recover(self) -> None:
        """Write the visits a previous run left open (it ended without close(),
        e.g. a hard kill) with "recovered": true, then delete the sidecar.
        A visit already in events.jsonl (same run_id and frame_seq) is skipped:
        the kill came between appending its line and updating the sidecar."""
        if not self.sidecar_path.exists():
            return
        sp = self._obs.span("recover", context={"path": str(self.sidecar_path)})
        try:
            visits = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
            if not isinstance(visits, list):
                raise ValueError("expected a JSON list")
        except (OSError, ValueError) as e:
            bad = self.sidecar_path.with_name(self.sidecar_path.name + ".corrupt")
            os.replace(self.sidecar_path, bad)
            logger.error(f"Unreadable {self.sidecar_path.name} ({e}); moved to {bad.name}")
            sp.fail("parse", describe(e), {"moved_to": str(bad)})
            return
        # A visit is identified by its visit_id; lines written before visit ids
        # existed, by (run_id, frame_seq). Two visits can open on one frame.
        written: set[Any] = set()
        if self.events_path.exists():
            with self.events_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        ev = json.loads(line)
                        written.add(ev.get("visit_id") or (ev["run_id"], ev["frame_seq"]))
                    except (ValueError, KeyError, TypeError, AttributeError):
                        continue
        recovered = skipped = 0
        for event in visits:
            if (event.get("visit_id") or (event.get("run_id"), event.get("frame_seq"))) in written:
                skipped += 1
                continue
            self._append(dict(event, recovered=True))
            recovered += 1
        self.sidecar_path.unlink()
        if recovered:
            logger.warning(f"Recovered {recovered} visit(s) left open by a run that did not "
                           f"close cleanly; written with \"recovered\": true")
        sp.success({"recovered": recovered, "already_written": skipped})

    def _append(self, event: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _open_visit(self, frame: Frame, det: Detection, index: int) -> dict[str, Any]:
        t = frame.captured_wall_time
        base = f"{stamp(t)}_seq{frame.seq:06d}_d{index}"
        x0, y0, x1, y1 = det.crop_xyxy
        crop = self._write_image(self.snapshots_dir / f"{base}_crop{self._ext}",
                                 frame.image[y0:y1, x0:x1], frame.seq)
        full = None
        if self._save_full_frame:
            full = self._write_image(self.snapshots_dir / f"{base}_full{self._ext}",
                                     frame.image, frame.seq)
        event = {
            "ts": iso(t),
            "run_id": self._run_id,
            "frame_seq": frame.seq,
            "class": det.class_name,
            "confidence": round(det.confidence, 4),
            "bbox_xywh": list(det.bbox_xywh),
            "snapshot_crop": self._display_path(crop),
            "snapshot_full": self._display_path(full),
            "last_seen": iso(t),
            "visit_frames": 1,
        }
        self._visit_counter += 1
        event["visit_id"] = f"{self._run_id}-{self._visit_counter}"
        self._open.append(_Visit(event, det.class_name, det.bbox_xywh, t, frame.seq, started=t))
        for visit in self._open:
            visit.concurrent_max = max(visit.concurrent_max, len(self._open))
        return event

    def _write_image(self, path: Path, image: np.ndarray, seq: int) -> Optional[Path]:
        if image.size and cv2.imwrite(str(path), image, self._write_params):
            return path
        message = f"could not write snapshot {path} (empty image or cv2.imwrite returned False)"
        logger.error(message)
        self._obs.emit("persist", "fail", frame_seq=seq, error_type="hardware",
                       error_message=message, context={"path": str(path)})
        return None

    def _write_event(self, visit: _Visit, truncated: bool = False) -> None:
        t0 = time.perf_counter()
        event = self._snapshot_event(visit, truncated=truncated)
        self._append(event)
        self._obs.emit(
            "persist", "success", frame_seq=event["frame_seq"],
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            context={"ts": event["ts"], "last_seen": event["last_seen"],
                     "visit_frames": event["visit_frames"], "truncated": truncated,
                     "snapshot_crop": event["snapshot_crop"], "visit_id": event["visit_id"],
                     "concurrent_max": event["concurrent_max"]},
        )
        logger.info(
            f"Bird visit logged: {event['ts']} seq={event['frame_seq']} "
            f"frames={event['visit_frames']} conf={event['confidence']}"
            + (" (truncated at max_visit_seconds)" if truncated else "")
        )

    def _display_path(self, path: Optional[Path]) -> Optional[str]:
        """Path relative to snapshots_dir's parent ("snapshots/<name>"), POSIX slashes."""
        if path is None:
            return None
        return path.relative_to(self.snapshots_dir.parent).as_posix()
