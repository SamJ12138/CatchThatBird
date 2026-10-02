"""The five tests that load the real yolov8n.pt (marker `slow`). They run
main.py in a subprocess so the test process itself never imports ultralytics."""
from __future__ import annotations

import json
from datetime import datetime
import subprocess
import sys

import pytest

from scripts import fetch_real_clip as fetch
from scripts import make_synth_video as synth
from tests.conftest import ROOT, read_log, write_test_config


@pytest.mark.slow
def test_real_yolo_end_to_end(synth_video_15s, tmp_path) -> None:
    if not (ROOT / "yolov8n.pt").exists():
        pytest.skip("yolov8n.pt not present (ultralytics would download it)")
    log_dir = tmp_path / "logs"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--source", str(synth_video_15s),
         "--headless", "--no-pace", "--log-dir", str(log_dir),
         "--roi-file", str(tmp_path / "no_roi.json"),
         "--config", str(write_test_config(tmp_path))],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]

    (log,) = log_dir.glob("run_*.jsonl")
    lines = read_log(log)
    init = [l for l in lines if l["stage"] == "detector_init" and l["event"] == "success"]
    assert init and init[0]["context"]["device"] in ("cpu", "cuda")
    assert init[0]["context"]["target_class_ids"] == [14]
    yolo = [l for l in lines if l["stage"] == "yolo_infer" and l["event"] == "success"]
    assert len(yolo) >= 5
    assert all("n_results" in l["context"] for l in yolo)


def iou(a, b) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


@pytest.mark.slow
def test_real_yolo_detects_the_synthetic_bird(tmp_path) -> None:
    """The quickstart clip (default bird, 1280x720, example ROI): the bird
    lands, perches ~13 s and leaves. Exactly one visit, labelled bird, where
    the script drew it, tracked through the perch (P9)."""
    if not (ROOT / "yolov8n.pt").exists():
        pytest.skip("yolov8n.pt not present (ultralytics would download it)")
    example = ROOT / "data" / "roi.example.json"
    roi = synth.scaled_roi(example, 1280, 720)
    clip = synth.make_video(tmp_path / "synth_bird.mp4", roi=roi)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--source", str(clip), "--headless", "--no-pace",
         "--yes", "--log-dir", str(tmp_path / "logs"), "--roi-file", str(example),
         "--config", str(write_test_config(tmp_path))],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]

    events = read_log(tmp_path / "data" / "events.jsonl")
    assert len(events) == 1, json.dumps(events)
    (ev,) = events
    assert ev["visit_frames"] >= 10
    stay = datetime.fromisoformat(ev["last_seen"]) - datetime.fromisoformat(ev["ts"])
    assert stay.total_seconds() >= 10
    sprite_wh = synth.sprite_size(roi)
    for ev in events:
        assert ev["class"] == "bird" and ev["confidence"] >= 0.35
        assert (tmp_path / "data" / ev["snapshot_crop"]).is_file()
        x, y = synth.bird_position((ev["frame_seq"] - 1) / 30, roi, sprite_wh)
        assert iou(ev["bbox_xywh"], (x, y, *sprite_wh)) > 0.3, json.dumps(ev)


def run_real_clip(tmp_path, *flags: str, **replacements: str) -> list[dict]:
    """main.py on the real hummingbird clip with its committed ROI; returns
    the events. Skips when the clip or the weights are not there."""
    if not fetch.OUT.exists():
        pytest.skip("real clip not fetched: python scripts/fetch_real_clip.py")
    if not (ROOT / "yolov8n.pt").exists():
        pytest.skip("yolov8n.pt not present (ultralytics would download it)")
    tmp_path.mkdir(exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--source", str(fetch.OUT), "--headless",
         "--yes", "--log-dir", str(tmp_path / "logs"), "--roi-file", str(fetch.ROI),
         "--config", str(write_test_config(tmp_path, **replacements)), *flags],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return read_log(tmp_path / "data" / "events.jsonl")


def clip_seconds(event: dict) -> float:
    """Where in the clip the visit opened (30 fps, frame 1 at 0 s)."""
    return (event["frame_seq"] - 1) / 30


# The hummingbird enters the frame at 2.8 s and YOLO first finds it at 3.0 s,
# still in the air (docs/observations.md, R2). A visit must open within 0.5 s
# of that.
BIRD_FIRST_FOUND_S = 3.0
OPEN_WITHIN_S = 0.5


@pytest.mark.slow
def test_real_clip_is_logged_as_a_visit(tmp_path) -> None:
    """The real hummingbird clip (scripts/fetch_real_clip.py; not in the
    repository), with its committed ROI and the default config: at least one
    visit, confirmed on at least 5 gated frames."""
    events = run_real_clip(tmp_path, "--no-pace")
    assert len(events) >= 1
    assert max(ev["visit_frames"] for ev in events) >= 5, json.dumps(events)
    assert all(ev["class"] == "bird" for ev in events)


@pytest.mark.slow
def test_real_clip_arrival_is_not_missed_when_the_gated_frame_is_blurred(tmp_path) -> None:
    """R2, made repeatable. Run c2c24c8d opened its visit at 5.93 s because
    load moved its gated frames to 96 (the bird blurred in flight) and 138 (a
    second after it settled, its still body already background). Warm-up 53
    and a cadence of 42 put the gated frames of an unpaced run on 54, 96, 138
    and 180: without the re-check the visit opens on frame 180, 5.97 s."""
    events = run_real_clip(tmp_path, "--no-pace",
                           **{"motion_warmup_frames: 60": "motion_warmup_frames: 53",
                              "process_every_n_frames: 30": "process_every_n_frames: 42"})
    assert len(events) == 1, json.dumps(events)
    assert clip_seconds(events[0]) <= BIRD_FIRST_FOUND_S + OPEN_WITHIN_S, json.dumps(events)
    assert events[0]["visit_frames"] >= 5


@pytest.mark.slow
def test_real_clip_paced_runs_open_the_visit_on_arrival(tmp_path) -> None:
    """Three runs paced like a camera, default config. Which frames are gated
    depends on load, so each run may differ; every one must open its visit
    within 0.5 s of the bird first being findable."""
    opened = []
    for n in range(3):
        events = run_real_clip(tmp_path / f"run{n}")
        assert len(events) == 1, json.dumps(events)
        opened.append(clip_seconds(events[0]))
    assert max(opened) <= BIRD_FIRST_FOUND_S + OPEN_WITHIN_S, opened
