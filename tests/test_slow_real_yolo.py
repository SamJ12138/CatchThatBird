"""The two tests that load the real yolov8n.pt (marker `slow`). They run
main.py in a subprocess so the test process itself never imports ultralytics."""
from __future__ import annotations

import json
from datetime import datetime
import subprocess
import sys

import pytest

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
