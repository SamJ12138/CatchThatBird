"""The only test that loads the real yolov8n.pt. It runs main.py in a
subprocess so the test process itself never imports ultralytics."""
from __future__ import annotations

import subprocess
import sys

import pytest

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
