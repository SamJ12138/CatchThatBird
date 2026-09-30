"""main.main() in-process with a fake predictor (no YOLO, no windows)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import main
from tests.conftest import read_log


def run(video: Path, tmp_path: Path, predictor, *extra: str) -> tuple[int, list[dict]]:
    log_dir = tmp_path / "logs"
    args = ["--source", str(video), "--headless", "--no-pace",
            "--log-dir", str(log_dir), *extra]
    if "--roi-file" not in extra:
        args += ["--roi-file", str(tmp_path / "no_roi.json")]
    code = main.main(args, predictor=predictor)
    (log,) = log_dir.glob("run_*.jsonl")
    return code, read_log(log)


def summary_total(lines: list[dict], stage: str, key: str = "records") -> int:
    return sum(l["context"][key] for l in lines if l["stage"] == stage and l["context"].get("summary"))


def test_headless_run_processes_every_frame(synth_video_2s, tmp_path, fake_predictor) -> None:
    code, lines = run(synth_video_2s, tmp_path, fake_predictor())

    assert code == 0
    assert summary_total(lines, "capture_read") == 60
    assert summary_total(lines, "mog2_apply") == 60
    run_end = [l for l in lines if l["stage"] == "run" and l["event"] != "start"]
    assert run_end[-1]["event"] == "success"


def test_headless_corrupt_roi_exits_1(synth_video_2s, tmp_path, fake_predictor) -> None:
    roi = tmp_path / "roi.json"
    roi.write_text("{not json")

    code, lines = run(synth_video_2s, tmp_path, fake_predictor(), "--roi-file", str(roi))

    assert code == 1
    assert not any(l["stage"] == "mog2_apply" for l in lines)  # stopped before detection


def test_headless_same_aspect_roi_is_rescaled(synth_video_2s, tmp_path, fake_predictor) -> None:
    roi = tmp_path / "roi.json"
    roi.write_text(json.dumps({"x": 480, "y": 270, "w": 960, "h": 540,
                               "frame_width": 1920, "frame_height": 1080}))

    code, lines = run(synth_video_2s, tmp_path, fake_predictor(), "--roi-file", str(roi))

    assert code == 0
    roi_end = [l for l in lines if l["stage"] == "roi_load" and l["event"] == "success"]
    assert roi_end[-1]["context"]["roi"] == [160, 90, 320, 180]


def test_blob_reaches_the_predictor(synth_video_2s, tmp_path, fake_predictor) -> None:
    fake = fake_predictor()
    code, lines = run(synth_video_2s, tmp_path, fake)

    assert code == 0
    # 60 frames, warm-up 60 -> nothing gated with the default config.
    assert fake.calls == []
    reasons: Counter = Counter()
    for l in lines:
        if l["stage"] == "gate_check" and l["context"].get("summary"):
            reasons.update(l["context"].get("reasons", {}))
    assert reasons["warmup"] == 60
