"""Detector: crop / coordinate mapping, warm-up, predictor injection."""
from __future__ import annotations

import subprocess
import sys
from collections import Counter

import pytest

import detector as detector_mod
from camera import Frame
from config_schema import DetectionConfig
from detector import Detector
from tests.conftest import ROOT, frame_with_block, grey_frame, read_log


def dcfg(**overrides) -> DetectionConfig:
    base = dict(motion_warmup_frames=2, process_every_n_frames=1,
                motion_min_area=50, motion_padding_px=10)
    base.update(overrides)
    return DetectionConfig(**base)


def make_frame(image, seq: int) -> Frame:
    return Frame(image=image, captured_at=0.0, seq=seq, captured_wall_time=1_700_000_000.0 + seq)


def run_frames(det: Detector, images) -> list:
    out = []
    for seq, img in enumerate(images, start=1):
        out += det.process(make_frame(img, seq))
    return out


# ---------------------------------------------------------------- coordinate mapping

@pytest.mark.parametrize(
    "roi, block_xy, expected_origin, expected_crop_hw",
    [
        # ROI at the frame origin: region coords == frame coords.
        ((0, 0, 200, 150), (60, 40), (50, 30), (40, 40)),
        # ROI offset: contour found in region coords, translated by (100, 80).
        ((100, 80, 150, 120), (150, 120), (140, 110), (40, 40)),
        # Padding clamped at the top-left frame edge.
        (None, (3, 2), (0, 0), (32, 33)),
        # Padding clamped at the bottom-right frame edge.
        (None, (300, 220), (290, 210), (30, 30)),
    ],
    ids=["roi-at-origin", "roi-offset", "clamp-top-left", "clamp-bottom-right"],
)
def test_crop_and_detection_map_to_full_frame(obs, fake_predictor, roi, block_xy,
                                              expected_origin, expected_crop_hw) -> None:
    fake = fake_predictor(crop_boxes=[(5, 6, 15, 26)])  # crop coords: 10x20 box at (5, 6)
    det = Detector(dcfg(), roi=roi, obs=obs, predictor=fake)

    detections = run_frames(det, [grey_frame(), grey_frame(), frame_with_block(*block_xy)])

    assert fake.calls[-1]["origin"] == expected_origin
    assert fake.calls[-1]["crop_shape"] == expected_crop_hw
    assert len(detections) == 1
    ox, oy = expected_origin
    assert detections[0].bbox_xywh == (ox + 5, oy + 6, 10, 20)
    ch, cw = expected_crop_hw
    assert detections[0].crop_xyxy == (ox, oy, ox + cw, oy + ch)  # for the crop snapshot
    assert detections[0].class_name == "bird"
    assert detections[0].frame_seq == 3


def test_full_frame_boxes_round_trip_through_crop(obs, fake_predictor) -> None:
    fake = fake_predictor({3: [(100, 90, 12, 8)]})
    det = Detector(dcfg(), roi=(40, 30, 200, 150), obs=obs, predictor=fake)

    detections = run_frames(det, [grey_frame(), grey_frame(), frame_with_block(100, 90)])

    assert [d.bbox_xywh for d in detections] == [(100, 90, 12, 8)]


def test_low_confidence_and_wrong_class_are_dropped(obs, fake_predictor) -> None:
    low = fake_predictor(crop_boxes=[(0, 0, 5, 5)], conf=0.1)
    det = Detector(dcfg(), obs=obs, predictor=low)
    assert run_frames(det, [grey_frame(), grey_frame(), frame_with_block(50, 50)]) == []

    person = fake_predictor(crop_boxes=[(0, 0, 5, 5)], cls_id=0)
    det = Detector(dcfg(), obs=obs, predictor=person)
    assert run_frames(det, [grey_frame(), grey_frame(), frame_with_block(50, 50)]) == []


# ---------------------------------------------------------------- warm-up

def test_warmup_skips_exactly_motion_warmup_frames(obs, fake_predictor) -> None:
    """obs F5: warm-up used `<`, so it skipped one frame fewer than configured."""
    det = Detector(dcfg(motion_warmup_frames=5), obs=obs, predictor=fake_predictor())

    run_frames(det, [grey_frame()] * 10)

    obs.close()
    lines = read_log(obs.path)
    reasons: Counter = Counter()
    for l in lines:
        if l["stage"] == "gate_check" and l["context"].get("summary"):
            reasons.update(l["context"].get("reasons", {}))
    assert reasons["warmup"] == 5
    complete = [l for l in lines if l["context"].get("reason") == "mog2_warmup_complete"]
    assert [l["frame_seq"] for l in complete] == [6]


def test_no_detection_during_warmup(obs, fake_predictor) -> None:
    fake = fake_predictor(crop_boxes=[(0, 0, 5, 5)])
    det = Detector(dcfg(motion_warmup_frames=3), obs=obs, predictor=fake)
    # Motion on frame 3 is inside the warm-up window and must not reach the predictor.
    run_frames(det, [grey_frame(), grey_frame(), frame_with_block(50, 50)])
    assert fake.calls == []


# ---------------------------------------------------------------- predictor injection

def test_default_predictor_is_built_from_config(monkeypatch, obs, fake_predictor) -> None:
    built = {}

    class StubYolo(type(fake_predictor())):
        def __init__(self, model_path: str) -> None:
            super().__init__()
            built["model_path"] = model_path

    monkeypatch.setattr(detector_mod, "YoloPredictor", StubYolo)
    det = Detector(dcfg(yolo_model="custom.pt"), obs=obs)

    assert built == {"model_path": "custom.pt"}
    assert det.stats["frames_processed"] == 0


def test_injected_predictor_is_loaded_once(obs, fake_predictor) -> None:
    fake = fake_predictor()
    Detector(dcfg(), obs=obs, predictor=fake)
    assert fake.loaded


def test_importing_detector_does_not_import_ultralytics() -> None:
    code = ("import sys; sys.path.insert(0, r'%s'); import detector, main; "
            "print('ultralytics' in sys.modules, 'torch' in sys.modules)" % ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "False"]
