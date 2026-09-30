"""Run main.main() in a subprocess with test doubles installed.

    CTB_FAKES='{"capture": {...}, "devices": [[0, "name"]], "predictor": {...}}' \
        python tests/main_with_fakes.py <main.py args>

capture   -> cv2.VideoCapture is replaced by FakeCapture(**capture)
devices   -> main.list_devices returns this list (no device enumeration)
predictor -> FakePredictor.from_spec(predictor) (always a fake: never YOLO)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

import main  # noqa: E402
from tests.fakes import FakeCapture, FakePredictor  # noqa: E402


def run() -> int:
    spec = json.loads(os.environ.get("CTB_FAKES", "{}"))
    if "capture" in spec:
        capture_spec = spec["capture"]
        cv2.VideoCapture = lambda *a, **k: FakeCapture(*a, **capture_spec)
    if "devices" in spec:
        devices = [tuple(d) for d in spec["devices"]]
        main.list_devices = lambda _obs: devices
    predictor = FakePredictor.from_spec(spec.get("predictor", {}))
    return main.main(sys.argv[1:], predictor=predictor)


if __name__ == "__main__":
    sys.exit(run())
