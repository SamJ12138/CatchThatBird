"""Run main.main() in a subprocess with test doubles installed.

    CTB_FAKES='{"capture": {...}, "devices": [[0, "name"]], "predictor": {...}}' \
        python tests/main_with_fakes.py <main.py args>

capture   -> cv2.VideoCapture is replaced by FakeCapture(**capture)
devices   -> main.list_devices returns this list (no device enumeration)
predictor -> FakePredictor.from_spec(predictor) (always a fake: never YOLO)
grabber   -> extra FrameGrabber keyword arguments (e.g. join_timeout_s)
"""
from __future__ import annotations

import functools
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
    if "grabber" in spec:
        main.FrameGrabber = functools.partial(main.FrameGrabber, **spec["grabber"])
    predictor = FakePredictor.from_spec(spec.get("predictor", {}))
    return main.main(sys.argv[1:], predictor=predictor)


if __name__ == "__main__":
    sys.exit(run())
