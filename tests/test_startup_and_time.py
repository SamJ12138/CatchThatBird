"""Startup order (obs F3) and capture-time wall clock on Frame / Detection."""
from __future__ import annotations

import time

import main
from camera import Frame, FrameGrabber
from config_schema import DetectionConfig
from detector import Detector
from tests.conftest import frame_with_block, grey_frame, read_log
from tests.fakes import FakeClock


def test_first_frame_the_detector_processes_has_seq_at_most_2(synth_video_2s, tmp_path,
                                                                fake_predictor) -> None:
    """Paced like a camera, with a 1 s model load: frames must not be lost
    while the model loads (before: the first processed seq was ~31).

    Lockstep fake clock: the load takes 1 s of fake time, and the paced
    grabber only moves when the main loop idles. Had capture started before
    the load, the grabber would owe 30 frames when the clock jumped and would
    publish them back to back, so the first frame processed would be ~31."""
    clock = FakeClock(lockstep=True)
    slow_loading = fake_predictor(load_delay=1.0, sleep=clock.sleep)
    log_dir = tmp_path / "logs"

    code = main.main(["--source", str(synth_video_2s), "--headless", "--log-dir", str(log_dir),
                      "--roi-file", str(tmp_path / "no_roi.json")], predictor=slow_loading,
                     clock=clock.monotonic, sleep=clock.sleep)

    assert code == 0
    assert clock.slept[0] == 1.0                 # the load ran first, in fake time
    (log,) = log_dir.glob("run_*.jsonl")
    summaries = [l for l in read_log(log) if l["stage"] == "mog2_apply" and l["context"].get("summary")]
    assert summaries[0]["context"]["seq_range"][0] <= 2


def test_model_is_loaded_before_capture_starts(synth_video_2s, tmp_path, fake_predictor) -> None:
    log_dir = tmp_path / "logs"
    main.main(["--source", str(synth_video_2s), "--headless", "--no-pace", "--log-dir", str(log_dir),
               "--roi-file", str(tmp_path / "no_roi.json")], predictor=fake_predictor())
    (log,) = log_dir.glob("run_*.jsonl")
    order = [l["stage"] for l in read_log(log) if l["event"] == "start"]
    assert order.index("detector_init") < order.index("capture_open")


def test_frame_carries_capture_wall_time(synth_video_2s, obs) -> None:
    before = time.time()
    with FrameGrabber(0, 640, 360, 30, obs=obs, source=str(synth_video_2s), pace=False) as grabber:
        deadline = time.monotonic() + 5
        while (frame := grabber.read_latest()) is None and time.monotonic() < deadline:
            time.sleep(0.005)
        after = time.time()
    assert frame is not None
    assert before <= frame.captured_wall_time <= after


def test_detection_copies_wall_time_from_frame(obs, fake_predictor) -> None:
    clock = FakeClock()  # "inference" takes 0.2 s of fake time; time.time() is untouched
    slow = fake_predictor(crop_boxes=[(0, 0, 5, 5)], predict_delay=0.2, sleep=clock.sleep)
    det = Detector(DetectionConfig(motion_warmup_frames=2, process_every_n_frames=1,
                                   motion_min_area=50), obs=obs, predictor=slow)
    images = [grey_frame(), grey_frame(), frame_with_block(60, 60)]
    detections = []
    for seq, img in enumerate(images, start=1):
        detections += det.process(Frame(image=img, captured_at=0.0, seq=seq,
                                        captured_wall_time=1_700_000_000.0 + seq))

    assert [d.captured_wall_time for d in detections] == [1_700_000_003.0]
