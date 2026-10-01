"""--annotate-out: every processed frame, drawn exactly as the live preview
draws it (ROI box, held detection boxes, 3-line HUD), written to a video file."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import main
from tests.conftest import read_log, write_test_config

BOX = (300, 150, 30, 20)   # inside the ROI
ROI = {"x": 160, "y": 90, "w": 320, "h": 180, "frame_width": 640, "frame_height": 360}


def roi_file(tmp_path: Path) -> Path:
    path = tmp_path / "roi.json"
    path.write_text(json.dumps(ROI))
    return path


def read_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()
    return frames


def processed(lines: list[dict]) -> int:
    """Frames that went through the pipeline: every one reaches mog2_apply."""
    return sum(l["context"]["records"] for l in lines
               if l["stage"] == "mog2_apply" and l["context"].get("summary"))


def annotate(video: Path, tmp_path: Path, predictor, *extra: str) -> tuple[int, Path, list[dict]]:
    out = tmp_path / "annotated.mp4"
    log_dir = tmp_path / "logs"
    code = main.main(["--source", str(video), "--headless", "--no-pace", "--log-dir", str(log_dir),
                      "--roi-file", str(roi_file(tmp_path)), "--annotate-out", str(out),
                      *extra], predictor=predictor)
    (log,) = log_dir.glob("run_*.jsonl")
    return code, out, read_log(log)


def edge_colour(img: np.ndarray, x0: int, x1: int, y: int) -> np.ndarray:
    """Mean BGR of the most coloured of the 3 rows around a rectangle's top
    edge at `y` (thickness 2 straddles y; the codec blurs it a little)."""
    rows = img[y - 1:y + 2, x0:x1].mean(axis=1)
    return rows[np.argmax(rows.max(axis=1) - rows.min(axis=1))]


def test_annotate_out_writes_every_processed_frame(synth_video_2s, tmp_path, fake_predictor) -> None:
    code, out, lines = annotate(synth_video_2s, tmp_path, fake_predictor())

    assert code == 0
    frames = read_frames(out)
    assert len(frames) == processed(lines) == 60
    assert frames[0].shape == (360, 640, 3)
    render = [l for l in lines if l["stage"] == "render" and l["context"].get("summary")]
    assert sum(l["context"]["records"] for l in render) == 60
    assert all(l["context"].get("reasons", {}).get("headless", 0) == 0 for l in render)
    # A paced run skips frames when the pipeline falls behind, so the frame seq
    # of every written frame goes in a sidecar next to the video.
    sidecar = json.loads((tmp_path / "annotated.mp4.frames.json").read_text(encoding="utf-8"))
    assert sidecar == {"fps": 30.0, "frame_seqs": list(range(1, 61))}


def test_annotated_frames_carry_the_preview_overlays(synth_video_2s, tmp_path, monkeypatch,
                                                      fake_predictor) -> None:
    """ROI box and HUD on every frame; a detection box from frame 11 is held
    for 1.5 s of capture time (media time for a file), then dropped."""
    monkeypatch.setattr(main, "CONFIG_FILE", write_test_config(
        tmp_path, **{"motion_warmup_frames: 60": "motion_warmup_frames: 10"}))
    fake = fake_predictor({11: [BOX]})

    code, out, _ = annotate(synth_video_2s, tmp_path, fake)

    assert code == 0
    assert 11 in [c["frame_seq"] for c in fake.calls]
    frames = read_frames(out)
    source = read_frames(synth_video_2s)
    b, g, r = edge_colour(frames[0], 170, 470, 90)          # ROI (160, 90, 320, 180), orange
    assert r > 180 and 100 < g < 210 and b < 90
    hud = (slice(14, 100), slice(16, 400))
    assert np.abs(frames[0][hud].astype(int) - source[0][hud].astype(int)).mean() > 8
    x, y, w, _ = BOX
    for seq, held in ((10, False), (11, True), (50, True), (60, False)):
        b, g, r = edge_colour(frames[seq - 1], x + 4, x + w - 4, y)
        assert (g > 180 and r < 100 and b < 100) == held, (seq, (b, g, r))


def test_preview_and_annotation_draw_the_same_image(synth_video_2s, tmp_path, monkeypatch,
                                                    fake_predictor) -> None:
    """With windows (GUI calls stubbed), each frame written is the frame shown."""
    shown: list[np.ndarray] = []
    monkeypatch.setattr(main.cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(main.cv2, "resizeWindow", lambda *a, **k: None)
    monkeypatch.setattr(main.cv2, "imshow", lambda _name, img: shown.append(img.copy()))
    monkeypatch.setattr(main.cv2, "waitKey", lambda *_: -1)
    monkeypatch.setattr(main.cv2, "destroyAllWindows", lambda: None)
    roi = roi_file(tmp_path)
    out = tmp_path / "annotated.mp4"

    code = main.main(["--source", str(synth_video_2s), "--no-pace", "--roi-file", str(roi),
                      "--log-dir", str(tmp_path / "logs"), "--annotate-out", str(out)],
                     predictor=fake_predictor())

    assert code == 0
    written = read_frames(out)
    source = read_frames(synth_video_2s)
    assert len(shown) == len(written) == len(source) == 60
    for a, b, src in zip(shown, written, source):
        a, b, src = (x.astype(int) for x in (a, b, src))
        overlay = np.abs(a - src).max(axis=2) > 60      # ROI box, label and HUD pixels
        assert overlay.sum() > 2000
        # Lossy codec: the written frame is close to the shown one on the
        # overlay pixels, and far from the bare source frame there.
        assert np.abs(a - b)[overlay].mean() < 20 < np.abs(src - b)[overlay].mean()


def test_annotate_out_needs_a_source(tmp_path, monkeypatch) -> None:
    """The project records no video of the live camera; the flag is for files."""
    monkeypatch.setattr(main, "list_devices", lambda *_: pytest.fail("camera touched"))

    code = main.main(["--annotate-out", str(tmp_path / "a.mp4"), "--yes",
                      "--log-dir", str(tmp_path / "logs")])

    assert code == 1
    assert not (tmp_path / "a.mp4").exists()


def test_hud_fps_reads_the_paced_frame_rate_from_the_first_frame(synth_video_2s, tmp_path,
                                                                 monkeypatch, fake_predictor) -> None:
    """The first frame was already waiting when the loop started; its "frame
    interval" is not one and must not seed the fps average (it showed 140+ fps
    for the first 2 s of a paced 30 fps run)."""
    import re

    from tests.fakes import FakeClock

    clock = FakeClock(lockstep=True)
    ticks = iter(range(10**9))
    huds: list[str] = []
    draw = main.draw_overlays
    monkeypatch.setattr(main, "draw_overlays",
                        lambda image, roi, dets, hud: huds.append(hud[0][0]) or draw(image, roi, dets, hud))

    code = main.main(["--source", str(synth_video_2s), "--headless", "--log-dir", str(tmp_path / "logs"),
                      "--roi-file", str(roi_file(tmp_path)), "--annotate-out", str(tmp_path / "a.mp4")],
                     predictor=fake_predictor(),
                     clock=lambda: clock.monotonic() + next(ticks) * 1e-6,  # time moves a hair per read
                     sleep=clock.sleep)

    assert code == 0
    fps = [float(re.search(r"fps=\s*([\d.]+)", h).group(1)) for h in huds]
    assert len(fps) == 60
    # No runaway seed: one jittered interval at most (frame 1 was picked up
    # after the first-frame wait), then the 30 fps the file is paced at.
    assert max(fps) < 60, fps[:5]
    assert all(28 < f < 33 for f in fps[30:]), fps[30:]   # 5 ms idle-poll granularity
