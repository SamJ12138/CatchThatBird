"""scripts/make_demo_gif.py: input and detection side by side, cut from the
first visit in events.jsonl, perch sped up to fit the length cap."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import make_demo_gif as demo
from scripts.make_synth_video import make_video
from tests.conftest import ROOT

TS = "2026-09-30T20:06:15.080-04:00"


def event(ts: str = TS, last_seen: str = "2026-09-30T20:06:28.080-04:00", frame_seq: int = 121,
          **extra) -> dict:
    return {"ts": ts, "run_id": "d0f537c9", "frame_seq": frame_seq, "class": "bird",
            "confidence": 0.92, "bbox_xywh": [583, 457, 103, 67], "snapshot_crop": None,
            "snapshot_full": None, "last_seen": last_seen, "visit_frames": 14,
            "truncated": False, "recovered": False, **extra}


def write_events(path: Path, *events: dict) -> Path:
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return path


def test_segment_runs_from_2s_before_ts_to_2s_after_last_seen() -> None:
    # frame_seq 121 at 30 fps is 4.0 s into the clip; last_seen is 13 s later.
    assert demo.segment(event(), fps=30, clip_seconds=20.0) == demo.Segment(2.0, 4.0, 17.0, 19.0)


def test_segment_is_clamped_to_the_clip() -> None:
    seg = demo.segment(event(frame_seq=31, last_seen="2026-09-30T20:06:19.080-04:00"),
                       fps=30, clip_seconds=5.5)
    assert seg == demo.Segment(0.0, 1.0, 5.0, 5.5)


def test_the_first_visit_is_the_earliest_ts(tmp_path) -> None:
    # Lines are written in close order; the earlier visit can come second.
    events = write_events(tmp_path / "events.jsonl",
                          event(ts="2026-09-30T20:07:00.000-04:00", frame_seq=900),
                          event())
    assert demo.first_visit(events)["frame_seq"] == 121


def test_no_visit_is_an_error(tmp_path) -> None:
    with pytest.raises(demo.DemoError, match="no visit"):
        demo.first_visit(write_events(tmp_path / "events.jsonl"))


def test_a_long_perch_plays_faster_and_landing_and_take_off_stay_at_1x() -> None:
    frames, factor = demo.plan(demo.Segment(2.0, 4.0, 17.0, 19.0), fps=10, max_seconds=12.0)

    assert factor == 2
    assert len(frames) / 10 <= 12.0
    times = [f.t for f in frames]
    assert times == sorted(times) and len(set(times)) == len(times)
    assert times[0] == pytest.approx(2.0) and times[-1] == pytest.approx(18.9)
    landing = [f for f in frames if f.t < 4.0]
    perch = [f for f in frames if 4.0 <= f.t < 17.0]
    take_off = [f for f in frames if f.t >= 17.0]
    assert len(landing) == 20 and len(take_off) == 20          # 2 s each at 1x
    assert len(perch) == 65                                    # 13 s at 2x
    assert {f.label for f in landing + take_off} == {None}
    assert {f.label for f in perch} == {"2x"}


def test_a_short_visit_plays_at_1x() -> None:
    frames, factor = demo.plan(demo.Segment(0.0, 2.0, 5.0, 7.0), fps=10, max_seconds=12.0)
    assert factor == 1 and len(frames) == 70
    assert {f.label for f in frames} == {None}


def test_a_real_clip_in_data_samples_replaces_the_synthetic_one(tmp_path) -> None:
    assert demo.choose_source(tmp_path) == tmp_path / "synth_bird.mp4"
    (tmp_path / "demo.mp4").write_bytes(b"")
    assert demo.choose_source(tmp_path) == tmp_path / "demo.mp4"


def test_pipeline_command_renders_the_annotated_video(tmp_path) -> None:
    cmd = demo.pipeline_command(Path("clip.mp4"), tmp_path)
    assert cmd[1].endswith("main.py")
    for flag in ("--headless", "--yes"):
        assert flag in cmd
    assert "--no-pace" not in cmd      # paced like a camera: the HUD shows live numbers
    assert cmd[cmd.index("--source") + 1] == "clip.mp4"
    assert cmd[cmd.index("--annotate-out") + 1] == str(tmp_path / "annotated.mp4")
    assert cmd[cmd.index("--data-root") + 1] == str(tmp_path)   # events.jsonl stays in tmp


def test_skipped_frames_show_the_last_processed_frame_on_both_sides() -> None:
    # Paced run: seqs 3 and 6 were never processed (the pipeline was busy).
    seqs = [1, 2, 4, 5, 7]
    # source frame index wanted -> (source index, annotated index) shown
    assert demo.align([0, 1, 2, 3, 5, 6], seqs) == [(0, 0), (1, 1), (1, 1), (3, 2), (4, 3), (6, 4)]
    assert demo.align([0, 1, 2], None) == [(0, 0), (1, 1), (2, 2)]   # no sidecar: one to one


def test_sidecar_of_the_annotated_video_is_read(tmp_path) -> None:
    video = tmp_path / "annotated.mp4"
    assert demo.annotated_seqs(video) is None
    (tmp_path / "annotated.mp4.frames.json").write_text(
        json.dumps({"fps": 30.0, "frame_seqs": [1, 2, 4]}), encoding="utf-8")
    assert demo.annotated_seqs(video) == [1, 2, 4]


def test_fixture_to_gif_under_the_limit(tmp_path, capsys) -> None:
    from PIL import Image

    clip = make_video(tmp_path / "clip.mp4", seconds=16, width=320, height=180, bird=False)
    events = write_events(tmp_path / "events.jsonl",   # 2.0 s .. 14.0 s into the clip
                          event(frame_seq=61, last_seen="2026-09-30T20:06:27.080-04:00"))
    out = tmp_path / "demo.gif"

    code = demo.main(["--source", str(clip), "--annotated", str(clip), "--events", str(events),
                      "--out", str(out)])

    assert code == 0
    assert out.read_bytes()[:6] == b"GIF89a"
    assert out.stat().st_size <= demo.HARD_LIMIT_BYTES
    with Image.open(out) as gif:
        assert gif.size[0] == 2 * 480
        assert gif.n_frames == 100                            # 2 + 12/2 + 2 s at 10 fps
        assert gif.info["duration"] == 100
    report = capsys.readouterr().out
    assert "2x" in report and "960x" in report


def test_ordered_dither_is_tried_only_after_error_diffusion(tmp_path, monkeypatch) -> None:
    """Real footage: every frame's background shimmers, so error diffusion
    re-dithers it all and the GIF stays large. The ladder then retries with
    ordered (Bayer) dithering on denoised frames, which repeat between frames."""
    tried = []

    def fake_render(source, annotated, frames, clip_fps, width, fps, out, work, *, style):
        tried.append((width, fps, style))
        out.write_bytes(b"x" * (7_000_000 if style == "diffusion" else 4_000_000))
        return 2 * width, 249

    monkeypatch.setattr(demo, "render", fake_render)
    clip = make_video(tmp_path / "clip.mp4", seconds=16, width=320, height=180, bird=False)
    events = write_events(tmp_path / "events.jsonl",
                          event(frame_seq=61, last_seen="2026-09-30T20:06:27.080-04:00"))
    out = tmp_path / "demo.gif"
    assert demo.main(["--source", str(clip), "--annotated", str(clip), "--events", str(events),
                      "--out", str(out)]) == 0
    assert tried == [(480, 10, "diffusion"), (480, 8, "diffusion"), (400, 8, "diffusion"),
                     (480, 10, "ordered")]
    assert out.stat().st_size == 4_000_000


def test_ordered_style_denoises_and_uses_a_bayer_palette(tmp_path, monkeypatch) -> None:
    cmds = []

    class Done:
        returncode, stderr = 0, ""

    monkeypatch.setattr(demo.subprocess, "run", lambda cmd, **kw: cmds.append(cmd) or Done())
    demo.encode_gif(tmp_path, 10, tmp_path / "x.gif", style="ordered")
    palettegen, paletteuse = (" ".join(c) for c in cmds)
    assert "hqdn3d" in palettegen and "max_colors=128" in palettegen
    assert "hqdn3d" in paletteuse and "dither=bayer" in paletteuse
    assert "diff_mode=rectangle" in paletteuse
    cmds.clear()
    demo.encode_gif(tmp_path, 10, tmp_path / "y.gif")                 # default: unchanged
    assert "sierra2_4a" in " ".join(cmds[1]) and "hqdn3d" not in " ".join(cmds[0])


def test_imageio_ffmpeg_is_a_dev_dependency_only() -> None:
    assert "imageio-ffmpeg" in (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "imageio" not in (ROOT / "requirements.txt").read_text(encoding="utf-8")
