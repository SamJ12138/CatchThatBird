"""scripts/corpus_eval.py: visits in events.jsonl scored against a clip's
hand-written ground truth, and the YOLO rate read from the run log."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from scripts import corpus_eval as ce

T0 = datetime.fromisoformat("2026-10-04T12:00:00.000+00:00")
FPS = 30.0


def visit(start_s: float, last_s: float, box, frames: int = 3, **extra) -> dict:
    """An events.jsonl line for a visit opened `start_s` into a 30 fps clip."""
    seq = round(start_s * FPS) + 1
    ts = T0 + timedelta(seconds=start_s)
    return {"ts": ts.isoformat(timespec="milliseconds"), "run_id": "abcd1234", "frame_seq": seq,
            "class": "bird", "confidence": 0.8, "bbox_xywh": list(box),
            "snapshot_crop": None, "snapshot_full": None,
            "last_seen": (ts + timedelta(seconds=last_s - start_s)).isoformat(timespec="milliseconds"),
            "visit_frames": frames, "truncated": False, "recovered": False, **extra}


def bird(id_, first, last, box, positions=None, uncertain=False) -> dict:
    b = {"id": id_, "first_s": first, "last_s": last, "landing_s": first, "box_at_landing": list(box)}
    if positions:
        b["positions"] = positions
    if uncertain:
        b["uncertain"] = True
    return b


def truth(*birds, seconds=20) -> dict:
    counts = [sum(b["first_s"] <= s <= b["last_s"] for b in birds) for s in range(seconds)]
    return {"clip": "fixture", "width": 1920, "height": 1080, "fps": FPS, "frames": seconds * 30,
            "duration_s": float(seconds), "birds": list(birds), "count_per_second": counts}


LEFT = (100, 500, 200, 150)
RIGHT = (1500, 500, 200, 150)


def test_one_bird_one_visit() -> None:
    r = ce.evaluate([visit(3.1, 12.0, (110, 505, 190, 140), frames=9)], truth(bird("A", 3, 12, LEFT)))
    assert (r["real"], r["logged"], r["missed"], r["split"], r["merged"], r["false"]) == (1, 1, 0, 0, 0, 0)
    assert r["visit_frames"] == [9]
    assert r["open_delay_s"] == [pytest.approx(0.1, abs=0.01)]


def test_a_bird_with_two_visits_is_one_split() -> None:
    events = [visit(3.0, 6.0, LEFT), visit(8.0, 12.0, LEFT)]
    r = ce.evaluate(events, truth(bird("A", 3, 12, LEFT)))
    assert (r["missed"], r["split"], r["false"]) == (0, 1, 0)


def test_a_bird_with_no_visit_is_missed() -> None:
    r = ce.evaluate([visit(3.0, 12.0, LEFT)], truth(bird("A", 3, 12, LEFT), bird("B", 5, 9, RIGHT)))
    assert (r["missed"], r["merged"], r["false"]) == (1, 0, 0)


def test_a_bird_inside_another_birds_visit_is_merged() -> None:
    """Two birds side by side, one visit: the second bird is merged into it
    (it was within the logger's reach of the visit's box), not missed."""
    beside = (330, 500, 200, 150)
    r = ce.evaluate([visit(3.0, 12.0, LEFT)], truth(bird("A", 3, 12, LEFT), bird("B", 4, 10, beside)))
    assert (r["real"], r["logged"], r["missed"], r["merged"]) == (2, 1, 0, 1)


def test_visits_with_no_bird_are_false() -> None:
    events = [visit(15.5, 18.0, LEFT),             # after the bird left
              visit(5.0, 8.0, (900, 100, 80, 60))]  # while it was there, far from it
    r = ce.evaluate(events, truth(bird("A", 3, 12, LEFT)))
    assert (r["false"], r["missed"]) == (2, 1)


def test_one_second_of_slack_for_the_one_fps_ground_truth() -> None:
    """The ground truth samples once a second: a bird seen first at 3 s may
    have arrived at 2.1 s, and one last seen at 12 s may leave at 12.9 s."""
    r = ce.evaluate([visit(2.2, 12.8, LEFT)], truth(bird("A", 3, 12, LEFT)))
    assert (r["missed"], r["false"]) == (0, 0)


def test_positions_say_where_a_moving_bird_is() -> None:
    walker = bird("A", 0, 19, LEFT, positions=[
        {"from_s": 0, "to_s": 9, "box": list(LEFT)}, {"from_s": 10, "to_s": 19, "box": list(RIGHT)}])
    r = ce.evaluate([visit(0.5, 9.0, LEFT), visit(12.0, 19.0, RIGHT)], truth(walker))
    assert (r["false"], r["split"]) == (0, 1)


def test_uncertain_birds_count_neither_way() -> None:
    maybe = bird("H", 5, 5, RIGHT, uncertain=True)
    r = ce.evaluate([], truth(maybe))
    assert (r["real"], r["missed"]) == (0, 0)
    r = ce.evaluate([visit(5.0, 5.0, RIGHT)], truth(maybe))
    assert (r["false"], r["on_uncertain"]) == (0, 1)


def test_no_bird_clip_scores_every_visit_false() -> None:
    r = ce.evaluate([visit(2.0, 4.0, LEFT)], truth())
    assert (r["real"], r["logged"], r["false"]) == (0, 1, 1)


def test_events_load_with_and_without_the_multi_bird_fields(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    lines = [visit(1.0, 2.0, LEFT), visit(3.0, 4.0, LEFT, visit_id="abcd1234-2", concurrent_max=2)]
    path.write_text("".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")
    assert len(ce.load_events(path)) == 2
    assert ce.load_events(tmp_path / "missing.jsonl") == []


def log_line(stage, event, seq, **context):
    return {"ts": "x", "run_id": "abcd1234", "stage": stage, "event": event, "duration_ms": 40.0,
            "frame_seq": seq, "error_type": None, "error_message": None, "context": context}


def test_yolo_rate_counts_calls_per_second_of_clip() -> None:
    lines = [log_line("yolo_infer", "start", 61), log_line("yolo_infer", "success", 61)]
    lines += [log_line("yolo_infer", "success", s) for s in (91, 92, 95, 100, 115)]   # second 3
    lines += [log_line("yolo_infer", "success", 121)]                                 # second 4
    rate = ce.yolo_rate(lines, FPS)
    assert rate == {"calls": 7, "max_per_s": 5}


def test_frames_skipped_come_from_the_last_render_summary() -> None:
    lines = [log_line("render", "success", 300, summary=True, records=300, skipped_total=4),
             log_line("render", "success", 450, summary=True, records=140, skipped_total=10)]
    assert ce.frames_seen(lines) == {"processed": 440, "skipped": 10}


def test_kept_runs_are_scored_again_against_the_current_ground_truth(tmp_path) -> None:
    """--keep-logs keeps run_<id>.jsonl and events_<id>.jsonl; --rescore reads
    the clip from the run's --source and scores the kept events again."""
    start = log_line("run", "start", None, argv=["--source", "./data/samples/corpus/parked_car_rain/clip.mp4",
                                                 "--headless", "--yes"])
    start["run_id"] = "feed0001"
    yolo = dict(log_line("yolo_infer", "success", 31), run_id="feed0001")
    (tmp_path / "run_feed0001.jsonl").write_text(
        json.dumps(start) + "\n" + json.dumps(yolo) + "\n", encoding="utf-8")
    (tmp_path / "events_feed0001.jsonl").write_text(json.dumps(visit(2.0, 3.0, LEFT)) + "\n",
                                                    encoding="utf-8")
    (result,) = ce.rescore(tmp_path)["parked_car_rain"]
    assert (result["run_id"], result["real"], result["false"], result["yolo_calls"]) == ("feed0001", 0, 1, 1)
