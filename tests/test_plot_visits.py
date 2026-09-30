"""scripts/plot_visits.py: visits per hour of day from events.jsonl."""
from __future__ import annotations

import cv2

from scripts import plot_visits
from tests.conftest import ROOT

FIXTURE = ROOT / "tests" / "fixtures" / "events_5.jsonl"


def test_counts_visits_by_local_hour_of_day() -> None:
    events, bad = plot_visits.load_events(FIXTURE)
    assert (len(events), bad) == (5, 0)
    counts = plot_visits.visits_per_hour(events)
    assert len(counts) == 24
    # 06:12, 06:47, 07:03, 13:30, 18:59:59.999 in the recording's own offset (+02:00)
    assert {h: n for h, n in enumerate(counts) if n} == {6: 2, 7: 1, 13: 1, 18: 1}


def test_writes_a_png(tmp_path, capsys) -> None:
    out = tmp_path / "visits.png"

    assert plot_visits.main([str(FIXTURE), "--out", str(out), "--title", "Test fixture"]) == 0

    image = cv2.imread(str(out))
    assert image is not None and image.shape[0] > 200 and image.shape[1] > 400
    assert "5 visits" in capsys.readouterr().out


def test_malformed_lines_are_skipped_and_counted(tmp_path, capsys) -> None:
    events = tmp_path / "events.jsonl"
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    events.write_text("\n".join([lines[0], "{broken", '{"no_ts": 1}', lines[1], ""]), encoding="utf-8")

    assert plot_visits.main([str(events), "--out", str(tmp_path / "v.png")]) == 0

    captured = capsys.readouterr()
    assert "2 visits" in captured.out
    assert "skipped 2 malformed line(s)" in captured.err


def test_missing_or_empty_events_file_exits_1(tmp_path, capsys) -> None:
    assert plot_visits.main([str(tmp_path / "none.jsonl"), "--out", str(tmp_path / "v.png")]) == 1
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    assert plot_visits.main([str(tmp_path / "empty.jsonl"), "--out", str(tmp_path / "v.png")]) == 1
    assert not (tmp_path / "v.png").exists()
    assert "No visits" in capsys.readouterr().out


def test_matplotlib_is_a_dev_dependency_only() -> None:
    assert "matplotlib" in (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "matplotlib" not in (ROOT / "requirements.txt").read_text(encoding="utf-8")


def test_readme_embeds_the_plot_with_an_honest_caption() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    output = readme.split("## Output", 1)[1].split("\n## ", 1)[0]
    assert "![" in output and "(docs/visits.png)" in output
    assert "tests/fixtures/events_5.jsonl" in output and "not real observations" in output
    assert "python scripts/plot_visits.py" in output
    assert (ROOT / "docs" / "visits.png").is_file()
