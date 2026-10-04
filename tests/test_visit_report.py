"""scripts/visit_report.py on a fixture: six visits over two days, old
lines (no visit_id, no concurrent_max) and new ones, and one bad line."""
from __future__ import annotations

from scripts import visit_report as vr
from tests.conftest import ROOT

FIXTURE = ROOT / "tests" / "fixtures" / "visit_report_events.jsonl"


def test_summary_of_the_fixture() -> None:
    events, bad = vr.load_events(FIXTURE)
    s = vr.summarise(events)
    assert (len(events), bad) == (6, 1)
    assert s["per_hour"][6] == 1 and s["per_hour"][7] == 4 and s["per_hour"][17] == 1
    assert sum(s["per_hour"]) == 6
    # lengths 30, 60, 120, 30, 10, 20 s
    assert s["median_length_s"] == 30.0
    # three visits overlap on 2026-10-02 07:10:45-07:10:55
    assert s["max_overlap"] == 3 and s["max_concurrent_field"] == 3
    assert s["busiest_hour"] == (7, 4)


def test_report_text(capsys) -> None:
    assert vr.main([str(FIXTURE)]) == 0
    out = capsys.readouterr()
    assert "6 visits" in out.out and "2026-10-01 to 2026-10-02" in out.out
    assert "07:00   4" in out.out
    assert "Median visit length: 30.0 s" in out.out
    assert "Max concurrent birds: 3" in out.out
    assert "Busiest hour: 07:00-08:00 (4 visits)" in out.out
    assert "skipped 1" in out.err


def test_old_lines_alone_have_no_concurrent_max_field(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text("".join(l + "\n" for l in FIXTURE.read_text(encoding="utf-8").splitlines()[:2]),
                    encoding="utf-8")
    s = vr.summarise(vr.load_events(path)[0])
    assert s["max_overlap"] == 1 and s["max_concurrent_field"] is None


def test_missing_or_empty_file(tmp_path, capsys) -> None:
    assert vr.main([str(tmp_path / "none.jsonl")]) == 1
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    assert vr.main([str(tmp_path / "empty.jsonl")]) == 1
