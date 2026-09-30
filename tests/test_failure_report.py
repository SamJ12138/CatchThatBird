"""scripts/failure_report.py on a hand-written 20-line log covering every
event type (start / success / fail / skip), per-frame summary lines, a
swallowed sub-error and skip reasons."""
from __future__ import annotations

import json
import os
from pathlib import Path

from scripts import failure_report
from tests.conftest import ROOT

FIXTURE = ROOT / "tests" / "fixtures" / "run_fixture.jsonl"

EXPECTED = """Files: run_fixture.jsonl
Runs: 1  Lines: 20  Unparseable lines: 0
  fixture1: 2026-09-30T12:00:00.000+00:00 .. 2026-09-30T12:00:20.000+00:00  run success, exit_code=0, 20.0s

== Errors: stage x error_type (fail/skip lines carrying an error_type) ==
stage             input_invalid  external_api  parse  timeout  hardware  unknown  total
----------------  -------------  ------------  -----  -------  --------  -------  -----
device_list                   .             1      .        .         .        .      1
capture_open                  .             .      .        .         1        .      1
capture_read                  .             .      .        .         1        .      1
first_frame_wait              .             .      .        1         .        .      1
TOTAL                         .             1      .        1         2        .      4

== Skips: stage x reason ==
stage                        reason  count
-------------  --------------------  -----
roi_load       headless_whole_frame      1
gate_check                  cadence    230
gate_check                   warmup     60
detection_map            yolo_empty      1

== Stages: totals and duration_ms ==
stage             success  fail  skip    p50_ms    p95_ms
----------------  -------  ----  ----  --------  --------
run                     1     0     0  20000.00  20000.00
config_load             1     0     0     12.50     12.50
device_list             1     1     0     30.00     30.00
capture_open            1     1     0    250.00    250.00
roi_load                0     0     1      1.00      1.00
capture_read *        298     2     0     ~1.50     ~3.00
gate_check *           10     0   290         -         -
mog2_apply *          300     0     0     ~4.00     ~6.00
yolo_infer              1     0     0     60.00     60.00
detection_map           0     0     1      0.20      0.20
first_frame_wait        0     1     0   5000.00   5000.00
TOTAL                 613     5   292

* per-frame stage: totals from 300-frame summary lines; ~ = approximate percentile (median of window p50s / max of window p95s).
  A stage that swallows a sub-error logs a fail line and then its normal terminal line, so fail + success can exceed the number of stage runs.
"""


def report(capsys, *argv: str) -> tuple[int, list[str], str]:
    code = failure_report.main(list(argv))
    captured = capsys.readouterr()
    return code, [line.rstrip() for line in captured.out.splitlines()], captured.err


def test_fixture_has_20_lines_and_every_event_type() -> None:
    lines = [json.loads(l) for l in FIXTURE.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 20
    assert {l["event"] for l in lines} == {"start", "success", "fail", "skip"}


def test_report_table(capsys) -> None:
    code, out, err = report(capsys, str(FIXTURE))
    assert code == 0 and err == ""
    assert out == EXPECTED.splitlines()


def test_default_reads_every_log_in_log_dir_and_latest_picks_the_newest(tmp_path, capsys) -> None:
    old, new = tmp_path / "run_old.jsonl", tmp_path / "run_new.jsonl"
    old.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    new.write_text(FIXTURE.read_text(encoding="utf-8").replace("fixture1", "fixture2"),
                   encoding="utf-8")
    os.utime(old, (1_700_000_000, 1_700_000_000))
    os.utime(new, (1_800_000_000, 1_800_000_000))

    _, out, _ = report(capsys, "--log-dir", str(tmp_path))
    assert out[0] == "Files: run_old.jsonl, run_new.jsonl"
    assert out[1] == "Runs: 2  Lines: 40  Unparseable lines: 0"

    _, out, _ = report(capsys, "--log-dir", str(tmp_path), "--latest")
    assert out[0] == "Files: run_new.jsonl"
    assert out[2].startswith("  fixture2:")


def test_no_logs(tmp_path, capsys) -> None:
    code, out, _ = report(capsys, "--log-dir", str(tmp_path))
    assert (code, out) == (1, ["No log files found."])


def test_bad_line_is_counted_and_a_crashed_run_is_flagged(tmp_path, capsys) -> None:
    log = tmp_path / "run_crash.jsonl"
    head = FIXTURE.read_text(encoding="utf-8").splitlines()[:3]
    log.write_text("\n".join([*head, "{truncated", ""]), encoding="utf-8")

    code, out, err = report(capsys, str(log))

    assert code == 0
    assert "run_crash.jsonl:4 is not valid JSON" in err
    assert out[1] == "Runs: 1  Lines: 3  Unparseable lines: 1"
    assert out[2].endswith("no terminal run line (crashed or still running)")
    assert "(none)" in out  # no error_type lines
