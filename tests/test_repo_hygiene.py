"""What a stranger's clone contains: license, dependency pins, ignore rules,
the example ROI, the architecture doc, and no personal data in tracked files."""
from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from tests.conftest import ROOT

# Assembled at runtime so this file does not match its own search. The second
# alternative is a Windows home-directory prefix: the word, then a backslash.
PERSONAL = "|".join(["isa" + "mb", "Use" + "rs" + "\\" * 2, "/ho" + "me/", "@gm" + "ail",
                     "@getty" + "sburg"])


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout


needs_git = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(), reason="not a git checkout")


@needs_git
def test_no_personal_data_in_tracked_files() -> None:
    hits = subprocess.run(["git", "grep", "-n", "-i", "-E", PERSONAL], cwd=ROOT,
                          capture_output=True, text=True)
    assert hits.returncode == 1, hits.stdout  # 1 = no match


@needs_git
def test_roi_example_is_tracked_and_the_personal_roi_is_not() -> None:
    tracked = set(git("ls-files", "data").split())
    assert "data/roi.example.json" in tracked
    assert "data/roi.json" not in tracked


@needs_git
@pytest.mark.parametrize("path", [
    ".venv/x", "yolov8n.pt", "__pycache__/x.pyc", "logs/run_x.jsonl", "data/events.jsonl",
    "data/snapshots/a.jpg", "data/samples/a.mp4", "data/roi.json", "snap_001.jpg",
    ".coverage", ".pytest_cache/x",
])
def test_gitignore(path: str) -> None:
    assert subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT).returncode == 0


def test_license_is_mit_2026() -> None:
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License")
    assert "Copyright (c) 2026 Tianyi Jia" in text


def test_torch_is_pinned_and_dev_tools_are_separate() -> None:
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"^torch==\d+\.\d+\.\d+$", req, re.M)
    assert re.search(r"^torchvision==\d+\.\d+\.\d+$", req, re.M)
    assert "cpu" in req.lower() and "cuda" in req.lower()   # the CPU / CUDA note
    assert "pytest" not in req
    assert "pytest" in (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")


def test_architecture_doc_replaces_the_project_report() -> None:
    assert not (ROOT / "PROJECT_REPORT.md").exists()
    text = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    assert len(text.splitlines()) < 400
    cited = set(re.findall(r"\brun_id `([0-9a-f]{8})`", text))
    assert cited, "no run_id citations"
    for run_id in cited:  # every cited run's log ships with the doc
        assert (ROOT / "docs" / "runs" / f"run_{run_id}.jsonl").is_file(), run_id


def test_synth_video_uses_the_example_roi_on_a_fresh_clone(tmp_path) -> None:
    from scripts.make_synth_video import default_roi_path

    (tmp_path / "roi.example.json").write_text("{}")
    assert default_roi_path(tmp_path) == tmp_path / "roi.example.json"
    (tmp_path / "roi.json").write_text("{}")
    assert default_roi_path(tmp_path) == tmp_path / "roi.json"
