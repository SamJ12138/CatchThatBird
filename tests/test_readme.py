"""README.md stays in step with the code: section order, the quickstart
commands, every config key, every events.jsonl field and every CLI flag."""
from __future__ import annotations

import re

import pytest
import yaml

import main
from tests.conftest import ROOT
from tests.test_event_logger import EVENT_KEYS

SECTIONS = [
    "Quickstart",
    "Running with a real camera",
    "How it works",
    "Configuration",
    "Output",
    "Development",
    "Status and roadmap",
    "License",
]
QUICKSTART = [
    "git clone",
    "-m venv .venv",
    "pip install -r requirements.txt",
    "python scripts/make_synth_video.py",
    "python main.py --source data/samples/synth_blob.mp4 --headless --no-pace --yes",
    "python scripts/failure_report.py --latest",
]


@pytest.fixture(scope="module")
def readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def section(text: str, title: str) -> str:
    match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match, f"no '## {title}' section"
    return match.group(1)


def test_sections_in_order(readme: str) -> None:
    headings = re.findall(r"^## (.+)$", readme, re.M)
    assert headings == SECTIONS
    intro = readme.split("\n## ", 1)[0]
    assert "parked car" in intro and "CPU" in intro


def test_quickstart_commands_in_order(readme: str) -> None:
    quick = section(readme, "Quickstart")
    positions = [quick.find(cmd) for cmd in QUICKSTART]
    assert -1 not in positions, dict(zip(QUICKSTART, positions))
    assert positions == sorted(positions)
    assert "Activate.ps1" in quick and "source .venv/bin/activate" in quick  # both shells


def test_every_config_key_is_documented(readme: str) -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    table = section(readme, "Configuration")
    for group, values in config.items():
        for key in values:
            assert f"`{group}.{key}`" in table, f"{group}.{key}"


def test_every_event_field_is_documented(readme: str) -> None:
    output = section(readme, "Output")
    for key in EVENT_KEYS:
        assert f"`{key}`" in output, key


def test_every_cli_flag_is_documented(readme: str, capsys) -> None:
    with pytest.raises(SystemExit):
        main.parse_args(["--help"])
    flags = set(re.findall(r"--[a-z][a-z-]+", capsys.readouterr().out)) - {"--help"}
    assert len(flags) >= 12
    documented = set(re.findall(r"--[a-z][a-z-]+", readme))
    assert flags <= documented, flags - documented


def test_no_emoji(readme: str) -> None:
    assert not re.search("[\U0001F300-\U0001FAFF☀-➿]", readme)
