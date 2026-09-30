"""Lint the CI workflow and pytest config (the CI itself runs on GitHub)."""
from __future__ import annotations

import tomllib

import yaml

from tests.conftest import ROOT


def load_workflow() -> dict:
    return yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))


def test_ci_runs_fast_suite_on_ubuntu_and_windows_with_python_312() -> None:
    jobs = load_workflow()["jobs"]
    runners = {job["runs-on"] for job in jobs.values()}
    # Pinned images: a runner-image migration cannot change CI without a commit.
    assert runners == {"ubuntu-24.04", "windows-2025"}
    for name, job in jobs.items():
        steps = job["steps"]
        setup = [s for s in steps if str(s.get("uses", "")).startswith("actions/setup-python")]
        assert setup and str(setup[0]["with"]["python-version"]) == "3.12", name
        runs = "\n".join(s.get("run", "") for s in steps)
        assert "requirements.txt" in runs and "requirements-dev.txt" in runs, name
        assert 'pytest -m "not slow"' in runs and "--cov" in runs, name


def test_ci_exposes_coverage_as_a_job_output() -> None:
    jobs = load_workflow()["jobs"]
    for name, job in jobs.items():
        assert "coverage" in job.get("outputs", {}), name


def test_pyproject_pytest_config() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pytest_cfg = config["tool"]["pytest"]["ini_options"]
    assert pytest_cfg["pythonpath"] == ["."]
    assert pytest_cfg["testpaths"] == ["tests"]
    assert any(m.startswith("slow:") for m in pytest_cfg["markers"])
    assert "--strict-markers" in pytest_cfg["addopts"]
    assert "subprocess" in config["tool"]["coverage"]["run"]["patch"]


# First major of each action whose action.yml declares `using: node24`
# (checked in each repo on 2026-09-30); older majors run on the deprecated Node 20.
NODE24_MAJORS = {"actions/checkout": 5, "actions/setup-python": 6, "actions/upload-artifact": 6}


def test_actions_run_on_node_24() -> None:
    uses = [step["uses"] for job in load_workflow()["jobs"].values()
            for step in job["steps"] if "uses" in step]
    assert {u.split("@")[0] for u in uses} == set(NODE24_MAJORS)
    for use in uses:
        name, version = use.split("@")
        assert int(version.lstrip("v").split(".")[0]) >= NODE24_MAJORS[name], use
