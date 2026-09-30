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
    assert runners == {"ubuntu-latest", "windows-latest"}
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
