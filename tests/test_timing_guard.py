"""The session-level timing guards in conftest.py."""
from __future__ import annotations

from tests.conftest import SLEEP_LIMIT_S, WALL_TIME_LIMIT_S, guard_sleep, over_budget


def test_limits() -> None:
    assert WALL_TIME_LIMIT_S == 5.0
    assert SLEEP_LIMIT_S == 0.1


def test_over_budget_adds_setup_call_and_teardown() -> None:
    durations = {
        "t::fast": {"setup": 1.0, "call": 3.0, "teardown": 0.5},
        "t::slow_setup": {"setup": 4.0, "call": 1.5, "teardown": 0.0},
        "t::slow_call": {"call": 6.0},
    }
    assert over_budget(durations, 5.0) == [("t::slow_call", 6.0), ("t::slow_setup", 5.5)]


def test_long_real_sleeps_are_recorded() -> None:
    requested: list[float] = []
    violations: list[float] = []
    sleep = guard_sleep(requested.append, 0.1, violations)
    for s in (0.0, 0.05, 0.1, 0.25, 3600):
        sleep(s)
    assert violations == [0.25, 3600]
    assert requested == [0.0, 0.05, 0.1, 0.25, 3600]  # still delegated
