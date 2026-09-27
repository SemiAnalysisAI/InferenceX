"""Behavioral tests for the review-time dashboard frontier counter."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from infx.workflows.pareto_coverage import assess_coverage

ROOT = Path(__file__).resolve().parents[3]


def assess(curves):
    result = subprocess.run(
        [sys.executable, "-m", "infx.workflows.pareto_coverage"],
        input=json.dumps(curves),
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip())
    return json.loads(result.stdout)


def curve(points, key="model/scenario/hw/precision/run/p90/image"):
    return {"key": key, "points": points}


@pytest.mark.parametrize("count,status", [(0, "WARN"), (1, "WARN"), (4, "WARN"), (5, "PASS"), (6, "PASS")])
def test_five_point_boundary(count, status):
    result = assess([curve([{"x": i, "y": i * 100} for i in range(1, count + 1)])])[0]
    assert result["frontierPoints"] == count
    assert result["status"] == status


def test_counts_frontier_not_raw_jobs_and_keeps_input_coordinates():
    points = [
        {"x": 3, "y": 300},
        {"x": 1, "y": 100},
        {"x": 2, "y": 90},  # dominated by lower-latency point
        {"x": 3, "y": 200},  # same-latency lower throughput
        {"x": 3, "y": 300},  # exact duplicate
        {"x": 4, "y": 400},
    ]
    result = assess([curve(points)])[0]
    assert result["measuredPoints"] == 6
    assert result["frontierPoints"] == 3
    assert result["status"] == "WARN"
    assert result["frontier"] == [{"x": 1, "y": 100}, {"x": 3, "y": 300}, {"x": 4, "y": 400}]


def test_equal_throughput_plateau_matches_dashboard_not_strict_dominance():
    points = [{"x": i, "y": 100} for i in range(1, 6)]
    result = assess([curve(points + [points[0]])])[0]
    assert result["frontier"] == points
    assert result["status"] == "PASS"


def test_curves_are_not_pooled():
    points = [{"x": i, "y": i * 100} for i in range(1, 4)]
    result = assess([curve(points, "model-a"), curve(points, "model-b")])
    assert [c["frontierPoints"] for c in result] == [3, 3]
    assert [c["status"] for c in result] == ["WARN", "WARN"]


@pytest.mark.parametrize("bad", [
    {"x": 0, "y": 100}, {"x": -1, "y": 100}, {"x": None, "y": 100},
    {"x": "1", "y": 100}, {"x": 1, "y": None}, {"x": 1, "y": 0},
    {"x": 1, "y": -1}, {"x": 1, "y": "100"}, {}, None,
    {"x": True, "y": 100}, {"x": 1, "y": False},
    {"x": float("inf"), "y": 1}, {"x": 1, "y": float("nan")},
])
def test_invalid_measurements_warn_even_when_five_good_points_exist(bad):
    points = [{"x": i, "y": i * 100} for i in range(1, 6)]
    result = assess([curve(points + [bad])])[0]
    assert result["frontierPoints"] == 5
    assert result["invalidPoints"] == 1
    assert result["status"] == "WARN"


def test_canonical_intersection_is_applied_after_full_e2el_frontier():
    points = [
        {"x": 1, "y": 500, "isOnNormalizedInteractivityFrontier": False},
        {"x": 2, "y": 100, "isOnNormalizedInteractivityFrontier": True},
        {"x": 3, "y": 600, "isOnNormalizedInteractivityFrontier": True},
    ]
    result = assess([curve(points)])[0]
    assert result["canonicalRestriction"] is True
    assert result["frontier"] == [points[2]]


def test_all_false_canonical_flags_cannot_fall_back_to_unrestricted_frontier():
    points = [{"x": i, "y": i * 100, "isOnNormalizedInteractivityFrontier": False}
              for i in range(1, 6)]
    result = assess([curve(points)])[0]
    assert result["frontierPoints"] == 0
    assert result["status"] == "WARN"


def test_unstamped_points_have_no_unconditional_agentic_restriction():
    result = assess([curve([{"x": i, "y": i * 100} for i in range(1, 6)])])[0]
    assert result["canonicalRestriction"] is False
    assert result["status"] == "PASS"


@pytest.mark.parametrize("curves", [
    [], {}, [curve([], "")], [{"key": "a"}], [curve([], "a"), curve([], "a")],
    [None], [curve([], [])],
])
def test_malformed_or_empty_coverage_is_not_a_pass(curves):
    with pytest.raises(ValueError):
        assess(curves)


def test_counter_does_not_reorder_input_and_requires_literal_canonical_true():
    points = [
        {"x": 3, "y": 300, "isOnNormalizedInteractivityFrontier": True},
        {"x": 1, "y": 100, "isOnNormalizedInteractivityFrontier": 1},
        {"x": 2, "y": 200, "isOnNormalizedInteractivityFrontier": None},
    ]
    result = assess_coverage([curve(points)])[0]
    assert [point["x"] for point in points] == [3, 1, 2]
    assert result["frontier"] == [points[0]]
