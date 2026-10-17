"""Synthetic frontend/backend parity; these scores are not measured evidence."""

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from evaluator.mvp_deployment_frontier import pareto_frontier, select_frontiers

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures/deployment-frontier.json").read_text()
)


@pytest.mark.parametrize("case", FIXTURES["cases"], ids=lambda case: case["name"])
def test_shared_dominance_cases(case):
    result = pareto_frontier(case["points"], case["xBetter"], case["yBetter"])
    assert [point["id"] for point in result] == case["expected"]


def test_nonfinite_missing_and_boolean_axes_do_not_become_measurements():
    invalid = [None, float("nan"), float("inf"), "12", True]
    points = [{"id": "measured", "x": 10, "y": 5}]
    points += [
        {"id": str(index), "x": value, "y": 100} for index, value in enumerate(invalid)
    ]
    assert pareto_frontier(points, "lower", "higher") == [
        {"id": "measured", "x": 10, "y": 5}
    ]
    assert pareto_frontier([], "lower", "higher") == []


def record(identity, **overrides):
    return {
        "id": identity,
        "hardwareKey": "h200",
        "cohortKey": "matched-workload",
        "deploymentKey": identity,
        "queueingStatus": "unqueued",
        "x": 150,
        "y": 6,
        **overrides,
    }


def test_cohorts_health_queue_and_quality_filter_before_dominance():
    four = assessed_record("four-gpu")
    other_evaluator = assessed_record("other-evaluator", x=2, y=100)
    other_evaluator["quality"]["metrics"]["audio_content"]["evaluatorVersion"] = "v2"
    points = [
        four,
        assessed_record("eight-gpu", x=80, y=5),
        assessed_record("dominated", x=160, y=5),
        assessed_record("other-workload", x=1, y=100, cohortKey="other"),
        other_evaluator,
        assessed_record(
            "failed-hardware", x=1, y=1000, hardwareHealth={"status": "fail"}
        ),
        assessed_record("queue", x=1, y=1000, queueingStatus="queueing"),
        assessed_record("unknown-capacity", x=1, y=1000, queueingStatus="unknown"),
        record("unjudged", x=1, y=1000),
        assessed_record("unknown-workload-a", cohortKey=None),
        assessed_record("unknown-workload-b", cohortKey=None, x=1, y=100),
    ]
    result = select_frontiers(points, "lower", "higher", quality_required=True)
    assert [point["id"] for point in result["plotted"]] == [
        "four-gpu",
        "eight-gpu",
        "other-workload",
        "other-evaluator",
        "unknown-workload-a",
        "unknown-workload-b",
    ]
    assert [
        [point["id"] for point in line] for line in result["frontiers"].values()
    ] == [
        ["eight-gpu", "four-gpu"],
        ["other-workload"],
        ["other-evaluator"],
        ["unknown-workload-a"],
        ["unknown-workload-b"],
    ]
    assert result["excluded"] == {
        "failed-hardware": "hardware_health_failed",
        "queue": "queueing",
        "unknown-capacity": "unknown_capacity",
        "unjudged": "quality_unqualified",
    }
    result = select_frontiers(
        points, "lower", "higher", quality_required=True, optimal=False
    )
    assert [point["id"] for point in result["plotted"]][:3] == [
        "four-gpu",
        "eight-gpu",
        "dominated",
    ]
    assert not result["plotted"][2]["optimal"]
    assert len(result["plotted"]) == 7


def test_repeat_observations_of_one_deployment_are_unconnected():
    result = select_frontiers(
        [
            record("c1", deploymentKey="same"),
            record("c2", deploymentKey="same", x=160, y=7),
        ],
        "lower",
        "higher",
    )
    assert len(result["plotted"]) == 2
    assert all(len(line) == 1 for line in result["frontiers"].values())
    assert result["multiLayout"] is False


def test_offline_cli_writes_selected_identities_without_rewriting_inputs(tmp_path):
    source = tmp_path / "records.json"
    source.write_text(json.dumps({"points": [record("fast", x=80), record("slow")]}))
    original = source.read_bytes()
    output = tmp_path / "frontier.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluator.mvp_deployment_frontier",
            str(source),
            "--output",
            str(output),
            "--x-better",
            "lower",
            "--y-better",
            "higher",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    result = json.loads(output.read_text())
    assert [point["id"] for point in result["plotted"]] == ["fast"]
    assert source.read_bytes() == original


def assessed_record(identity="qualified", **overrides):
    metric = {
        "value": 4,
        "status": "pass",
        "direction": "higher",
        "evaluatorId": "fixture-human",
        "evaluatorVersion": "v1",
        "evaluatorSha256": "a" * 64,
        "samples": 20,
        "total": 20,
        "calibration": {
            "status": "calibrated",
            "cohortId": "pilot-v1",
            "threshold": 3,
            "provenance": "fixture://frozen-rule",
            "frozenAt": "2026-09-01T00:00:00Z",
        },
    }
    return record(
        identity,
        samples=20,
        quality={
            "scale": "ordinal_0_to_4",
            "contractId": "fixture-contract",
            "contractSha256": "b" * 64,
            "rubricVersion": "rubric-v1",
            "rubricSha256": "c" * 64,
            "metrics": {
                name: deepcopy(metric)
                for name in (
                    "prompt_adherence",
                    "visual_fidelity",
                    "temporal_consistency",
                    "motion_plausibility",
                    "audio_quality",
                    "audio_content",
                    "av_sync",
                )
            },
        },
        **overrides,
    )


def test_complete_original_rubric_accepts_a_calibrated_zero_without_coercion():
    point = assessed_record()
    metric = point["quality"]["metrics"]["prompt_adherence"]
    metric["value"] = metric["calibration"]["threshold"] = 0
    result = select_frontiers([point], "lower", "higher", quality_required=True)
    assert [p["id"] for p in result["plotted"]] == ["qualified"]
    assert result["plotted"][0]["quality"]["metrics"]["prompt_adherence"]["value"] == 0


@pytest.mark.parametrize(
    "defect",
    [
        "legacy_scale",
        "missing_scale",
        "missing_dimension",
        "failed_dimension",
        "coverage",
        "uncalibrated",
    ],
)
def test_opaque_claim_cannot_admit_incomplete_or_rejected_quality(defect):
    point = assessed_record(qualifiedQualityCohort="producer-claimed-pass")
    quality = point["quality"]
    if defect == "legacy_scale":
        quality["scale"] = "ordinal_1_to_5"
    elif defect == "missing_scale":
        del quality["scale"]
    elif defect == "missing_dimension":
        del quality["metrics"]["audio_content"]
    elif defect == "failed_dimension":
        quality["metrics"]["audio_content"]["status"] = "fail"
    elif defect == "coverage":
        quality["metrics"]["audio_content"]["samples"] = 19
    else:
        quality["metrics"]["audio_content"]["calibration"]["status"] = "uncalibrated"
    result = select_frontiers([point], "lower", "higher", quality_required=True)
    assert result["plotted"] == []
    assert result["excluded"] == {"qualified": "quality_unqualified"}


def test_reader_threshold_only_tightens_the_selected_dimension():
    point = assessed_record()
    audio = point["quality"]["metrics"]["audio_content"]
    audio["value"] = audio["calibration"]["threshold"] = 1
    result = select_frontiers(
        [point],
        "lower",
        "higher",
        quality_required=True,
        quality_metric="prompt_adherence",
        quality_threshold=4,
    )
    assert [p["id"] for p in result["plotted"]] == ["qualified"]
    result = select_frontiers(
        [point],
        "lower",
        "higher",
        quality_required=True,
        quality_metric="audio_content",
        quality_threshold=2,
    )
    assert result["plotted"] == []
    audio["value"] = 0
    result = select_frontiers(
        [point],
        "lower",
        "higher",
        quality_required=True,
        quality_metric="audio_content",
        quality_threshold=0,
    )
    assert result["plotted"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("evaluatorVersion", "v2"),
        ("calibration.threshold", 2),
        ("calibration.provenance", "fixture://another-rule"),
    ],
)
def test_unselected_dimensions_keep_incompatible_quality_cohorts_separate(field, value):
    first = assessed_record("original")
    second = assessed_record("other-rule", x=10, y=100)
    metric = second["quality"]["metrics"]["av_sync"]
    if field.startswith("calibration."):
        metric["calibration"][field.split(".")[1]] = value
    else:
        metric[field] = value
    result = select_frontiers([first, second], "lower", "higher", quality_required=True)
    assert [p["id"] for p in result["plotted"]] == ["original", "other-rule"]
    assert [len(line) for line in result["frontiers"].values()] == [1, 1]
    assert (
        result["plotted"][0]["qualifiedQualityCohort"]
        != result["plotted"][1]["qualifiedQualityCohort"]
    )


def test_opaque_quality_claim_alone_never_qualifies():
    result = select_frontiers(
        [record("claim", qualifiedQualityCohort="pass")],
        "lower",
        "higher",
        quality_required=True,
    )
    assert result["excluded"] == {"claim": "quality_unqualified"}
