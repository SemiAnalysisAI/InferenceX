"""Tests for the agentic aiperf result validity gate."""

from __future__ import annotations

import json
from pathlib import Path

from utils.agentic.validation.validate_agentic_result import validate_result


def _write_aggregate(tmp_path: Path, aggregate: dict, *, per_run: bool = False) -> Path:
    artifact_dir = tmp_path / "aiperf_artifacts"
    output_dir = artifact_dir / "run_0" if per_run else artifact_dir
    output_dir.mkdir(parents=True)
    with open(output_dir / "profile_export_aiperf.json", "w") as f:
        json.dump(aggregate, f)
    return artifact_dir


def test_passes_when_request_error_rate_is_within_limit(tmp_path: Path):
    artifact_dir = _write_aggregate(
        tmp_path,
        {
            "request_count": {"avg": 90},
            "error_request_count": {"avg": 10},
            "completed_request_count": {"avg": 100},
        },
    )

    assert validate_result(artifact_dir, 0.10) == []


def test_fails_when_request_error_rate_exceeds_limit(tmp_path: Path):
    artifact_dir = _write_aggregate(
        tmp_path,
        {
            "request_count": {"avg": 2},
            "error_request_count": {"avg": 65},
            "completed_request_count": {"avg": 67},
        },
    )

    errors = validate_result(artifact_dir, 0.10)
    assert errors == [
        "aiperf request error rate exceeded the benchmark limit: "
        "65/67 = 97.015% > 10.000%"
    ]


def test_treats_missing_error_count_as_zero(tmp_path: Path):
    artifact_dir = _write_aggregate(
        tmp_path,
        {"request_count": {"avg": 12}},
    )

    assert validate_result(artifact_dir, 0.10) == []


def test_supports_per_run_artifact_layout(tmp_path: Path):
    artifact_dir = _write_aggregate(
        tmp_path,
        {"request_count": {"avg": 12}},
        per_run=True,
    )

    assert validate_result(artifact_dir, 0.10) == []


def test_fails_when_aggregate_is_missing(tmp_path: Path):
    artifact_dir = tmp_path / "aiperf_artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "profile_export.jsonl").write_text('{"partial": true}\n')

    errors = validate_result(artifact_dir, 0.10)

    assert len(errors) == 1
    assert errors[0].endswith("profile_export_aiperf.json not found")


def test_workflow_runs_agentic_validity_gate():
    workflow = (
        Path(__file__).parents[3] / ".github/workflows/benchmark-tmpl.yml"
    ).read_text()

    assert "python3 -m utils.agentic.validation.validate_agentic_result" in workflow
    assert '--failed-request-threshold "$AIPERF_FAILED_REQUEST_THRESHOLD"' in workflow
