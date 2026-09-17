from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from infx.workflows.validate_reusable_sweep_artifacts import validate_agentic_artifacts


def agentic_result() -> dict:
    return {
        "scenario_type": "agentic-coding",
        "hw": "test-gpu",
        "infmax_model_prefix": "test-model",
        "framework": "sglang",
        "precision": "fp8",
        "tp": 1,
        "conc": 16,
    }


def write_agentic_artifacts(root: Path) -> None:
    point = root / "bmk_agentic_success"
    point.mkdir()
    (point / "result.json").write_text(json.dumps(agentic_result()))
    (root / "agentic_success").mkdir()


@pytest.mark.parametrize("total,raw_present", [(0, False), (349.0, True)])
def test_agentic_reuse_keeps_success_and_excludes_failed_attempt_without_deleting_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    total: float,
    raw_present: bool,
) -> None:
    from infx.workflows.validate_reusable_sweep_artifacts import main

    write_agentic_artifacts(tmp_path)
    failed = {
        **agentic_result(),
        "num_requests_successful": 0.0,
        "num_requests_total": total,
    }
    point = tmp_path / "bmk_agentic_failed"
    point.mkdir()
    payload = json.dumps(failed)
    (point / "result.json").write_text(payload)
    if raw_present:
        (tmp_path / "agentic_failed").mkdir()
    aggregate = tmp_path / "results_bmk"
    aggregate.mkdir()
    aggregate_payload = json.dumps([agentic_result(), failed])
    (aggregate / "results.json").write_text(aggregate_payload)
    monkeypatch.setattr(sys, "argv", ["validate", "--artifacts-dir", str(tmp_path)])

    assert main() == 0
    assert capsys.readouterr().out == (
        "Reusable sweep artifacts validated: "
        "0 fixed-sequence row(s), 1 agentic row(s), 0 eval row(s).\n"
    )
    assert (point / "result.json").read_text() == payload
    assert (aggregate / "results.json").read_text() == aggregate_payload
    assert (tmp_path / "agentic_failed").exists() is raw_present


@pytest.mark.parametrize(
    "counts",
    [
        {"num_requests_successful": 1, "num_requests_total": 349},
        {"num_requests_successful": "0", "num_requests_total": 349},
        {"num_requests_successful": False, "num_requests_total": 349},
        {"num_requests_successful": None, "num_requests_total": 349},
        {"num_requests_successful": -1, "num_requests_total": 349},
        {"num_requests_successful": 0},
        {"num_requests_successful": 0, "num_requests_total": "349"},
        {"num_requests_successful": 0, "num_requests_total": False},
        {"num_requests_successful": 0, "num_requests_total": None},
        {"num_requests_successful": 0, "num_requests_total": -1},
        {"num_requests_successful": 0, "num_requests_total": 0.5},
        {"num_requests_successful": 0, "num_requests_total": float("inf")},
        {"num_requests_successful": 0, "num_requests_total": float("nan")},
    ],
)
def test_agentic_nonfailed_or_unknown_counts_still_require_raw_artifacts(
    tmp_path: Path,
    counts: dict,
) -> None:
    point = tmp_path / "bmk_agentic_unverified"
    point.mkdir()
    (point / "result.json").write_text(json.dumps({**agentic_result(), **counts}))

    assert validate_agentic_artifacts(tmp_path) == [
        "missing raw agentic artifact dir: agentic_unverified"
    ]


@pytest.mark.parametrize(
    "other_payload",
    [
        [],
        None,
        ["unrecognized"],
        [agentic_result()],
    ],
)
def test_failed_agentic_directory_with_other_payload_remains_strict(
    tmp_path: Path,
    other_payload: object,
) -> None:
    point = tmp_path / "bmk_agentic_mixed"
    point.mkdir()
    failed = {
        **agentic_result(),
        "num_requests_successful": 0,
        "num_requests_total": 12,
    }
    (point / "failed.json").write_text(json.dumps(failed))
    (point / "other.json").write_text(json.dumps(other_payload))

    assert validate_agentic_artifacts(tmp_path) == [
        "missing raw agentic artifact dir: agentic_mixed"
    ]


def test_wholly_failed_agentic_sweep_has_no_reusable_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    from infx.workflows.validate_reusable_sweep_artifacts import main

    point = tmp_path / "bmk_agentic_failed"
    point.mkdir()
    failed = {
        **agentic_result(),
        "num_requests_successful": 0,
        "num_requests_total": 12,
    }
    (point / "result.json").write_text(json.dumps(failed))
    monkeypatch.setattr(sys, "argv", ["validate", "--artifacts-dir", str(tmp_path)])

    assert main() == 1
    assert (
        "no reusable benchmark, agentic, or eval result rows found"
        in capsys.readouterr().err
    )
    assert (point / "result.json").is_file()
