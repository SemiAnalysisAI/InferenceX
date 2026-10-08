"""Tests for batched eval score and manifest validation."""

import json
import sys
from pathlib import Path

from infx.evals.validate_scores import main as validate_scores_main
from infx.evals.validate_scores import validate_batch_manifest


def test_batched_eval_requires_a_valid_manifest(tmp_path: Path) -> None:
    result_path = tmp_path / "results_test_conc4.json"
    result_path.write_text('{"lm_eval_version":"0.4.0"}')

    errors = validate_batch_manifest(
        str(tmp_path / "meta_env.json"),
        [str(result_path)],
    )

    assert any("unavailable or invalid" in error for error in errors)


def test_validate_scores_fails_when_expected_batch_metadata_is_unreadable(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    meta_path = tmp_path / "meta_env.json"
    meta_path.write_text("{invalid")
    result_path = tmp_path / "results_test.json"
    result_path.write_text(
        json.dumps({
            "results": {
                "gsm8k": {
                    "exact_match,strict-match": 1.0,
                },
            },
        })
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_scores.py",
            "--meta-env",
            str(meta_path),
            "--results-glob",
            str(result_path),
            "--expected-concs",
            "1 4 8",
        ],
    )

    assert validate_scores_main() == 1
    captured = capsys.readouterr()
    assert "unavailable or invalid" in captured.err


def test_workflow_concurrencies_are_independent_of_eval_metadata(
    tmp_path: Path,
) -> None:
    meta_path = tmp_path / "meta_env.json"
    meta_path.write_text(json.dumps({
        "eval_concs": [8],
        "completed_eval_concs": [8],
        "failed_eval_concs": [],
    }))
    result_path = tmp_path / "results_test_conc8.json"
    result_path.write_text('{"results": {}}')

    errors = validate_batch_manifest(
        str(meta_path),
        [str(result_path)],
        expected_concs=[1, 4, 8],
    )

    assert "batched eval metadata does not match workflow concurrencies" in errors
    assert any("missing completed concurrency: 1, 4" in error for error in errors)
    assert any("missing result files for concurrency: 1, 4" in error for error in errors)


def test_validate_scores_checks_threshold_for_every_concurrency(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "meta_env.json").write_text(json.dumps({
        "eval_concs": [1, 4],
        "completed_eval_concs": [1, 4],
        "failed_eval_concs": [],
    }))
    for conc, score in ((1, 0.9), (4, 0.8)):
        (tmp_path / f"results_test_conc{conc}.json").write_text(json.dumps({
            "results": {
                "gsm8k": {
                    "exact_match,strict-match": score,
                },
            },
        }))
    monkeypatch.setattr(sys, "argv", [
        "validate_scores.py",
        "--meta-env",
        str(tmp_path / "meta_env.json"),
        "--results-glob",
        str(tmp_path / "results*.json"),
        "--expected-concs",
        "1 4",
    ])

    assert validate_scores_main() == 1

    # Each score line is attributed to the concurrency that produced it, so a
    # failing concurrency is identifiable from the log (conc 4 here).
    captured = capsys.readouterr()
    assert "PASS: [conc=1] gsm8k exact_match,strict-match" in captured.out
    assert "FAIL: [conc=4] gsm8k exact_match,strict-match" in captured.err


def test_validate_scores_reports_integration_failure_without_thresholding(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    result_path = tmp_path / "results_test.json"
    result_path.write_text(json.dumps({
        "integration_error": {
            "type": "RuntimeError",
            "message": "vendor verifier checkout failed",
        },
        "results": {
            "gsm8k": {
                "exact_match,strict-match": 0.0,
            },
        },
        "n-samples": {"gsm8k": {"effective": 0}},
    }))
    monkeypatch.setattr(sys, "argv", [
        "validate_scores.py",
        "--meta-env",
        str(tmp_path / "meta_env.json"),
        "--results-glob",
        str(result_path),
    ])

    assert validate_scores_main() == 1
    captured = capsys.readouterr()
    assert "integration failure: RuntimeError: vendor verifier checkout failed" in captured.err
    assert "gsm8k exact_match,strict-match" not in captured.err


def test_validate_scores_rejects_invalid_effective_count_without_thresholding(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    result_path = tmp_path / "results_test.json"
    result_path.write_text(json.dumps({
        "results": {
            "gsm8k": {
                "exact_match,strict-match": 0.0,
            },
            "other": {
                "exact_match,strict-match": 0.0,
            },
            "nonfinite": {
                "exact_match,strict-match": 1.0,
            },
        },
        "n-samples": {
            "gsm8k": {"effective": "unknown"},
            "other": {"effective": 0},
            "nonfinite": {"effective": float("inf")},
        },
    }))
    monkeypatch.setattr(sys, "argv", [
        "validate_scores.py",
        "--meta-env",
        str(tmp_path / "meta_env.json"),
        "--results-glob",
        str(result_path),
    ])

    assert validate_scores_main() == 1
    captured = capsys.readouterr()
    assert "gsm8k invalid effective sample count: 'unknown'" in captured.err
    assert "gsm8k exact_match,strict-match" not in captured.err
    assert "other invalid effective sample count: 0" in captured.err
    assert "other exact_match,strict-match" not in captured.err

    assert "nonfinite invalid effective sample count: inf" in captured.err
    assert "nonfinite exact_match,strict-match" not in captured.err

def test_validate_scores_accepts_legacy_result_without_effective_count(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    result_path = tmp_path / "results_test.json"
    result_path.write_text(json.dumps({
        "results": {
            "gsm8k": {
                "exact_match,strict-match": 1.0,
            },
        },
    }))
    monkeypatch.setattr(sys, "argv", [
        "validate_scores.py",
        "--meta-env",
        str(tmp_path / "meta_env.json"),
        "--results-glob",
        str(result_path),
    ])

    assert validate_scores_main() == 0
    captured = capsys.readouterr()
    assert "PASS: gsm8k exact_match,strict-match" in captured.out
