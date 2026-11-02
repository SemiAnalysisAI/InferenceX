"""Tests for batched eval score and manifest validation."""

import json
import sys
from pathlib import Path

import pytest

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


@pytest.mark.parametrize(
    ("score", "shown"), [(float("nan"), "nan"), (float("inf"), "inf"), (True, "True")]
)
def test_validate_scores_fails_a_non_finite_or_boolean_score(
    tmp_path: Path,
    monkeypatch,
    capsys,
    score: object,
    shown: str,
) -> None:
    result_path = tmp_path / "results_test.json"
    result_path.write_text(json.dumps({"results": {"gsm8k": {"exact_match,strict-match": score}}}))
    monkeypatch.setattr(sys, "argv", [
        "validate_scores.py",
        "--meta-env",
        str(tmp_path / "meta_env.json"),
        "--results-glob",
        str(result_path),
    ])

    assert validate_scores_main() == 1
    captured = capsys.readouterr()
    assert f"FAIL: gsm8k exact_match,strict-match = {shown} is not a finite number" in captured.err
    assert "PASS" not in captured.out


def test_validate_scores_fails_when_thresholds_cannot_be_loaded(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    thresholds = tmp_path / "thresholds.yaml"
    thresholds.write_text("default: [gsm8k\n")
    result_path = tmp_path / "results_test.json"
    result_path.write_text(json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.87}}}))
    monkeypatch.setattr(sys, "argv", [
        "validate_scores.py",
        "--thresholds",
        str(thresholds),
        "--meta-env",
        str(tmp_path / "meta_env.json"),
        "--results-glob",
        str(result_path),
    ])

    assert validate_scores_main() == 1
    captured = capsys.readouterr()
    assert f"FAIL: could not load thresholds from {thresholds}: invalid YAML" in captured.err
    assert "PASS" not in captured.out


@pytest.mark.parametrize(
    ("thresholds_text", "expected"),
    [
        (
            "default:\n  gsm8k: 0.9\nmodels:\n",
            "FAIL: gsm8k exact_match,strict-match = 0.8700 (< 0.9 from default)",
        ),
        (
            "default:\n  gsm8k: 0.9\nmodels:\n  dsr1:\n",
            "FAIL: gsm8k exact_match,strict-match = 0.8700 (< 0.9 from default)",
        ),
        (
            "default:\n  gsm8k: 0.9\nmodels: [dsr1]\n",
            "thresholds config 'models' must be a mapping",
        ),
        (
            "default: []\n",
            "thresholds config 'default' must be a mapping",
        ),
        (
            "default:\n  gsm8k: 0.9\nmodels:\n  dsr1: 0\n",
            "thresholds config 'models.dsr1' must be a mapping",
        ),
        (
            "default:\n  gsm8k: .nan\n",
            "thresholds config default.gsm8k must be a finite number, got nan",
        ),
        (
            "default:\n  gsm8k: 0.8\nmodels:\n  dsr1:\n    gsm8k: '0.9'\n",
            "thresholds config models.dsr1.gsm8k must be a finite number, got '0.9'",
        ),
        (
            "default:\n  gsm8k: 0.8\nmodel:\n  dsr1:\n    gsm8k: 0.9\n",
            "thresholds config has unknown section(s): model",
        ),
    ],
)
def test_validate_scores_handles_empty_or_malformed_threshold_sections(
    tmp_path: Path,
    monkeypatch,
    capsys,
    thresholds_text: str,
    expected: str,
) -> None:
    thresholds = tmp_path / "thresholds.yaml"
    thresholds.write_text(thresholds_text)
    result_path = tmp_path / "results_test.json"
    result_path.write_text(json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.87}}}))
    monkeypatch.setattr(sys, "argv", [
        "validate_scores.py",
        "--thresholds",
        str(thresholds),
        "--model-prefix",
        "dsr1",
        "--meta-env",
        str(tmp_path / "meta_env.json"),
        "--results-glob",
        str(result_path),
    ])

    assert validate_scores_main() == 1
    assert expected in capsys.readouterr().err


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (b'{"results": {"gsm8k": ', "invalid JSON"),
        (b"[1]", "expected a JSON object with a 'results' mapping"),
        (b'{"results": null}', "expected a JSON object with a 'results' mapping"),
        (b"\xff{}", "'utf-8' codec can't decode byte 0xff"),
    ],
)
def test_validate_scores_fails_a_malformed_result_file_and_checks_the_rest(
    tmp_path: Path,
    monkeypatch,
    capsys,
    content: bytes,
    reason: str,
) -> None:
    bad = tmp_path / "results_a.json"
    bad.write_bytes(content)
    (tmp_path / "results_b.json").write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.95}}})
    )
    monkeypatch.setattr(sys, "argv", [
        "validate_scores.py",
        "--meta-env",
        str(tmp_path / "meta_env.json"),
        "--results-glob",
        str(tmp_path / "results*.json"),
    ])

    assert validate_scores_main() == 1
    captured = capsys.readouterr()
    assert f"FAIL: unreadable result file {bad}: {reason}" in captured.err
    assert "PASS: gsm8k exact_match,strict-match = 0.9500" in captured.out


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        ({"results": {"gsm8k": None}}, "'results' entry is not a mapping"),
        ({"configs": [], "results": {"gsm8k": {}}}, "'configs' is not a mapping"),
        ({"configs": {"gsm8k": []}, "results": {"gsm8k": {}}}, "'configs' entry is not a mapping"),
        (
            {"configs": {"gsm8k": {"metric_list": 5}}, "results": {"gsm8k": {}}},
            "'metric_list' is not a list",
        ),
    ],
)
def test_validate_scores_fails_a_malformed_task_entry_and_checks_the_rest(
    tmp_path: Path,
    monkeypatch,
    capsys,
    data: dict,
    reason: str,
) -> None:
    (tmp_path / "results_a.json").write_text(json.dumps(data))
    (tmp_path / "results_b.json").write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.95}}})
    )
    monkeypatch.setattr(sys, "argv", [
        "validate_scores.py",
        "--meta-env",
        str(tmp_path / "meta_env.json"),
        "--results-glob",
        str(tmp_path / "results*.json"),
    ])

    assert validate_scores_main() == 1
    captured = capsys.readouterr()
    assert f"FAIL: gsm8k malformed result: {reason}" in captured.err
    assert "PASS: gsm8k exact_match,strict-match = 0.9500" in captured.out


def test_validate_scores_fails_a_result_file_with_no_checked_metric(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "meta_env.json").write_text(json.dumps({
        "eval_concs": [1, 4],
        "completed_eval_concs": [1, 4],
        "failed_eval_concs": [],
    }))
    (tmp_path / "results_test_conc1.json").write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.95}}})
    )
    (tmp_path / "results_test_conc4.json").write_text(
        json.dumps({"results": {"gsm8k": {"acc,none": 0.95}}})
    )
    monkeypatch.setattr(sys, "argv", [
        "validate_scores.py",
        "--meta-env",
        str(tmp_path / "meta_env.json"),
        "--results-glob",
        str(tmp_path / "results*.json"),
    ])

    assert validate_scores_main() == 1
    captured = capsys.readouterr()
    assert "PASS: [conc=1] gsm8k exact_match,strict-match = 0.9500" in captured.out
    assert (
        "FAIL: [conc=4] no metrics matched 'declared task metrics' in "
        f"{tmp_path / 'results_test_conc4.json'}"
    ) in captured.err
