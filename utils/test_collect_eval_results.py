"""Tests for eval result aggregation."""

import json
from pathlib import Path

from collect_eval_results import (
    build_row,
    collect_eval_rows,
    extract_lm_metrics,
)


def test_build_row_preserves_sequence_lengths() -> None:
    row = build_row(
        {
            "infmax_model_prefix": "gptoss",
            "hw": "h100",
            "framework": "vllm",
            "precision": "fp4",
            "isl": "1024",
            "osl": "1024",
        },
        {"task": "gsm8k"},
    )

    assert row["isl"] == 1024
    assert row["osl"] == 1024


def test_build_row_uses_flexible_metric_as_primary_score() -> None:
    row = build_row(
        {
            "infmax_model_prefix": "test",
            "hw": "h100",
            "framework": "vllm",
            "precision": "fp8",
        },
        {
            "task": "gpqa_diamond_cot_n_shot",
            "flex": 0.42,
            "flex_se": 0.02,
        },
    )

    assert row["score"] == 0.42
    assert row["score_name"] == "em_flexible"
    assert row["score_se"] == 0.02


def test_extract_lm_metrics_supports_default_none_filter(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "results_accuracy.json"
    result_path.write_text(json.dumps({
        "lm_eval_version": "0.4.9",
        "results": {
            "multiple_choice": {
                "acc,none": 0.75,
                "acc_stderr,none": 0.03,
            },
        },
        "configs": {
            "multiple_choice": {
                "metric_list": [{"metric": "acc"}],
                "filter_list": [],
            },
        },
    }))

    metrics = extract_lm_metrics(result_path)

    assert metrics[0]["accuracy"] == 0.75
    assert metrics[0]["accuracy_se"] == 0.03


def _write_lm_eval_result(path: Path, score: float) -> None:
    path.write_text(json.dumps({
        "lm_eval_version": "0.4.0",
        "model_name": "test-model",
        "results": {
            "gsm8k": {
                "exact_match,strict-match": score,
                "exact_match_stderr,strict-match": 0.01,
            },
        },
        "configs": {
            "gsm8k": {
                "metric_list": [{"metric": "exact_match"}],
                "filter_list": [{"name": "strict-match"}],
            },
        },
        "n-samples": {"gsm8k": {"effective": 10}},
    }))


def test_collect_eval_rows_expands_batched_concurrencies(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "eval_batch"
    artifact_dir.mkdir()
    (artifact_dir / "meta_env.json").write_text(json.dumps({
        "is_multinode": True,
        "infmax_model_prefix": "gptoss",
        "hw": "gb200",
        "framework": "dynamo-sglang",
        "precision": "fp8",
        "spec_decoding": "none",
        "isl": 8192,
        "osl": 1024,
        "prefill_tp": 4,
        "prefill_ep": 1,
        "prefill_num_workers": 1,
        "decode_tp": 8,
        "decode_ep": 1,
        "decode_num_workers": 2,
        "eval_exit_code": 0,
        "eval_concs": [4, 16],
        "completed_eval_concs": [4, 16],
        "failed_eval_concs": [],
        "conc": 4,
    }))
    _write_lm_eval_result(
        artifact_dir / "results_test_conc4.json",
        0.90,
    )
    _write_lm_eval_result(
        artifact_dir / "results_test_conc16.json",
        0.91,
    )

    rows = collect_eval_rows(tmp_path)

    assert [row["conc"] for row in rows] == [4, 16]
    assert [row["score"] for row in rows] == [0.90, 0.91]


def test_collect_eval_rows_rejects_failed_batch(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "eval_batch"
    artifact_dir.mkdir()
    (artifact_dir / "meta_env.json").write_text(json.dumps({
        "is_multinode": True,
        "eval_exit_code": 1,
        "eval_concs": [4, 16],
        "completed_eval_concs": [4],
        "failed_eval_concs": [16],
        "conc": 4,
    }))
    _write_lm_eval_result(
        artifact_dir / "results_test_conc4.json",
        0.90,
    )
    _write_lm_eval_result(
        artifact_dir / "results_test_conc16.json",
        0.91,
    )

    rows = collect_eval_rows(tmp_path)

    assert rows == []


def test_collect_eval_rows_ignores_failed_single_eval(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "eval_failed"
    artifact_dir.mkdir()
    (artifact_dir / "meta_env.json").write_text(json.dumps({
        "is_multinode": False,
        "eval_exit_code": 7,
        "conc": 4,
    }))
    _write_lm_eval_result(artifact_dir / "results_test.json", 0.99)

    assert collect_eval_rows(tmp_path) == []


def test_collect_eval_rows_rejects_inconsistent_batch_metadata(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "eval_batch"
    artifact_dir.mkdir()
    (artifact_dir / "meta_env.json").write_text(json.dumps({
        "is_multinode": True,
        "eval_exit_code": 0,
        "eval_concs": [4, 16],
        "completed_eval_concs": [4, 16],
        "failed_eval_concs": [16],
        "conc": 4,
    }))
    _write_lm_eval_result(
        artifact_dir / "results_test_conc4.json",
        0.90,
    )

    assert collect_eval_rows(tmp_path) == []


def test_collect_eval_rows_rejects_incomplete_batch_results(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "eval_batch"
    artifact_dir.mkdir()
    (artifact_dir / "meta_env.json").write_text(json.dumps({
        "is_multinode": True,
        "eval_exit_code": 0,
        "eval_concs": [4, 16],
        "completed_eval_concs": [4, 16],
        "failed_eval_concs": [],
        "conc": 4,
    }))
    _write_lm_eval_result(
        artifact_dir / "results_test_conc4.json",
        0.90,
    )

    assert collect_eval_rows(tmp_path) == []
