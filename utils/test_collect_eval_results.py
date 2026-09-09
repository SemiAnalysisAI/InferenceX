"""Tests for eval result aggregation."""

import json
import os
from pathlib import Path

import pytest

from collect_eval_results import (
    EVAL_RESULT_FORMAT,
    build_row,
    collect_eval_rows,
)
from evals.kimi_vendor_eval import RESULT_FORMAT as KIMI_VENDOR_RESULT_FORMAT
from evals.minimax_provider_eval import RESULT_FORMAT as MINIMAX_RESULT_FORMAT


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
    assert "eval_suite" not in row


def test_build_row_preserves_explicit_eval_suite() -> None:
    row = build_row(
        {"eval_suite": "kimi_tool_call_schema"},
        {"task": "kimi_tool_call_schema"},
    )

    assert row["eval_suite"] == "kimi_tool_call_schema"


def _write_lm_eval_result(
    path: Path,
    score: float,
    task: str = "gsm8k",
) -> None:
    path.write_text(json.dumps({
        "lm_eval_version": "0.4.0",
        "model_name": "test-model",
        "results": {
            task: {
                "exact_match,strict-match": score,
                "exact_match_stderr,strict-match": 0.01,
            },
        },
        "configs": {
            task: {
                "metric_list": [{"metric": "exact_match"}],
                "filter_list": [{"name": "strict-match"}],
            },
        },
        "n-samples": {task: {"effective": 10}},
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
        "eval_concs": [4, 16],
        "completed_eval_concs": [4, 16],
        "failed_eval_concs": [],
        "conc": 4,
        "eval_suite": "gsm8k",
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
    assert {row["eval_suite"] for row in rows} == {"gsm8k"}


def test_collect_eval_rows_ignores_failed_batch_points(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "eval_batch"
    artifact_dir.mkdir()
    (artifact_dir / "meta_env.json").write_text(json.dumps({
        "is_multinode": True,
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

    assert [row["conc"] for row in rows] == [4]



@pytest.mark.parametrize("result_format", [KIMI_VENDOR_RESULT_FORMAT, MINIMAX_RESULT_FORMAT])
def test_collect_eval_rows_accepts_provider_compatibility_result(
    tmp_path: Path, result_format: str,
) -> None:
    artifact_dir = tmp_path / "eval_minimax"
    artifact_dir.mkdir()
    (artifact_dir / "meta_env.json").write_text(
        json.dumps({"eval_suite": "minimax_m3_smoke"})
    )
    result_path = artifact_dir / "results_minimax_vendor.json"
    _write_lm_eval_result(result_path, 1.0, task="minimax_m3_smoke")
    result = json.loads(result_path.read_text())
    result.pop("lm_eval_version")
    result["result_format"] = result_format
    result["eval_adapter"] = "minimax-provider-verifier"
    result_path.write_text(json.dumps(result))

    rows = collect_eval_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["task"] == "minimax_m3_smoke"
    assert rows[0]["score"] == 1.0
    assert rows[0]["eval_suite"] == "minimax_m3_smoke"


def test_collect_eval_rows_retains_integration_and_sample_failures(
    tmp_path: Path,
) -> None:
    for name, invalid in (
        ("integration", "integration"),
        ("zero", 0),
        ("nonnumeric", "unknown"),
        ("nonfinite", float("nan")),
        ("malformed", []),
    ):
        artifact_dir = tmp_path / f"eval_{name}"
        artifact_dir.mkdir()
        (artifact_dir / "meta_env.json").write_text(json.dumps({
            "eval_suite": "gsm8k",
        }))
        result_path = artifact_dir / f"results_{name}.json"
        _write_lm_eval_result(result_path, 0.0)
        result = json.loads(result_path.read_text())
        if invalid == "integration":
            result["integration_error"] = {
                "type": "RuntimeError",
                "message": "vendor verifier checkout failed",
            }
        else:
            result["n-samples"]["gsm8k"]["effective"] = invalid
        result_path.write_text(json.dumps(result))

    rows = collect_eval_rows(tmp_path)
    assert len(rows) == 5
    assert all(row["infrastructure_success"] is False for row in rows)
    assert all(row["score"] is None for row in rows)
    assert all(row["n_eff"] == 0 for row in rows)
    assert {
        row["integration_error"]["type"]
        for row in rows
    } == {"RuntimeError", "InvalidEffectiveSampleCount"}


def test_collect_eval_rows_handles_malformed_failure_metadata(
    tmp_path: Path,
) -> None:
    for index, (configs, sample_counts) in enumerate(
        (
            (None, None),
            ({"gsm8k": None}, {"gsm8k": None}),
        )
    ):
        artifact_dir = tmp_path / f"eval_malformed_metadata_{index}"
        artifact_dir.mkdir()
        (artifact_dir / "meta_env.json").write_text(
            json.dumps({"eval_suite": "gsm8k"})
        )
        result_path = artifact_dir / f"results_{index}.json"
        _write_lm_eval_result(result_path, 0.0)
        result = json.loads(result_path.read_text())
        result["configs"] = configs
        result["n-samples"] = sample_counts
        result["integration_error"] = {
            "type": "RuntimeError",
            "message": "setup failed",
        }
        result_path.write_text(json.dumps(result))

    rows = collect_eval_rows(tmp_path)
    assert len(rows) == 2
    assert all(row["score"] is None for row in rows)
    assert all(row["n_eff"] == 0 for row in rows)
    assert all(row["infrastructure_success"] is False for row in rows)


def test_collect_eval_rows_accepts_legacy_missing_effective_count(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "eval_legacy"
    artifact_dir.mkdir()
    (artifact_dir / "meta_env.json").write_text(json.dumps({
        "eval_suite": "gsm8k",
    }))
    result_path = artifact_dir / "results_legacy.json"
    _write_lm_eval_result(result_path, 0.9)
    result = json.loads(result_path.read_text())
    result.pop("n-samples")
    result_path.write_text(json.dumps(result))

    rows = collect_eval_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["score"] == 0.9
    assert rows[0]["n_eff"] is None


def test_collect_eval_rows_does_not_resurrect_stale_valid_result(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "eval_retry"
    artifact_dir.mkdir()
    (artifact_dir / "meta_env.json").write_text(json.dumps({
        "eval_suite": "kimi_tool_call_schema",
    }))
    stale_path = (
        artifact_dir / "results_kimi_vendor_2026-08-12T01-00-00.000000.json"
    )
    _write_lm_eval_result(stale_path, 1.0, task="kimi_tool_call_schema")
    current_path = (
        artifact_dir / "results_kimi_vendor_2026-08-12T02-00-00.000000.json"
    )
    _write_lm_eval_result(current_path, 0.0, task="kimi_tool_call_schema")
    result = json.loads(current_path.read_text())
    result["integration_error"] = {
        "type": "RuntimeError",
        "message": "vendor verifier checkout failed",
    }
    current_path.write_text(json.dumps(result))
    current_path.touch()
    stale_path.touch()

    rows = collect_eval_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["infrastructure_success"] is False
    assert rows[0]["integration_error"]["message"] == (
        "vendor verifier checkout failed"
    )


def test_collect_eval_rows_uses_mtime_for_newer_legacy_name(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "eval_retry"
    artifact_dir.mkdir()
    (artifact_dir / "meta_env.json").write_text(
        json.dumps({"eval_suite": "kimi_tool_call_schema"})
    )
    stale_path = (
        artifact_dir / "results_kimi_vendor_2026-08-12T01-00-00.000000.json"
    )
    _write_lm_eval_result(stale_path, 1.0, task="kimi_tool_call_schema")
    current_path = artifact_dir / "results.json"
    _write_lm_eval_result(current_path, 0.0, task="kimi_tool_call_schema")
    current = json.loads(current_path.read_text())
    current["integration_error"] = {
        "type": "RuntimeError",
        "message": "latest attempt failed",
    }
    current_path.write_text(json.dumps(current))
    os.utime(current_path, (2_000_000_000, 2_000_000_000))

    rows = collect_eval_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["infrastructure_success"] is False
    assert rows[0]["integration_error"]["message"] == "latest attempt failed"


def test_collect_eval_rows_retains_missing_or_out_of_range_scores(
    tmp_path: Path,
) -> None:
    for index, score in enumerate(
        (None, True, float("nan"), float("inf"), -0.1, 1.1)
    ):
        artifact_dir = tmp_path / f"eval_invalid_{index}"
        artifact_dir.mkdir()
        (artifact_dir / "meta_env.json").write_text(
            json.dumps({"eval_suite": "kimi_tool_call_schema"})
        )
        _write_lm_eval_result(
            artifact_dir / f"results_{index}.json",
            score,
            task="kimi_tool_call_schema",
        )

    rows = collect_eval_rows(tmp_path)
    assert len(rows) == 6
    assert all(row["score"] is None for row in rows)
    assert all(row["infrastructure_success"] is False for row in rows)
    assert {
        row["integration_error"]["type"] for row in rows
    } == {"InvalidPrimaryScore"}


def test_collect_eval_rows_falls_back_for_invalid_filename_timestamp(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "eval_invalid_timestamp"
    artifact_dir.mkdir()
    (artifact_dir / "meta_env.json").write_text(
        json.dumps({"eval_suite": "kimi_tool_call_schema"})
    )
    _write_lm_eval_result(
        artifact_dir / "results_2026-99-99T99-99-99.json",
        1.0,
        task="kimi_tool_call_schema",
    )

    rows = collect_eval_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["score"] == 1.0


def test_collect_eval_rows_uses_extract_filter_as_primary_score(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "eval_gpqa"
    artifact_dir.mkdir()
    (artifact_dir / "meta_env.json").write_text(
        json.dumps({"eval_suite": "gpqa_diamond_cot_n_shot"})
    )
    (artifact_dir / "results_gpqa.json").write_text(json.dumps({
        "lm_eval_version": "0.4.0",
        "results": {
            "gpqa_diamond_cot_n_shot": {
                "exact_match,extract_abcd": 0.75,
                "exact_match_stderr,extract_abcd": 0.02,
            },
        },
        "configs": {
            "gpqa_diamond_cot_n_shot": {
                "metric_list": [{"metric": "exact_match"}],
                "filter_list": [{"name": "extract_abcd"}],
            },
        },
        "n-samples": {
            "gpqa_diamond_cot_n_shot": {"effective": 8},
        },
    }))

    rows = collect_eval_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["score"] == 0.75
    assert rows[0]["score_name"] == "em_flexible"


def test_collect_eval_rows_accepts_bfcl_compatibility_and_ignores_native_report(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "eval_bfcl"
    artifact_dir.mkdir()
    (artifact_dir / "meta_env.json").write_text(
        json.dumps({"eval_suite": "bfcl_smoke"})
    )
    (artifact_dir / "bfcl_report.json").write_text(
        json.dumps(
            {
                "results": {"native_only": {"acc,none": 1.0}},
            }
        )
    )
    compatibility_path = artifact_dir / "results_bfcl.json"
    tasks = {
        "bfcl_smoke": 0.75,
        "bfcl_simple_python": 1.0,
    }
    compatibility_path.write_text(
        json.dumps(
            {
                "result_format": EVAL_RESULT_FORMAT,
                "model_name": "test-model",
                "results": {
                    task: {
                        "acc,none": score,
                        "acc_stderr,none": 0.0,
                    }
                    for task, score in tasks.items()
                },
                "configs": {
                    task: {
                        "metric_list": [{"metric": "acc"}],
                        "filter_list": [{"name": "none"}],
                    }
                    for task in tasks
                },
                "n-samples": {
                    "bfcl_smoke": {"effective": 4},
                    "bfcl_simple_python": {"effective": 1},
                },
            }
        )
    )

    rows = collect_eval_rows(tmp_path)

    assert {row["task"]: row["score"] for row in rows} == tasks
    assert {row["score_name"] for row in rows} == {"accuracy"}
    assert {row["source"] for row in rows} == {str(compatibility_path)}
