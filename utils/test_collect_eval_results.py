"""Tests for eval result aggregation."""

import json
from pathlib import Path

from collect_eval_results import (
    EVAL_RESULT_FORMAT,
    build_row,
    collect_eval_rows,
)
from evals.kimi_vendor_eval import RESULT_FORMAT as KIMI_VENDOR_RESULT_FORMAT


def test_kimi_vendor_result_format_matches_collector_contract() -> None:
    assert KIMI_VENDOR_RESULT_FORMAT == EVAL_RESULT_FORMAT


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



def test_collect_eval_rows_accepts_neutral_result_format(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "eval_provider"
    artifact_dir.mkdir()
    (artifact_dir / "meta_env.json").write_text(
        json.dumps({"eval_suite": "provider_smoke"})
    )
    result_path = artifact_dir / "results_provider.json"
    _write_lm_eval_result(result_path, 1.0)
    result = json.loads(result_path.read_text())
    result.pop("lm_eval_version")
    result["result_format"] = EVAL_RESULT_FORMAT
    result_path.write_text(json.dumps(result))

    rows = collect_eval_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["score"] == 1.0
    assert rows[0]["eval_suite"] == "provider_smoke"


def test_collect_eval_rows_excludes_integration_and_sample_failures(
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

    assert collect_eval_rows(tmp_path) == []


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
        "eval_suite": "gsm8k",
    }))
    stale_path = artifact_dir / "results_older.json"
    _write_lm_eval_result(stale_path, 1.0)
    current_path = artifact_dir / "results_current.json"
    _write_lm_eval_result(current_path, 0.0)
    result = json.loads(current_path.read_text())
    result["integration_error"] = {
        "type": "RuntimeError",
        "message": "vendor verifier checkout failed",
    }
    current_path.write_text(json.dumps(result))
    stale_path.touch()
    current_path.touch()

    assert collect_eval_rows(tmp_path) == []