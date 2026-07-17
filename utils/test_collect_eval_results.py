"""Tests for eval result aggregation."""

import json
from pathlib import Path

from collect_eval_results import build_row, collect_eval_rows, extract_lm_metrics


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


def _write_lm_eval_result(path: Path, score: float) -> None:
    path.write_text(
        json.dumps(
            {
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
            }
        )
    )


def test_collect_eval_rows_expands_batched_concurrencies(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "eval_batch"
    artifact_dir.mkdir()
    (artifact_dir / "meta_env.json").write_text(
        json.dumps(
            {
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
            }
        )
    )
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


def test_collect_eval_rows_ignores_failed_batch_points(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "eval_batch"
    artifact_dir.mkdir()
    (artifact_dir / "meta_env.json").write_text(
        json.dumps(
            {
                "is_multinode": True,
                "eval_concs": [4, 16],
                "completed_eval_concs": [4],
                "failed_eval_concs": [16],
                "conc": 4,
            }
        )
    )
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


def test_extracts_current_lm_eval_metrics_for_new_tasks(tmp_path: Path) -> None:
    result_path = tmp_path / "results.json"
    result_path.write_text(
        json.dumps(
            {
                "results": {
                    "aime26": {
                        "exact_match,none": 0.5,
                        "exact_match_stderr,none": 0.09,
                    },
                    "gpqa_diamond_cot_n_shot": {
                        "exact_match,extract_abcd": 0.74,
                        "exact_match_stderr,extract_abcd": 0.02,
                    },
                },
                "configs": {
                    "aime26": {
                        "metric_list": [{"metric": "exact_match"}],
                        "filter_list": [],
                    },
                    "gpqa_diamond_cot_n_shot": {
                        "metric_list": [{"metric": "exact_match"}],
                        "filter_list": [{"name": "extract_abcd"}],
                    },
                },
                "n-samples": {
                    "aime26": {"effective": 30},
                    "gpqa_diamond_cot_n_shot": {"effective": 396},
                },
            }
        )
    )

    metrics = {item["task"]: item for item in extract_lm_metrics(result_path)}
    aime_row = build_row({}, metrics["aime26"])
    gpqa_row = build_row({}, metrics["gpqa_diamond_cot_n_shot"])

    assert (aime_row["score"], aime_row["score_name"], aime_row["n_eff"]) == (
        0.5,
        "em_strict",
        30,
    )
    assert (gpqa_row["score"], gpqa_row["score_name"], gpqa_row["n_eff"]) == (
        0.74,
        "em_flexible",
        396,
    )
