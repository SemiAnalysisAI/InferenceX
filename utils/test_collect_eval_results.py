"""Tests for eval result aggregation."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from infx.results.collect_eval_results import (
    EVAL_RESULT_FORMAT,
    build_row,
    collect_eval_rows,
    detect_lm_eval_jsons,
    result_concurrency,
)
from evals.kimi_vendor_eval import RESULT_FORMAT as KIMI_VENDOR_RESULT_FORMAT
from evals.minimax_provider_eval import RESULT_FORMAT as MINIMAX_RESULT_FORMAT
from infx.results.evals import (
    build_rows, extract_metrics, select_latest_result, select_latest_results,
)


@pytest.mark.parametrize("batched", [False, True])
def test_result_selection_accepts_empty_candidates(batched: bool) -> None:
    assert select_latest_results(iter(()), batched=batched) == []


def test_result_selection_filters_concurrency_before_ordering(tmp_path: Path) -> None:
    chosen = tmp_path / "results_2026-06-27T01-00-00_conc4.json"
    other = tmp_path / "results_2026-06-28T01-00-00_conc8.json"
    missing = tmp_path / "results_legacy_conc16.json"
    # Timestamped candidates do not require stat; the legacy candidate does.
    assert select_latest_result(iter([other, missing, chosen]), concurrency=4) == chosen
    assert select_latest_result([other, missing, chosen], concurrency=32) is None
    with pytest.raises(FileNotFoundError):
        select_latest_result([missing], concurrency=16)
    # Batch collection can retain an uncontested candidate without ranking it.
    assert select_latest_results([missing], batched=True) == [missing]


def test_build_rows_uses_explicit_inputs_and_preserves_them(tmp_path: Path, monkeypatch) -> None:
    data = {
        "results": {"valid": {"acc": 0.75}, "invalid": {"acc": -0.1}},
        "configs": {"valid": {"metadata": {"model": "task-model"}}},
    }
    meta = {"model": "metadata-model", "conc": "4"}
    before = deepcopy((data, meta))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL", "unrelated-process-model")

    rows = build_rows(data, meta, source="artifact/results.json")

    assert (data, meta) == before
    assert list(tmp_path.iterdir()) == []
    assert [row["task"] for row in rows] == ["valid", "invalid"]
    assert [row["model"] for row in rows] == ["task-model", "metadata-model"]
    assert [row["score"] for row in rows] == [0.75, None]
    assert [row["conc"] for row in rows] == [4, 4]
    assert [row["source"] for row in rows] == ["artifact/results.json"] * 2
    assert rows[1]["integration_error"] == {
        "type": "InvalidPrimaryScore", "message": "invalid primary score: -0.1",
    }


def test_extract_metrics_keeps_raw_values_for_other_callers() -> None:
    data = {
        "model_name": "result-model",
        "results": {"task": {"acc": -0.1, "acc_stderr": 0.02}},
        "configs": {"task": {"metadata": {"model": "task-model"}}},
        "n-samples": {"task": {"effective": 2}},
    }

    assert extract_metrics(data, source="result.json") == [{
        "task": "task", "strict": None, "strict_se": None, "flex": None,
        "flex_se": None, "accuracy": -0.1, "accuracy_se": 0.02, "n_eff": 2,
        "model": "result-model", "source": "result.json",
        "infrastructure_success": True, "integration_error": None,
    }]


@pytest.mark.parametrize("metrics,config,expected", [
    ({"acc": 0.5, "acc_stderr": 0.02, "exact_match": 0.75}, {},
     (0.5, "accuracy", 0.02)),
    ({"custom": 0.25, "custom_stderr": 0.01},
     {"metric_list": [{"metric": "custom"}]}, (0.25, "em_strict", 0.01)),
    ({"acc,strict": 0.0, "acc,none": 0.5, "acc,flex": 0.75},
     {"metric_list": [{"metric": "acc"}],
      "filter_list": [{"name": "flex"}, {"name": "none"}, {"name": "strict"}]},
     (0.0, "em_strict", None)),
    ({"acc,strict": None, "acc,none": 0.5, "acc,flex": 0.75},
     {"metric_list": [{"metric": "acc"}],
      "filter_list": [{"name": "strict"}, {"name": "none"}, {"name": "flex"}]},
     (0.5, "accuracy", None)),
    ({"exact_match,strict-first": 0.25, "exact_match,strict-last": 0.75},
     {"filter_list": [{"name": "strict-first"}, {"name": "strict-last"}]},
     (0.75, "em_strict", None)),
    ({"exact_match,strict-first": 0.25, "exact_match,extract": 0.5},
     {"filter_list": [{"name": "strict-first"}, {"name": "strict-missing"},
                      {"name": "extract"}]}, (0.5, "em_flexible", None)),
    ({"exact_match,resolved": 1.0, "exact_match_stderr,resolved": 0.03},
     {"filter_list": [{"name": "resolved"}]}, (1.0, "em_strict", 0.03)),
    ({"exact_match,extract": 0.75, "exact_match_stderr,extract": 0.04},
     {"filter_list": [{"name": "extract"}]}, (0.75, "em_flexible", 0.04)),
])
def test_collector_metric_precedence(
    tmp_path: Path, metrics: dict, config: dict, expected: tuple,
) -> None:
    (tmp_path / "meta_env.json").write_text("{}")
    (tmp_path / "results.json").write_text(json.dumps({
        "lm_eval_version": "test", "results": {"task": metrics},
        "configs": {"task": config},
    }))

    [row] = collect_eval_rows(tmp_path)

    assert (row["score"], row["score_name"], row["score_se"]) == expected
    assert row["infrastructure_success"] is True


@pytest.mark.parametrize("score", [True, "0.5", -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_primary_does_not_fall_back_to_valid_secondary(tmp_path: Path, score: object) -> None:
    (tmp_path / "meta_env.json").write_text("{}")
    (tmp_path / "results.json").write_text(json.dumps({
        "lm_eval_version": "test",
        "results": {"task": {"acc,strict": score, "acc,none": 0.5}},
        "configs": {"task": {"metric_list": [{"metric": "acc"}],
                              "filter_list": [{"name": "strict"}, {"name": "none"}]}},
        "n-samples": {"task": {"effective": 8}},
    }))

    [row] = collect_eval_rows(tmp_path)

    assert row["score"] is None
    assert row["em_strict"] is None
    assert row["em_flexible"] is None
    assert row["n_eff"] == 8
    assert row["infrastructure_success"] is False
    assert row["integration_error"] == {
        "type": "InvalidPrimaryScore", "message": f"invalid primary score: {score!r}",
    }


@pytest.mark.parametrize("error,expected", [
    (None, None),
    ("", {"type": "IntegrationError", "message": ""}),
    ({}, {}),
])
def test_collector_retains_error_value_semantics(tmp_path: Path, error: object, expected: object) -> None:
    (tmp_path / "meta_env.json").write_text("{}")
    (tmp_path / "results.json").write_text(json.dumps({
        "lm_eval_version": "test", "results": {"task": {"acc": 0.5}},
        "integration_error": error, "n-samples": {"task": {"effective": 8}},
    }))

    [row] = collect_eval_rows(tmp_path)

    assert row["integration_error"] == expected
    assert row["infrastructure_success"] is (error is None)
    assert row["score"] == (0.5 if error is None else None)
    assert row["n_eff"] == (8 if error is None else 0)


@pytest.mark.parametrize("effective,success", [
    (True, False), (None, False), (0, False), (-1, False), ("8", False),
    (float("inf"), False), (0.5, True), (1, True),
])
def test_effective_count_accepts_only_positive_finite_numbers(
    tmp_path: Path, effective: object, success: bool,
) -> None:
    from infx.workflows.validate_reusable_sweep_artifacts import _raw_result_error

    (tmp_path / "meta_env.json").write_text("{}")
    path = tmp_path / "results.json"
    path.write_text(json.dumps({
        "lm_eval_version": "test", "results": {"task": {"acc": 0.5}},
        "n-samples": {"task": {"effective": effective}},
    }))

    [row] = collect_eval_rows(tmp_path)

    assert row["infrastructure_success"] is success
    assert row["n_eff"] == (effective if success else 0)
    assert (_raw_result_error(path) is None) is success


@pytest.mark.parametrize("config,error", [
    ({"metric_list": [None]}, TypeError),
    ({"metric_list": [{}]}, KeyError),
    ({"filter_list": [None]}, TypeError),
    ({"filter_list": [{}]}, KeyError),
])
def test_collector_preserves_malformed_config_errors(tmp_path: Path, config: dict, error: type) -> None:
    (tmp_path / "meta_env.json").write_text("{}")
    (tmp_path / "results.json").write_text(json.dumps({
        "lm_eval_version": "test", "results": {"task": {"acc": 0.5}},
        "configs": {"task": config},
    }))
    with pytest.raises(error):
        collect_eval_rows(tmp_path)


def test_build_row_preserves_topology_defaults_and_score_precedence() -> None:
    row = build_row({
        "is_multinode": "true", "tp": "4", "ep": "bad", "decode_tp": 8,
        "decode_num_workers": "2", "dp_attention": True, "decode_dp_attention": "true",
        "hw": "test-hw", "framework": "TEST-FRAMEWORK", "precision": "FP8",
        "infmax_model_prefix": "prefix", "model": "metadata-model", "eval_suite": None,
    }, {"task": "task", "strict": 0, "accuracy": 0.5, "flex": 1, "strict_se": 0.01})

    assert row == {
        "is_multinode": True, "model_prefix": "prefix", "model": "metadata-model",
        "hw": "TEST-HW", "framework": "test-framework", "precision": "fp8",
        "spec_decoding": "unknown", "isl": 0, "osl": 0, "tp": 4, "ep": 1,
        "prefill_tp": 4, "prefill_ep": 1, "prefill_num_workers": 1,
        "decode_tp": 8, "decode_ep": 1, "decode_num_workers": 2, "conc": 0,
        "dp_attention": "prefill=true,decode=true", "prefill_dp_attention": "true",
        "decode_dp_attention": "true", "task": "task", "em_strict": 0,
        "em_strict_se": 0.01, "em_flexible": 1, "em_flexible_se": None,
        "n_eff": None, "source": None, "infrastructure_success": True,
        "integration_error": None, "eval_suite": None,
        "score": 0, "score_name": "em_strict", "score_se": 0.01,
    }


@pytest.mark.parametrize("payload,recognized", [
    ({"lm_eval_version": "0.4.0"}, True),
    ({"lm_eval_version": None}, True),
    ({"result_format": "inferencex-eval-v1"}, True),
    ({"result_format": "foreign", "lm_eval_version": False}, True),
    ({"result_format": "foreign"}, False),
    ({"result_format": ["inferencex-eval-v1"]}, False),
    ({"results": {}}, False), (None, False), ([], False),
])
def test_result_readers_recognize_format_markers(
    tmp_path: Path, payload: object, recognized: bool,
) -> None:
    from infx.workflows.validate_reusable_sweep_artifacts import _recognized_eval_result_paths

    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload))
    expected = [path] if recognized else []
    assert detect_lm_eval_jsons(tmp_path) == expected
    assert _recognized_eval_result_paths([path]) == expected


@pytest.mark.parametrize("name,expected", [
    ("results_conc16.json", 16), ("results_conc16_2.json", 16),
    ("results_conc0004.json", 4), ("results_conc0.json", 0),
    ("results_conc4_conc16_2.json", 16),
    ("results_conc-4.json", None), ("results_conc4_extra.json", None),
    ("results_conc4.json.bak", None), ("results_conc4_2_3.json", None),
    ("results.json", None),
])
def test_result_readers_parse_concurrency_suffixes(name: str, expected: int | None) -> None:
    from infx.workflows.validate_reusable_sweep_artifacts import _result_concurrency

    assert result_concurrency(Path(name)) == expected
    assert _result_concurrency(name) == expected


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


def test_collector_discovers_custom_names_but_not_metadata_or_nested_files(
    tmp_path: Path,
) -> None:
    custom = tmp_path / "custom.json"
    _write_lm_eval_result(custom, 0.75)
    _write_lm_eval_result(tmp_path / "meta_env.json", 0.25)
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_lm_eval_result(nested / "results.json", 0.5)
    assert detect_lm_eval_jsons(tmp_path) == [custom]


@pytest.mark.parametrize("batched,expected_names", [
    (False, ["custom_2026-06-27T02-00-00.json"]),
    (True, [
        "results_2026-06-27T01-00-00_conc0.json",
        "results_2026-06-27T01-00-00_conc0004_2.json",
        "results_2026-06-27T01-00-00_conc16.json",
    ]),
])
def test_collector_selects_retries_and_orders_concurrencies(
    tmp_path: Path, batched: bool, expected_names: list[str],
) -> None:
    for name in (
        "results_2026-06-27T01-00-00_conc16.json",
        "custom_2026-06-27T02-00-00.json",
        "results_2026-06-27T01-00-00_conc0004_2.json",
        "results_2026-06-27T01-00-00_conc0004_10.json",
        "results_2026-06-27T01-00-00_conc0.json",
        "results_2026-06-27T00-00-00_conc4.json",
    ):
        _write_lm_eval_result(tmp_path / name, 0.75)

    assert [path.name for path in detect_lm_eval_jsons(tmp_path, batched)] == expected_names


def test_collector_cli_runs_outside_checkout(tmp_path: Path) -> None:
    artifact = tmp_path / "eval_input"
    artifact.mkdir()
    (artifact / "meta_env.json").write_text('{"conc": 4}')
    result = artifact / "custom.json"
    _write_lm_eval_result(result, 0.75)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("collect_eval_results.py")),
         str(artifact), "test"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert "### Single-Node Eval Results" in completed.stdout
    rows = json.loads((tmp_path / "agg_eval_test.json").read_text())
    assert len(rows) == 1
    assert rows[0]["score"] == 0.75
    assert rows[0]["conc"] == 4
    assert rows[0]["source"] == str(result)


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
