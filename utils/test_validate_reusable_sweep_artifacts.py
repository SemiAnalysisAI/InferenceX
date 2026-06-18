from __future__ import annotations

import json
import sys
from pathlib import Path

from validate_reusable_sweep_artifacts import (
    actual_benchmark_keys,
    agentic_key,
    expected_agentic_keys,
    expected_benchmark_keys,
    expected_eval_artifact_prefixes,
    expected_eval_keys,
    main,
    validate_agentic_artifacts,
    validate_eval_artifacts,
    validate_fixed_artifacts,
    validate_identity_set,
)


def write_eval_aggregate(
    root: Path,
    rows: list[dict] | None = None,
) -> None:
    eval_dir = root / "eval_results_all"
    eval_dir.mkdir()
    (eval_dir / "agg_eval_all.json").write_text(
        json.dumps(rows or [{"task": "gsm8k"}])
    )


def single_eval_entry(conc: int) -> dict:
    return {
        "exp-name": "gptoss_8k1k",
        "runner": "h100-dgxc-slurm",
        "model-prefix": "gptoss",
        "precision": "fp4",
        "framework": "vllm",
        "tp": 2,
        "ep": 1,
        "dp-attn": False,
        "disagg": False,
        "spec-decoding": "none",
        "conc": conc,
    }


def single_eval_result(conc: int) -> dict:
    return {
        "is_multinode": False,
        "hw": "H100-DGXC-SLURM",
        "model_prefix": "gptoss",
        "framework": "vllm",
        "precision": "fp4",
        "spec_decoding": "none",
        "tp": 2,
        "ep": 1,
        "dp_attention": False,
        "conc": conc,
        "task": "gsm8k",
    }


def single_fixed_entry(conc: int) -> dict:
    return {
        "runner": "h100",
        "model-prefix": "gptoss",
        "framework": "vllm",
        "precision": "fp8",
        "spec-decoding": "none",
        "disagg": False,
        "isl": 1024,
        "osl": 1024,
        "tp": 2,
        "ep": 1,
        "dp-attn": False,
        "conc": conc,
    }


def fixed_result(conc: int) -> dict:
    return {
        "hw": "h100",
        "infmax_model_prefix": "gptoss",
        "framework": "vllm",
        "precision": "fp8",
        "spec_decoding": "none",
        "disagg": False,
        "isl": 1024,
        "osl": 1024,
        "tp": 2,
        "ep": 1,
        "dp_attention": False,
        "conc": conc,
        "is_multinode": False,
    }


def single_agentic_entry(conc: int = 16) -> dict:
    return {
        "runner": "b200-dgxc",
        "model-prefix": "dsv4",
        "framework": "vllm",
        "precision": "fp4",
        "tp": 8,
        "ep": 8,
        "dp-attn": True,
        "conc": conc,
        "offloading": "cpu",
    }


def agentic_result(conc: int = 16) -> dict:
    return {
        "hw": "b200-dgxc",
        "infmax_model_prefix": "dsv4",
        "framework": "vllm",
        "precision": "fp4",
        "scenario_type": "agentic-coding",
        "is_multinode": False,
        "tp": 8,
        "ep": 8,
        "dp_attention": "true",
        "conc": conc,
        "offloading": "cpu",
    }


def test_multinode_agentic_identity_fields_match() -> None:
    config = {
        "single_node": {"agentic": []},
        "multi_node": {
            "agentic": [
                {
                    "runner": "gb200",
                    "model-prefix": "dsv4",
                    "framework": "dynamo-sglang",
                    "precision": "fp8",
                    "spec-decoding": "none",
                    "disagg": True,
                    "prefill": {
                        "tp": 4,
                        "ep": 2,
                        "dp-attn": True,
                        "num-worker": 2,
                    },
                    "decode": {
                        "tp": 8,
                        "ep": 4,
                        "dp-attn": False,
                        "num-worker": 3,
                    },
                    "conc": 64,
                }
            ]
        },
    }
    row = {
        "hw": "gb200",
        "infmax_model_prefix": "dsv4",
        "framework": "dynamo-sglang",
        "precision": "fp8",
        "spec_decoding": "none",
        "disagg": True,
        "scenario_type": "agentic-coding",
        "is_multinode": True,
        "prefill_tp": 4,
        "prefill_ep": 2,
        "prefill_dp_attention": "true",
        "prefill_num_workers": 2,
        "decode_tp": 8,
        "decode_ep": 4,
        "decode_dp_attention": "false",
        "decode_num_workers": 3,
        "conc": 64,
    }

    assert expected_agentic_keys(config) == {agentic_key(row)}


def write_agentic_artifacts(
    root: Path,
    conc: int = 16,
    *,
    aggregate: bool = True,
) -> None:
    result_name = f"dsv4_tp8_conc{conc}_offloadcpu_result"
    point_dir = root / f"bmk_agentic_{result_name}"
    point_dir.mkdir()
    (point_dir / f"{result_name}.json").write_text(
        json.dumps(agentic_result(conc))
    )
    (root / f"agentic_{result_name}").mkdir()
    if aggregate:
        aggregate_dir = root / "agentic_aggregated"
        aggregate_dir.mkdir()
        (aggregate_dir / "summary.csv").write_text(
            f"exp_name,status\nagentic_{result_name},SUCCESS\n"
        )


def test_eval_validation_requires_raw_result_dirs_not_eval_debug_dirs(
    tmp_path: Path,
) -> None:
    config = {
        "evals": [single_eval_entry(32), single_eval_entry(64)],
        "multinode_evals": [],
    }
    prefixes = expected_eval_artifact_prefixes(config)
    write_eval_aggregate(tmp_path)

    (tmp_path / "eval_server_logs_gptoss_8k1k_runner").mkdir()
    (tmp_path / "eval_gpu_metrics_gptoss_8k1k_runner").mkdir()
    (tmp_path / f"{prefixes[0]}00").mkdir()

    errors = validate_eval_artifacts(tmp_path, prefixes)

    assert "missing 1 expected raw eval result artifact dir(s)" in errors
    assert f"  missing eval artifact prefix: {prefixes[1]}" in errors


def test_eval_validation_accepts_all_expected_raw_result_dirs(tmp_path: Path) -> None:
    config = {
        "evals": [single_eval_entry(32), single_eval_entry(64)],
        "multinode_evals": [],
    }
    prefixes = expected_eval_artifact_prefixes(config)
    write_eval_aggregate(tmp_path)
    for index, prefix in enumerate(prefixes):
        (tmp_path / f"{prefix}{index:02d}").mkdir()

    assert validate_eval_artifacts(tmp_path, prefixes) == []


def test_eval_validation_rejects_unexpected_result_dir(tmp_path: Path) -> None:
    config = {"evals": [single_eval_entry(32)], "multinode_evals": []}
    prefixes = expected_eval_artifact_prefixes(config)
    write_eval_aggregate(tmp_path)
    (tmp_path / f"{prefixes[0]}00").mkdir()
    (tmp_path / "eval_unrelated_config").mkdir()

    errors = validate_eval_artifacts(tmp_path, prefixes)

    assert "found 1 unexpected raw eval artifact dir(s)" in errors


def test_eval_validation_rejects_duplicate_raw_identity(tmp_path: Path) -> None:
    config = {"evals": [single_eval_entry(32)], "multinode_evals": []}
    prefixes = expected_eval_artifact_prefixes(config)
    write_eval_aggregate(tmp_path)
    (tmp_path / f"{prefixes[0]}00").mkdir()
    (tmp_path / f"{prefixes[0]}01").mkdir()

    errors = validate_eval_artifacts(tmp_path, prefixes)

    assert any("matched 2 raw result artifact dirs" in error for error in errors)


def test_eval_aggregate_validation_is_exact(tmp_path: Path) -> None:
    config = {
        "evals": [single_eval_entry(32)],
        "multinode_evals": [],
    }
    prefixes = expected_eval_artifact_prefixes(config)
    write_eval_aggregate(
        tmp_path,
        [single_eval_result(32), single_eval_result(64)],
    )
    (tmp_path / f"{prefixes[0]}00").mkdir()

    errors = validate_eval_artifacts(
        tmp_path,
        prefixes,
        expected_eval_keys(config),
    )

    assert "eval aggregate artifacts contain 1 unexpected row(s)" in errors


def test_eval_aggregate_validation_rejects_duplicate_identity(
    tmp_path: Path,
) -> None:
    config = {
        "evals": [single_eval_entry(32)],
        "multinode_evals": [],
    }
    prefixes = expected_eval_artifact_prefixes(config)
    write_eval_aggregate(
        tmp_path,
        [single_eval_result(32), single_eval_result(32)],
    )
    (tmp_path / f"{prefixes[0]}00").mkdir()

    errors = validate_eval_artifacts(
        tmp_path,
        prefixes,
        expected_eval_keys(config),
    )

    assert "eval aggregate artifacts contain 1 duplicate row(s)" in errors


def test_fixed_sequence_validation_is_exact(tmp_path: Path) -> None:
    config = {
        "single_node": {
            "1k1k": [single_fixed_entry(8)],
            "8k1k": [],
        },
        "multi_node": {"1k1k": [], "8k1k": []},
    }
    results = tmp_path / "results_bmk"
    results.mkdir()
    (results / "agg_bmk.json").write_text(
        json.dumps([fixed_result(8), fixed_result(16)])
    )

    errors = validate_identity_set(
        "fixed-sequence",
        expected_benchmark_keys(config),
        actual_benchmark_keys(tmp_path),
    )

    assert "fixed-sequence artifacts contain 1 unexpected row(s)" in errors


def test_fixed_sequence_validation_rejects_duplicate_identity(
    tmp_path: Path,
) -> None:
    config = {
        "single_node": {
            "1k1k": [single_fixed_entry(8)],
            "8k1k": [],
        },
        "multi_node": {"1k1k": [], "8k1k": []},
    }
    results = tmp_path / "results_bmk"
    results.mkdir()
    (results / "agg_bmk.json").write_text(
        json.dumps([fixed_result(8), fixed_result(8)])
    )

    errors = validate_fixed_artifacts(
        tmp_path,
        expected_benchmark_keys(config),
    )

    assert "fixed-sequence artifacts contain 1 duplicate row(s)" in errors


def test_agentic_validation_checks_points_raw_and_aggregate(tmp_path: Path) -> None:
    config = {
        "single_node": {"agentic": [single_agentic_entry()]},
        "multi_node": {"agentic": []},
    }
    write_agentic_artifacts(tmp_path)

    assert (
        validate_agentic_artifacts(
            tmp_path,
            expected_agentic_keys(config),
        )
        == []
    )


def test_agentic_validation_accepts_run_sweep_point_artifacts(
    tmp_path: Path,
) -> None:
    config = {
        "single_node": {"agentic": [single_agentic_entry()]},
        "multi_node": {"agentic": []},
    }
    write_agentic_artifacts(tmp_path, aggregate=False)

    assert (
        validate_agentic_artifacts(
            tmp_path,
            expected_agentic_keys(config),
        )
        == []
    )


def test_agentic_validation_rejects_extra_identity(tmp_path: Path) -> None:
    config = {
        "single_node": {"agentic": [single_agentic_entry()]},
        "multi_node": {"agentic": []},
    }
    write_agentic_artifacts(tmp_path)
    extra_dir = tmp_path / "bmk_agentic_extra"
    extra_dir.mkdir()
    (extra_dir / "extra.json").write_text(json.dumps(agentic_result(32)))
    (tmp_path / "agentic_extra").mkdir()
    summary = tmp_path / "agentic_aggregated" / "summary.csv"
    summary.write_text(summary.read_text() + "agentic_extra,SUCCESS\n")

    errors = validate_agentic_artifacts(
        tmp_path,
        expected_agentic_keys(config),
    )

    assert "agentic artifacts contain 1 unexpected row(s)" in errors


def test_agentic_validation_requires_point_and_raw_artifacts(
    tmp_path: Path,
) -> None:
    config = {
        "single_node": {"agentic": [single_agentic_entry()]},
        "multi_node": {"agentic": []},
    }
    aggregate = tmp_path / "results_bmk"
    aggregate.mkdir()
    (aggregate / "agg_bmk.json").write_text(
        json.dumps([agentic_result()])
    )

    errors = validate_agentic_artifacts(
        tmp_path,
        expected_agentic_keys(config),
    )

    assert "agentic artifacts are missing 1 expected row(s)" in errors


def test_agentic_validation_rejects_duplicate_point_identity(
    tmp_path: Path,
) -> None:
    config = {
        "single_node": {"agentic": [single_agentic_entry()]},
        "multi_node": {"agentic": []},
    }
    write_agentic_artifacts(tmp_path, aggregate=False)
    point_dir = (
        tmp_path / "bmk_agentic_dsv4_tp8_conc16_offloadcpu_result"
    )
    result_path = next(point_dir.glob("*.json"))
    result_path.write_text(
        json.dumps([agentic_result(), agentic_result()])
    )

    errors = validate_agentic_artifacts(
        tmp_path,
        expected_agentic_keys(config),
    )

    assert "agentic point artifacts contain 1 duplicate row(s)" in errors


def test_eval_only_main_does_not_require_benchmark_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = {
        "single_node": {"1k1k": [], "8k1k": [], "agentic": []},
        "multi_node": {"1k1k": [], "8k1k": [], "agentic": []},
        "evals": [single_eval_entry(32)],
        "multinode_evals": [],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    prefixes = expected_eval_artifact_prefixes(config)
    write_eval_aggregate(tmp_path, [single_eval_result(32)])
    (tmp_path / f"{prefixes[0]}00").mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_reusable_sweep_artifacts.py",
            "--config-json",
            str(config_path),
            "--artifacts-dir",
            str(tmp_path),
        ],
    )

    assert main() == 0
