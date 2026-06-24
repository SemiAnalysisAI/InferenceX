from __future__ import annotations

import json
import sys
from pathlib import Path

from validate_reusable_sweep_artifacts import (
    agentic_key,
    main,
    validate_agentic_artifacts,
    validate_eval_artifacts,
    validate_fixed_artifacts,
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


def single_eval_result(
    conc: int,
    runner: str = "h100-dgxc-slurm",
    isl: int = 8192,
    osl: int = 1024,
) -> dict:
    return {
        "is_multinode": False,
        "hw": runner.upper(),
        "model_prefix": "gptoss",
        "framework": "vllm",
        "precision": "fp4",
        "spec_decoding": "none",
        "isl": isl,
        "osl": osl,
        "tp": 2,
        "ep": 1,
        "dp_attention": False,
        "conc": conc,
        "task": "gsm8k",
    }


def single_eval_meta(
    conc: int,
    runner: str = "h100-dgxc-slurm",
    isl: int = 8192,
    osl: int = 1024,
) -> dict:
    row = single_eval_result(conc, runner, isl, osl)
    row["infmax_model_prefix"] = row.pop("model_prefix")
    return row


def write_raw_eval_artifact(
    root: Path,
    conc: int,
    *,
    logical_runner: str = "h100-dgxc-slurm",
    physical_runner: str = "h100-dgxc-slurm_00",
    isl: int = 8192,
    osl: int = 1024,
) -> None:
    artifact_dir = root / f"eval_result_conc{conc}_{physical_runner}"
    artifact_dir.mkdir()
    (artifact_dir / "meta_env.json").write_text(
        json.dumps(single_eval_meta(conc, logical_runner, isl, osl))
    )


def multinode_eval_result(conc: int) -> dict:
    return {
        "is_multinode": True,
        "hw": "GB200",
        "model_prefix": "gptoss",
        "framework": "dynamo-sglang",
        "precision": "fp8",
        "spec_decoding": "none",
        "isl": 8192,
        "osl": 1024,
        "prefill_tp": 4,
        "prefill_ep": 1,
        "prefill_dp_attention": False,
        "prefill_num_workers": 1,
        "decode_tp": 8,
        "decode_ep": 1,
        "decode_dp_attention": True,
        "decode_num_workers": 2,
        "conc": conc,
        "task": "gsm8k",
    }


def write_raw_batched_eval_artifact(
    root: Path,
    concs: list[int],
    *,
    completed_concs: list[int] | None = None,
    failed_concs: list[int] | None = None,
) -> None:
    artifact_dir = root / "eval_gptoss_8k1k_batch"
    artifact_dir.mkdir()
    meta = multinode_eval_result(concs[0])
    meta["infmax_model_prefix"] = meta.pop("model_prefix")
    meta["eval_concs"] = concs
    meta["completed_eval_concs"] = (
        concs if completed_concs is None else completed_concs
    )
    meta["failed_eval_concs"] = (
        [] if failed_concs is None else failed_concs
    )
    (artifact_dir / "meta_env.json").write_text(json.dumps(meta))


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

    assert agentic_key(row) == (
        "multi",
        "gb200",
        "dsv4",
        "dynamo-sglang",
        "fp8",
        "none",
        True,
        4,
        2,
        True,
        2,
        8,
        4,
        False,
        3,
        64,
    )


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
    write_eval_aggregate(
        tmp_path,
        [single_eval_result(32), single_eval_result(64)],
    )

    (tmp_path / "eval_server_logs_gptoss_8k1k_runner").mkdir()
    (tmp_path / "eval_gpu_metrics_gptoss_8k1k_runner").mkdir()
    write_raw_eval_artifact(tmp_path, 32)

    errors = validate_eval_artifacts(tmp_path)

    assert any("unexpected" in error for error in errors)


def test_eval_validation_accepts_matching_raw_and_aggregate(
    tmp_path: Path,
) -> None:
    write_eval_aggregate(
        tmp_path,
        [single_eval_result(32), single_eval_result(64)],
    )
    write_raw_eval_artifact(tmp_path, 32)
    write_raw_eval_artifact(
        tmp_path,
        64,
        physical_runner="h100-dgxc-slurm_01",
    )

    assert validate_eval_artifacts(tmp_path) == []


def test_eval_validation_distinguishes_sequence_lengths(tmp_path: Path) -> None:
    write_eval_aggregate(
        tmp_path,
        [
            single_eval_result(32, isl=1024),
            single_eval_result(32, isl=8192),
        ],
    )
    write_raw_eval_artifact(tmp_path, 32, isl=1024)
    write_raw_eval_artifact(
        tmp_path,
        32,
        physical_runner="h100-dgxc-slurm_01",
        isl=8192,
    )

    assert validate_eval_artifacts(tmp_path) == []


def test_eval_validation_rejects_raw_aggregate_mismatch(tmp_path: Path) -> None:
    write_eval_aggregate(tmp_path, [single_eval_result(32)])
    write_raw_eval_artifact(tmp_path, 32)
    write_raw_eval_artifact(
        tmp_path,
        64,
        physical_runner="h100-dgxc-slurm_01",
    )

    errors = validate_eval_artifacts(tmp_path)

    assert any("missing" in error for error in errors)


def test_eval_validation_rejects_duplicate_raw_identity(tmp_path: Path) -> None:
    write_eval_aggregate(tmp_path, [single_eval_result(32)])
    write_raw_eval_artifact(tmp_path, 32)
    write_raw_eval_artifact(
        tmp_path,
        32,
        physical_runner="h100-dgxc-slurm_01",
    )

    errors = validate_eval_artifacts(tmp_path)

    assert any("duplicate" in error for error in errors)


def test_eval_validation_uses_logical_runner_from_metadata(
    tmp_path: Path,
) -> None:
    write_eval_aggregate(tmp_path, [single_eval_result(64, "mi300x")])
    write_raw_eval_artifact(
        tmp_path,
        64,
        logical_runner="mi300x",
        physical_runner="mi300x-amds_04",
    )

    assert validate_eval_artifacts(tmp_path) == []


def test_eval_validation_expands_one_batched_multinode_artifact(
    tmp_path: Path,
) -> None:
    concs = [4, 16, 64]
    write_eval_aggregate(
        tmp_path,
        [multinode_eval_result(conc) for conc in concs],
    )
    write_raw_batched_eval_artifact(tmp_path, concs)

    assert validate_eval_artifacts(tmp_path) == []


def test_eval_validation_accepts_completed_points_from_failed_batch(
    tmp_path: Path,
) -> None:
    requested_concs = [4, 16, 64]
    completed_concs = [4, 64]
    write_eval_aggregate(
        tmp_path,
        [multinode_eval_result(conc) for conc in completed_concs],
    )
    write_raw_batched_eval_artifact(
        tmp_path,
        requested_concs,
        completed_concs=completed_concs,
        failed_concs=[16],
    )

    assert validate_eval_artifacts(tmp_path) == []


def test_eval_aggregate_validation_is_exact(tmp_path: Path) -> None:
    write_eval_aggregate(
        tmp_path,
        [single_eval_result(32), single_eval_result(64)],
    )
    write_raw_eval_artifact(tmp_path, 32)

    errors = validate_eval_artifacts(tmp_path)

    assert any(
        "eval aggregate" in error and "unexpected" in error
        for error in errors
    )


def test_eval_aggregate_validation_rejects_duplicate_identity(
    tmp_path: Path,
) -> None:
    write_eval_aggregate(
        tmp_path,
        [single_eval_result(32), single_eval_result(32)],
    )
    write_raw_eval_artifact(tmp_path, 32)

    errors = validate_eval_artifacts(tmp_path)

    assert any(
        "eval aggregate" in error and "duplicate" in error
        for error in errors
    )


def test_fixed_sequence_validation_accepts_unique_source_rows(tmp_path: Path) -> None:
    results = tmp_path / "results_bmk"
    results.mkdir()
    (results / "agg_bmk.json").write_text(
        json.dumps([fixed_result(8), fixed_result(16)])
    )

    assert validate_fixed_artifacts(tmp_path) == []


def test_fixed_sequence_validation_rejects_duplicate_identity(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results_bmk"
    results.mkdir()
    (results / "agg_bmk.json").write_text(
        json.dumps([fixed_result(8), fixed_result(8)])
    )

    errors = validate_fixed_artifacts(tmp_path)

    assert "fixed-sequence artifacts contain 1 duplicate row(s)" in errors


def test_agentic_validation_checks_points_raw_and_aggregate(tmp_path: Path) -> None:
    write_agentic_artifacts(tmp_path)

    assert validate_agentic_artifacts(tmp_path) == []


def test_agentic_validation_accepts_run_sweep_point_artifacts(
    tmp_path: Path,
) -> None:
    write_agentic_artifacts(tmp_path, aggregate=False)

    assert validate_agentic_artifacts(tmp_path) == []


def test_agentic_validation_accepts_additional_source_identity(
    tmp_path: Path,
) -> None:
    write_agentic_artifacts(tmp_path)
    extra_dir = tmp_path / "bmk_agentic_extra"
    extra_dir.mkdir()
    (extra_dir / "extra.json").write_text(json.dumps(agentic_result(32)))
    (tmp_path / "agentic_extra").mkdir()
    summary = tmp_path / "agentic_aggregated" / "summary.csv"
    summary.write_text(summary.read_text() + "agentic_extra,SUCCESS\n")

    assert validate_agentic_artifacts(tmp_path) == []


def test_agentic_validation_requires_point_and_raw_artifacts(
    tmp_path: Path,
) -> None:
    aggregate = tmp_path / "results_bmk"
    aggregate.mkdir()
    (aggregate / "agg_bmk.json").write_text(
        json.dumps([agentic_result()])
    )

    errors = validate_agentic_artifacts(tmp_path)

    assert "agentic aggregate artifacts contain 1 unexpected row(s)" in errors


def test_agentic_validation_rejects_duplicate_point_identity(
    tmp_path: Path,
) -> None:
    write_agentic_artifacts(tmp_path, aggregate=False)
    point_dir = (
        tmp_path / "bmk_agentic_dsv4_tp8_conc16_offloadcpu_result"
    )
    result_path = next(point_dir.glob("*.json"))
    result_path.write_text(
        json.dumps([agentic_result(), agentic_result()])
    )

    errors = validate_agentic_artifacts(tmp_path)

    assert "agentic point artifacts contain 1 duplicate row(s)" in errors


def test_eval_only_main_does_not_require_benchmark_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    write_eval_aggregate(tmp_path, [single_eval_result(32)])
    write_raw_eval_artifact(tmp_path, 32)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_reusable_sweep_artifacts.py",
            "--artifacts-dir",
            str(tmp_path),
        ],
    )

    assert main() == 0
