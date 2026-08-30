from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from validate_reusable_sweep_artifacts import (
    agentic_key,
    benchmark_key,
    eval_key,
    eval_result_key,
    dedupe_reran_evals,
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
    eval_suite: str | None = None,
) -> dict:
    row = {
        "is_multinode": False,
        "hw": runner.upper(),
        "model_prefix": "gptoss",
        "framework": "vllm",
        "precision": "fp4",
        "spec_decoding": "none",
        "isl": isl,
        "osl": osl,
        "tp": 2,
        "pp": 1,
        "dcp_size": 1,
        "pcp_size": 1,
        "ep": 1,
        "dp_attention": False,
        "conc": conc,
        "task": "gsm8k",
    }
    if eval_suite is not None:
        row["eval_suite"] = eval_suite
    return row


def single_eval_meta(
    conc: int,
    runner: str = "h100-dgxc-slurm",
    isl: int = 8192,
    osl: int = 1024,
    eval_suite: str | None = None,
) -> dict:
    row = single_eval_result(conc, runner, isl, osl, eval_suite)
    row["infmax_model_prefix"] = row.pop("model_prefix")
    return row


def raw_eval_result(
    score: float = 0.9,
    *,
    effective: object = 10,
    task: str = "gsm8k",
) -> dict:
    return {
        "lm_eval_version": "0.4.0",
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
        "n-samples": {task: {"effective": effective}},
    }


def write_raw_eval_artifact(
    root: Path,
    conc: int,
    *,
    logical_runner: str = "h100-dgxc-slurm",
    physical_runner: str = "h100-dgxc-slurm_00",
    isl: int = 8192,
    osl: int = 1024,
    eval_suite: str | None = None,
) -> None:
    artifact_dir = root / f"eval_result_conc{conc}_{physical_runner}"
    artifact_dir.mkdir()
    (artifact_dir / "meta_env.json").write_text(
        json.dumps(
            single_eval_meta(
                conc,
                logical_runner,
                isl,
                osl,
                eval_suite,
            )
        )
    )
    (artifact_dir / "results_test.json").write_text(
        json.dumps(raw_eval_result())
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
    completed = concs if completed_concs is None else completed_concs
    meta = multinode_eval_result(concs[0])
    meta["infmax_model_prefix"] = meta.pop("model_prefix")
    meta["eval_concs"] = concs
    meta["completed_eval_concs"] = completed
    meta["failed_eval_concs"] = (
        [] if failed_concs is None else failed_concs
    )
    (artifact_dir / "meta_env.json").write_text(json.dumps(meta))
    for conc in completed:
        (artifact_dir / f"results_test_conc{conc}.json").write_text(
            json.dumps(raw_eval_result())
        )


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
        "pp": 1,
        "dcp_size": 1,
        "pcp_size": 1,
        "ep": 1,
        "dp_attention": False,
        "conc": conc,
        "is_multinode": False,
    }


def agentic_result(conc: int = 16) -> dict:
    return {
        "hw": "b200-nscale",
        "infmax_model_prefix": "dsv4",
        "framework": "vllm",
        "precision": "fp4",
        "scenario_type": "agentic-coding",
        "is_multinode": False,
        "tp": 8,
        "pp": 1,
        "dcp_size": 1,
        "pcp_size": 1,
        "ep": 8,
        "dp_attention": "true",
        "conc": conc,
        "offloading": "cpu",
    }


def test_single_node_reusable_keys_normalize_legacy_parallelism_and_separate_variants() -> None:
    cases = (
        ("fixed", benchmark_key, fixed_result(16)),
        ("agentic", agentic_key, agentic_result()),
        ("eval", eval_key, single_eval_result(16)),
    )

    for name, identity, row in cases:
        legacy_row = dict(row)
        legacy_row.pop("pp")
        legacy_row.pop("dcp_size")
        legacy_row.pop("pcp_size")
        assert identity(legacy_row) == identity(row), name
        assert identity({**row, "pp": 2}) != identity(row), name
        assert identity({**row, "dcp_size": 2}) != identity(row), name
        assert identity({**row, "pcp_size": 2}) != identity(row), name


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
        "prefill_pp": 2,
        "prefill_dcp_size": 2,
        "prefill_pcp_size": 2,
        "prefill_ep": 2,
        "prefill_dp_attention": "true",
        "prefill_num_workers": 2,
        "decode_tp": 8,
        "decode_pp": 2,
        "decode_dcp_size": 4,
        "decode_pcp_size": 1,
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
        2,
        2,
        2,
        True,
        2,
        8,
        2,
        4,
        1,
        4,
        False,
        3,
        64,
    )

    for identity in (benchmark_key, agentic_key, eval_key):
        legacy_row = dict(row)
        for field in (
            "prefill_pp",
            "prefill_dcp_size",
            "prefill_pcp_size",
            "decode_pp",
            "decode_dcp_size",
            "decode_pcp_size",
        ):
            legacy_row.pop(field)
        default_row = {
            **row,
            "prefill_pp": 1,
            "prefill_dcp_size": 1,
            "prefill_pcp_size": 1,
            "decode_pp": 1,
            "decode_dcp_size": 1,
            "decode_pcp_size": 1,
        }
        assert identity(legacy_row) == identity(default_row)
        for field in (
            "prefill_pp",
            "prefill_dcp_size",
            "prefill_pcp_size",
            "decode_pp",
            "decode_dcp_size",
            "decode_pcp_size",
        ):
            assert identity({**default_row, field: 2}) != identity(default_row)


def test_agentic_identity_freezes_nested_kv_offload_backend() -> None:
    row = {
        **agentic_result(),
        "kv_offloading": "dram",
        "kv_offload_backend": {
            "name": "native",
            "options": {"layers": ["cpu", "gpu"]},
        },
    }
    row.pop("offloading")

    identity = agentic_key(row)

    assert isinstance(hash(identity), int)
    assert identity == agentic_key(
        {
            **row,
            "kv_offload_backend": {
                "options": {"layers": ["cpu", "gpu"]},
                "name": "native",
            },
        }
    )
    assert identity != agentic_key(
        {
            **row,
            "kv_offload_backend": {
                "name": "native",
                "options": {"layers": ["cpu"]},
            },
        }
    )


def write_agentic_artifacts(
    root: Path,
    conc: int = 16,
) -> None:
    result_name = f"dsv4_tp8_conc{conc}_offloadcpu_result"
    point_dir = root / f"bmk_agentic_{result_name}"
    point_dir.mkdir()
    (point_dir / f"{result_name}.json").write_text(
        json.dumps(agentic_result(conc))
    )
    (root / f"agentic_{result_name}").mkdir()


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


def test_eval_validation_accepts_matching_legacy_artifacts_without_suite(
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


def test_eval_validation_separates_explicit_suite_identities(
    tmp_path: Path,
) -> None:
    gsm8k = single_eval_result(32, eval_suite="gsm8k")
    tool_use = single_eval_result(
        32,
        eval_suite="kimi_tool_call_schema",
    )
    write_eval_aggregate(tmp_path, [gsm8k, tool_use])
    write_raw_eval_artifact(
        tmp_path,
        32,
        eval_suite="gsm8k",
    )
    write_raw_eval_artifact(
        tmp_path,
        32,
        physical_runner="h100-dgxc-slurm_01",
        eval_suite="kimi_tool_call_schema",
    )

    assert eval_key(gsm8k) != eval_key(tool_use)
    assert validate_eval_artifacts(tmp_path) == []


def test_eval_result_key_includes_task_identity() -> None:
    gsm8k = single_eval_result(32, eval_suite="tool_use")
    bfcl = {**gsm8k, "task": "bfcl_smoke"}

    assert eval_result_key(gsm8k) != eval_result_key(bfcl)


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
        physical_runner="mi300x-amd_04",
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


def test_eval_validation_accepts_legacy_batch_without_failed_list(
    tmp_path: Path,
) -> None:
    concs = [4, 16]
    write_eval_aggregate(
        tmp_path,
        [multinode_eval_result(conc) for conc in concs],
    )
    write_raw_batched_eval_artifact(tmp_path, concs)
    meta_path = tmp_path / "eval_gptoss_8k1k_batch" / "meta_env.json"
    meta = json.loads(meta_path.read_text())
    meta.pop("failed_eval_concs")
    meta_path.write_text(json.dumps(meta))

    assert validate_eval_artifacts(tmp_path) == []


def test_eval_validation_rejects_failed_batch(
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

    errors = validate_eval_artifacts(tmp_path)

    assert any("reports failed eval concurrencies" in error for error in errors)


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


def test_eval_aggregate_validation_rejects_non_list_file(
    tmp_path: Path,
) -> None:
    write_raw_eval_artifact(tmp_path, 32)
    eval_dir = tmp_path / "eval_results_all"
    eval_dir.mkdir()
    (eval_dir / "agg_eval_all.json").write_text(
        json.dumps(single_eval_result(32))
    )

    errors = validate_eval_artifacts(tmp_path)

    assert any("is not a list" in error for error in errors)


def test_eval_aggregate_validation_rejects_non_object_row(
    tmp_path: Path,
) -> None:
    write_raw_eval_artifact(tmp_path, 32)
    eval_dir = tmp_path / "eval_results_all"
    eval_dir.mkdir()
    (eval_dir / "agg_eval_all.json").write_text(
        json.dumps([single_eval_result(32), "not-a-row"])
    )

    errors = validate_eval_artifacts(tmp_path)

    assert any("row 1 is not an object" in error for error in errors)


def test_eval_validation_rejects_scores_outside_unit_interval(
    tmp_path: Path,
) -> None:
    for index, score in enumerate((-0.01, 1.01)):
        root = tmp_path / str(index)
        root.mkdir()
        write_raw_eval_artifact(root, 32)
        result_path = next(
            (root / "eval_result_conc32_h100-dgxc-slurm_00").glob(
                "results*.json"
            )
        )
        result_path.write_text(json.dumps(raw_eval_result(score)))
        write_eval_aggregate(root, [single_eval_result(32)])

        errors = validate_eval_artifacts(root)

        assert any("invalid score" in error for error in errors)


def test_eval_validation_rejects_directory_with_only_markerless_results(
    tmp_path: Path,
) -> None:
    write_raw_eval_artifact(tmp_path, 32)
    result_path = next(
        (
            tmp_path / "eval_result_conc32_h100-dgxc-slurm_00"
        ).glob("results*.json")
    )
    data = json.loads(result_path.read_text())
    data.pop("lm_eval_version")
    result_path.write_text(json.dumps(data))
    write_eval_aggregate(tmp_path, [single_eval_result(32)])

    errors = validate_eval_artifacts(tmp_path)

    assert any("has no recognized eval result" in error for error in errors)


def test_eval_validation_accepts_neutral_result_format_marker(
    tmp_path: Path,
) -> None:
    write_raw_eval_artifact(tmp_path, 32)
    result_path = next(
        (
            tmp_path / "eval_result_conc32_h100-dgxc-slurm_00"
        ).glob("results*.json")
    )
    data = json.loads(result_path.read_text())
    data.pop("lm_eval_version")
    data["result_format"] = "inferencex-eval-v1"
    result_path.write_text(json.dumps(data))
    write_eval_aggregate(tmp_path, [single_eval_result(32)])

    assert validate_eval_artifacts(tmp_path) == []


def test_eval_validation_accepts_neutral_filter_primary_score(
    tmp_path: Path,
) -> None:
    write_raw_eval_artifact(tmp_path, 32, eval_suite="bfcl_smoke")
    result_path = next(tmp_path.glob("eval_*/results*.json"))
    data = json.loads(result_path.read_text())
    data["results"]["gsm8k"] = {
        "acc,none": 1.0,
        "acc_stderr,none": 0.0,
    }
    data["configs"]["gsm8k"] = {
        "metric_list": [{"metric": "acc"}],
        "filter_list": [{"name": "none"}],
    }
    result_path.write_text(json.dumps(data))
    write_eval_aggregate(
        tmp_path,
        [single_eval_result(32, eval_suite="bfcl_smoke")],
    )

    assert validate_eval_artifacts(tmp_path) == []


def test_eval_validation_accepts_multiple_tasks_from_one_artifact(
    tmp_path: Path,
) -> None:
    write_raw_eval_artifact(tmp_path, 32, eval_suite="tool_use")
    result_path = next(tmp_path.glob("eval_*/results*.json"))
    data = json.loads(result_path.read_text())
    data["results"]["bfcl_smoke"] = {
        "exact_match,strict-match": 1.0,
        "exact_match_stderr,strict-match": 0.0,
    }
    data["configs"]["bfcl_smoke"] = data["configs"]["gsm8k"]
    data["n-samples"]["bfcl_smoke"] = {"effective": 1}
    result_path.write_text(json.dumps(data))
    base_row = single_eval_result(32, eval_suite="tool_use")
    write_eval_aggregate(
        tmp_path,
        [base_row, {**base_row, "task": "bfcl_smoke"}],
    )

    assert validate_eval_artifacts(tmp_path) == []


def test_eval_validation_rejects_invalid_legacy_concurrency(
    tmp_path: Path,
) -> None:
    for index, conc in enumerate((0, -1, True, "4")):
        root = tmp_path / str(index)
        root.mkdir()
        artifact_dir = root / "eval_invalid_legacy"
        artifact_dir.mkdir()
        (artifact_dir / "meta_env.json").write_text(
            json.dumps({**single_eval_meta(4), "conc": conc})
        )
        (artifact_dir / "results_test.json").write_text(
            json.dumps(raw_eval_result())
        )

        errors = validate_eval_artifacts(root)

        assert any("invalid legacy concurrency" in error for error in errors)


def test_eval_validation_rejects_malformed_batch_metadata(
    tmp_path: Path,
) -> None:
    cases = (
        ({"eval_concs": [4, True]}, "invalid eval_concs"),
        ({"eval_concs": [4, 4]}, "duplicate eval_concs"),
        ({"completed_eval_concs": [4, 4]}, "duplicate completed_eval_concs"),
        ({"failed_eval_concs": [16, 16]}, "duplicate failed_eval_concs"),
        ({"completed_eval_concs": [4, 16], "failed_eval_concs": [16]}, "overlapping"),
        ({"completed_eval_concs": [4, 32]}, "unexpected"),
        ({"completed_eval_concs": [4], "failed_eval_concs": [32]}, "failed unexpected"),
        ({"completed_eval_concs": "4"}, "invalid batched concurrency metadata"),
        ({"completed_eval_concs": [4], "failed_eval_concs": [16]}, "reports failed"),
        ({"failed_eval_concs": None}, "invalid batched concurrency metadata"),
        ({"eval_concs": [], "completed_eval_concs": []}, "empty eval_concs"),
    )
    for index, (overrides, expected) in enumerate(cases):
        root = tmp_path / str(index)
        root.mkdir()
        artifact_dir = root / "eval_invalid_batch"
        artifact_dir.mkdir()
        meta = _dd_meta(0)
        meta.update(
            {
                "eval_concs": [4, 16],
                "completed_eval_concs": [4, 16],
                "failed_eval_concs": [],
            }
        )
        meta.update(overrides)
        (artifact_dir / "meta_env.json").write_text(json.dumps(meta))

        errors = validate_eval_artifacts(root)

        assert any(expected in error for error in errors), errors


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


def test_agentic_validation_checks_points_and_raw_artifacts(tmp_path: Path) -> None:
    write_agentic_artifacts(tmp_path)

    assert validate_agentic_artifacts(tmp_path) == []


def test_agentic_validation_accepts_run_sweep_point_artifacts(
    tmp_path: Path,
) -> None:
    write_agentic_artifacts(tmp_path)

    assert validate_agentic_artifacts(tmp_path) == []


def test_agentic_validation_accepts_additional_source_identity(
    tmp_path: Path,
) -> None:
    write_agentic_artifacts(tmp_path)
    extra_dir = tmp_path / "bmk_agentic_extra"
    extra_dir.mkdir()
    (extra_dir / "extra.json").write_text(json.dumps(agentic_result(32)))
    (tmp_path / "agentic_extra").mkdir()

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
    write_agentic_artifacts(tmp_path)
    point_dir = (
        tmp_path / "bmk_agentic_dsv4_tp8_conc16_offloadcpu_result"
    )
    result_path = next(point_dir.glob("*.json"))
    result_path.write_text(
        json.dumps([agentic_result(), agentic_result()])
    )

    errors = validate_agentic_artifacts(tmp_path)

    assert "agentic point artifacts contain 1 duplicate row(s)" in errors


def test_agentic_validation_handles_mapping_kv_offload_backend(
    tmp_path: Path,
) -> None:
    row = {
        **agentic_result(),
        "kv_offloading": "dram",
        "kv_offload_backend": {"name": "native"},
    }
    row.pop("offloading")
    point_dir = tmp_path / "bmk_agentic_native_offload"
    point_dir.mkdir()
    (point_dir / "result.json").write_text(json.dumps([row, row]))
    (tmp_path / "agentic_native_offload").mkdir()

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


# ── dedupe_reran_evals ────────────────────────────────────────────────────────


def _dd_meta(conc: int) -> dict:
    return {
        "is_multinode": True,
        "hw": "b300",
        "infmax_model_prefix": "minimaxm3",
        "framework": "dynamo-vllm",
        "precision": "fp4",
        "spec_decoding": "none",
        "isl": 8192,
        "osl": 1024,
        "prefill_tp": 2,
        "prefill_ep": 2,
        "prefill_dp_attention": True,
        "prefill_num_workers": 4,
        "decode_tp": 8,
        "decode_ep": 8,
        "decode_dp_attention": True,
        "decode_num_workers": 2,
        "conc": conc,
    }


def _dd_agg_row(conc: int, source: str, em_strict: float) -> dict:
    row = _dd_meta(conc)
    row["model_prefix"] = row.pop("infmax_model_prefix")
    row["task"] = "gsm8k"
    row["em_strict"] = em_strict
    row["source"] = source
    return row


def _dd_write_aggregate(root: Path, rows: list[dict]) -> Path:
    eval_dir = root / "eval_results_all"
    eval_dir.mkdir(exist_ok=True)
    path = eval_dir / "agg_eval_all.json"
    path.write_text(json.dumps(rows, indent=2))
    return path


def _dd_write_legacy_raw(
    root: Path, name: str, conc: int, timestamp: str | None
) -> None:
    artifact_dir = root / name
    artifact_dir.mkdir()
    (artifact_dir / "meta_env.json").write_text(json.dumps(_dd_meta(conc)))
    if timestamp is not None:
        (artifact_dir / f"results_{timestamp}.json").write_text(
            json.dumps(raw_eval_result())
        )


def test_dedupe_keeps_latest_legacy_rerun(tmp_path: Path) -> None:
    # Three reruns of one eval plus a result-less attempt, mirroring a flaky
    # config retried until it passed.
    old, mid, new, empty = (
        "eval_minimaxm3_conc4096_b300-nv_15",
        "eval_minimaxm3_conc4096_b300-nv_16",
        "eval_minimaxm3_conc4096_b300-nv_12",
        "eval_minimaxm3_conc4096_b300-nv_03",
    )
    _dd_write_legacy_raw(tmp_path, old, 4096, "2026-06-26T13-00-22.596040")
    _dd_write_legacy_raw(tmp_path, mid, 4096, "2026-06-26T19-00-52.356121")
    _dd_write_legacy_raw(tmp_path, new, 4096, "2026-06-27T04-28-31.838775")
    _dd_write_legacy_raw(tmp_path, empty, 4096, None)
    _dd_write_aggregate(
        tmp_path,
        [
            _dd_agg_row(4096, f"eval_results/{old}/results_2026-06-26T13-00-22.596040.json", 0.83),
            _dd_agg_row(4096, f"eval_results/{new}/results_2026-06-27T04-28-31.838775.json", 0.95),
            _dd_agg_row(4096, f"eval_results/{mid}/results_2026-06-26T19-00-52.356121.json", 0.78),
        ],
    )

    messages = dedupe_reran_evals(tmp_path)

    assert validate_eval_artifacts(tmp_path) == []
    rows = json.loads((tmp_path / "eval_results_all" / "agg_eval_all.json").read_text())
    assert [r["em_strict"] for r in rows] == [0.95]
    assert (tmp_path / new).is_dir()
    for superseded in (old, mid, empty):
        assert not (tmp_path / superseded).exists()
    assert any("kept 1 of 3" in message for message in messages)


def test_dedupe_leaves_ambiguous_artifacts_for_validation(tmp_path: Path) -> None:
    # Result-less raw artifacts cannot be ordered or reused. Dedupe leaves them
    # for validation to reject as missing recognized results.
    for name in ("eval_minimaxm3_conc4096_b300-nv_01", "eval_minimaxm3_conc4096_b300-nv_02"):
        _dd_write_legacy_raw(tmp_path, name, 4096, None)
    _dd_write_aggregate(
        tmp_path,
        [_dd_agg_row(4096, "eval_results/eval_minimaxm3_conc4096_b300-nv_01/x.json", 0.9)],
    )

    assert dedupe_reran_evals(tmp_path) == []
    assert any(
        "no recognized eval result" in error
        for error in validate_eval_artifacts(tmp_path)
    )


def test_dedupe_is_noop_for_clean_artifacts(tmp_path: Path) -> None:
    name = "eval_minimaxm3_conc4096_b300-nv_01"
    _dd_write_legacy_raw(tmp_path, name, 4096, "2026-06-27T04-28-31.838775")
    agg_path = _dd_write_aggregate(
        tmp_path,
        [_dd_agg_row(4096, f"eval_results/{name}/results_2026-06-27T04-28-31.838775.json", 0.95)],
    )
    before = agg_path.read_text()

    assert dedupe_reran_evals(tmp_path) == []
    assert agg_path.read_text() == before
    assert (tmp_path / name).is_dir()
    assert validate_eval_artifacts(tmp_path) == []


def test_dedupe_prunes_superseded_batched_conc(tmp_path: Path) -> None:
    # Two batched reruns overlap on conc 32; the newer run wins that conc while
    # each run keeps the concurrencies unique to it.
    older = tmp_path / "eval_minimaxm3_batch_b300-nv_05"
    newer = tmp_path / "eval_minimaxm3_batch_b300-nv_09"
    for artifact_dir, concs, stamp in (
        (older, [16, 32], "2026-06-26T10-00-00.000000"),
        (newer, [32, 64], "2026-06-26T20-00-00.000000"),
    ):
        artifact_dir.mkdir()
        meta = _dd_meta(0)
        meta["eval_concs"] = concs
        meta["completed_eval_concs"] = list(concs)
        meta["failed_eval_concs"] = []
        (artifact_dir / "meta_env.json").write_text(json.dumps(meta))
        for conc in concs:
            (artifact_dir / f"results_{stamp}_conc{conc}.json").write_text(
                json.dumps(raw_eval_result())
            )
    _dd_write_aggregate(
        tmp_path,
        [
            _dd_agg_row(16, f"eval_results/{older.name}/results_2026-06-26T10-00-00.000000_conc16.json", 0.50),
            _dd_agg_row(32, f"eval_results/{older.name}/results_2026-06-26T10-00-00.000000_conc32.json", 0.40),
            _dd_agg_row(32, f"eval_results/{newer.name}/results_2026-06-26T20-00-00.000000_conc32.json", 0.90),
            _dd_agg_row(64, f"eval_results/{newer.name}/results_2026-06-26T20-00-00.000000_conc64.json", 0.70),
        ],
    )

    dedupe_reran_evals(tmp_path)

    assert validate_eval_artifacts(tmp_path) == []
    assert json.loads((older / "meta_env.json").read_text())["completed_eval_concs"] == [16]
    assert json.loads((older / "meta_env.json").read_text())["eval_concs"] == [16]
    assert not (older / "results_2026-06-26T10-00-00.000000_conc32.json").exists()
    assert (older / "results_2026-06-26T10-00-00.000000_conc16.json").exists()
    rows = json.loads((tmp_path / "eval_results_all" / "agg_eval_all.json").read_text())
    assert [r["em_strict"] for r in rows if r["conc"] == 32] == [0.90]


def test_dedupe_orders_timestamped_and_legacy_results_coherently(
    tmp_path: Path,
) -> None:
    timestamped = "eval_minimaxm3_conc4096_b300-nv_timestamped"
    legacy = "eval_minimaxm3_conc4096_b300-nv_legacy"
    _dd_write_legacy_raw(
        tmp_path,
        timestamped,
        4096,
        "2026-06-27T04-28-31.838775",
    )
    _dd_write_legacy_raw(tmp_path, legacy, 4096, "retry")
    legacy_result = next((tmp_path / legacy).glob("results*.json"))
    future_ns = 2_000_000_000_000_000_000
    os.utime(legacy_result, ns=(future_ns, future_ns))
    _dd_write_aggregate(
        tmp_path,
        [
            _dd_agg_row(
                4096,
                f"eval_results/{timestamped}/results_2026-06-27T04-28-31.838775.json",
                0.5,
            ),
            _dd_agg_row(
                4096,
                f"eval_results/{legacy}/results_retry.json",
                0.9,
            ),
        ],
    )

    dedupe_reran_evals(tmp_path)

    assert validate_eval_artifacts(tmp_path) == []
    assert (tmp_path / legacy).is_dir()
    assert not (tmp_path / timestamped).exists()
    rows = json.loads(
        (tmp_path / "eval_results_all" / "agg_eval_all.json").read_text()
    )
    assert [row["em_strict"] for row in rows] == [0.9]


def test_dedupe_collapses_identity_across_aggregate_files(
    tmp_path: Path,
) -> None:
    old = "eval_minimaxm3_conc4096_b300-nv_old"
    new = "eval_minimaxm3_conc4096_b300-nv_new"
    _dd_write_legacy_raw(
        tmp_path,
        old,
        4096,
        "2026-06-26T01-00-00.000000",
    )
    _dd_write_legacy_raw(
        tmp_path,
        new,
        4096,
        "2026-06-27T01-00-00.000000",
    )
    aggregate_dir = tmp_path / "eval_results_all"
    aggregate_dir.mkdir()
    old_path = aggregate_dir / "old.json"
    new_path = aggregate_dir / "new.json"
    old_path.write_text(
        json.dumps(
            [
                _dd_agg_row(
                    4096,
                    f"eval_results/{old}/results_2026-06-26T01-00-00.000000.json",
                    0.4,
                )
            ]
        )
    )
    new_path.write_text(
        json.dumps(
            [
                _dd_agg_row(
                    4096,
                    f"eval_results/{new}/results_2026-06-27T01-00-00.000000.json",
                    0.9,
                )
            ]
        )
    )

    messages = dedupe_reran_evals(tmp_path)

    assert json.loads(old_path.read_text()) == []
    assert [
        row["em_strict"]
        for row in json.loads(new_path.read_text())
    ] == [0.9]
    assert not (tmp_path / old).exists()
    assert (tmp_path / new).is_dir()
    assert validate_eval_artifacts(tmp_path) == []
    assert any("old.json: kept 0 of 1" in message for message in messages)


def test_dedupe_accepts_zero_score_as_structurally_valid(
    tmp_path: Path,
) -> None:
    old = "eval_minimaxm3_conc4096_b300-nv_old"
    new = "eval_minimaxm3_conc4096_b300-nv_new"
    _dd_write_legacy_raw(tmp_path, old, 4096, "2026-06-26T01-00-00.000000")
    _dd_write_legacy_raw(tmp_path, new, 4096, "2026-06-27T01-00-00.000000")
    new_result = next((tmp_path / new).glob("results*.json"))
    new_result.write_text(json.dumps(raw_eval_result(0.0)))
    _dd_write_aggregate(
        tmp_path,
        [
            _dd_agg_row(4096, f"eval_results/{old}/results_2026-06-26T01-00-00.000000.json", 0.8),
            _dd_agg_row(4096, f"eval_results/{new}/results_2026-06-27T01-00-00.000000.json", 0.0),
        ],
    )

    dedupe_reran_evals(tmp_path)

    assert validate_eval_artifacts(tmp_path) == []
    assert (tmp_path / new).is_dir()
    assert not (tmp_path / old).exists()


def test_dedupe_does_not_resurrect_stale_result_after_integration_error(
    tmp_path: Path,
) -> None:
    name = "eval_minimaxm3_conc4096_b300-nv_retry"
    _dd_write_legacy_raw(
        tmp_path,
        name,
        4096,
        "2026-06-26T01-00-00.000000",
    )
    new_result = (
        tmp_path
        / name
        / "results_2026-06-27T01-00-00.000000.json"
    )
    failed = raw_eval_result(0.0)
    failed["integration_error"] = {
        "type": "RuntimeError",
        "message": "verifier checkout failed",
    }
    new_result.write_text(json.dumps(failed))
    _dd_write_aggregate(
        tmp_path,
        [_dd_agg_row(4096, f"eval_results/{name}/results_old.json", 0.8)],
    )

    assert dedupe_reran_evals(tmp_path) == []
    errors = validate_eval_artifacts(tmp_path)

    assert any("integration error" in error for error in errors)
    assert (tmp_path / name).is_dir()


def test_newer_foreign_result_does_not_suppress_recognized_result(
    tmp_path: Path,
) -> None:
    markerless = raw_eval_result()
    markerless.pop("lm_eval_version")
    for index, contents in enumerate(("{not-json", json.dumps(markerless))):
        root = tmp_path / str(index)
        root.mkdir()
        name = "eval_minimaxm3_conc4096_b300-nv_retry"
        _dd_write_legacy_raw(
            root,
            name,
            4096,
            "2026-06-26T01-00-00.000000",
        )
        (
            root
            / name
            / "results_2026-06-27T01-00-00.000000.json"
        ).write_text(contents)
        _dd_write_aggregate(
            root,
            [_dd_agg_row(4096, f"eval_results/{name}/results_old.json", 0.8)],
        )

        assert dedupe_reran_evals(root) == []
        assert validate_eval_artifacts(root) == []


def test_dedupe_does_not_prune_when_latest_recognized_result_is_invalid(
    tmp_path: Path,
) -> None:
    cases = (
        (json.dumps({**raw_eval_result(), "results": {}}), "empty or malformed"),
        (json.dumps({**raw_eval_result(), "results": []}), "empty or malformed"),
        (
            json.dumps(
                {
                    **raw_eval_result(),
                    "results": {
                        "gsm8k": {"exact_match,strict-match": "invalid"}
                    },
                }
            ),
            "invalid score",
        ),
        (
            json.dumps(
                {
                    **raw_eval_result(),
                    "results": {"gsm8k": {"alias": "gsm8k"}},
                }
            ),
            "no score",
        ),
        (json.dumps(raw_eval_result(effective=0)), "invalid effective sample count"),
        (
            json.dumps(raw_eval_result(effective="unknown")),
            "invalid effective sample count",
        ),
        (json.dumps(raw_eval_result(effective=True)), "invalid effective sample count"),
        (
            json.dumps(
                {
                    **raw_eval_result(),
                    "n-samples": {"gsm8k": {}},
                }
            ),
            "malformed effective sample count",
        ),
    )
    for index, (contents, expected) in enumerate(cases):
        root = tmp_path / str(index)
        root.mkdir()
        name = "eval_minimaxm3_conc4096_b300-nv_retry"
        _dd_write_legacy_raw(
            root,
            name,
            4096,
            "2026-06-26T01-00-00.000000",
        )
        (
            root
            / name
            / "results_2026-06-27T01-00-00.000000.json"
        ).write_text(contents)
        _dd_write_aggregate(
            root,
            [_dd_agg_row(4096, f"eval_results/{name}/results_old.json", 0.8)],
        )

        assert dedupe_reran_evals(root) == []
        errors = validate_eval_artifacts(root)

        assert any(expected in error for error in errors), errors
        assert (root / name).is_dir()


def test_dedupe_requires_aggregate_row_for_latest_raw_directory(
    tmp_path: Path,
) -> None:
    old = "eval_minimaxm3_conc4096_b300-nv_old"
    new = "eval_minimaxm3_conc4096_b300-nv_new"
    _dd_write_legacy_raw(tmp_path, old, 4096, "2026-06-26T01-00-00.000000")
    _dd_write_legacy_raw(tmp_path, new, 4096, "2026-06-27T01-00-00.000000")
    agg_path = _dd_write_aggregate(
        tmp_path,
        [
            _dd_agg_row(
                4096,
                f"eval_results/{new}-unrelated/results_old.json",
                0.8,
            )
        ],
    )
    before = agg_path.read_text()

    assert dedupe_reran_evals(tmp_path) == []
    assert agg_path.read_text() == before
    assert (tmp_path / old).is_dir()
    assert (tmp_path / new).is_dir()
    assert any("duplicate" in error for error in validate_eval_artifacts(tmp_path))



def test_eval_validation_accepts_extract_filter_primary_score(
    tmp_path: Path,
) -> None:
    write_raw_eval_artifact(tmp_path, 32, eval_suite="gpqa")
    raw_path = next(tmp_path.glob("eval_*/results*.json"))
    data = raw_eval_result(task="gpqa")
    data["results"]["gpqa"] = {
        "exact_match,extract_abcd": 0.75,
        "exact_match_stderr,extract_abcd": 0.02,
        "answer_token_count": 42,
    }
    data["configs"]["gpqa"]["filter_list"] = [{"name": "extract_abcd"}]
    raw_path.write_text(json.dumps(data))
    write_eval_aggregate(
        tmp_path,
        [
            {
                **single_eval_result(32, eval_suite="gpqa"),
                "task": "gpqa",
            }
        ],
    )

    assert validate_eval_artifacts(tmp_path) == []


def test_eval_dedupe_leaves_invalid_suite_for_validation(
    tmp_path: Path,
) -> None:
    write_raw_eval_artifact(tmp_path, 32)
    raw_dir = next(tmp_path.glob("eval_*"))
    meta_path = raw_dir / "meta_env.json"
    meta = json.loads(meta_path.read_text())
    meta["eval_suite"] = []
    meta_path.write_text(json.dumps(meta))
    row = single_eval_result(32)
    row["eval_suite"] = []
    write_eval_aggregate(tmp_path, [row])

    assert dedupe_reran_evals(tmp_path) == []
    errors = validate_eval_artifacts(tmp_path)
    assert any("invalid eval_suite" in error for error in errors)


def test_dedupe_uses_winning_legacy_result_mtime_for_aggregate(
    tmp_path: Path,
) -> None:
    artifact_name = "eval_minimaxm3_conc4096_b300-nv_retry"
    _dd_write_legacy_raw(tmp_path, artifact_name, 4096, "a")
    artifact_dir = tmp_path / artifact_name
    older = artifact_dir / "results_b.json"
    older.write_text(json.dumps(raw_eval_result()))
    newer = artifact_dir / "results_a.json"
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))
    _dd_write_aggregate(
        tmp_path,
        [
            _dd_agg_row(
                4096,
                f"eval_results/{artifact_name}/results_b.json",
                0.1,
            ),
            _dd_agg_row(
                4096,
                f"eval_results/{artifact_name}/results_a.json",
                0.9,
            ),
        ],
    )

    dedupe_reran_evals(tmp_path)

    rows = json.loads(
        (tmp_path / "eval_results_all" / "agg_eval_all.json").read_text()
    )
    assert [row["em_strict"] for row in rows] == [0.9]
    assert validate_eval_artifacts(tmp_path) == []
