from __future__ import annotations

import json
import sys
from pathlib import Path

from validate_reusable_sweep_artifacts import (
    agentic_key,
    benchmark_key,
    eval_key,
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
        "pp": 1,
        "dcp_size": 1,
        "pcp_size": 1,
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
        "hw": "b200-dgxc",
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
        (artifact_dir / f"results_{timestamp}.json").write_text("{}")


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


def test_dedupe_leaves_ambiguous_duplicates_for_validation(tmp_path: Path) -> None:
    # Duplicate raw identities with no result timestamps cannot be ordered, so
    # dedupe must leave them and validation must still reject them.
    for name in ("eval_minimaxm3_conc4096_b300-nv_01", "eval_minimaxm3_conc4096_b300-nv_02"):
        _dd_write_legacy_raw(tmp_path, name, 4096, None)
    _dd_write_aggregate(
        tmp_path,
        [_dd_agg_row(4096, "eval_results/eval_minimaxm3_conc4096_b300-nv_01/x.json", 0.9)],
    )

    assert dedupe_reran_evals(tmp_path) == []
    assert any("duplicate" in e for e in validate_eval_artifacts(tmp_path))


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
        (artifact_dir / "meta_env.json").write_text(json.dumps(meta))
        for conc in concs:
            (artifact_dir / f"results_{stamp}_conc{conc}.json").write_text("{}")
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
    assert not (older / "results_2026-06-26T10-00-00.000000_conc32.json").exists()
    assert (older / "results_2026-06-26T10-00-00.000000_conc16.json").exists()
    rows = json.loads((tmp_path / "eval_results_all" / "agg_eval_all.json").read_text())
    assert [r["em_strict"] for r in rows if r["conc"] == 32] == [0.90]
