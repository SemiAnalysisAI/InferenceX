"""Smoke tests for process_agentic_result.py against synthetic aiperf output.

The processor consumes three files in $RESULT_DIR/aiperf_artifacts/:
profile_export.jsonl, profile_export_aiperf.json, and
(optionally) server_metrics_export.json. It writes one
$RESULT_FILENAME.json under $AGENTIC_OUTPUT_DIR. We build a minimal
fixture, run the processor, and assert the agg JSON has the expected
metadata plus nested request/server metric schema.

These tests run entirely in tmpdir; no aiperf install or HF cache
required.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from utils.agentic.aggregation.request_metrics import (
    compute_request_metrics,
    load_aggregate,
    load_records,
)
from utils.agentic.aggregation.process_agentic_result import _gpu_shape
from utils.agentic.aggregation.server_metrics import (
    compute_server_metrics,
    load_server_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[3]

AGG_TOP_LEVEL_KEYS = {
    "infmax_model_prefix",
    "model",
    "hw",
    "framework",
    "precision",
    "conc",
    "scenario_type",
    "is_multinode",
    "tp",
    "pp",
    "dcp_size",
    "pcp_size",
    "ep",
    "dp_attention",
    "kv_offloading",
    "kv_offload_backend",
    "allocated_cpu_dram_gb",
    "num_requests_total",
    "num_requests_successful",
    "request_accounting",
    "request_metrics",
    "server_metrics",
    "kv_cache_pool_tokens",
}
REQUEST_ACCOUNTING_KEYS = {
    "records_total",
    "records_profiled",
    "records_dropped_total",
    "records_warmup_dropped",
    "records_error_dropped",
    "error_categories",
}

SERVER_METRICS_KEYS = {
    "present",
    "adapter",
    "metric_count",
    "cache",
    "kv_cache",
    "kv_offload",
    "tokens",
    "sources",
}
SERVER_CACHE_KEYS = {
    "gpu_cache_hit_rate",
    "cpu_cache_hit_rate",
    "external_cache_hit_rate",
    "overall_cache_hit_rate",
    "prefix_cache_hits",
    "prefix_cache_queries",
    "external_prefix_cache_hits",
    "external_prefix_cache_queries",
    "cached_tokens_by_source",
    "frontend_cache_hit_rate",
    "router_kv_hit_rate",
    "router_shared_cache_hit_rate",
    "frontend_cached_tokens",
    "frontend_input_tokens",
}
SERVER_KV_CACHE_KEYS = {
    "gpu_usage_pct",
    "gpu_total_tokens",
    "cpu_usage_pct",
    "cpu_used_tokens",
    "cpu_total_tokens",
}
SERVER_KV_OFFLOAD_KEYS = {
    "bytes_gpu_to_cpu",
    "bytes_cpu_to_gpu",
    "time_gpu_to_cpu",
    "time_cpu_to_gpu",
    "bandwidth_gpu_to_cpu_bytes_per_second",
    "bandwidth_cpu_to_gpu_bytes_per_second",
}
SERVER_TOKEN_KEYS = {
    "prompt_total",
    "generation_total",
    "requests_completed",
    "prompt_by_source",
}
SERVER_PROMPT_SOURCE_KEYS = {
    "gpu_cache_hit",
    "cpu_or_external_cache_hit",
    "computed",
    "raw",
}
REQUEST_METRICS_KEYS = {"qps", "latency", "tokens", "throughput", "cache", "stability"}
REQUEST_LATENCY_KEYS = {"ttft", "e2el", "itl", "tpot", "intvty"}
REQUEST_TOKEN_KEYS = {"input", "output_actual", "output_expected"}
REQUEST_THROUGHPUT_KEYS = {
    "input",
    "output",
    "total",
    "duration_seconds",
    "per_gpu",
}
REQUEST_CACHE_KEYS = {"theoretical_cache_hit_rate"}
REQUEST_STABILITY_KEYS = {
    "window_seconds",
    "expected_window_count",
    "observed_window_count",
    "min_window_requests",
    "root_trajectory_count",
    "root_trajectory_kish_effective_count",
    "root_trajectory_largest_share",
    "observed_ranges",
    "convergence",
}
REQUEST_CONVERGENCE_KEYS = {
    "checkpoint_seconds",
    "tolerance_ratio",
    "min_confirmation_seconds",
    "horizon_seconds",
    "metrics",
}


def _assert_stable_server_metrics_schema(agg: dict) -> None:
    server_metrics = agg["server_metrics"]
    assert set(server_metrics) == SERVER_METRICS_KEYS
    assert set(server_metrics["cache"]) == SERVER_CACHE_KEYS
    assert set(server_metrics["kv_cache"]) == SERVER_KV_CACHE_KEYS
    assert set(server_metrics["kv_offload"]) == SERVER_KV_OFFLOAD_KEYS
    assert set(server_metrics["tokens"]) == SERVER_TOKEN_KEYS
    assert set(server_metrics["tokens"]["prompt_by_source"]) == SERVER_PROMPT_SOURCE_KEYS


def _assert_stable_request_metrics_schema(agg: dict) -> None:
    request_metrics = agg["request_metrics"]
    assert set(agg["request_accounting"]) == REQUEST_ACCOUNTING_KEYS
    assert set(request_metrics) == REQUEST_METRICS_KEYS
    assert set(request_metrics["latency"]) == REQUEST_LATENCY_KEYS
    assert set(request_metrics["tokens"]) == REQUEST_TOKEN_KEYS
    assert set(request_metrics["throughput"]) == REQUEST_THROUGHPUT_KEYS
    assert set(request_metrics["cache"]) == REQUEST_CACHE_KEYS
    assert set(request_metrics["stability"]) == REQUEST_STABILITY_KEYS
    assert set(request_metrics["stability"]["convergence"]) == REQUEST_CONVERGENCE_KEYS


def _flat_request_keys(result_dir: Path) -> set[str]:
    artifact = result_dir / "aiperf_artifacts"
    records = load_records(artifact / "profile_export.jsonl")
    aggregate_path = artifact / "profile_export_aiperf.json"
    aggregate = load_aggregate(aggregate_path) if aggregate_path.exists() else {}
    flat, _ = compute_request_metrics(records, aggregate)
    return set(flat)


def _flat_server_keys(result_dir: Path, framework: str = "vllm") -> set[str]:
    records = load_records(result_dir / "aiperf_artifacts" / "profile_export.jsonl")
    server_metrics = load_server_metrics(
        result_dir / "aiperf_artifacts" / "server_metrics_export.json"
    )
    flat, _, _ = compute_server_metrics(
        server_metrics,
        framework=framework,
        records=records,
    )
    return set(flat)


def _make_record(
    *,
    conv_id: str,
    turn_index: int,
    isl: int,
    osl: int,
    ttft_ms: float,
    e2e_ms: float,
    itl_ms: float,
    start_ns: int,
    end_ns: int,
) -> dict:
    return {
        "metadata": {
            "session_num": 0,
            "x_correlation_id": "x" * 36,
            "conversation_id": conv_id,
            "turn_index": turn_index,
            "request_start_ns": start_ns,
            "request_ack_ns": start_ns + 100,
            "request_end_ns": end_ns,
            "worker_id": "worker_test",
            "benchmark_phase": "profiling",
            "was_cancelled": False,
            "cancellation_time_ns": None,
        },
        "metrics": {
            "input_sequence_length": {"value": isl, "unit": "tokens"},
            "output_sequence_length": {"value": osl, "unit": "tokens"},
            "time_to_first_token": {"value": ttft_ms, "unit": "ms"},
            "request_latency": {"value": e2e_ms, "unit": "ms"},
            "inter_token_latency": {"value": itl_ms, "unit": "ms"},
        },
        "error": None,
    }


def _write_fixture(tmp_path: Path) -> Path:
    """Build a $RESULT_DIR with aiperf-shaped artifacts. Returns RESULT_DIR."""
    result_dir = tmp_path / "results"
    artifact = result_dir / "aiperf_artifacts"
    artifact.mkdir(parents=True)

    # 5 records across 2 conversations; turn indices grow within each.
    records = [
        _make_record(
            conv_id="trace-A",
            turn_index=0,
            isl=100,
            osl=50,
            ttft_ms=30.0,
            e2e_ms=1000.0,
            itl_ms=18.0,
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
        ),
        _make_record(
            conv_id="trace-A",
            turn_index=1,
            isl=180,
            osl=60,
            ttft_ms=35.0,
            e2e_ms=1100.0,
            itl_ms=18.5,
            start_ns=2_500_000_000,
            end_ns=3_700_000_000,
        ),
        _make_record(
            conv_id="trace-B",
            turn_index=0,
            isl=120,
            osl=40,
            ttft_ms=32.0,
            e2e_ms=900.0,
            itl_ms=17.5,
            start_ns=1_500_000_000,
            end_ns=2_300_000_000,
        ),
        _make_record(
            conv_id="trace-B",
            turn_index=1,
            isl=200,
            osl=70,
            ttft_ms=40.0,
            e2e_ms=1400.0,
            itl_ms=19.0,
            start_ns=3_000_000_000,
            end_ns=4_500_000_000,
        ),
        _make_record(
            conv_id="trace-A",
            turn_index=2,
            isl=240,
            osl=55,
            ttft_ms=33.0,
            e2e_ms=1050.0,
            itl_ms=18.2,
            start_ns=4_000_000_000,
            end_ns=5_100_000_000,
        ),
    ]
    with open(artifact / "profile_export.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Aggregate file. Processor uses jsonl as the canonical source so this
    # only needs to be parsable — the values aren't asserted on.
    with open(artifact / "profile_export_aiperf.json", "w") as f:
        json.dump(
            {
                "request_count": len(records),
                "benchmark_duration": {"avg": 4.1, "unit": "sec"},
                "request_latency": {"avg": 1090.0, "unit": "ms"},
                "metadata": {
                    "dataset": {
                        "source_type": "public_dataset",
                        "loader": "semianalysis_cc_traces_weka_with_subagents",
                        "hf_dataset_name": "semianalysisai/cc-traces-weka-062126",
                        "hf_split": "train",
                        "num_dataset_entries": 393,
                    }
                },
            },
            f,
        )

    # No server_metrics_export.json — exercises the missing-file path.
    return result_dir


def _run_processor(
    result_dir: Path,
    output_dir: Path,
    env_overrides: dict[str, str] | None = None,
) -> dict:
    env = os.environ.copy()
    env.pop("PREFILL_HARDWARE", None)
    env.pop("DECODE_HARDWARE", None)
    env.update(
        {
            "RESULT_DIR": str(result_dir),
            "AGENTIC_OUTPUT_DIR": str(output_dir),
            "RESULT_FILENAME": "agg_test",
            "MODEL": "test-model",
            "MODEL_PREFIX": "test/prefix",
            "FRAMEWORK": "vllm",
            "PRECISION": "fp4",
            "TP": "4",
            "PP_SIZE": "1",
            "DCP_SIZE": "1",
            "PCP_SIZE": "1",
            "EP_SIZE": "1",
            "DP_ATTENTION": "false",
            "CONC": "8",
            "KV_OFFLOADING": "none",
            "RUNNER_TYPE": "b200-x4",
            "IMAGE": "test/image:0.1",
            "SPEC_DECODING": "none",
            "DISAGG": "false",
            "IS_MULTINODE": "false",
            # No aiperf theoretical cache metric in this fixture.
            "HF_HUB_CACHE": str(result_dir / "_no_such_cache"),
        }
    )
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-m", "utils.agentic.aggregation.process_agentic_result"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"processor exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    out = output_dir / "agg_test.json"
    assert out.exists(), f"missing output {out}; stdout:\n{proc.stdout}"
    return json.loads(out.read_text())


def test_processor_emits_nested_request_and_server_metrics(tmp_path: Path):
    result_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "out"
    agg = _run_processor(result_dir, output_dir)
    missing = AGG_TOP_LEVEL_KEYS - set(agg.keys())
    assert not missing, f"agg JSON missing top-level keys: {sorted(missing)}"
    assert not (_flat_request_keys(result_dir) & set(agg.keys()))
    assert not (_flat_server_keys(result_dir) & set(agg.keys()))
    _assert_stable_request_metrics_schema(agg)
    _assert_stable_server_metrics_schema(agg)


def test_processor_preserves_dataset_provenance(tmp_path: Path):
    result_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "out"
    agg = _run_processor(result_dir, output_dir)
    assert agg["dataset"] == {
        "source_type": "public_dataset",
        "loader": "semianalysis_cc_traces_weka_with_subagents",
        "hf_dataset_name": "semianalysisai/cc-traces-weka-062126",
        "hf_split": "train",
        "num_dataset_entries": 393,
    }


def test_processor_emits_component_metadata_when_present(tmp_path: Path):
    result_dir = _write_fixture(tmp_path)
    agg = _run_processor(
        result_dir,
        tmp_path / "out",
        env_overrides={
            "ROUTER_METADATA": json.dumps({"name": "vllm-router", "version": "0.1.14"}),
            "KV_P2P_TRANSFER": "mooncake",
        },
    )

    assert agg["router"] == {"name": "vllm-router", "version": "0.1.14"}
    assert agg["kv_p2p_transfer"] == "mooncake"


def test_processor_omits_component_metadata_when_absent(tmp_path: Path):
    result_dir = _write_fixture(tmp_path)
    agg = _run_processor(result_dir, tmp_path / "out")

    assert "router" not in agg
    assert "kv_p2p_transfer" not in agg


@pytest.mark.parametrize(
    "metadata",
    [
        {"name": "lmcache"},
        {"name": "lmcache", "version": "0.5.1"},
    ],
)
def test_processor_emits_kv_offload_backend_metadata(
    tmp_path: Path,
    metadata: dict[str, str],
):
    result_dir = _write_fixture(tmp_path)
    agg = _run_processor(
        result_dir,
        tmp_path / "out",
        env_overrides={
            "KV_OFFLOADING": "dram",
            "KV_OFFLOAD_BACKEND": "lmcache",
            "KV_OFFLOAD_BACKEND_METADATA": json.dumps(metadata),
        },
    )

    assert agg["kv_offload_backend"] == metadata


def test_processor_latency_units_are_seconds(tmp_path: Path):
    """aiperf reports ms; legacy schema is seconds. Verify conversion."""
    result_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "out"
    agg = _run_processor(result_dir, output_dir)
    latency = agg["request_metrics"]["latency"]
    # Fixture mean ttft = (30+35+32+40+33)/5 = 34.0 ms = 0.034 s.
    assert 0.020 < latency["ttft"]["mean"] < 0.050, latency["ttft"]["mean"]
    assert 0.020 < latency["ttft"]["p50"] < 0.050, latency["ttft"]["p50"]
    # Fixture mean e2e = (1000+1100+900+1400+1050)/5 = 1090 ms = 1.09 s.
    assert 0.5 < latency["e2el"]["mean"] < 2.0, latency["e2el"]["mean"]
    # mean itl = ~18 ms = 0.018 s.
    assert 0.010 < latency["itl"]["mean"] < 0.030, latency["itl"]["mean"]
    _assert_stable_request_metrics_schema(agg)
    assert "mean_ttft" not in agg
    assert "p50_ttft" not in agg
    assert "p90_itl" not in agg


def test_request_metrics_reads_aiperf_benchmark_duration_shape() -> None:
    records = [
        _make_record(
            conv_id="trace-A",
            turn_index=0,
            isl=100,
            osl=20,
            ttft_ms=100,
            e2e_ms=1000,
            itl_ms=20,
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
        ),
        _make_record(
            conv_id="trace-B",
            turn_index=0,
            isl=100,
            osl=20,
            ttft_ms=200,
            e2e_ms=2000,
            itl_ms=40,
            start_ns=602_000_000_000,
            end_ns=604_000_000_000,
        ),
    ]

    _, nested = compute_request_metrics(
        records,
        {
            # The observed wall time can land just below a clean boundary;
            # fixed-window count must follow the configured run duration.
            "benchmark_duration": {"avg": 1199.9, "unit": "sec"},
            "input_config": {"loadgen": {"benchmark_duration": 1200}},
        },
    )

    stability = nested["stability"]
    assert stability["expected_window_count"] == 2
    assert stability["observed_window_count"] == 2
    assert stability["observed_ranges"]["ttft"]["p90"] == pytest.approx(
        {"min": 0.1, "max": 0.2}
    )


def test_processor_derives_interactivity_from_matching_itl_percentile(
    tmp_path: Path,
):
    result_dir = tmp_path / "results"
    artifact = result_dir / "aiperf_artifacts"
    artifact.mkdir(parents=True)

    for idx, itl_ms in enumerate((10.0, 20.0, 100.0)):
        rec = _make_record(
            conv_id=f"trace-{idx}",
            turn_index=0,
            isl=100,
            osl=50,
            ttft_ms=30.0,
            e2e_ms=1000.0,
            itl_ms=itl_ms,
            start_ns=(idx + 1) * 1_000_000_000,
            end_ns=(idx + 2) * 1_000_000_000,
        )
        with open(artifact / "profile_export.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    with open(artifact / "profile_export_aiperf.json", "w") as f:
        json.dump({"request_count": 3}, f)

    agg = _run_processor(result_dir, tmp_path / "out")

    latency = agg["request_metrics"]["latency"]
    assert latency["intvty"]["p90"] == pytest.approx(1 / latency["itl"]["p90"], rel=0.01)
    assert latency["intvty"]["p75"] == pytest.approx(1 / latency["itl"]["p75"], rel=0.01)
    assert latency["intvty"]["p50"] == pytest.approx(1 / latency["itl"]["p50"], rel=0.01)
    assert latency["intvty"]["mean"] == pytest.approx(1 / latency["itl"]["mean"], rel=0.01)
    assert latency["intvty"]["p90"] < 20


def test_processor_throughput_per_gpu(tmp_path: Path):
    result_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "out"
    agg = _run_processor(
        result_dir,
        output_dir,
        env_overrides={"TP": "4", "PP_SIZE": "2", "DCP_SIZE": "2", "PCP_SIZE": "2"},
    )
    throughput = agg["request_metrics"]["throughput"]
    per_gpu = throughput["per_gpu"]
    assert agg["pp"] == 2
    assert agg["dcp_size"] == 2
    assert agg["pcp_size"] == 2
    assert per_gpu["total_tput_tps"] == pytest.approx(
        throughput["total"]["tokens_per_second"] / 16
    )
    assert per_gpu["input_tput_tps"] == pytest.approx(
        throughput["input"]["tokens_per_second"] / 16
    )
    assert per_gpu["output_tput_tps"] == pytest.approx(
        throughput["output"]["tokens_per_second"] / 16
    )


def test_processor_surfaces_allocated_cpu_dram(tmp_path: Path):
    result_dir = _write_fixture(tmp_path)

    agg = _run_processor(
        result_dir,
        tmp_path / "out",
        env_overrides={"TOTAL_CPU_DRAM_GB": "2400"},
    )

    assert agg["allocated_cpu_dram_gb"] == 2400


def test_multinode_processor_surfaces_heterogeneous_hardware(tmp_path: Path):
    result_dir = _write_fixture(tmp_path)
    agg = _run_processor(
        result_dir,
        tmp_path / "out",
        env_overrides={
            "IS_MULTINODE": "true",
            "DISAGG": "true",
            "PREFILL_NUM_WORKERS": "1",
            "PREFILL_TP": "8",
            "PREFILL_PP_SIZE": "2",
            "PREFILL_DCP_SIZE": "2",
            "PREFILL_PCP_SIZE": "2",
            "PREFILL_EP": "8",
            "PREFILL_DP_ATTN": "false",
            "PREFILL_HARDWARE": "b200",
            "DECODE_NUM_WORKERS": "2",
            "DECODE_TP": "8",
            "DECODE_PP_SIZE": "2",
            "DECODE_DCP_SIZE": "4",
            "DECODE_PCP_SIZE": "1",
            "DECODE_EP": "8",
            "DECODE_DP_ATTN": "false",
            "DECODE_HARDWARE": "h100",
        },
    )

    assert agg["prefill_hw"] == "b200"
    assert agg["decode_hw"] == "h100"
    assert (
        agg["prefill_pp"],
        agg["prefill_dcp_size"],
        agg["prefill_pcp_size"],
        agg["num_prefill_gpu"],
    ) == (2, 2, 2, 32)
    assert (
        agg["decode_pp"],
        agg["decode_dcp_size"],
        agg["decode_pcp_size"],
        agg["num_decode_gpu"],
    ) == (2, 4, 1, 32)


def test_multinode_processor_omits_homogeneous_hardware(tmp_path: Path):
    result_dir = _write_fixture(tmp_path)
    agg = _run_processor(
        result_dir,
        tmp_path / "out",
        env_overrides={
            "IS_MULTINODE": "true",
            "DISAGG": "true",
            "PREFILL_NUM_WORKERS": "1",
            "PREFILL_TP": "8",
            "DECODE_NUM_WORKERS": "2",
            "DECODE_TP": "8",
        },
    )

    assert "prefill_hw" not in agg
    assert "decode_hw" not in agg


@pytest.mark.parametrize(
    ("present_var", "missing_var"),
    [
        ("PREFILL_HARDWARE", "DECODE_HARDWARE"),
        ("DECODE_HARDWARE", "PREFILL_HARDWARE"),
    ],
)
def test_multinode_processor_rejects_one_sided_hardware(
    monkeypatch: pytest.MonkeyPatch,
    present_var: str,
    missing_var: str,
):
    monkeypatch.setenv("IS_MULTINODE", "true")
    monkeypatch.setenv(present_var, "b200")
    monkeypatch.delenv(missing_var, raising=False)

    with pytest.raises(SystemExit, match="must be specified together"):
        _gpu_shape()


def test_processor_surfaces_request_accounting(tmp_path: Path):
    result_dir = tmp_path / "results"
    artifact = result_dir / "aiperf_artifacts"
    artifact.mkdir(parents=True)

    profiling = _make_record(
        conv_id="trace-A",
        turn_index=0,
        isl=100,
        osl=50,
        ttft_ms=30.0,
        e2e_ms=1_000.0,
        itl_ms=10.0,
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
    )
    warmup = _make_record(
        conv_id="trace-A",
        turn_index=1,
        isl=100,
        osl=50,
        ttft_ms=30.0,
        e2e_ms=1_000.0,
        itl_ms=10.0,
        start_ns=2_000_000_000,
        end_ns=3_000_000_000,
    )
    warmup["metadata"]["benchmark_phase"] = "warmup"
    errored = _make_record(
        conv_id="trace-A",
        turn_index=2,
        isl=100,
        osl=50,
        ttft_ms=30.0,
        e2e_ms=1_000.0,
        itl_ms=10.0,
        start_ns=3_000_000_000,
        end_ns=4_000_000_000,
    )
    errored["error"] = {"type": "HTTPStatusError", "message": "500 server error"}

    with open(artifact / "profile_export.jsonl", "w") as f:
        for record in (profiling, warmup, errored):
            f.write(json.dumps(record) + "\n")
    with open(artifact / "profile_export_aiperf.json", "w") as f:
        json.dump({"request_count": 1, "error_request_count": 1}, f)

    agg = _run_processor(result_dir, tmp_path / "out")

    assert agg["num_requests_total"] == 3
    assert agg["num_requests_successful"] == 1
    assert agg["request_accounting"] == {
        "records_total": 3,
        "records_profiled": 1,
        "records_dropped_total": 2,
        "records_warmup_dropped": 1,
        "records_error_dropped": 1,
        "error_categories": {"HTTPStatusError": 1},
    }
    assert agg["server_metrics"]["tokens"]["requests_completed"] == 1


def test_processor_handles_missing_server_metrics(tmp_path: Path):
    """No server_metrics_export.json -> server cache fields are None, not error."""
    result_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "out"
    agg = _run_processor(result_dir, output_dir)
    server_metrics = agg["server_metrics"]
    assert server_metrics["cache"]["gpu_cache_hit_rate"] is None
    assert server_metrics["kv_cache"]["gpu_total_tokens"] is None
    assert agg["kv_cache_pool_tokens"] is None
    assert agg["request_metrics"]["cache"]["theoretical_cache_hit_rate"] is None
    # Non-server-derived totals fall back to per-record sums.
    assert server_metrics["tokens"]["prompt_total"] == 100 + 180 + 120 + 200 + 240
    assert server_metrics["tokens"]["generation_total"] == 50 + 60 + 40 + 70 + 55
    assert server_metrics["tokens"]["requests_completed"] == 5
    _assert_stable_server_metrics_schema(agg)
    assert agg["server_metrics"]["present"] is False


def test_processor_reads_gpu_kv_cache_capacity_from_server_log(tmp_path: Path):
    result_dir = _write_fixture(tmp_path)
    (result_dir / "server.log").write_text(
        "\n".join(
            [
                "INFO (EngineCore_DP0 pid=100) GPU KV cache size: 5,000,000 tokens",
                "INFO (EngineCore_DP0 pid=100) GPU KV cache size: 5,000,000 tokens",
                "INFO (EngineCore_DP1 pid=101) GPU KV cache size: 6,500,000 tokens",
            ]
        )
    )

    agg = _run_processor(result_dir, tmp_path / "out")

    assert agg["server_metrics"]["kv_cache"]["gpu_total_tokens"] == 11_500_000
    assert agg["kv_cache_pool_tokens"] == 11_500_000
    _assert_stable_server_metrics_schema(agg)


def test_processor_emits_sglang_kv_pool_from_server_log(tmp_path: Path):
    result_dir = _write_fixture(tmp_path)
    (result_dir / "server.log").write_text(
        "\n".join(
            [
                "[2026-07-08 16:43:35] server_args=ServerArgs(tp_size=4, dp_size=4)",
                "[2026-07-08 16:49:59 DP0 TP0 EP0] "
                "max_total_num_tokens=4602880, chunked_prefill_size=4096",
            ]
        )
    )

    agg = _run_processor(
        result_dir,
        tmp_path / "out",
        env_overrides={"FRAMEWORK": "sglang"},
    )

    assert agg["server_metrics"]["kv_cache"]["gpu_total_tokens"] == 18_411_520
    assert agg["kv_cache_pool_tokens"] == 18_411_520
    _assert_stable_server_metrics_schema(agg)


def test_processor_reads_multinode_gpu_kv_cache_capacity_from_worker_logs(
    tmp_path: Path,
):
    result_dir = _write_fixture(tmp_path)
    log_root = result_dir.parent
    (log_root / "watchtower-node-a_prefill_w0.out").write_text(
        "\n".join(
            [
                "INFO (EngineCore_DP0 pid=100) GPU KV cache size: 5,000,000 tokens",
                "INFO (EngineCore_DP1 pid=101) GPU KV cache size: 6,500,000 tokens",
            ]
        )
    )
    (log_root / "watchtower-node-b_decode_w0.out").write_text(
        "INFO (EngineCore_DP0 pid=200) GPU KV cache size: 7,000,000 tokens"
    )

    agg = _run_processor(result_dir, tmp_path / "out")

    assert agg["server_metrics"]["kv_cache"]["gpu_total_tokens"] == 18_500_000
    _assert_stable_server_metrics_schema(agg)


def test_processor_falls_back_to_sglang_max_total_num_tokens(tmp_path: Path):
    result_dir = _write_fixture(tmp_path)
    artifact = result_dir / "aiperf_artifacts"
    server_metrics = {
        "metrics": {
            "sglang:max_total_num_tokens": {
                "type": "gauge",
                "series": [
                    {"labels": {"dp_rank": "0"}, "stats": {"max": 1000.0}},
                    {"labels": {"dp_rank": "1"}, "stats": {"max": 1200.0}},
                ],
            },
            "sglang:token_usage": {
                "type": "gauge",
                "series": [{"stats": {"max": 0.25}}],
            },
        }
    }
    with open(artifact / "server_metrics_export.json", "w") as f:
        json.dump(server_metrics, f)

    agg = _run_processor(
        result_dir,
        tmp_path / "out",
        env_overrides={"FRAMEWORK": "sglang"},
    )

    assert agg["server_metrics"]["kv_cache"]["gpu_total_tokens"] == 2200
    assert agg["server_metrics"]["kv_cache"]["gpu_usage_pct"] == pytest.approx(0.25)
    _assert_stable_server_metrics_schema(agg)


def test_server_metrics_reject_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported agentic server metrics backend"):
        compute_server_metrics(
            {
                "metrics": {
                    "unknown_backend:cache_hits": {
                        "type": "counter",
                        "series": [{"stats": {"total": 1.0}}],
                    }
                }
            },
            framework="unknown",
            records=[],
        )


def test_processor_excludes_warmup_phase_records(tmp_path: Path):
    result_dir = tmp_path / "results"
    artifact = result_dir / "aiperf_artifacts"
    artifact.mkdir(parents=True)

    warmup = _make_record(
        conv_id="trace-A",
        turn_index=0,
        isl=10_000,
        osl=5_000,
        ttft_ms=9_000.0,
        e2e_ms=30_000.0,
        itl_ms=900.0,
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
    )
    warmup["metadata"]["benchmark_phase"] = "warmup"

    profiling = _make_record(
        conv_id="trace-A",
        turn_index=1,
        isl=100,
        osl=50,
        ttft_ms=30.0,
        e2e_ms=1_000.0,
        itl_ms=10.0,
        start_ns=3_000_000_000,
        end_ns=4_000_000_000,
    )

    with open(artifact / "profile_export.jsonl", "w") as f:
        f.write(json.dumps(warmup) + "\n")
        f.write(json.dumps(profiling) + "\n")
    with open(artifact / "profile_export_aiperf.json", "w") as f:
        json.dump({"request_count": 1}, f)

    agg = _run_processor(result_dir, tmp_path / "out")

    assert agg["num_requests_total"] == 2
    assert agg["num_requests_successful"] == 1
    assert agg["request_accounting"]["records_total"] == 2
    assert agg["request_accounting"]["records_profiled"] == 1
    assert agg["request_accounting"]["records_dropped_total"] == 1
    assert agg["request_accounting"]["records_warmup_dropped"] == 1
    assert agg["request_accounting"]["records_error_dropped"] == 0
    assert agg["server_metrics"]["tokens"]["requests_completed"] == 1
    assert agg["server_metrics"]["tokens"]["prompt_total"] == 100
    assert agg["server_metrics"]["tokens"]["generation_total"] == 50
    assert agg["request_metrics"]["latency"]["ttft"]["mean"] == pytest.approx(0.03)


def test_processor_rounds_decimal_outputs_to_five_decimal_places(tmp_path: Path):
    result_dir = _write_fixture(tmp_path)
    artifact = result_dir / "aiperf_artifacts"
    server_metrics = {
        "metrics": {
            "vllm:prefix_cache_hits": {
                "type": "counter",
                "series": [{"stats": {"total": 12345.678901}}],
            },
            "vllm:prefix_cache_queries": {
                "type": "counter",
                "series": [{"stats": {"total": 37037.036703}}],
            },
            "vllm:kv_cache_usage_perc": {
                "type": "gauge",
                "series": [{"stats": {"max": 0.123456789}}],
            },
        }
    }
    with open(artifact / "server_metrics_export.json", "w") as f:
        json.dump(server_metrics, f)

    agg = _run_processor(result_dir, tmp_path / "out")

    assert agg["server_metrics"]["cache"]["gpu_cache_hit_rate"] == 0.33333
    assert agg["server_metrics"]["cache"]["prefix_cache_hits"] == 12345.6789
    assert agg["server_metrics"]["kv_cache"]["gpu_usage_pct"] == 0.12346


def test_processor_parses_real_server_metrics_schema(tmp_path: Path):
    """Verify aiperf's actual server_metrics_export.json shape is parsed.

    Real schema: ``{"metrics": {<name>: {"series": [{"stats": {...}}, ...]}}}``
    — keyed by metric name, with stats nested inside each series entry.
    Regression guard: the v1 of the processor crashed with
    ``AttributeError: 'str' object has no attribute 'get'`` because it
    iterated the metrics dict like a list.
    """
    result_dir = _write_fixture(tmp_path)
    artifact = result_dir / "aiperf_artifacts"
    server_metrics = {
        "schema_version": "1.0",
        "summary": {
            "endpoints_configured": ["http://localhost:8888/metrics"],
            "endpoints_successful": ["http://localhost:8888/metrics"],
        },
        "metrics": {
            "vllm:prefix_cache_hits": {
                "type": "counter",
                "unit": "tokens",
                "series": [
                    {
                        "endpoint_url": "http://localhost:8888/metrics",
                        "labels": {"model": "test"},
                        "stats": {"total": 800.0, "rate": 8.0},
                    }
                ],
            },
            "vllm:prefix_cache_queries": {
                "type": "counter",
                "unit": "tokens",
                "series": [
                    {
                        "endpoint_url": "http://localhost:8888/metrics",
                        "labels": {"model": "test"},
                        "stats": {"total": 1000.0, "rate": 10.0},
                    }
                ],
            },
            "vllm:prompt_tokens": {
                "type": "counter",
                "unit": "tokens",
                "series": [
                    {
                        "endpoint_url": "http://localhost:8888/metrics",
                        "labels": {"model": "test"},
                        "stats": {"total": 12345.0, "rate": 100.0},
                    }
                ],
            },
            "vllm:generation_tokens": {
                "type": "counter",
                "unit": "tokens",
                "series": [
                    {
                        "endpoint_url": "http://localhost:8888/metrics",
                        "labels": {"model": "test"},
                        "stats": {"total": 6789.0, "rate": 50.0},
                    }
                ],
            },
        },
    }
    with open(artifact / "server_metrics_export.json", "w") as f:
        json.dump(server_metrics, f)

    output_dir = tmp_path / "out"
    agg = _run_processor(result_dir, output_dir)
    assert agg["server_metrics"]["cache"]["gpu_cache_hit_rate"] == pytest.approx(0.8)
    assert agg["server_metrics"]["tokens"]["prompt_total"] == 12345
    assert agg["server_metrics"]["tokens"]["generation_total"] == 6789
    _assert_stable_server_metrics_schema(agg)
    assert agg["server_metrics"]["adapter"] == "vllm"
    assert agg["server_metrics"]["cache"]["gpu_cache_hit_rate"] == pytest.approx(0.8)


def test_processor_aggregates_across_multiple_series(tmp_path: Path):
    """Counters with multiple series (multi-endpoint) sum across them."""
    result_dir = _write_fixture(tmp_path)
    artifact = result_dir / "aiperf_artifacts"
    server_metrics = {
        "metrics": {
            "vllm:prefix_cache_hits": {
                "type": "counter",
                "series": [
                    {"stats": {"total": 100.0}},
                    {"stats": {"total": 200.0}},
                ],
            },
            "vllm:prefix_cache_queries": {
                "type": "counter",
                "series": [
                    {"stats": {"total": 400.0}},
                    {"stats": {"total": 600.0}},
                ],
            },
        }
    }
    with open(artifact / "server_metrics_export.json", "w") as f:
        json.dump(server_metrics, f)

    output_dir = tmp_path / "out"
    agg = _run_processor(result_dir, output_dir)
    assert agg["server_metrics"]["cache"]["gpu_cache_hit_rate"] == pytest.approx(0.3)


def test_processor_surfaces_vllm_kv_offload_transfer_stats(tmp_path: Path):
    result_dir = _write_fixture(tmp_path)
    artifact = result_dir / "aiperf_artifacts"
    server_metrics = {
        "metrics": {
            "vllm:kv_offload_bytes_gpu_to_cpu": {
                "type": "counter",
                "series": [{"stats": {"total": 10_000_000.0}}],
            },
            "vllm:kv_offload_bytes_cpu_to_gpu": {
                "type": "counter",
                "series": [{"stats": {"total": 5_000_000.0}}],
            },
            "vllm:kv_offload_time_gpu_to_cpu": {
                "type": "counter",
                "series": [{"stats": {"total": 2.0}}],
            },
            "vllm:kv_offload_time_cpu_to_gpu": {
                "type": "counter",
                "series": [{"stats": {"total": 0.5}}],
            },
        }
    }
    with open(artifact / "server_metrics_export.json", "w") as f:
        json.dump(server_metrics, f)

    agg = _run_processor(result_dir, tmp_path / "out")
    kv_offload = agg["server_metrics"]["kv_offload"]

    assert kv_offload["bytes_gpu_to_cpu"] == 10_000_000
    assert kv_offload["bytes_cpu_to_gpu"] == 5_000_000
    assert kv_offload["time_gpu_to_cpu"] == 2
    assert kv_offload["time_cpu_to_gpu"] == 0.5
    assert kv_offload["bandwidth_gpu_to_cpu_bytes_per_second"] == 5_000_000
    assert kv_offload["bandwidth_cpu_to_gpu_bytes_per_second"] == 10_000_000


def test_processor_ignores_server_warmup_metrics_for_headline_stats(
    tmp_path: Path,
):
    result_dir = _write_fixture(tmp_path)
    artifact = result_dir / "aiperf_artifacts"
    server_metrics = {
        "metrics_phase": "profiling",
        "metrics": {
            "vllm:prefix_cache_hits": {
                "type": "counter",
                "series": [{"stats": {"total": 100.0}}],
            },
            "vllm:prefix_cache_queries": {
                "type": "counter",
                "series": [{"stats": {"total": 200.0}}],
            },
            "vllm:prompt_tokens": {
                "type": "counter",
                "series": [{"stats": {"total": 1000.0}}],
            },
        },
        "warmup_metrics": {
            "vllm:prefix_cache_hits": {
                "type": "counter",
                "series": [{"stats": {"total": 900000.0}}],
            },
            "vllm:prefix_cache_queries": {
                "type": "counter",
                "series": [{"stats": {"total": 900000.0}}],
            },
            "vllm:prompt_tokens": {
                "type": "counter",
                "series": [{"stats": {"total": 900000.0}}],
            },
        },
    }
    with open(artifact / "server_metrics_export.json", "w") as f:
        json.dump(server_metrics, f)

    agg = _run_processor(result_dir, tmp_path / "out")

    assert agg["server_metrics"]["cache"]["gpu_cache_hit_rate"] == pytest.approx(0.5)
    assert agg["server_metrics"]["tokens"]["prompt_total"] == 1000


def test_processor_normalizes_sglang_server_metrics(tmp_path: Path):
    result_dir = _write_fixture(tmp_path)
    artifact = result_dir / "aiperf_artifacts"
    server_metrics = {
        "metrics": {
            "sglang:prompt_tokens": {
                "type": "counter",
                "series": [{"stats": {"total": 1000.0}}],
            },
            "sglang:generation_tokens": {
                "type": "counter",
                "series": [{"stats": {"total": 200.0}}],
            },
            "sglang:cached_tokens": {
                "type": "counter",
                "series": [
                    {"labels": {"cache_source": "device"}, "stats": {"total": 400.0}},
                    {"labels": {"cache_source": "host"}, "stats": {"total": 100.0}},
                ],
            },
            "sglang:token_usage": {
                "type": "gauge",
                "series": [{"stats": {"max": 0.75}}],
            },
            "sglang:hicache_host_used_tokens": {
                "type": "gauge",
                "series": [{"stats": {"max": 300.0}}],
            },
            "sglang:hicache_host_total_tokens": {
                "type": "gauge",
                "series": [{"stats": {"max": 1000.0}}],
            },
            "sglang:realtime_tokens": {
                "type": "counter",
                "series": [
                    {"labels": {"mode": "prefill_compute"}, "stats": {"total": 500.0}}
                ],
            },
        }
    }
    with open(artifact / "server_metrics_export.json", "w") as f:
        json.dump(server_metrics, f)

    agg = _run_processor(
        result_dir,
        tmp_path / "out",
        env_overrides={"FRAMEWORK": "sglang"},
    )

    assert agg["server_metrics"]["adapter"] == "sglang"
    _assert_stable_server_metrics_schema(agg)
    assert agg["server_metrics"]["cache"]["gpu_cache_hit_rate"] == pytest.approx(0.4)
    assert agg["server_metrics"]["cache"]["cpu_cache_hit_rate"] == pytest.approx(0.1)
    assert agg["server_metrics"]["cache"]["external_cache_hit_rate"] == pytest.approx(0.1)
    assert agg["server_metrics"]["cache"]["overall_cache_hit_rate"] == pytest.approx(0.5)
    assert agg["server_metrics"]["kv_cache"]["gpu_usage_pct"] == pytest.approx(0.75)
    assert agg["server_metrics"]["kv_cache"]["cpu_usage_pct"] == pytest.approx(0.3)
    assert agg["server_metrics"]["tokens"]["prompt_by_source"]["computed"] == 500.0


def test_processor_normalizes_dynamo_server_metrics(tmp_path: Path):
    result_dir = _write_fixture(tmp_path)
    artifact = result_dir / "aiperf_artifacts"
    server_metrics = {
        "metrics": {
            "dynamo_frontend_input_sequence_tokens": {
                "type": "counter",
                "series": [{"stats": {"total": 900.0}}],
            },
            "dynamo_frontend_output_tokens": {
                "type": "counter",
                "series": [{"stats": {"total": 300.0}}],
            },
            "dynamo_frontend_cached_tokens": {
                "type": "counter",
                "series": [{"stats": {"total": 450.0}}],
            },
            "dynamo_component_router_shared_cache_hit_rate": {
                "type": "gauge",
                "series": [{"stats": {"avg": 0.55}}],
            },
            "dynamo_component_gpu_cache_usage_percent": {
                "type": "gauge",
                "series": [{"stats": {"max": 75.0}}],
            },
            "vllm:prefix_cache_hits": {
                "type": "counter",
                "series": [
                    {
                        "labels": {"dynamo_component": "prefill", "worker_id": "p0"},
                        "stats": {"total": 400.0},
                    }
                ],
            },
            "vllm:prefix_cache_queries": {
                "type": "counter",
                "series": [
                    {
                        "labels": {"dynamo_component": "prefill", "worker_id": "p0"},
                        "stats": {"total": 800.0},
                    }
                ],
            },
            "vllm:prompt_tokens_by_source": {
                "type": "counter",
                "series": [
                    {
                        "labels": {"source": "local_cache_hit"},
                        "stats": {"total": 300.0},
                    },
                    {
                        "labels": {"source": "external_kv_transfer"},
                        "stats": {"total": 150.0},
                    },
                    {
                        "labels": {"source": "local_compute"},
                        "stats": {"total": 450.0},
                    },
                ],
            },
        }
    }
    with open(artifact / "server_metrics_export.json", "w") as f:
        json.dump(server_metrics, f)

    agg = _run_processor(
        result_dir,
        tmp_path / "out",
        env_overrides={"FRAMEWORK": "dynamo-vllm", "IS_MULTINODE": "true"},
    )

    assert agg["server_metrics"]["adapter"] == "dynamo-vllm"
    _assert_stable_server_metrics_schema(agg)
    assert agg["server_metrics"]["tokens"]["prompt_total"] == 900
    assert agg["server_metrics"]["tokens"]["generation_total"] == 300
    assert agg["server_metrics"]["cache"]["gpu_cache_hit_rate"] == pytest.approx(0.33333)
    assert agg["server_metrics"]["cache"]["cpu_cache_hit_rate"] == pytest.approx(0.16667)
    assert agg["server_metrics"]["cache"]["overall_cache_hit_rate"] == pytest.approx(0.5)
    assert agg["server_metrics"]["kv_cache"]["gpu_usage_pct"] == pytest.approx(0.75)
    assert agg["server_metrics"]["cache"]["frontend_cache_hit_rate"] == pytest.approx(0.5)
    assert agg["server_metrics"]["cache"]["router_shared_cache_hit_rate"] == pytest.approx(0.55)
    assert any(source["role"] == "prefill" for source in agg["server_metrics"]["sources"])


def test_processor_uses_aiperf_theoretical_cache_metric(tmp_path: Path):
    """Aiperf's exported profile aggregate is the theoretical cache source."""
    result_dir = _write_fixture(tmp_path)
    artifact = result_dir / "aiperf_artifacts"
    with open(artifact / "profile_export_aiperf.json", "w") as f:
        json.dump(
            {
                "request_count": 5,
                "theoretical_prefix_cache_hit": {
                    "unit": "%",
                    "avg": 25.0,
                    "count": 4,
                    "sum": 1,
                },
            },
            f,
        )

    # Keep trace metadata present so this also proves theoretical cache does
    # not depend on recomputing from local HF traces.
    hf_cache = tmp_path / "_hf"
    snapshot = hf_cache / "datasets--semianalysisai--cc-traces-weka-042026" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    # Real corpus uses the ``out`` alias (Pydantic's external name for
    # output_length). Mix both to verify the loader accepts either.
    traces = [
        {
            "id": "trace-A",
            "requests": [
                {"type": "n", "hash_ids": [1, 2, 3], "out": 50},
                {"type": "n", "hash_ids": [1, 2, 3, 4], "out": 60},
                {"type": "n", "hash_ids": [1, 2, 3, 4, 5], "output_length": 55},
            ],
        },
        {
            "id": "trace-B",
            "requests": [
                {"type": "n", "hash_ids": [10, 11], "out": 40},
                {"type": "n", "hash_ids": [10, 11, 12, 13], "out": 70},
            ],
        },
    ]
    with open(snapshot / "traces.jsonl", "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")

    env = os.environ.copy()
    env.update(
        {
            "RESULT_DIR": str(result_dir),
            "AGENTIC_OUTPUT_DIR": str(tmp_path / "out"),
            "RESULT_FILENAME": "agg_test",
            "MODEL": "test-model",
            "TP": "4",
            "CONC": "8",
            "KV_OFFLOADING": "none",
            "RUNNER_TYPE": "h100-x4",
            "HF_HUB_CACHE": str(hf_cache),
        }
    )
    proc = subprocess.run(
        [sys.executable, "-m", "utils.agentic.aggregation.process_agentic_result"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    agg = json.loads((tmp_path / "out" / "agg_test.json").read_text())
    assert agg["request_metrics"]["cache"]["theoretical_cache_hit_rate"] == pytest.approx(0.25)
    # output_tokens_expected populated from trace metadata (5 records: A turns 0,1,2 + B turns 0,1)
    assert agg["request_metrics"]["tokens"]["output_expected"]["mean"] == pytest.approx(
        (50 + 60 + 55 + 40 + 70) / 5
    )


def test_processor_supports_per_run_subdir_layout(tmp_path: Path):
    """When --num-profile-runs > 1, aiperf writes into a per-run subdir."""
    result_dir = tmp_path / "results"
    artifact = result_dir / "aiperf_artifacts" / "run_0"
    artifact.mkdir(parents=True)
    rec = _make_record(
        conv_id="trace-A",
        turn_index=0,
        isl=100,
        osl=50,
        ttft_ms=30.0,
        e2e_ms=1000.0,
        itl_ms=18.0,
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
    )
    with open(artifact / "profile_export.jsonl", "w") as f:
        f.write(json.dumps(rec) + "\n")
    with open(artifact / "profile_export_aiperf.json", "w") as f:
        json.dump({"request_count": 1}, f)

    output_dir = tmp_path / "out"
    agg = _run_processor(result_dir, output_dir)
    assert agg["num_requests_total"] == 1
