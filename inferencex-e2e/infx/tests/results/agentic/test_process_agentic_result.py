"""Smoke tests for process_agentic_result.py against synthetic aiperf output.

The processor consumes profile_export.jsonl and profile_export_aiperf.json in
$RESULT_DIR/aiperf_artifacts/. It writes one $RESULT_FILENAME.json under
$AGENTIC_OUTPUT_DIR. We build a minimal fixture, run the processor, and assert
the agg JSON has the expected metadata plus nested request metric schema.

These tests run entirely in tmpdir; no aiperf install or HF cache
required.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType

import pytest

from infx.results.agentic.request_metrics import compute_qps_stats, compute_request_metrics
from infx.results.agentic import _gpu_shape
from infx.results.agentic import (
    build_result,
    _optional_component_metadata,
    _optional_kv_offload_backend_metadata,
)


REPO_ROOT = Path(__file__).resolve().parents[4]

AGG_TOP_LEVEL_KEYS = {
    "infmax_model_prefix",
    "model",
    "hw",
    "framework",
    "precision",
    "conc",
    "scenario_type",
    "is_multinode",
    "num_gpus",
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
}
REQUEST_ACCOUNTING_KEYS = {
    "records_total",
    "records_profiled",
    "records_dropped_total",
    "records_warmup_dropped",
    "records_error_dropped",
    "error_categories",
}

REQUEST_METRICS_KEYS = {"qps", "latency", "tokens", "throughput", "cache"}
REQUEST_LATENCY_KEYS = {
    "ttft",
    "e2el",
    "itl",
    "tpot",
    "intvty",
    "e2e_norm_intvty",
    "full_response_itl",
    "full_response_intvty",
}
REQUEST_TOKEN_KEYS = {"input", "output_actual", "output_expected"}
REQUEST_THROUGHPUT_KEYS = {
    "input",
    "output",
    "total",
    "duration_seconds",
    "per_gpu",
}
REQUEST_CACHE_KEYS = {"theoretical_cache_hit_rate"}


def _assert_stable_request_metrics_schema(agg: dict) -> None:
    request_metrics = agg["request_metrics"]
    assert set(agg["request_accounting"]) == REQUEST_ACCOUNTING_KEYS
    assert set(request_metrics) == REQUEST_METRICS_KEYS
    assert set(request_metrics["latency"]) == REQUEST_LATENCY_KEYS
    assert set(request_metrics["tokens"]) == REQUEST_TOKEN_KEYS
    assert set(request_metrics["throughput"]) == REQUEST_THROUGHPUT_KEYS
    assert set(request_metrics["cache"]) == REQUEST_CACHE_KEYS


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
                "benchmark_duration": 4.1,
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
            "RECIPE_FINGERPRINT": "b" * 64,
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
        [sys.executable, "-m", "infx.results.agentic.process_agentic_result"],
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


def test_processor_emits_nested_request_metrics_without_server_aggregates(tmp_path: Path):
    result_dir = _write_fixture(tmp_path)
    (result_dir / "aiperf_artifacts" / "server_metrics_export.json").write_text(
        json.dumps({"metrics": {"vllm:prefix_cache_hits": {"series": [{"stats": {"total": 1}}]}}})
    )
    (result_dir / "server.log").write_text("GPU KV cache size: 1,000 tokens\n")
    output_dir = tmp_path / "out"
    agg = _run_processor(result_dir, output_dir)
    assert agg["recipe_fingerprint"] == "b" * 64
    missing = AGG_TOP_LEVEL_KEYS - set(agg.keys())
    assert not missing, f"agg JSON missing top-level keys: {sorted(missing)}"
    assert not {"server_metrics", "kv_cache_pool_tokens", "warnings"} & set(agg)
    _assert_stable_request_metrics_schema(agg)


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


@pytest.mark.parametrize("parser", [
    _optional_component_metadata, _optional_kv_offload_backend_metadata,
])
@pytest.mark.parametrize("raw", [None, "", "null"])
def test_optional_metadata_accepts_unset_values(parser, raw):
    env = {} if raw is None else {"TEST_METADATA": raw}
    assert parser(env, "TEST_METADATA") is None


@pytest.mark.parametrize(("parser", "raw", "message"), [
    (_optional_component_metadata, "{", "must contain valid JSON"),
    (_optional_component_metadata, "[]", "must contain exactly 'name' and 'version'"),
    (_optional_component_metadata, '{"name":"router"}', "must contain exactly 'name' and 'version'"),
    (_optional_component_metadata, '{"name":"router","version":0}', "name and version must be non-empty strings"),
    (_optional_kv_offload_backend_metadata, "{", "must contain valid JSON"),
    (_optional_kv_offload_backend_metadata, "[]", "may contain only 'name' and 'version'"),
    (_optional_kv_offload_backend_metadata, '{"name":"cache","extra":1}', "may contain only 'name' and 'version'"),
    (_optional_kv_offload_backend_metadata, "{}", "must contain 'name' and optional 'version'"),
    (_optional_kv_offload_backend_metadata, '{"version":"1"}', "must contain 'name' and optional 'version'"),
    (_optional_kv_offload_backend_metadata, '{"name":"cache","version":""}', "values must be non-empty strings"),
])
def test_optional_metadata_preserves_cli_errors(parser, raw, message):
    with pytest.raises(SystemExit) as error:
        parser({"TEST_METADATA": raw}, "TEST_METADATA")
    assert error.value.code == f"TEST_METADATA {message}"


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
    assert latency["ttft"]["mean"] == pytest.approx(0.034)
    assert latency["ttft"]["p50"] == pytest.approx(0.033)
    # Fixture mean e2e = (1000+1100+900+1400+1050)/5 = 1090 ms = 1.09 s.
    assert latency["e2el"]["mean"] == pytest.approx(1.09)
    assert latency["itl"]["mean"] == pytest.approx(0.01824)
    _assert_stable_request_metrics_schema(agg)
    assert "mean_ttft" not in agg
    assert "p50_ttft" not in agg
    assert "p90_itl" not in agg


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

    metric = agg["request_metrics"]["latency"]["intvty"]
    # Hand-worked ITL p90/p75/p50/mean: 84, 60, 20, 43 1/3 ms.
    assert metric["p90"] == pytest.approx(11.90476)
    assert metric["p75"] == pytest.approx(16.66667)
    assert metric["p50"] == pytest.approx(50.0)
    assert metric["mean"] == pytest.approx(23.07692)


def test_processor_aggregates_e2e_normalized_interactivity_from_slow_tail(
    tmp_path: Path,
):
    result_dir = tmp_path / "results"
    artifact = result_dir / "aiperf_artifacts"
    artifact.mkdir(parents=True)

    # E2EL / OSL ratios are 0.02 and 0.04 seconds per output token.
    records = [
        _make_record(
            conv_id="trace-fast",
            turn_index=0,
            isl=100,
            osl=50,
            ttft_ms=30.0,
            e2e_ms=1_000.0,
            itl_ms=10.0,
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
        ),
        _make_record(
            conv_id="trace-slow",
            turn_index=0,
            isl=100,
            osl=50,
            ttft_ms=30.0,
            e2e_ms=2_000.0,
            itl_ms=10.0,
            start_ns=2_000_000_000,
            end_ns=4_000_000_000,
        ),
    ]
    with open(artifact / "profile_export.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    with open(artifact / "profile_export_aiperf.json", "w") as f:
        json.dump({"request_count": len(records)}, f)

    agg = _run_processor(result_dir, tmp_path / "out")
    metric = agg["request_metrics"]["latency"]["e2e_norm_intvty"]

    assert metric["mean"] == pytest.approx(33.33333)
    assert metric["p75"] == pytest.approx(28.57143)
    assert metric["p90"] == pytest.approx(26.31579)
    assert metric["std"] == pytest.approx(12.5)
    assert metric["p50"] >= metric["p75"] >= metric["p90"] >= metric["p95"]
    assert "p99" not in metric
    _assert_stable_request_metrics_schema(agg)


def test_e2e_normalized_interactivity_pairs_metrics_within_each_record():
    missing_osl = _make_record(
        conv_id="trace-missing-osl",
        turn_index=0,
        isl=100,
        osl=50,
        ttft_ms=30.0,
        e2e_ms=1_000.0,
        itl_ms=10.0,
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
    )
    del missing_osl["metrics"]["output_sequence_length"]
    missing_e2el = _make_record(
        conv_id="trace-missing-e2el",
        turn_index=0,
        isl=100,
        osl=50,
        ttft_ms=30.0,
        e2e_ms=1_000.0,
        itl_ms=10.0,
        start_ns=2_000_000_000,
        end_ns=3_000_000_000,
    )
    del missing_e2el["metrics"]["request_latency"]

    _, nested = compute_request_metrics([missing_osl, missing_e2el])

    assert nested["latency"]["e2e_norm_intvty"] == {}


def test_e2e_normalized_interactivity_skips_nonpositive_and_nonfinite_values():
    invalid_pairs = [
        (0.0, 50),
        (-1.0, 50),
        (float("nan"), 50),
        (float("inf"), 50),
        (1_000.0, 0),
        (1_000.0, -1),
        (1_000.0, float("nan")),
        (1_000.0, float("inf")),
    ]
    records = []
    for idx, (e2e_ms, osl) in enumerate(invalid_pairs):
        records.append(
            _make_record(
                conv_id=f"trace-invalid-{idx}",
                turn_index=0,
                isl=100,
                osl=osl,
                ttft_ms=30.0,
                e2e_ms=e2e_ms,
                itl_ms=10.0,
                start_ns=(idx + 1) * 1_000_000_000,
                end_ns=(idx + 2) * 1_000_000_000,
            )
        )
    records.append(
        _make_record(
            conv_id="trace-valid",
            turn_index=0,
            isl=100,
            osl=50,
            ttft_ms=30.0,
            e2e_ms=1_000.0,
            itl_ms=10.0,
            start_ns=10_000_000_000,
            end_ns=11_000_000_000,
        )
    )

    _, nested = compute_request_metrics(records)
    metric = nested["latency"]["e2e_norm_intvty"]

    assert metric == {
        "mean": 50.0,
        "p50": 50.0,
        "p75": 50.0,
        "p90": 50.0,
        "p95": 50.0,
        "std": 0.0,
    }


def test_e2e_normalized_interactivity_empty_without_valid_samples():
    _, nested = compute_request_metrics([])

    assert nested["latency"]["e2e_norm_intvty"] == {}


def test_processor_throughput_per_gpu(tmp_path: Path):
    result_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "out"
    agg = _run_processor(
        result_dir,
        output_dir,
        env_overrides={"TP": "4", "PP_SIZE": "2", "DCP_SIZE": "2", "PCP_SIZE": "2"},
    )
    per_gpu = agg["request_metrics"]["throughput"]["per_gpu"]
    assert agg["pp"] == 2
    assert agg["dcp_size"] == 2
    assert agg["pcp_size"] == 2
    assert agg["num_gpus"] == 16
    # 840 input + 275 output tokens over 4.1 seconds on 16 GPUs.
    assert per_gpu["total_tput_tps"] == pytest.approx(16.99695)
    assert per_gpu["input_tput_tps"] == pytest.approx(12.80488)
    assert per_gpu["output_tput_tps"] == pytest.approx(4.19207)


def test_processor_serializes_shared_expert_gpus(tmp_path: Path):
    result_dir = _write_fixture(tmp_path)
    agg = _run_processor(result_dir, tmp_path / "out", env_overrides={"TP": "4", "EP_SIZE": "4"})
    # EP shares devices with TP.
    assert agg["num_gpus"] == 4


def test_processor_aggregates_full_response_itl_and_interactivity(tmp_path: Path):
    result_dir = tmp_path / "results"
    artifact = result_dir / "aiperf_artifacts"
    artifact.mkdir(parents=True)

    full_response_itls_ms = (5.469791, 5.0, 4.0)
    with open(artifact / "profile_export.jsonl", "w") as f:
        for idx, full_response_itl_ms in enumerate(full_response_itls_ms):
            record = _make_record(
                conv_id=f"trace-{idx}",
                turn_index=0,
                isl=100,
                osl=26_571,
                ttft_ms=529.058811,
                e2e_ms=610.559573,
                itl_ms=0.003067398,
                start_ns=(idx + 1) * 1_000_000_000,
                end_ns=(idx + 1) * 1_000_000_000 + 145_861_451_008,
            )
            record["metrics"]["full_response_inter_token_latency"] = {
                "value": full_response_itl_ms,
                "unit": "ms",
            }
            f.write(json.dumps(record) + "\n")

    with open(artifact / "profile_export_aiperf.json", "w") as f:
        json.dump({"request_count": len(full_response_itls_ms)}, f)

    agg = _run_processor(result_dir, tmp_path / "out")
    latency = agg["request_metrics"]["latency"]
    full_response_itl = latency["full_response_itl"]
    full_response_intvty = latency["full_response_intvty"]

    assert full_response_itl["p50"] == pytest.approx(0.005)
    assert full_response_itl["p75"] == pytest.approx(0.00523)
    assert full_response_intvty["p50"] == pytest.approx(200.0)
    # p75 interpolates to 5.2348955 ms before conversion and rounding.
    assert full_response_intvty["p75"] == pytest.approx(191.02578)


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
        _gpu_shape(os.environ)


@pytest.mark.parametrize("env", [{}, {"PP_SIZE": "", "DCP_SIZE": "", "PCP_SIZE": ""}])
def test_gpu_shape_defaults_empty_parallelism_to_one(env):
    assert _gpu_shape(env) == ({"pp": 1, "dcp_size": 1, "pcp_size": 1}, 1, 1, 1, "false")


@pytest.mark.parametrize("decode_workers,expected_gpus,expected_decode", [
    ("3", 78, (2, 11, 1, 9, 5)),
    ("0", 48, (0, 0, 1, 1, 1)),
])
def test_gpu_shape_counts_workers_and_normalizes_absent_decode(
    decode_workers, expected_gpus, expected_decode,
):
    env = {
        "IS_MULTINODE": "true", "PREFILL_NUM_WORKERS": "2", "PREFILL_TP": "3",
        "PREFILL_PP_SIZE": "2", "PREFILL_PCP_SIZE": "4", "PREFILL_DCP_SIZE": "7",
        "PREFILL_EP": "8", "PREFILL_DP_ATTN": "YES", "DECODE_NUM_WORKERS": decode_workers,
        "DECODE_TP": "2", "DECODE_EP": "11", "DECODE_PP_SIZE": "1",
        "DECODE_DCP_SIZE": "9", "DECODE_PCP_SIZE": "5", "DECODE_DP_ATTN": "false",
        "PREFILL_GPUS": "999", "DECODE_GPUS": "999",
    }
    fields, num_gpus, tp, ep, attention = _gpu_shape(env)
    # Two 24-GPU prefill workers plus either three 10-GPU decode workers or
    # no decode workers. EP and DCP do not allocate extra GPUs.
    assert num_gpus == expected_gpus
    assert fields["num_prefill_gpu"] == 48
    assert fields["num_decode_gpu"] == (30 if decode_workers == "3" else 0)
    assert tuple(fields[key] for key in ("decode_tp", "decode_ep", "decode_pp",
                                        "decode_dcp_size", "decode_pcp_size")) == expected_decode
    assert (tp, ep, attention) == ((5, 11, "true") if decode_workers == "3" else (3, 8, "true"))
    assert fields["prefill_dp_attention"] == "YES"
    assert fields["decode_dp_attention"] == "false"


@pytest.mark.parametrize("env,error_type,message", [
    ({"IS_MULTINODE": "true", "TP": "bad"}, ValueError, "invalid literal for int"),
    ({"IS_MULTINODE": "true", "PREFILL_PP_SIZE": "0", "PREFILL_HARDWARE": "gpu"},
     SystemExit, "Multinode PP, DCP, and PCP sizes must be positive integers."),
    ({"IS_MULTINODE": "true", "DECODE_NUM_WORKERS": "0", "DECODE_PCP_SIZE": "0"},
     SystemExit, "Multinode PP, DCP, and PCP sizes must be positive integers."),
])
def test_gpu_shape_preserves_parsing_and_validation_order(env, error_type, message):
    with pytest.raises(error_type, match=message):
        _gpu_shape(env)


@pytest.mark.parametrize("error,category", [
    ({"type": "HTTPStatusError", "message": "500 server error"}, "HTTPStatusError"),
    (" \t\r\n", "unknown"),
    ({"message": " \t\r\n"}, "unknown"),
])
def test_processor_surfaces_request_accounting(tmp_path: Path, error, category):
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
        e2e_ms=10_000.0,
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
        e2e_ms=20_000.0,
        itl_ms=10.0,
        start_ns=3_000_000_000,
        end_ns=4_000_000_000,
    )
    errored["error"] = error

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
        "error_categories": {category: 1},
    }
    e2e_norm_intvty = agg["request_metrics"]["latency"]["e2e_norm_intvty"]
    assert e2e_norm_intvty["mean"] == pytest.approx(50.0)
    assert e2e_norm_intvty["p95"] == pytest.approx(50.0)


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
    assert agg["request_metrics"]["latency"]["ttft"]["mean"] == pytest.approx(0.03)


def test_processor_rounds_decimal_outputs_to_five_decimal_places(tmp_path: Path):
    result_dir = tmp_path / "results"
    artifact = result_dir / "aiperf_artifacts"
    artifact.mkdir(parents=True)
    record = _make_record(
        conv_id="trace-A",
        turn_index=0,
        isl=100,
        osl=50,
        ttft_ms=12.3456789,
        e2e_ms=1_000.0,
        itl_ms=10.0,
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
    )
    (artifact / "profile_export.jsonl").write_text(json.dumps(record) + "\n")
    (artifact / "profile_export_aiperf.json").write_text(json.dumps({"request_count": 1}))

    agg = _run_processor(result_dir, tmp_path / "out")

    assert agg["request_metrics"]["latency"]["ttft"]["mean"] == 0.01235


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
                "metadata": {
                    "dataset": {"hf_dataset_name": "semianalysisai/cc-traces-weka-042026"}
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
        [sys.executable, "-m", "infx.results.agentic.process_agentic_result"],
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
        55.0
    )


@pytest.mark.parametrize("cache_state", ["matching", "missing", "ambiguous", "no_identity"])
def test_processor_expected_output_uses_declared_dataset(tmp_path: Path, cache_state: str):
    result_dir = _write_fixture(tmp_path)
    profile_path = result_dir / "aiperf_artifacts" / "profile_export_aiperf.json"
    profile = json.loads(profile_path.read_text())
    if cache_state == "no_identity":
        del profile["metadata"]["dataset"]["hf_dataset_name"]
    profile["theoretical_prefix_cache_hit"] = {"unit": "%", "avg": 25.0}
    profile_path.write_text(json.dumps(profile))

    hf_cache = tmp_path / "hf"
    snapshots = [("cc-traces-weka-062126-256k", "newer", 900, 200)]
    if cache_state != "missing":
        snapshots.append(("cc-traces-weka-062126", "correct", 100, 100))
    if cache_state == "ambiguous":
        snapshots.append(("cc-traces-weka-062126", "another", 200, 300))
    for dataset, revision, output, modified_at in snapshots:
        snapshot = hf_cache / f"datasets--semianalysisai--{dataset}" / "snapshots" / revision
        snapshot.mkdir(parents=True)
        traces = [
            {"id": trace_id, "requests": [{"type": "n", "out": output}] * turns}
            for trace_id, turns in [("trace-A", 3), ("trace-B", 2)]
        ]
        (snapshot / "traces.jsonl").write_text("\n".join(json.dumps(trace) for trace in traces))
        os.utime(snapshot, (modified_at, modified_at))

    agg = _run_processor(result_dir, tmp_path / "out", {"HF_HUB_CACHE": str(hf_cache)})

    expected = agg["request_metrics"]["tokens"]["output_expected"]
    if cache_state == "matching":
        assert expected["mean"] == 100
    else:
        assert expected == {}
    assert agg["request_metrics"]["tokens"]["output_actual"]["mean"] == 55
    assert agg["num_requests_successful"] == 5
    assert agg["request_metrics"]["cache"]["theoretical_cache_hit_rate"] == 0.25


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


def test_builder_uses_explicit_inputs_without_environment_or_file_access(monkeypatch):
    records = [
        _make_record(conv_id="root", turn_index=0, isl=100, osl=50,
                     ttft_ms=30, e2e_ms=1000, itl_ms=10,
                     start_ns=1_000_000_000, end_ns=2_000_000_000),
        _make_record(conv_id="root::sa:child", turn_index=1, isl=200, osl=100,
                     ttft_ms=50, e2e_ms=2000, itl_ms=20,
                     start_ns=2_000_000_000, end_ns=4_000_000_000),
    ]
    aggregate = {"metadata": {"dataset": {"hf_dataset_name": "example/traces"}}}
    traces = [{"id": "root", "requests": [
        {"type": "n", "out": 11}, {"type": "tool", "out": 999},
        {"type": "s", "output_length": 29},
    ]}]
    accounting = {"records_total": 4, "records_profiled": 2, "records_dropped_total": 2,
                  "records_warmup_dropped": 1, "records_error_dropped": 1,
                  "error_categories": {"Timeout": 1}}
    env = {"KV_OFFLOADING": "none", "TP": "4", "FRAMEWORK": "vllm"}
    before = deepcopy((records, aggregate, traces, accounting, env))

    def forbidden(*args, **kwargs):
        raise AssertionError("builder accessed ambient environment or filesystem")

    class NoEnvironment(dict):
        get = __getitem__ = __iter__ = forbidden

    with monkeypatch.context() as isolated:
        isolated.setattr(os, "environ", NoEnvironment())
        isolated.setattr("builtins.open", forbidden)
        isolated.setattr(Path, "open", forbidden)
        result = build_result(
            records, aggregate, MappingProxyType(env),
            request_accounting=accounting, traces=iter(traces),
        )
        second = build_result(records, aggregate, {**env, "TP": "2"})

    # 450 tokens over three seconds, divided among four GPUs.
    assert result["request_metrics"]["throughput"]["per_gpu"] == {
        "total_tput_tps": 37.5, "output_tput_tps": 12.5, "input_tput_tps": 25.0,
    }
    assert second["request_metrics"]["throughput"]["per_gpu"]["total_tput_tps"] == 75.0
    assert result["request_metrics"]["tokens"]["output_expected"]["mean"] == 20
    assert second["request_metrics"]["tokens"]["output_expected"] == {}
    assert result["num_requests_total"] == 4
    assert result["num_requests_successful"] == 2
    assert result["dataset"] == {"hf_dataset_name": "example/traces"}
    assert (records, aggregate, traces, accounting, env) == before


@pytest.mark.parametrize("override,message", [
    ({"KV_OFFLOADING": ""}, "Missing required environment variable: KV_OFFLOADING"),
    ({"PP_SIZE": "0"}, "PP_SIZE, DCP_SIZE, and PCP_SIZE must be positive integers."),
    ({"ROUTER_METADATA": "{"}, "ROUTER_METADATA must contain valid JSON"),
    ({}, "invalid literal for int() with base 10: 'bad-output'"),
])
def test_processor_preserves_validation_before_optional_trace_reads(tmp_path, override, message):
    result_dir = _write_fixture(tmp_path)
    cache = tmp_path / "hf"
    snapshot = cache / "datasets--semianalysisai--cc-traces-weka-062126" / "snapshots" / "one"
    snapshot.mkdir(parents=True)
    (snapshot / "traces.json").write_text(json.dumps({
        "id": "trace-A", "requests": [{"type": "n", "out": "bad-output"}],
    }))
    out = tmp_path / "out"
    env = {
        "PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(REPO_ROOT),
        "RESULT_DIR": str(result_dir), "AGENTIC_OUTPUT_DIR": str(out),
        "RESULT_FILENAME": "agg_test", "KV_OFFLOADING": "none",
        "FRAMEWORK": "vllm", "HF_HUB_CACHE": str(cache), **override,
    }
    proc = subprocess.run(
        [sys.executable, "-m", "infx.results.agentic.process_agentic_result"],
        cwd=REPO_ROOT, env=env, text=True, capture_output=True, timeout=30,
    )
    assert proc.returncode == 1
    assert proc.stderr.rstrip().endswith(message)
    assert proc.stdout == ""
    assert not out.exists()


@pytest.mark.parametrize("cache_env", ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"])
def test_processor_trace_cache_precedence_and_last_duplicate(tmp_path, cache_env):
    result_dir = _write_fixture(tmp_path)
    cache = tmp_path / "hf"
    snapshot = cache / "datasets--semianalysisai--cc-traces-weka-062126" / "snapshots" / "one"
    snapshot.mkdir(parents=True)
    trace = {"id": "trace-A", "requests": [{"type": "n", "out": 17}] * 3}
    # Bad JSON is skipped; later JSON overrides earlier JSONL for this trace.
    (snapshot / "a.jsonl").write_text("{bad json\n" + json.dumps(trace) + "\n")
    (snapshot / "z.json").write_text(json.dumps({
        **trace, "requests": [{"type": "s", "output_length": 23}] * 3,
    }))
    (snapshot / "zz.json").write_text(json.dumps({"id": "trace-A", "requests": []}))
    env = {"HF_HUB_CACHE": "", "HUGGINGFACE_HUB_CACHE": "", "HF_HOME": ""}
    env[cache_env] = str(cache)
    if cache_env == "HF_HOME":
        hub = cache / "hub"
        hub.mkdir()
        (cache / "datasets--semianalysisai--cc-traces-weka-062126").rename(
            hub / "datasets--semianalysisai--cc-traces-weka-062126"
        )
    else:
        # A configured direct cache takes precedence over HF_HOME.
        env["HF_HOME"] = str(tmp_path / "wrong-home")
    if cache_env == "HF_HUB_CACHE":
        env["HUGGINGFACE_HUB_CACHE"] = str(tmp_path / "wrong-hub")
    result = _run_processor(result_dir, tmp_path / "out", env)
    expected = result["request_metrics"]["tokens"]["output_expected"]
    assert expected["mean"] == 23
    assert expected["std"] == 0


@pytest.mark.parametrize("offsets,expected_mean,expected_p95,windows", [
    ([0, 500_000_000, 1_000_000_000, 1_500_000_000, 2_000_000_000], 2, 2, 2),
    ([0, 1_000_000_000, 1_000_000_000, 2_000_000_000], 1.5, 1.95, 2),
    ([0, 250_000_000, 500_000_000], 6, None, 0),
])
def test_qps_windows_exclude_right_boundary_and_preserve_duplicates(offsets, expected_mean, expected_p95, windows):
    records = [{"metadata": {"request_end_ns": 1_000_000_000 + offset}} for offset in reversed(offsets)]
    flat, nested = compute_qps_stats(records)
    assert nested["samples"] == windows
    assert flat["mean_qps"] == pytest.approx(expected_mean)
    if expected_p95 is None:
        assert "p95_qps" not in flat
    else:
        assert flat["p95_qps"] == pytest.approx(expected_p95)


@pytest.mark.parametrize("invalid_turn", [-1, -3, 2])
def test_expected_output_tokens_ignore_out_of_range_trace_turns(invalid_turn):
    records = [{"metadata": {"conversation_id": "root", "turn_index": index}}
               for index in (0, 1, invalid_turn)]
    traces = [{"id": "root", "requests": [{"type": "n", "out": 11},
                                           {"type": "s", "out": 29}]}]
    _, metrics = compute_request_metrics(records, traces=traces)
    assert metrics["tokens"]["output_expected"]["mean"] == 20
    assert metrics["tokens"]["output_expected"]["std"] == 9
