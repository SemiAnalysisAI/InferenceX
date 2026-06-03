"""Smoke tests for process_agentic_result.py against synthetic aiperf output.

The processor consumes three files in $RESULT_DIR/aiperf_artifacts/:
profile_export.jsonl, profile_export_aiperf.json, and
(optionally) server_metrics_export.json. It writes one
$RESULT_FILENAME.json under $AGENTIC_OUTPUT_DIR. We build a minimal
fixture, run the processor, and assert the agg JSON has every key
utils/summarize.py reads.

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


PROCESSOR = Path(__file__).resolve().parent / "process_agentic_result.py"

# Keys consumed by utils/summarize.py:78-195 for agentic results. The
# processor must emit every one of these for downstream aggregation.
SUMMARIZE_KEYS = {
    "infmax_model_prefix",
    "hw",
    "framework",
    "precision",
    "conc",
    "scenario_type",
    "is_multinode",
    "mean_ttft",
    "p75_ttft",
    "p90_ttft",
    "p95_ttft",
    "mean_tpot",
    "p75_tpot",
    "mean_intvty",
    "p75_intvty",
    "p90_intvty",
    "p95_intvty",
    "mean_e2el",
    "p75_e2el",
    "p90_e2el",
    "p95_e2el",
    "tput_per_gpu",
    "output_tput_per_gpu",
    "input_tput_per_gpu",
}


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
            },
            f,
        )

    # No server_metrics_export.json — exercises the missing-file path.
    return result_dir


def _run_processor(result_dir: Path, output_dir: Path) -> dict:
    env = os.environ.copy()
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
            "EP_SIZE": "1",
            "DP_ATTENTION": "false",
            "CONC": "8",
            "OFFLOADING": "none",
            "RUNNER_TYPE": "b200-x4",
            "IMAGE": "test/image:0.1",
            "SPEC_DECODING": "none",
            "DISAGG": "false",
            "IS_MULTINODE": "false",
            # Hide HF cache so theoretical_cache_hit_rate stays None and we
            # don't depend on a downloaded dataset in CI.
            "HF_HUB_CACHE": str(result_dir / "_no_such_cache"),
        }
    )
    proc = subprocess.run(
        [sys.executable, str(PROCESSOR)],
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


def test_processor_emits_required_summarize_keys(tmp_path: Path):
    result_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "out"
    agg = _run_processor(result_dir, output_dir)
    missing = SUMMARIZE_KEYS - set(agg.keys())
    assert not missing, f"agg JSON missing summarize keys: {sorted(missing)}"


def test_processor_latency_units_are_seconds(tmp_path: Path):
    """aiperf reports ms; legacy schema is seconds. Verify conversion."""
    result_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "out"
    agg = _run_processor(result_dir, output_dir)
    # Fixture mean ttft = (30+35+32+40+33)/5 = 34.0 ms = 0.034 s.
    assert 0.020 < agg["mean_ttft"] < 0.050, agg["mean_ttft"]
    # Fixture mean e2e = (1000+1100+900+1400+1050)/5 = 1090 ms = 1.09 s.
    assert 0.5 < agg["mean_e2el"] < 2.0, agg["mean_e2el"]
    # mean itl = ~18 ms = 0.018 s.
    assert 0.010 < agg["mean_itl"] < 0.030, agg["mean_itl"]


def test_processor_throughput_per_gpu(tmp_path: Path):
    result_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "out"
    agg = _run_processor(result_dir, output_dir)
    assert agg["tput_per_gpu"] > 0
    assert agg["input_tput_per_gpu"] > 0
    assert agg["output_tput_per_gpu"] > 0


def test_processor_handles_missing_server_metrics(tmp_path: Path):
    """No server_metrics_export.json -> server cache fields are None, not error."""
    result_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "out"
    agg = _run_processor(result_dir, output_dir)
    assert agg["server_gpu_cache_hit_rate"] is None
    assert agg["theoretical_cache_hit_rate"] is None
    # Non-server-derived totals fall back to per-record sums.
    assert agg["total_prompt_tokens"] == 100 + 180 + 120 + 200 + 240
    assert agg["total_generation_tokens"] == 50 + 60 + 40 + 70 + 55
    assert agg["total_requests_completed"] == 5


def test_processor_response_cache_hit_rate_populated_when_cached_tokens_present(
    tmp_path: Path,
):
    result_dir = tmp_path / "results"
    artifact = result_dir / "aiperf_artifacts"
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
    rec["metrics"]["usage_prompt_cache_read_tokens"] = {
        "value": 60,
        "unit": "tokens",
    }
    with open(artifact / "profile_export.jsonl", "w") as f:
        f.write(json.dumps(rec) + "\n")
    with open(artifact / "profile_export_aiperf.json", "w") as f:
        json.dump({"request_count": 1}, f)

    output_dir = tmp_path / "out"
    agg = _run_processor(result_dir, output_dir)
    assert agg["response_cache_hit_rate"] == pytest.approx(0.6)


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
    assert agg["server_gpu_cache_hit_rate"] == pytest.approx(0.8)
    assert agg["total_prompt_tokens"] == 12345
    assert agg["total_generation_tokens"] == 6789


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
    assert agg["server_gpu_cache_hit_rate"] == pytest.approx(0.3)


def test_processor_loads_traces_jsonl_for_theoretical_cache(tmp_path: Path):
    """JSONL trace format (one trace per line) populates theoretical_cache_hit_rate.

    The published HF dataset ships a single traces.jsonl — the loader must
    accept it in addition to the legacy per-trace *.json layout.
    """
    result_dir = _write_fixture(tmp_path)
    # Build a fake HF cache with traces.jsonl matching the conv_ids the
    # fixture references (trace-A, trace-B).
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
            "RUNNER_TYPE": "h100-x4",
            "HF_HUB_CACHE": str(hf_cache),
        }
    )
    proc = subprocess.run(
        [sys.executable, str(PROCESSOR)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    agg = json.loads((tmp_path / "out" / "agg_test.json").read_text())
    # Walking trace-A turns 0..2 yields hash_ids
    #   [1,2,3] | [1,2,3,4] | [1,2,3,4,5]
    # Hits = blocks seen earlier in the same trace = 3 (turn 1) + 4 (turn 2) = 7
    # Total = 3 + 4 + 5 = 12
    # Walking trace-B turns 0..1: hits = 2 (turn 1's [10,11]) ; total = 2 + 4 = 6
    # Aggregate: hits=9, total=18 -> 0.5
    assert agg["theoretical_cache_hit_rate"] == pytest.approx(0.5)
    # output_tokens_expected populated from trace metadata (5 records: A turns 0,1,2 + B turns 0,1)
    assert agg["mean_output_tokens_expected"] == pytest.approx(
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
