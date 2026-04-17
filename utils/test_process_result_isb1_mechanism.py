"""Integration test for mechanism_eval wiring in process_result_isb1.py.

Runs process_result_isb1.py in a subprocess with a minimal replay fixture and
verifies that the aggregated JSON carries the mechanism_eval schema fields and
the mechanism_eval_validation record.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parent
SCRIPT = UTILS_DIR / "process_result_isb1.py"


def _minimal_replay_payload(export_file: str) -> dict:
    return {
        "model_id": "deepseek-ai/DeepSeek-R1-0528",
        "max_concurrency": 2,
        "num_sessions": 2,
        "max_turns": 3,
        "num_warmup_sessions": 0,
        "aggregate_metrics": {
            "total_token_throughput_tps": 1000.0,
            "output_throughput_tps": 800.0,
            "total_sessions": 2,
            "completed_sessions": 2,
            "session_throughput_sps": 0.05,
            "median_ttft_ms": 120.0,
            "p99_ttft_ms": 240.0,
            "median_tpot_ms": 15.0,
            "p99_tpot_ms": 20.0,
            "total_wall_time_s": 600.0,
        },
        "per_turn_metrics": {},
        "server_metrics_summary": {
            "gpu_cache_usage_peak": 0.25,
            "cpu_cache_usage_peak": 0.0,
            "cpu_cache_metric_available": True,
            "kv_offload_observed": False,
            "preemption_count": 0,
            "observability_status": "ok",
        },
        "selection": {
            "support_statuses": ["reviewed_preview"],
            "benchmark_certification_statuses": ["dataset_replay_verified"],
        },
        "mode": "multi-turn",
        "harness_request_mode": "auto",
        "depth_telemetry": {
            "total_actual_input_tokens": 12000,
            "max_actual_context_len_per_turn": 10000,
        },
    }


def _run_process_result(
    tmp_path: Path,
    export_file: Path,
    replay_payload: dict,
    extra_env: dict | None = None,
) -> dict:
    result_filename = "mechanism_test"
    replay_path = tmp_path / f"{result_filename}.json"
    replay_path.write_text(json.dumps(replay_payload))

    env = os.environ.copy()
    # Strip any mechanism env vars from the outer environment so we control
    # them explicitly per-case.
    for key in list(env):
        if (
            key.startswith(("MECHANISM", "COMPRESSION_", "DECOMPRESSION_",
                            "QUALITY_", "DRAFT_MODEL_", "SPECULATIVE_"))
        ):
            env.pop(key, None)

    env.update(
        {
            "RUNNER_TYPE": "h100",
            "FRAMEWORK": "vllm",
            "PRECISION": "fp8",
            "RESULT_FILENAME": result_filename,
            "MODEL_PREFIX": "dsr1",
            "IMAGE": "vllm/vllm-openai:v0.11.0",
            "TP": "8",
            "EP_SIZE": "1",
            "DP_ATTENTION": "false",
            "BENCHMARK_TYPE": "isb1_replay",
            "EXPORT_FILE": str(export_file),
            "RUNTIME_STACK_ID": "standalone:vllm",
            "HARDWARE_PROFILE_ID": "nvidia:h100_sxm_80gb",
            "CANONICAL_MODEL_ID": "deepseek_r1_0528",
            "REQUEST_MODE": "multi-turn",
            "MAX_CONCURRENCY": "2",
            "SUPPORT_STATUS": "reviewed_preview",
        }
    )
    if extra_env:
        env.update(extra_env)

    subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=tmp_path,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
    )

    aggregated = json.loads((tmp_path / f"agg_{result_filename}.json").read_text())
    return aggregated


def test_process_result_defaults_to_baseline_when_no_env(tmp_path):
    export_file = tmp_path / "code_8k1k.json"
    export_file.write_text(
        json.dumps(
            {
                "adapter_id": "test",
                "bundle_id": "test_bundle",
                "served_shape": {"isl": 8192, "osl": 1024},
                "exports": [{"context_band": "lc3_8k"}],
            }
        )
    )

    aggregated = _run_process_result(tmp_path, export_file, _minimal_replay_payload(str(export_file)))

    assert aggregated["mechanism"] == "baseline"
    assert aggregated["mechanism_variant"] is None
    assert aggregated["compression_method"] is None
    assert aggregated["quality_eval_id"] is None
    assert "mechanism_eval_validation" in aggregated
    # Baseline is always considered registered (even with variant=None).
    assert aggregated["mechanism_eval_validation"]["mechanism_eval_registered"] is True


def test_process_result_surfaces_registered_fp8_kv_fields(tmp_path):
    export_file = tmp_path / "code_131k1k.json"
    export_file.write_text(
        json.dumps(
            {
                "adapter_id": "test",
                "bundle_id": "test_bundle",
                "served_shape": {"isl": 131072, "osl": 1024},
                "exports": [{"context_band": "xlc2_384k_512k"}],
            }
        )
    )

    aggregated = _run_process_result(
        tmp_path,
        export_file,
        _minimal_replay_payload(str(export_file)),
        extra_env={
            "MECHANISM": "kv_quantization",
            "MECHANISM_VARIANT": "fp8_e4m3",
            "COMPRESSION_METHOD": "fp8_e4m3",
            "COMPRESSION_SCOPE": "kv_cache",
            "COMPRESSION_RATIO": "0.5",
            "QUALITY_EVAL_ID": "ruler_v1",
            "QUALITY_EVAL_STATUS": "pending",
        },
    )

    assert aggregated["mechanism"] == "kv_quantization"
    assert aggregated["mechanism_variant"] == "fp8_e4m3"
    assert aggregated["compression_method"] == "fp8_e4m3"
    assert aggregated["compression_scope"] == "kv_cache"
    assert aggregated["compression_ratio"] == 0.5
    assert aggregated["quality_eval_id"] == "ruler_v1"
    assert aggregated["quality_eval_status"] == "pending"
    validation = aggregated["mechanism_eval_validation"]
    assert validation["mechanism_eval_registered"] is True
    assert validation["quality_eval_registered"] is True
    assert validation["quality_eval_status_known"] is True
    assert validation["issues"] == []


def test_process_result_flags_unregistered_variant(tmp_path):
    export_file = tmp_path / "code_8k1k.json"
    export_file.write_text(
        json.dumps(
            {
                "adapter_id": "test",
                "served_shape": {"isl": 8192, "osl": 1024},
                "exports": [{"context_band": "lc3_8k"}],
            }
        )
    )

    aggregated = _run_process_result(
        tmp_path,
        export_file,
        _minimal_replay_payload(str(export_file)),
        extra_env={
            "MECHANISM": "kv_quantization",
            "MECHANISM_VARIANT": "invented_variant",
        },
    )

    validation = aggregated["mechanism_eval_validation"]
    assert validation["mechanism_eval_registered"] is False
    assert any(
        "not registered in mechanism_variant_registry.json" in issue
        for issue in validation["issues"]
    )
