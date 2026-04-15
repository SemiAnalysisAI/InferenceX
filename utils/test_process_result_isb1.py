import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent / "process_result_isb1.py"


def write_export_fixture(tmp_path: Path, relative_path: str, payload: dict) -> str:
    export_path = tmp_path / relative_path
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(json.dumps(payload))
    return str(export_path.relative_to(tmp_path))


@pytest.fixture
def sample_replay_result():
    return {
        "model_id": "deepseek-ai/DeepSeek-R1-0528",
        "mode": "export_replay",
        "max_concurrency": 8,
        "num_sessions": 2,
        "max_turns": 4,
        "num_warmup_sessions": 1,
        "harness_request_mode": "auto",
        "selection": {
            "adapter_id": "inferencex_multiturn",
            "selected_sessions": 2,
            "runtime_stack_ids": ["vllm-0.8.5-h200"],
            "hardware_profile_ids": ["h200-8gpu"],
            "canonical_model_ids": ["deepseek-r1-0528"],
            "support_statuses": ["supported"],
            "support_status_counts": {"supported": 2},
            "benchmark_certification_statuses": ["dataset_replay_verified"],
            "benchmark_certification_status_counts": {
                "dataset_replay_verified": 2
            },
            "request_mode_mix": {"chat": 2},
        },
        "server_metrics_summary": {
            "cache_usage_avg": 0.45,
            "cache_hit_rate_avg": 0.15,
            "gpu_cache_usage_avg": 0.45,
            "gpu_cache_usage_peak": 0.78,
            "gpu_cache_metric_name": "vllm:gpu_cache_usage_perc",
            "cpu_cache_usage_avg": 0.12,
            "cpu_cache_usage_peak": 0.31,
            "cpu_cache_metric_name": "vllm:cpu_cache_usage_perc",
            "cpu_cache_metric_available": True,
            "observability_status": "direct_cpu_cache_metric",
            "kv_offload_observed": True,
            "samples": 5,
        },
        "per_turn_metrics": {
            "turn_1": {
                "completed": 2,
                "mean_context_len": 8192.0,
                "mean_ttft_ms": 180.0,
                "p99_ttft_ms": 300.0,
                "mean_e2el_ms": 1000.0,
            }
        },
        "aggregate_metrics": {
            "completed_sessions": 2,
            "total_sessions": 2,
            "total_input_tokens": 1000,
            "total_output_tokens": 300,
            "total_wall_time_s": 2.0,
            "session_throughput_sps": 1.0,
            "output_throughput_tps": 150.0,
            "total_token_throughput_tps": 650.0,
            "mean_ttft_ms": 200.0,
            "median_ttft_ms": 180.0,
            "p99_ttft_ms": 500.0,
            "mean_tpot_ms": 20.0,
            "median_tpot_ms": 25.0,
            "p99_tpot_ms": 50.0,
            "mean_e2el_ms": 1200.0,
            "median_e2el_ms": 1100.0,
            "p99_e2el_ms": 2000.0,
        },
    }


@pytest.fixture
def base_env():
    return {
        "RUNNER_TYPE": "h200-cw-1",
        "FRAMEWORK": "vllm",
        "PRECISION": "fp8",
        "RESULT_FILENAME": "isb1_result",
        "MODEL_PREFIX": "dsr1",
        "IMAGE": "vllm/vllm-openai:v0.8.5",
        "TP": "8",
        "EP_SIZE": "1",
        "DP_ATTENTION": "false",
        "BENCHMARK_TYPE": "isb1_replay",
        "EXPORT_FILE": "datasets/isb1/exports/core/chat_8k1k.json",
        "RUNTIME_STACK_ID": "vllm-0.8.5-h200",
        "HARDWARE_PROFILE_ID": "h200-8gpu",
        "CANONICAL_MODEL_ID": "deepseek-r1-0528",
        "SUPPORT_STATUS": "supported",
        "REQUEST_MODE": "multi-turn",
        "MAX_CONCURRENCY": "8",
        "SPEC_DECODING": "none",
        "IGNORE_WAITS": "true",
        "GITHUB_REF": "refs/heads/test-isb1-traceability",
    }


def run_script(tmp_path, env, replay_result, result_filename="isb1_result"):
    result_file = tmp_path / f"{result_filename}.json"
    result_file.write_text(json.dumps(replay_result))

    env = env.copy()
    env["RESULT_FILENAME"] = result_filename

    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )


def assert_traceability_fields(
    output_data: dict, result_filename: str, dispatch_ref: str = "refs/heads/test-isb1-traceability"
):
    assert output_data["result_filename"] == result_filename
    assert output_data["artifact_stems"] == {
        "processed": f"isb1_{result_filename}",
        "raw_replay": f"replay_{result_filename}",
        "server_logs": f"server_logs_{result_filename}",
        "gpu_metrics": f"gpu_metrics_{result_filename}",
    }
    assert output_data["dispatch_ref"] == dispatch_ref


def test_isb1_replay_processing(tmp_path, sample_replay_result, base_env):
    export_file = write_export_fixture(
        tmp_path,
        "datasets/isb1/exports/core/chat_8k1k.json",
        {
            "adapter_id": "inferencex_multiturn",
            "bundle_id": "bundle-core-chat",
            "surface": "chat",
            "exports": [
                {
                    "trace_id": "trace-1",
                    "runtime_stack_id": "vllm-0.8.5-h200",
                    "hardware_profile_id": "h200-8gpu",
                    "canonical_model_id": "deepseek-r1-0528",
                    "support_status": "supported",
                }
            ],
        },
    )
    env = base_env.copy()
    env["EXPORT_FILE"] = export_file

    result = run_script(tmp_path, env, sample_replay_result)
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    output_data = json.loads(result.stdout)

    assert output_data["benchmark_type"] == "isb1_replay"
    assert output_data["request_mode"] == "multi-turn"
    assert output_data["harness_request_mode"] == "auto"
    assert output_data["isl"] == 8192
    assert output_data["osl"] == 1024
    assert output_data["export_lane"] == "core"
    assert output_data["benchmark_surface"] == "chat"
    assert output_data["support_status"] == "supported"
    assert output_data["benchmark_certification_status"] == "dataset_replay_verified"
    assert output_data["effective_max_context_depth"] == 8192 + 1024 + 200
    assert output_data["context_pressure_class"] == "standard"
    assert output_data["context_pressure_signal"]["status"] == "not_applicable"
    assert output_data["context_pressure_suspicious"] is False
    assert output_data["completed_sessions"] == 2
    assert output_data["session_throughput_sps"] == pytest.approx(1.0)
    assert output_data["tput_per_gpu"] == pytest.approx(650.0 / 8)
    assert output_data["output_tput_per_gpu"] == pytest.approx(150.0 / 8)
    assert output_data["input_tput_per_gpu"] == pytest.approx((650.0 - 150.0) / 8)
    assert output_data["median_ttft"] == pytest.approx(0.18)
    assert output_data["median_intvty"] == pytest.approx(40.0)
    assert output_data["median_e2el"] == pytest.approx(1.1)
    assert output_data["kv_offload_observed"] is True
    assert output_data["peak_gpu_cache_usage"] == pytest.approx(0.78)
    assert output_data["peak_cpu_cache_usage"] == pytest.approx(0.31)
    assert output_data["selection"]["request_mode_mix"] == {"chat": 2}
    assert output_data["selection"]["support_status_counts"] == {"supported": 2}
    assert output_data["per_turn_metrics"]["turn_1"]["completed"] == 2
    assert output_data["runtime_overrides"] == {
        "vllm_cpu_offload_gb": None,
        "vllm_swap_space_gb": None,
        "sglang_mem_fraction_override": None,
        "sglang_chunked_prefill_override": None,
    }
    assert_traceability_fields(output_data, "isb1_result")

    output_file = tmp_path / "agg_isb1_result.json"
    assert output_file.exists()
    persisted_output = json.loads(output_file.read_text())
    assert_traceability_fields(persisted_output, "isb1_result")


def test_offload_mode_env_propagation(tmp_path, sample_replay_result, base_env):
    export_file = write_export_fixture(
        tmp_path,
        "datasets/isb1/exports/core/chat_8k1k.json",
        {
            "adapter_id": "inferencex_multiturn",
            "surface": "chat",
            "exports": [
                {
                    "trace_id": "trace-1",
                    "runtime_stack_id": "vllm-0.8.5-h200",
                    "hardware_profile_id": "h200-8gpu",
                    "canonical_model_id": "deepseek-r1-0528",
                    "support_status": "supported",
                }
            ],
        },
    )
    env = base_env.copy()
    env["EXPORT_FILE"] = export_file
    env["OFFLOAD_MODE"] = "noprefix"
    env["KV_CACHE_DTYPE"] = "fp8"
    env["DISABLE_PREFIX_CACHING"] = "true"

    result = run_script(tmp_path, env, sample_replay_result, result_filename="isb1_offload_env")
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    output_data = json.loads(result.stdout)
    assert output_data["offload_mode"] == "noprefix"
    assert output_data["kv_cache_dtype"] == "fp8"
    assert output_data["disable_prefix_caching"] is True


def test_support_status_mismatch_fails(tmp_path, sample_replay_result, base_env):
    export_file = write_export_fixture(
        tmp_path,
        "datasets/isb1/exports/core/chat_8k1k.json",
        {
            "adapter_id": "inferencex_multiturn",
            "surface": "chat",
            "exports": [
                {
                    "trace_id": "trace-1",
                    "runtime_stack_id": "vllm-0.8.5-h200",
                    "hardware_profile_id": "h200-8gpu",
                    "canonical_model_id": "deepseek-r1-0528",
                    "support_status": "supported",
                }
            ],
        },
    )
    replay_result = {
        **sample_replay_result,
        "selection": {
            **sample_replay_result["selection"],
            "support_statuses": ["supported"],
            "support_status_counts": {"supported": 2},
        },
    }
    env = base_env.copy()
    env["EXPORT_FILE"] = export_file
    env["SUPPORT_STATUS"] = "reviewed_preview"

    result = run_script(tmp_path, env, replay_result, result_filename="isb1_mismatch")
    assert result.returncode != 0
    assert "support-status mismatch" in result.stderr


def test_certification_status_mismatch_fails(tmp_path, sample_replay_result, base_env):
    export_file = write_export_fixture(
        tmp_path,
        "datasets/isb1/exports/core/chat_8k1k.json",
        {
            "adapter_id": "inferencex_multiturn",
            "surface": "chat",
            "exports": [
                {
                    "trace_id": "trace-1",
                    "runtime_stack_id": "vllm-0.8.5-h200",
                    "hardware_profile_id": "h200-8gpu",
                    "canonical_model_id": "deepseek-r1-0528",
                    "support_status": "supported",
                    "benchmark_certification_status": "dataset_replay_verified",
                }
            ],
        },
    )
    replay_result = {
        **sample_replay_result,
        "selection": {
            **sample_replay_result["selection"],
            "benchmark_certification_statuses": ["pending_review"],
            "benchmark_certification_status_counts": {"pending_review": 2},
        },
    }
    env = base_env.copy()
    env["EXPORT_FILE"] = export_file

    result = run_script(tmp_path, env, replay_result, result_filename="isb1_cert_mismatch")
    assert result.returncode != 0
    assert "benchmark-certification mismatch" in result.stderr


def test_missing_required_env_vars_fails(tmp_path, sample_replay_result):
    result_file = tmp_path / "isb1_result.json"
    result_file.write_text(json.dumps(sample_replay_result))

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=tmp_path,
        env={"PATH": "/usr/bin", "RESULT_FILENAME": "isb1_result"},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Missing required environment variables" in result.stderr


def test_dispatch_ref_prefers_explicit_override(tmp_path, sample_replay_result, base_env):
    export_file = write_export_fixture(
        tmp_path,
        "datasets/isb1/exports/core/chat_8k1k.json",
        {
            "adapter_id": "inferencex_multiturn",
            "bundle_id": "bundle-core-chat",
            "surface": "chat",
            "exports": [
                {
                    "trace_id": "trace-1",
                    "runtime_stack_id": "vllm-0.8.5-h200",
                    "hardware_profile_id": "h200-8gpu",
                    "canonical_model_id": "deepseek-r1-0528",
                    "support_status": "supported",
                }
            ],
        },
    )
    env = base_env.copy()
    env["EXPORT_FILE"] = export_file
    env["DISPATCH_REF"] = "refs/tags/isb1-dispatch-override"

    result = run_script(tmp_path, env, sample_replay_result, result_filename="isb1_dispatch_override")
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    output_data = json.loads(result.stdout)
    assert_traceability_fields(
        output_data,
        "isb1_dispatch_override",
        dispatch_ref="refs/tags/isb1-dispatch-override",
    )


def test_preview_offload_core_processing(tmp_path, sample_replay_result, base_env):
    preview_export = (
        write_export_fixture(
            tmp_path,
            "datasets/isb1/exports/preview/offload_core/"
            "inferencex_multiturn__chat_hopper_blackwell_offload_core_v1__smoke.json",
            {
                "adapter_id": "inferencex_multiturn",
                "profile_id": "chat_hopper_blackwell_offload_core_v1",
                "duration_tier": "smoke",
                "adapter_surface": "chat",
                "tier": "reviewed_preview",
                "adapter_support_status": "reviewed_preview",
                "exports": [
                    {
                        "context_band": "lc1_8k_16k",
                    },
                    {
                        "context_band": "lc3_96k_128k",
                    },
                ],
                "producer_handoff_metadata": {
                    "class": "phase_2_offload_core_preview",
                    "claim_boundary": "Not blanket certification.",
                },
            },
        )
    )

    env = base_env.copy()
    env["EXPORT_FILE"] = preview_export
    env["SUPPORT_STATUS"] = "reviewed_preview"
    env["MAX_MODEL_LEN"] = "131272"
    replay_result = {
        **sample_replay_result,
        "selection": {
            **sample_replay_result["selection"],
            "support_statuses": ["reviewed_preview"],
            "support_status_counts": {"reviewed_preview": 2},
        },
    }

    result = run_script(tmp_path, env, replay_result, result_filename="isb1_preview")
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    output_data = json.loads(result.stdout)
    assert output_data["export_lane"] == "preview/offload_core"
    assert output_data["benchmark_surface"] == "chat"
    assert output_data["profile_id"] == "chat_hopper_blackwell_offload_core_v1"
    assert output_data["duration_tier"] == "smoke"
    assert output_data["context_bands"] == ["lc1_8k_16k", "lc3_96k_128k"]
    assert output_data["producer_handoff_class"] == "phase_2_offload_core_preview"
    assert output_data["support_status"] == "reviewed_preview"
    assert output_data["isl"] == 0
    assert output_data["osl"] == 0
    assert_traceability_fields(output_data, "isb1_preview")


def test_qwen_500k_preview_processing_preserves_served_shape_and_context_band(
    tmp_path, sample_replay_result, base_env
):
    preview_export = write_export_fixture(
        tmp_path,
        "datasets/isb1/exports/preview/long_context_500k/"
        "inferencex_trace_replay__coding_qwen3.5_xlc2_500k_preview_v1__vllm.json",
        {
            "adapter_id": "inferencex_trace_replay",
            "bundle_id": "isb1_preview_long_context_500k_vllm_code_xlc2_qwen3_5",
            "profile_id": "coding_qwen3.5_xlc2_500k_preview_v1",
            "duration_tier": "standard",
            "surface": "code",
            "served_shape": {"shape_family": "131k1k", "isl": 131072, "osl": 1024},
            "tier": "reviewed_preview",
            "adapter_support_status": "reviewed_preview",
            "producer_handoff_metadata": {
                "class": "bounded_500k_class",
                "claim_boundary": "Replay-derived 500k preview only.",
            },
            "exports": [
                {
                    "context_band": "xlc2_384k_512k",
                    "support_status": "reviewed_preview",
                    "benchmark_certification_status": "dataset_replay_verified",
                    "runtime_stack_id": "standalone:vllm",
                    "hardware_profile_id": "nvidia:b200_sxm_180gb",
                    "canonical_model_id": "qwen3_5_397b_a17b",
                    "kv_mode": "offload_cliff",
                },
                {
                    "context_band": "xlc2_384k_512k",
                    "support_status": "reviewed_preview",
                    "benchmark_certification_status": "dataset_replay_verified",
                    "runtime_stack_id": "standalone:vllm",
                    "hardware_profile_id": "nvidia:h100_sxm_80gb",
                    "canonical_model_id": "qwen3_5_397b_a17b",
                    "kv_mode": "offload_cliff",
                },
                {
                    "context_band": "xlc2_384k_512k",
                    "support_status": "reviewed_preview",
                    "benchmark_certification_status": "dataset_replay_verified",
                    "runtime_stack_id": "standalone:vllm",
                    "hardware_profile_id": "nvidia:h200_sxm_141gb",
                    "canonical_model_id": "qwen3_5_397b_a17b",
                    "kv_mode": "offload_cliff",
                },
            ],
        },
    )

    env = base_env.copy()
    env.update(
        {
            "RUNNER_TYPE": "b200-cw-1",
            "FRAMEWORK": "vllm",
            "MODEL_PREFIX": "qwen3.5",
            "IMAGE": "vllm/vllm-openai:v0.8.5",
            "EXPORT_FILE": preview_export,
            "RUNTIME_STACK_ID": "standalone:vllm",
            "HARDWARE_PROFILE_ID": "nvidia:b200_sxm_180gb",
            "CANONICAL_MODEL_ID": "qwen3_5_397b_a17b",
            "SUPPORT_STATUS": "reviewed_preview",
            "MAX_MODEL_LEN": "524288",
            "VLLM_CPU_OFFLOAD_GB": "120",
            "VLLM_SWAP_SPACE_GB": "24",
        }
    )
    replay_result = {
        **sample_replay_result,
        "model_id": "Qwen/Qwen3.5-397B-A17B-FP8",
        "vllm_cpu_offload_gb": "128",
        "vllm_swap_space_gb": "32",
        "selection": {
            **sample_replay_result["selection"],
            "runtime_stack_ids": ["standalone:vllm"],
            "hardware_profile_ids": ["nvidia:b200_sxm_180gb"],
            "canonical_model_ids": ["qwen3_5_397b_a17b"],
            "support_statuses": ["reviewed_preview"],
            "support_status_counts": {"reviewed_preview": 3},
            "request_mode_mix": {"code": 3},
        },
    }

    result = run_script(tmp_path, env, replay_result, result_filename="isb1_qwen_500k")
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    output_data = json.loads(result.stdout)
    assert output_data["export_lane"] == "preview/long_context_500k"
    assert output_data["benchmark_surface"] == "code"
    assert output_data["profile_id"] == "coding_qwen3.5_xlc2_500k_preview_v1"
    assert output_data["context_bands"] == ["xlc2_384k_512k"]
    assert output_data["producer_handoff_class"] == "bounded_500k_class"
    assert output_data["support_status"] == "reviewed_preview"
    assert output_data["benchmark_certification_status"] == "dataset_replay_verified"
    assert output_data["isl"] == 131072
    assert output_data["osl"] == 1024
    assert output_data["max_model_len"] == 524288
    assert output_data["effective_max_context_depth"] == 524288
    assert output_data["context_pressure_class"] == "extended_500k"
    assert output_data["context_pressure_signal"]["status"] == "ok"
    assert output_data["context_pressure_suspicious"] is False
    assert output_data["kv_offload_observed"] is True
    assert output_data["runtime_overrides"] == {
        "vllm_cpu_offload_gb": "128",
        "vllm_swap_space_gb": "32",
        "sglang_mem_fraction_override": None,
        "sglang_chunked_prefill_override": None,
    }
    assert_traceability_fields(output_data, "isb1_qwen_500k")


def test_qwen_1m_preview_processing_preserves_8k_served_shape_and_offload_metadata(
    tmp_path, sample_replay_result, base_env
):
    preview_export = write_export_fixture(
        tmp_path,
        "datasets/isb1/exports/preview/long_context_1m/"
        "inferencex_trace_replay__coding_qwen3.5_ulc2_1m_preview_v1__vllm.json",
        {
            "adapter_id": "inferencex_trace_replay",
            "bundle_id": "isb1_preview_long_context_1m_vllm_code_ulc2_qwen3_5",
            "profile_id": "coding_qwen3.5_ulc2_1m_preview_v1",
            "duration_tier": "standard",
            "surface": "code",
            "served_shape": {"shape_family": "8k1k", "isl": 8192, "osl": 1024},
            "tier": "reviewed_preview",
            "adapter_support_status": "reviewed_preview",
            "producer_handoff_metadata": {
                "class": "bounded_1m_class",
                "claim_boundary": "Manual 1M preview only.",
            },
            "exports": [
                {
                    "context_band": "ulc2_1m_plus",
                    "support_status": "reviewed_preview",
                    "benchmark_certification_status": "dataset_replay_verified",
                    "runtime_stack_id": "standalone:vllm",
                    "hardware_profile_id": "nvidia:b200_sxm_180gb",
                    "canonical_model_id": "qwen3_5_397b_a17b",
                    "kv_mode": "offload_cliff",
                }
            ],
        },
    )

    env = base_env.copy()
    env.update(
        {
            "RUNNER_TYPE": "b200-cw-1",
            "FRAMEWORK": "vllm",
            "MODEL_PREFIX": "qwen3.5",
            "IMAGE": "vllm/vllm-openai:v0.8.5",
            "EXPORT_FILE": preview_export,
            "RUNTIME_STACK_ID": "standalone:vllm",
            "HARDWARE_PROFILE_ID": "nvidia:b200_sxm_180gb",
            "CANONICAL_MODEL_ID": "qwen3_5_397b_a17b",
            "SUPPORT_STATUS": "reviewed_preview",
            "MAX_MODEL_LEN": "1048576",
            "MAX_SESSIONS": "1",
            "MAX_TURNS_PER_SESSION": "3",
        }
    )
    replay_result = {
        **sample_replay_result,
        "model_id": "Qwen/Qwen3.5-397B-A17B-FP8",
        "selection": {
            **sample_replay_result["selection"],
            "runtime_stack_ids": ["standalone:vllm"],
            "hardware_profile_ids": ["nvidia:b200_sxm_180gb"],
            "canonical_model_ids": ["qwen3_5_397b_a17b"],
            "support_statuses": ["reviewed_preview"],
            "support_status_counts": {"reviewed_preview": 1},
            "request_mode_mix": {"code": 1},
        },
    }

    result = run_script(tmp_path, env, replay_result, result_filename="isb1_qwen_1m")
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    output_data = json.loads(result.stdout)
    assert output_data["export_lane"] == "preview/long_context_1m"
    assert output_data["benchmark_surface"] == "code"
    assert output_data["profile_id"] == "coding_qwen3.5_ulc2_1m_preview_v1"
    assert output_data["context_bands"] == ["ulc2_1m_plus"]
    assert output_data["producer_handoff_class"] == "bounded_1m_class"
    assert output_data["support_status"] == "reviewed_preview"
    assert output_data["benchmark_certification_status"] == "dataset_replay_verified"
    assert output_data["isl"] == 8192
    assert output_data["osl"] == 1024
    assert output_data["max_model_len"] == 1048576
    assert output_data["effective_max_context_depth"] == 1048576
    assert output_data["context_pressure_class"] == "extended_1m"
    assert output_data["context_pressure_signal"]["status"] == "ok"
    assert output_data["context_pressure_suspicious"] is False
    assert output_data["max_sessions"] == 1
    assert output_data["max_turns_per_session"] == 3
    assert output_data["kv_offload_observed"] is True
    assert_traceability_fields(output_data, "isb1_qwen_1m")


def test_context_pressure_warning_on_high_context_without_cpu_cache(
    tmp_path, sample_replay_result, base_env
):
    preview_export = write_export_fixture(
        tmp_path,
        "datasets/isb1/exports/preview/long_context_500k/"
        "inferencex_trace_replay__coding_qwen3.5_xlc2_500k_preview_v1__vllm.json",
        {
            "adapter_id": "inferencex_trace_replay",
            "bundle_id": "isb1_preview_long_context_500k_vllm_code_xlc2_qwen3_5",
            "profile_id": "coding_qwen3.5_xlc2_500k_preview_v1",
            "duration_tier": "standard",
            "surface": "code",
            "served_shape": {"shape_family": "131k1k", "isl": 131072, "osl": 1024},
            "tier": "reviewed_preview",
            "adapter_support_status": "reviewed_preview",
            "exports": [
                {
                    "context_band": "xlc2_384k_512k",
                    "support_status": "reviewed_preview",
                    "benchmark_certification_status": "dataset_replay_verified",
                    "runtime_stack_id": "standalone:vllm",
                    "hardware_profile_id": "nvidia:b200_sxm_180gb",
                    "canonical_model_id": "qwen3_5_397b_a17b",
                    "kv_mode": "offload_cliff",
                }
            ],
        },
    )

    env = base_env.copy()
    env.update(
        {
            "RUNNER_TYPE": "b200-cw-1",
            "FRAMEWORK": "vllm",
            "MODEL_PREFIX": "qwen3.5",
            "IMAGE": "vllm/vllm-openai:v0.8.5",
            "EXPORT_FILE": preview_export,
            "RUNTIME_STACK_ID": "standalone:vllm",
            "HARDWARE_PROFILE_ID": "nvidia:b200_sxm_180gb",
            "CANONICAL_MODEL_ID": "qwen3_5_397b_a17b",
            "SUPPORT_STATUS": "reviewed_preview",
            "MAX_MODEL_LEN": "524288",
        }
    )
    replay_result = {
        **sample_replay_result,
        "model_id": "Qwen/Qwen3.5-397B-A17B-FP8",
        "selection": {
            **sample_replay_result["selection"],
            "runtime_stack_ids": ["standalone:vllm"],
            "hardware_profile_ids": ["nvidia:b200_sxm_180gb"],
            "canonical_model_ids": ["qwen3_5_397b_a17b"],
            "support_statuses": ["reviewed_preview"],
            "support_status_counts": {"reviewed_preview": 1},
            "request_mode_mix": {"code": 1},
        },
        "server_metrics_summary": {
            "cache_usage_avg": 0.45,
            "cache_hit_rate_avg": 0.15,
            "gpu_cache_usage_avg": 0.45,
            "gpu_cache_usage_peak": 0.91,
            "gpu_cache_metric_name": "vllm:gpu_cache_usage_perc",
            "cpu_cache_usage_avg": 0.0,
            "cpu_cache_usage_peak": 0.0,
            "cpu_cache_metric_name": "vllm:cpu_cache_usage_perc",
            "cpu_cache_metric_available": True,
            "observability_status": "direct_cpu_cache_metric",
            "kv_offload_observed": False,
            "samples": 5,
        },
    }

    result = run_script(tmp_path, env, replay_result, result_filename="isb1_qwen_500k_warn")
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "saw no CPU cache usage" in result.stderr

    output_data = json.loads(result.stdout)
    assert output_data["context_pressure_signal"]["status"] == "suspicious"
    assert output_data["context_pressure_suspicious"] is True
    assert_traceability_fields(output_data, "isb1_qwen_500k_warn")


def test_context_pressure_signal_marks_sglang_observability_gap(
    tmp_path, sample_replay_result, base_env
):
    preview_export = write_export_fixture(
        tmp_path,
        "datasets/isb1/exports/preview/long_context_500k/"
        "inferencex_trace_replay__coding_qwen3.5_xlc2_500k_preview_v1__sglang.json",
        {
            "adapter_id": "inferencex_trace_replay",
            "bundle_id": "isb1_preview_long_context_500k_sglang_code_xlc2_qwen3_5",
            "profile_id": "coding_qwen3.5_xlc2_500k_preview_v1",
            "duration_tier": "standard",
            "surface": "code",
            "served_shape": {"shape_family": "131k1k", "isl": 131072, "osl": 1024},
            "tier": "reviewed_preview",
            "adapter_support_status": "reviewed_preview",
            "exports": [
                {
                    "context_band": "xlc2_384k_512k",
                    "support_status": "reviewed_preview",
                    "benchmark_certification_status": "dataset_replay_verified",
                    "runtime_stack_id": "standalone:sglang",
                    "hardware_profile_id": "nvidia:b200_sxm_180gb",
                    "canonical_model_id": "qwen3_5_397b_a17b",
                    "kv_mode": "offload_cliff",
                }
            ],
        },
    )

    env = base_env.copy()
    env.update(
        {
            "RUNNER_TYPE": "b200-cw-1",
            "FRAMEWORK": "sglang",
            "MODEL_PREFIX": "qwen3.5",
            "IMAGE": "lmsysorg/sglang:v0.5.9-cu130",
            "EXPORT_FILE": preview_export,
            "RUNTIME_STACK_ID": "standalone:sglang",
            "HARDWARE_PROFILE_ID": "nvidia:b200_sxm_180gb",
            "CANONICAL_MODEL_ID": "qwen3_5_397b_a17b",
            "SUPPORT_STATUS": "reviewed_preview",
            "MAX_MODEL_LEN": "524288",
            "SGLANG_MEM_FRACTION_OVERRIDE": "0.77",
            "SGLANG_CHUNKED_PREFILL_OVERRIDE": "65536",
        }
    )
    replay_result = {
        **sample_replay_result,
        "model_id": "Qwen/Qwen3.5-397B-A17B-FP8",
        "selection": {
            **sample_replay_result["selection"],
            "runtime_stack_ids": ["standalone:sglang"],
            "hardware_profile_ids": ["nvidia:b200_sxm_180gb"],
            "canonical_model_ids": ["qwen3_5_397b_a17b"],
            "support_statuses": ["reviewed_preview"],
            "support_status_counts": {"reviewed_preview": 1},
            "request_mode_mix": {"code": 1},
        },
        "server_metrics_summary": {
            "cache_usage_avg": 0.52,
            "cache_hit_rate_avg": 0.23,
            "gpu_cache_usage_avg": 0.52,
            "gpu_cache_usage_peak": 0.88,
            "gpu_cache_metric_name": "sglang:token_usage",
            "cpu_cache_usage_avg": 0.0,
            "cpu_cache_usage_peak": 0.0,
            "cpu_cache_metric_name": None,
            "cpu_cache_metric_available": False,
            "observability_status": "indirect_without_cpu_cache_metric",
            "kv_offload_observed": False,
            "samples": 5,
        },
    }

    result = run_script(tmp_path, env, replay_result, result_filename="isb1_qwen_500k_sglang")
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "lacks a direct CPU cache metric" in result.stderr

    output_data = json.loads(result.stdout)
    assert output_data["context_pressure_signal"]["status"] == "observability_gap"
    assert output_data["context_pressure_signal"]["requires_log_review"] is True
    assert output_data["context_pressure_suspicious"] is False
    assert output_data["runtime_overrides"] == {
        "vllm_cpu_offload_gb": None,
        "vllm_swap_space_gb": None,
        "sglang_mem_fraction_override": "0.77",
        "sglang_chunked_prefill_override": "65536",
    }
    assert_traceability_fields(output_data, "isb1_qwen_500k_sglang")


def test_depth_coverage_ratio_for_500k_preview(tmp_path, base_env, sample_replay_result):
    """Verify depth coverage ratio and class for a 500k preview with 131k actual tokens."""
    export_payload = {
        "served_shape": {"shape_family": "131k1k", "isl": 131072, "osl": 1024},
        "surface": "code",
        "exports": [
            {
                "runtime_stack_id": "standalone:vllm",
                "hardware_profile_id": "h200-8gpu",
                "canonical_model_id": "qwen3_5_397b_a17b",
                "support_status": "reviewed_preview",
                "benchmark_certification_status": "dataset_replay_verified",
                "context_band": "xlc2_384k_512k",
                "trace_metadata": {
                    "estimated_kv_bytes_peak": 27294647296,
                    "context_pressure_profile": {
                        "expected_offload_mode": "soft_offload",
                    },
                    "expected_offload_mode": "soft_offload",
                },
            }
        ],
    }
    export_file = write_export_fixture(
        tmp_path, "datasets/isb1/exports/preview/long_context_500k/test_500k.json", export_payload
    )

    env = base_env.copy()
    env["EXPORT_FILE"] = export_file
    env["MODEL_PREFIX"] = "qwen3.5"
    env["CANONICAL_MODEL_ID"] = "qwen3_5_397b_a17b"
    env["SUPPORT_STATUS"] = "reviewed_preview"
    env["MAX_MODEL_LEN"] = "524288"
    env["FRAMEWORK"] = "vllm"

    replay_result = sample_replay_result.copy()
    replay_result["selection"] = {
        **replay_result["selection"],
        "support_statuses": ["reviewed_preview"],
    }
    replay_result["server_metrics_summary"] = {
        "gpu_cache_usage_avg": 0.35,
        "gpu_cache_usage_peak": 0.42,
        "cpu_cache_usage_avg": 0.15,
        "cpu_cache_usage_peak": 0.25,
        "cpu_cache_metric_available": True,
        "observability_status": "direct_cpu_cache_metric",
        "kv_offload_observed": True,
        "samples": 10,
    }
    replay_result["depth_telemetry"] = {
        "total_estimated_input_tokens": 500000,
        "total_actual_input_tokens": 131072,
        "max_actual_context_len_per_turn": 131072,
    }

    result = run_script(tmp_path, env, replay_result, result_filename="isb1_qwen_500k_depth")
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    output_data = json.loads(result.stdout)

    # Depth coverage ratio: 131072 / 524288 ≈ 0.25
    assert output_data["depth_coverage_ratio"] is not None
    assert 0.24 < output_data["depth_coverage_ratio"] < 0.26
    assert output_data["depth_coverage_class"] == "bounded_preview"
    assert output_data["max_actual_context_len_per_turn"] == 131072
    assert output_data["depth_gap_tokens"] == 524288 - 131072

    # Producer expectation validation
    assert output_data["producer_estimated_kv_bytes_peak"] == 27294647296
    assert output_data["producer_expected_offload_mode"] == "soft_offload"
    assert output_data["producer_expectation_validation"]["offload_mode_match"] is True
    assert output_data["producer_expectation_validation"]["depth_exercised"] is False

    # Preemption count
    assert output_data["preemption_count"] == 0


def test_depth_mismatch_warning_for_configuration_only(tmp_path, base_env, sample_replay_result):
    """Verify depth_mismatch status when actual context is <10% of configured."""
    export_payload = {
        "served_shape": {"shape_family": "8k1k", "isl": 8192, "osl": 1024},
        "surface": "code",
        "exports": [
            {
                "runtime_stack_id": "standalone:vllm",
                "hardware_profile_id": "h200-8gpu",
                "canonical_model_id": "qwen3_5_397b_a17b",
                "support_status": "reviewed_preview",
                "benchmark_certification_status": "dataset_replay_verified",
                "context_band": "ulc2_1m_plus",
                "trace_metadata": {
                    "estimated_kv_bytes_peak": 39500000000,
                    "expected_offload_mode": "hard_offload",
                },
            }
        ],
    }
    export_file = write_export_fixture(
        tmp_path, "datasets/isb1/exports/preview/long_context_1m/test_1m.json", export_payload
    )

    env = base_env.copy()
    env["EXPORT_FILE"] = export_file
    env["MODEL_PREFIX"] = "qwen3.5"
    env["CANONICAL_MODEL_ID"] = "qwen3_5_397b_a17b"
    env["SUPPORT_STATUS"] = "reviewed_preview"
    env["MAX_MODEL_LEN"] = "1048576"
    env["FRAMEWORK"] = "vllm"

    replay_result = sample_replay_result.copy()
    replay_result["selection"] = {
        **replay_result["selection"],
        "support_statuses": ["reviewed_preview"],
    }
    replay_result["server_metrics_summary"] = {
        "gpu_cache_usage_avg": 0.10,
        "gpu_cache_usage_peak": 0.15,
        "cpu_cache_usage_avg": 0.05,
        "cpu_cache_usage_peak": 0.10,
        "cpu_cache_metric_available": True,
        "observability_status": "direct_cpu_cache_metric",
        "kv_offload_observed": True,
        "samples": 5,
    }
    # 1M preview sends only 8k actual tokens
    replay_result["depth_telemetry"] = {
        "total_estimated_input_tokens": 1600000,
        "total_actual_input_tokens": 8192,
        "max_actual_context_len_per_turn": 8192,
    }

    result = run_script(tmp_path, env, replay_result, result_filename="isb1_qwen_1m_depth")
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    output_data = json.loads(result.stdout)

    # 8192 / 1048576 ≈ 0.0078 — less than 0.1 threshold
    assert output_data["depth_coverage_ratio"] < 0.01
    assert output_data["depth_coverage_class"] == "configuration_only"
    assert output_data["context_pressure_signal"]["status"] == "depth_mismatch"
    assert output_data["context_pressure_signal"]["reason"] == "configured_depth_not_exercised"
    assert "depth_coverage_ratio" in output_data["context_pressure_signal"]
    assert "configured for" in result.stderr


def test_producer_expectation_offload_mismatch(tmp_path, base_env, sample_replay_result):
    """Verify producer expectation validation when offload is expected but not observed."""
    export_payload = {
        "served_shape": {"shape_family": "131k1k", "isl": 131072, "osl": 1024},
        "surface": "code",
        "exports": [
            {
                "runtime_stack_id": "standalone:vllm",
                "hardware_profile_id": "h200-8gpu",
                "canonical_model_id": "gpt_oss_120b",
                "support_status": "reviewed_preview",
                "benchmark_certification_status": "dataset_replay_verified",
                "context_band": "xlc2_384k_512k",
                "trace_metadata": {
                    "estimated_kv_bytes_peak": 27000000000,
                    "context_pressure_profile": {
                        "expected_offload_mode": "hard_offload",
                    },
                },
            }
        ],
    }
    export_file = write_export_fixture(
        tmp_path, "datasets/isb1/exports/preview/long_context_500k/test_mismatch.json", export_payload
    )

    env = base_env.copy()
    env["EXPORT_FILE"] = export_file
    env["MODEL_PREFIX"] = "gptoss"
    env["CANONICAL_MODEL_ID"] = "gpt_oss_120b"
    env["SUPPORT_STATUS"] = "reviewed_preview"
    env["MAX_MODEL_LEN"] = "524288"

    replay_result = sample_replay_result.copy()
    replay_result["selection"] = {
        **replay_result["selection"],
        "support_statuses": ["reviewed_preview"],
    }
    replay_result["server_metrics_summary"] = {
        "gpu_cache_usage_avg": 0.50,
        "gpu_cache_usage_peak": 0.60,
        "cpu_cache_usage_avg": 0.0,
        "cpu_cache_usage_peak": 0.0,
        "cpu_cache_metric_available": True,
        "observability_status": "direct_cpu_cache_metric",
        "kv_offload_observed": False,
        "samples": 10,
    }
    replay_result["depth_telemetry"] = {
        "total_estimated_input_tokens": 400000,
        "total_actual_input_tokens": 131072,
        "max_actual_context_len_per_turn": 131072,
    }

    result = run_script(tmp_path, env, replay_result, result_filename="isb1_mismatch")
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    output_data = json.loads(result.stdout)

    # Producer expected hard_offload, but kv_offload_observed is False
    assert output_data["producer_expectation_validation"]["offload_mode_match"] is False
    assert output_data["producer_expected_offload_mode"] == "hard_offload"
    assert output_data["kv_offload_observed"] is False
