import json
from pathlib import Path

from summarize_isb1 import generate_summary


def write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def make_row(**overrides):
    row = {
        "benchmark_type": "isb1_replay",
        "result_filename": "isb1_control_vllm_b200",
        "artifact_stems": {
            "processed": "isb1_isb1_control_vllm_b200",
            "raw_replay": "replay_isb1_control_vllm_b200",
            "server_logs": "server_logs_isb1_control_vllm_b200",
            "gpu_metrics": "gpu_metrics_isb1_control_vllm_b200",
        },
        "dispatch_ref": "refs/heads/test-summary",
        "infmax_model_prefix": "dsr1",
        "hw": "b200-cw-1",
        "framework": "vllm",
        "support_status": "supported",
        "benchmark_certification_status": "dataset_replay_verified",
        "effective_max_context_depth": 9416,
        "context_pressure_class": "standard",
        "context_pressure_signal": {
            "status": "not_applicable",
            "requires_log_review": False,
        },
        "context_pressure_suspicious": False,
        "completed_sessions": 2,
        "total_sessions": 2,
        "session_throughput_sps": 1.25,
        "median_ttft": 0.18,
        "kv_offload_observed": True,
        "peak_gpu_cache_usage": 0.78,
        "peak_cpu_cache_usage": 0.31,
        "runtime_overrides": {
            "vllm_cpu_offload_gb": None,
            "vllm_swap_space_gb": None,
            "sglang_mem_fraction_override": None,
            "sglang_chunked_prefill_override": None,
        },
    }
    row.update(overrides)
    return row


def test_generate_summary_surfaces_lane_override_and_action_sections(tmp_path):
    control_row = make_row()
    review_row = make_row(
        result_filename="isb1_qwen_500k_sglang",
        artifact_stems={
            "processed": "isb1_isb1_qwen_500k_sglang",
            "raw_replay": "replay_isb1_qwen_500k_sglang",
            "server_logs": "server_logs_isb1_qwen_500k_sglang",
            "gpu_metrics": "gpu_metrics_isb1_qwen_500k_sglang",
        },
        infmax_model_prefix="qwen3.5",
        hw="h200-cw-1",
        framework="sglang",
        support_status="reviewed_preview",
        effective_max_context_depth=524288,
        context_pressure_class="extended_500k",
        context_pressure_signal={
            "status": "observability_gap",
            "requires_log_review": True,
        },
        runtime_overrides={
            "vllm_cpu_offload_gb": None,
            "vllm_swap_space_gb": None,
            "sglang_mem_fraction_override": "0.77",
            "sglang_chunked_prefill_override": "65536",
        },
        kv_offload_observed=False,
        peak_gpu_cache_usage=0.88,
        peak_cpu_cache_usage=0.0,
    )
    non_isb1_row = {"benchmark_type": "throughput", "ignored": True}

    write_result(tmp_path / "results" / "control.json", control_row)
    write_result(tmp_path / "results" / "review.json", review_row)
    write_result(tmp_path / "results" / "non_isb1.json", non_isb1_row)

    summary = generate_summary(tmp_path / "results")

    assert "## ISB1 Operator Summary" in summary
    assert "### Lane Summary" in summary
    assert "### Runtime Overrides" in summary
    assert "### Action Items" in summary
    assert "isb1_qwen_500k_sglang" in summary
    assert "observability_gap" in summary
    assert "65536" in summary
    assert "server_logs_isb1_qwen_500k_sglang" in summary
    assert "non_isb1" not in summary


def test_generate_summary_handles_empty_results(tmp_path):
    summary = generate_summary(tmp_path / "results")
    assert "No ISB1 replay rows found." in summary
    assert "Lane Summary" not in summary
