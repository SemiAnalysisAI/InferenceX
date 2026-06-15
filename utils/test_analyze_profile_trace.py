import gzip
import json

import pytest

from utils.analyze_profile_trace import (
    analyze_trace,
    validate_categories,
    validate_kernel_count,
    validate_kernel_stream_count,
)


def _write_trace(tmp_path, events):
    path = tmp_path / "trace.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as trace_file:
        json.dump({"traceEvents": events}, trace_file)
    return path


def test_analyze_trace_selects_steady_state_decode(tmp_path):
    trace = _write_trace(
        tmp_path,
        [
            {
                "name": "execute_context_0(0)_generation_16(16)",
                "ph": "X",
                "ts": 100,
                "dur": 100,
            },
            {
                "name": "_gemma_fused_add_rms_norm_kernel",
                "cat": "kernel",
                "ph": "X",
                "ts": 110,
                "dur": 20,
            },
            {
                "name": "ncclDevKernel_Generic",
                "cat": "kernel",
                "ph": "X",
                "ts": 140,
                "dur": 30,
            },
        ],
    )

    summary = analyze_trace(trace, expected_concurrency=16)

    assert summary["generation_requests"] == 16
    assert summary["decode_kernel_span_us"] == 60
    assert summary["kernel_busy_time_us"] == 50
    assert summary["kernel_idle_time_us"] == 10
    assert summary["kernel_stream_count"] == 1
    assert summary["cross_stream_overlap_us"] == 0
    assert summary["max_concurrent_kernel_streams"] == 1
    assert summary["categories"]["gemma_rmsnorm"]["duration_us"] == 20
    assert summary["categories"]["allreduce"]["duration_us"] == 30


def test_analyze_trace_reports_cross_stream_overlap(tmp_path):
    trace = _write_trace(
        tmp_path,
        [
            {
                "name": "execute_context_0(0)_generation_16(16)",
                "cat": "gpu_user_annotation",
                "ph": "X",
                "ts": 100,
                "dur": 200,
            },
            {
                "name": "main_stream_kernel",
                "cat": "kernel",
                "ph": "X",
                "ts": 110,
                "dur": 40,
                "args": {"device": 0, "stream": 4},
            },
            {
                "name": "aux_stream_kernel",
                "cat": "kernel",
                "ph": "X",
                "ts": 130,
                "dur": 40,
                "args": {"device": 0, "stream": 7},
            },
            {
                "name": "main_stream_tail",
                "cat": "kernel",
                "ph": "X",
                "ts": 180,
                "dur": 10,
                "args": {"device": 0, "stream": 4},
            },
        ],
    )

    summary = analyze_trace(trace, expected_concurrency=16)

    assert summary["kernel_stream_count"] == 2
    assert summary["cross_stream_overlap_us"] == 20
    assert summary["max_concurrent_kernel_streams"] == 2
    assert [
        (item["stream"], item["count"]) for item in summary["kernel_streams"]
    ] == [("4", 2), ("7", 1)]


def test_analyze_trace_reports_kernel_launch_starvation(tmp_path):
    trace = _write_trace(
        tmp_path,
        [
            {
                "name": "execute_context_0(0)_generation_16(16)",
                "cat": "gpu_user_annotation",
                "ph": "X",
                "ts": 100,
                "dur": 200,
            },
            {
                "name": "hipLaunchKernel",
                "cat": "cuda_runtime",
                "ph": "X",
                "ts": 110,
                "dur": 5,
                "args": {"correlation": 1},
            },
            {
                "name": "first_kernel",
                "cat": "kernel",
                "ph": "X",
                "ts": 120,
                "dur": 20,
                "args": {"device": 0, "stream": 4, "correlation": 1},
            },
            {
                "name": "hipModuleLaunchKernel",
                "cat": "cuda_runtime",
                "ph": "X",
                "ts": 150,
                "dur": 5,
                "args": {"correlation": 2},
            },
            {
                "name": "second_kernel",
                "cat": "kernel",
                "ph": "X",
                "ts": 160,
                "dur": 10,
                "args": {"device": 0, "stream": 4, "correlation": 2},
            },
        ],
    )

    launch = analyze_trace(trace, expected_concurrency=16)["kernel_launch_analysis"]

    assert launch["correlated_kernel_count"] == 2
    assert launch["correlated_kernel_percent"] == 100
    assert launch["launch_submission_span_us"] == 45
    assert launch["launch_to_kernel_start_us"]["p50"] == 5
    assert launch["same_stream_gap_us"] == 20
    assert launch["host_late_launch_count"] == 1
    assert launch["definite_host_starvation_us"] == 10
    assert launch["launch_call_in_gap_us"] == 5
    assert launch["post_launch_device_gap_us"] == 5
    assert launch["global_gap_count"] == 1
    assert launch["global_gap_us"] == 20
    assert launch["global_host_late_launch_count"] == 1
    assert launch["global_definite_host_starvation_us"] == 10
    assert launch["global_launch_call_in_gap_us"] == 5
    assert launch["global_post_launch_device_gap_us"] == 5
    assert launch["global_unattributed_gap_count"] == 0


def test_analyze_trace_reports_global_multistream_idle_without_double_counting(
    tmp_path,
):
    events = [
        {
            "name": "execute_context_0(0)_generation_16(16)",
            "cat": "gpu_user_annotation",
            "ph": "X",
            "ts": 0,
            "dur": 300,
        }
    ]
    kernels = [
        ("stream4_first", 20, 20, 4, 1),
        ("stream7_first", 45, 20, 7, 2),
        ("stream7_second", 70, 5, 7, 3),
        ("stream4_second", 80, 10, 4, 4),
    ]
    for name, timestamp, duration, stream, correlation in kernels:
        events.extend(
            [
                {
                    "name": "hipLaunchKernel",
                    "cat": "cuda_runtime",
                    "ph": "X",
                    "ts": timestamp - 4,
                    "dur": 2,
                    "args": {"correlation": correlation},
                },
                {
                    "name": name,
                    "cat": "kernel",
                    "ph": "X",
                    "ts": timestamp,
                    "dur": duration,
                    "args": {
                        "device": 0,
                        "stream": stream,
                        "correlation": correlation,
                    },
                },
            ]
        )
    trace = _write_trace(tmp_path, events)

    summary = analyze_trace(trace, expected_concurrency=16)
    launch = summary["kernel_launch_analysis"]

    assert summary["kernel_idle_time_us"] == 15
    assert launch["same_stream_gap_us"] == 45
    assert launch["global_gap_count"] == 3
    assert launch["global_gap_us"] == 15
    assert launch["global_definite_host_starvation_us"] == 3
    assert launch["global_launch_call_in_gap_us"] == 6
    assert launch["global_post_launch_device_gap_us"] == 6


def test_analyze_trace_prefers_gpu_annotation_window(tmp_path):
    trace = _write_trace(
        tmp_path,
        [
            {
                "name": "execute_context_0(0)_generation_16(16)",
                "cat": "user_annotation",
                "ph": "X",
                "ts": 100,
                "dur": 100,
            },
            {
                "name": "execute_context_0(0)_generation_16(16)",
                "cat": "gpu_user_annotation",
                "ph": "X",
                "ts": 250,
                "dur": 100,
            },
            {
                "name": "cross_device_reduce_1stage",
                "cat": "kernel",
                "ph": "X",
                "ts": 275,
                "dur": 20,
            },
        ],
    )

    summary = analyze_trace(trace, expected_concurrency=16)

    assert summary["annotation_category"] == "gpu_user_annotation"
    assert summary["kernel_count"] == 1
    assert summary["categories"]["allreduce"]["count"] == 1


def test_analyze_trace_allows_bounded_underfilled_decode(tmp_path):
    trace = _write_trace(
        tmp_path,
        [
            {
                "name": "execute_context_0(0)_generation_217(217)",
                "cat": "gpu_user_annotation",
                "ph": "X",
                "ts": 100,
                "dur": 100,
            },
            {
                "name": "cross_device_reduce_1stage",
                "cat": "kernel",
                "ph": "X",
                "ts": 110,
                "dur": 20,
            },
        ],
    )

    summary = analyze_trace(
        trace,
        expected_concurrency=256,
        minimum_concurrency_fraction=0.8,
    )

    assert summary["generation_requests"] == 217
    assert summary["minimum_concurrency_fraction"] == 0.8

    with pytest.raises(ValueError, match="no steady-state decode annotation"):
        analyze_trace(trace, expected_concurrency=256)


def test_analyze_trace_rejects_invalid_concurrency_fraction(tmp_path):
    trace = _write_trace(tmp_path, [])

    with pytest.raises(ValueError, match=r"must be in \(0, 1\]"):
        analyze_trace(trace, minimum_concurrency_fraction=0)


def test_analyze_trace_classifies_one_stage_aiter_fusion(tmp_path):
    trace = _write_trace(
        tmp_path,
        [
            {
                "name": "execute_context_0(0)_generation_1(1)",
                "ph": "X",
                "ts": 100,
                "dur": 100,
            },
            {
                "name": "allreduce_fusion_kernel_1stage",
                "cat": "kernel",
                "ph": "X",
                "ts": 110,
                "dur": 20,
            },
        ],
    )

    summary = analyze_trace(trace, expected_concurrency=1)

    category = summary["categories"]["aiter_fused_allreduce_rmsnorm"]
    assert category["count"] == 1
    assert category["duration_us"] == 20


def test_analyze_trace_classifies_two_stage_aiter_fusion(tmp_path):
    trace = _write_trace(
        tmp_path,
        [
            {
                "name": "execute_context_0(0)_generation_256(256)",
                "ph": "X",
                "ts": 100,
                "dur": 100,
            },
            {
                "name": "reduce_scatter_cross_device_store",
                "cat": "kernel",
                "ph": "X",
                "ts": 110,
                "dur": 20,
            },
            {
                "name": "local_device_load_rmsnorm_512n",
                "cat": "kernel",
                "ph": "X",
                "ts": 140,
                "dur": 30,
            },
        ],
    )

    summary = analyze_trace(trace, expected_concurrency=256)

    category = summary["categories"]["aiter_fused_allreduce_rmsnorm"]
    assert category["count"] == 2
    assert category["duration_us"] == 50


def test_analyze_trace_classifies_custom_cross_device_allreduce(tmp_path):
    trace = _write_trace(
        tmp_path,
        [
            {
                "name": "execute_context_0(0)_generation_1(1)",
                "ph": "X",
                "ts": 100,
                "dur": 100,
            },
            {
                "name": "cross_device_reduce_1stage",
                "cat": "kernel",
                "ph": "X",
                "ts": 110,
                "dur": 20,
            },
        ],
    )

    summary = analyze_trace(trace, expected_concurrency=1)

    category = summary["categories"]["allreduce"]
    assert category["count"] == 1
    assert category["duration_us"] == 20


def test_analyze_trace_rejects_context_work(tmp_path):
    trace = _write_trace(
        tmp_path,
        [
            {
                "name": "execute_context_1(128)_generation_15(15)",
                "ph": "X",
                "ts": 100,
                "dur": 100,
            },
            {
                "name": "kernel",
                "cat": "kernel",
                "ph": "X",
                "ts": 110,
                "dur": 20,
            },
        ],
    )

    with pytest.raises(ValueError, match="no steady-state decode annotation"):
        analyze_trace(trace, expected_concurrency=16)


def test_category_validation_rejects_out_of_window_fallback():
    summary = {
        "annotation_window_used": False,
        "categories": {"aiter_fused_allreduce_rmsnorm": {"count": 1}},
    }

    with pytest.raises(ValueError, match="inside the decode annotation"):
        validate_categories(
            summary,
            required=["aiter_fused_allreduce_rmsnorm"],
            forbidden=[],
        )


def test_category_validation_checks_required_and_forbidden():
    summary = {
        "annotation_window_used": True,
        "categories": {"aiter_fused_allreduce_rmsnorm": {"count": 1}},
    }

    validate_categories(
        summary,
        required=["aiter_fused_allreduce_rmsnorm"],
        forbidden=[],
    )
    with pytest.raises(ValueError, match="forbidden kernel categories"):
        validate_categories(
            summary,
            required=[],
            forbidden=["aiter_fused_allreduce_rmsnorm"],
        )


def test_kernel_count_validation_rejects_partial_trace():
    with pytest.raises(ValueError, match="below required minimum"):
        validate_kernel_count({"kernel_count": 999}, minimum=1000)

    validate_kernel_count({"kernel_count": 1000}, minimum=1000)
    validate_kernel_count({"kernel_count": 0}, minimum=None)


def test_kernel_stream_count_validation_rejects_single_stream_trace():
    with pytest.raises(ValueError, match="kernel stream count 1"):
        validate_kernel_stream_count({"kernel_stream_count": 1}, minimum=2)

    validate_kernel_stream_count({"kernel_stream_count": 2}, minimum=2)
    validate_kernel_stream_count({"kernel_stream_count": 0}, minimum=None)


def test_analyze_trace_segments_minimax_m3_layers(tmp_path):
    kernel_names = [
        "decode_setup",
        "Cijk_dense_qkv",
        "fusedMiniMaxM3QNormRopeKVInsertKernel<c10::BFloat16, false, false>",
        "reshape_and_cache_kernel_flash",
        "kernel_unified_attention",
        "reduce_segments",
        "Cijk_dense_o",
        "cross_device_reduce_1stage",
        "_gemma_fused_add_rmsnorm_kernel",
        "Cijk_dense_gate_up",
        "_swiglu_oai_kernel",
        "Cijk_dense_down",
        "cross_device_reduce_1stage",
        "_gemma_fused_add_rmsnorm_kernel",
        "Cijk_sparse_qkv",
        "fusedMiniMaxM3QNormRopeKVInsertKernel<c10::BFloat16, true, true>",
        "_decode_index_score_kernel",
        "_topk_index_partial_kernel",
        "_topk_index_merge_kernel",
        "_gqa_sparse_decode_kernel",
        "_merge_topk_attn_out_kernel",
        "Cijk_sparse_o",
        "reduce_scatter_cross_device_store",
        "local_device_load_rmsnorm_512n",
        "bfloat16tofloat32_copy_kernel_cuda",
        "Cijk_router",
        "Cijk_shared_gate_up",
        "_swiglu_oai_kernel",
        "Cijk_shared_down",
        "topkGating",
        "moe_align_block_size_kernel",
        "count_and_sort_expert_tokens_kernel",
        "_kernel",
        "fused_moe_kernel",
        "_swiglu_oai_quant_kernel",
        "fused_moe_kernel",
        "moe_sum_kernel",
        "vectorized_elementwise_mul",
        "vectorized_elementwise_add",
        "reduce_scatter_cross_device_store",
        "local_device_load_rmsnorm_512n",
        "Cijk_logits",
        "ncclDevKernel_Generic",
    ]
    events = [
        {
            "name": "execute_context_0(0)_generation_16(16)",
            "cat": "gpu_user_annotation",
            "ph": "X",
            "ts": 0,
            "dur": 1000,
        }
    ]
    events.extend(
        {
            "name": name,
            "cat": "kernel",
            "ph": "X",
            "ts": 10 + index * 10,
            "dur": 5,
        }
        for index, name in enumerate(kernel_names)
    )
    trace = _write_trace(tmp_path, events)

    summary = analyze_trace(trace, expected_concurrency=16)

    m3 = summary["minimax_m3"]
    assert m3["recognized"]
    assert m3["layer_count"] == 2
    assert m3["dense_layer_count"] == 1
    assert m3["sparse_moe_layer_count"] == 1
    assert m3["phases"]["attention_qkv_projection"]["count"] == 2
    assert m3["phases"]["moe_expert_gemm_1"]["count"] == 1
    assert m3["phases"]["moe_expert_gemm_2"]["count"] == 1
    assert m3["phases"]["logits_projection"]["count"] == 1
    assert m3["phases"]["logits_all_gather"]["count"] == 1
    assert m3["largest_kernel_gaps"][0]["previous_phase"] == "decode_setup"
    assert m3["largest_kernel_gaps"][0]["next_phase"] == "attention_qkv_projection"
    assert m3["kernel_gap_transitions"][0]["duration_us"] > 0
    assert (
        m3["phases"]["sparse_index_score"]["top_kernels"][0]["name"]
        == "_decode_index_score_kernel"
    )


def test_analyze_trace_sorts_multistream_m3_kernels_by_timestamp(tmp_path):
    main_stream = {"device": 0, "stream": 4}
    aux_stream = {"device": 0, "stream": 12}
    main_kernel_names = [
        "Cijk_sparse_qkv",
        "fusedMiniMaxM3QNormRopeKVInsertKernel<c10::BFloat16, true, true>",
        "_decode_index_score_kernel",
        "_topk_index_partial_kernel",
        "_topk_index_merge_kernel",
        "_gqa_sparse_decode_kernel",
        "_merge_topk_attn_out_kernel",
        "Cijk_sparse_o",
        "cross_device_reduce_1stage",
        "_gemma_fused_add_rmsnorm_kernel",
        "bfloat16tofloat32_copy_kernel_cuda",
        "Cijk_router",
        "topkGating",
        "moe_align_block_size_kernel",
        "count_and_sort_expert_tokens_kernel",
        "fused_moe_kernel",
        "_swiglu_oai_quant_kernel",
        "fused_moe_kernel",
        "moe_sum_kernel",
        "vectorized_elementwise_add",
        "cross_device_reduce_1stage",
        "_gemma_fused_add_rmsnorm_kernel",
        "Cijk_logits",
    ]
    main_events = [
        {
            "name": name,
            "cat": "kernel",
            "ph": "X",
            "ts": 10 + index * 10,
            "dur": 5,
            "args": main_stream,
        }
        for index, name in enumerate(main_kernel_names)
    ]
    aux_events = [
        {
            "name": "Cijk_shared_gate_up",
            "cat": "kernel",
            "ph": "X",
            "ts": 185,
            "dur": 4,
            "args": aux_stream,
        },
        {
            "name": "_swiglu_oai_kernel",
            "cat": "kernel",
            "ph": "X",
            "ts": 191,
            "dur": 2,
            "args": aux_stream,
        },
        {
            "name": "Cijk_shared_down",
            "cat": "kernel",
            "ph": "X",
            "ts": 195,
            "dur": 4,
            "args": aux_stream,
        },
    ]
    events = [
        {
            "name": "execute_context_0(0)_generation_16(16)",
            "cat": "gpu_user_annotation",
            "ph": "X",
            "ts": 0,
            "dur": 1000,
        },
        *main_events,
        # Perfetto may append an auxiliary stream after later main-stream
        # events even when its timestamps belong in the middle.
        *aux_events,
    ]
    trace = _write_trace(tmp_path, events)

    summary = analyze_trace(trace, expected_concurrency=16)

    m3 = summary["minimax_m3"]
    assert m3["recognized"]
    assert summary["kernel_stream_count"] == 2
    assert m3["phases"]["moe_shared_gate_up_projection"]["count"] == 1
    assert m3["phases"]["moe_shared_activation"]["count"] == 1
    assert m3["phases"]["moe_shared_down_projection"]["count"] == 1
    assert m3["phases"]["logits_projection"]["count"] == 1
