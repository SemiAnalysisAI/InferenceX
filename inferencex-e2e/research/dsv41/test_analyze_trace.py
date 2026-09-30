import pytest

from research.dsv41.analyze_trace import family, interval_union, summarize


def test_union_does_not_add_overlapping_kernels():
    assert interval_union([(0, 4), (2, 7), (10, 12)]) == 9


@pytest.mark.parametrize(
    "scope, stage",
    [
        ("step[VERIFY bs=1]", "VERIFY"),
        ("execute_8_context_0_generation_1", "VLLM_EXECUTE"),
    ],
)
def test_model_span_follows_launch_correlation_not_cpu_scope_or_unrelated_work(
    scope, stage
):
    trace = {
        "traceEvents": [
            {
                "cat": "user_annotation",
                "name": scope,
                "pid": 100,
                "tid": 2,
                "ts": 0,
                "dur": 4,
            },
            {
                "cat": "cuda_runtime",
                "name": "cudaGraphLaunch",
                "pid": 100,
                "tid": 2,
                "ts": 1,
                "dur": 0.2,
                "args": {"correlation": 7},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "gemm",
                "pid": 0,
                "tid": 8,
                "ts": 10,
                "dur": 2,
                "args": {"correlation": 7},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "allreduce",
                "pid": 0,
                "tid": 8,
                "ts": 15,
                "dur": 3,
                "args": {"correlation": 7},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "unrelated",
                "pid": 0,
                "tid": 9,
                "ts": 11,
                "dur": 100,
                "args": {"correlation": 9},
            },
        ]
    }
    result = summarize(trace)
    assert result["stages"][stage]["samples"] == 1
    row = result["iterations"][0]
    assert row["device_span_us"] == 8
    assert row["device_active_union_us"] == 5
    assert row["uncovered_device_span_us"] == 3
    assert row["kernel_family_sums_us"] == {"dense_gemm": 2, "communication": 3}


def test_fused_attention_is_not_misclassified_as_only_rope():
    assert (
        family("sm100::fused_norm_rope_attn_rope_cast_fwd::core_attn::fwd_kernel")
        == "fused_attention_rope_cast"
    )
    assert family("standalone_rope_kernel") == "quantization_rope_norm"
