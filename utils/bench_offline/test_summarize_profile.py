import json

from summarize_profile import (
    architecture_tag,
    discover_traces,
    interval_summary,
    kernel_family,
    summarize_trace,
)


def test_summarize_trace_groups_gpu_and_cpu_events(tmp_path):
    trace = tmp_path / "run_torch_profile-rank-0.json"
    trace.write_text(
        json.dumps(
            {
                "deviceProperties": [
                    {
                        "id": 0,
                        "name": "NVIDIA B300 SXM6 AC",
                        "totalGlobalMem": 287428640768,
                        "computeMajor": 10,
                        "computeMinor": 3,
                        "numSms": 148,
                    }
                ],
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "gemm",
                        "ts": 100,
                        "dur": 20,
                    },
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "gemm",
                        "ts": 125,
                        "dur": 30,
                    },
                    {
                        "ph": "X",
                        "cat": "cpu_op",
                        "name": "aten::matmul",
                        "ts": 90,
                        "dur": 10,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    assert discover_traces([tmp_path]) == [trace.resolve()]
    summary = summarize_trace(trace, limit=5)
    assert summary["trace_span_ms"] == 0.065
    assert summary["gpu_event_sum_ms"] == 0.05
    assert summary["gpu_busy_union_ms"] == 0.05
    assert summary["gpu_kernel_window_ms"] == 0.055
    assert summary["gpu_idle_within_window_ms"] == 0.005
    assert summary["max_gpu_idle_gap_ms"] == 0.005
    assert summary["gpu_overlap_factor"] == 1.0
    assert summary["device_properties"][0] == {
        "id": 0,
        "name": "NVIDIA B300 SXM6 AC",
        "total_global_mem_bytes": 287428640768,
        "compute_capability": "10.3",
        "num_sms": 148,
    }
    assert summary["top_gpu_families"][0]["name"] == "other"
    assert summary["top_gpu_events"][0] == {
        "name": "gemm",
        "total_ms": 0.05,
        "count": 2,
        "mean_us": 25.0,
    }
    assert summary["top_cpu_events"][0]["name"] == "aten::matmul"


def test_interval_summary_accounts_for_overlap_and_idle():
    summary = interval_summary([(0.0, 20.0), (10.0, 30.0), (40.0, 50.0)])
    assert summary == {
        "sum_us": 50.0,
        "busy_union_us": 40.0,
        "window_us": 50.0,
        "idle_within_window_us": 10.0,
        "overlap_factor": 1.25,
        "max_idle_gap_us": 10.0,
    }


def test_kernel_family_and_architecture_classification():
    assert kernel_family("bmm_MxE4m3_anything_sm100f") == "moe_gemm1"
    assert kernel_family("moeA2ADispatchKernel") == "moe_dispatch"
    assert kernel_family("deep_gemm::sm100_fp8_fp4_gemm") == "dense_deepgemm"
    assert architecture_tag("nvjet_sm103_kernel") == "sm103"
    assert architecture_tag("bmm_kernel_sm100f") == "sm100f"
