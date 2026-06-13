import json

from summarize_profile import discover_traces, summarize_trace


def test_summarize_trace_groups_gpu_and_cpu_events(tmp_path):
    trace = tmp_path / "run_torch_profile-rank-0.json"
    trace.write_text(
        json.dumps(
            {
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
    assert summary["top_gpu_events"][0] == {
        "name": "gemm",
        "total_ms": 0.05,
        "count": 2,
        "mean_us": 25.0,
    }
    assert summary["top_cpu_events"][0]["name"] == "aten::matmul"
