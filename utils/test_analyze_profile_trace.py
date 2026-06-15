import gzip
import json

import pytest

from utils.analyze_profile_trace import analyze_trace


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
    assert summary["categories"]["gemma_rmsnorm"]["duration_us"] == 20
    assert summary["categories"]["allreduce"]["duration_us"] == 30


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
