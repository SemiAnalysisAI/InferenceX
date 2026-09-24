import sys
import types

sys.modules.setdefault("torch", types.ModuleType("torch"))  # _replay_stats is pure Python

from operatorx.runners.common import profiling  # noqa: E402


def _k(name, ts, dur, stream=7):
    return {"name": name, "ts": ts, "dur": dur, "args": {"stream": stream}}


def test_replay_stats_overlap_and_gap():
    flush = profiling._FLUSH_KERNEL_MARKER
    spin = profiling._SHIELD_KERNEL_MARKER
    events = [_k(flush, 0, 50), _k(spin, 50, 40), _k("quant", 100, 3), _k("shared", 101, 6, stream=9), _k("gemm", 105, 5),
              _k(flush, 200, 50), _k("quant", 300, 3), _k("gemm", 304, 5),
              _k(flush, 400, 50), _k("quant", 500, 3), _k("gemm", 504, 5.5)]
    r = profiling._replay_stats(events)
    # median-span replay is the third: [500,503] + [504,509.5]
    assert (r["span_us"], r["busy_us"], r["gap_us"], r["overlap_us"], r["streams"]) == (9.5, 8.5, 1.0, 0.0, 1)
    assert [t["name"] for t in r["timeline"]] == ["quant", "gemm"] and r["timeline"][1]["start_us"] == 4.0
    concurrent = profiling._replay_stats(events[:5])
    assert concurrent["overlap_us"] == 4.0 and concurrent["streams"] == 2 and concurrent["gap_us"] == 0.0
