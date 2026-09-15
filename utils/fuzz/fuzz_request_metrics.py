import pytest
from hypothesis import given, strategies as st

from infx.results.agentic.request_metrics import compute_request_metrics
from agentic.aggregation.test_process_agentic_result import _make_record


@given(rates=st.lists(st.sampled_from([0, 1, 2, 4, 8, 16, 32]), min_size=1, max_size=20),
       latency_scale=st.integers(1, 100), reverse=st.booleans(), unwrapped=st.booleans(), bursts=st.booleans())
def test_agentic_request_metrics_preserve_units_and_window_counts(rates, latency_scale, reverse, unwrapped, bursts):
    rates = [max(1, rates[0]), *rates[1:]]
    base = 1_700_000_000_000_000_000
    ends = [base + second * 1_000_000_000 + (0 if bursts else index * (1_000_000_000 // rate))
            for second, rate in enumerate(rates) for index in range(rate)]
    ends.append(base + len(rates) * 1_000_000_000)
    records = []
    for index, end in enumerate(ends):
        record = _make_record(conv_id=f"request-{index}", turn_index=0, isl=200, osl=100,
                              ttft_ms=10 * latency_scale, e2e_ms=1000 * latency_scale,
                              itl_ms=20 * latency_scale, start_ns=end - 1_000_000_000 * latency_scale, end_ns=end)
        if unwrapped:
            record["metrics"] = {key: value["value"] for key, value in record["metrics"].items()}
        records.append(record)
    if reverse:
        records.reverse()
    _, metrics = compute_request_metrics(records)
    assert metrics["qps"]["samples"] == len(rates)
    assert metrics["qps"]["mean"] == pytest.approx(sum(rates) / len(rates))
    latency = metrics["latency"]
    assert latency["ttft"]["mean"] == pytest.approx(0.01 * latency_scale)
    assert latency["itl"]["p95"] == pytest.approx(0.02 * latency_scale)
    assert latency["intvty"]["p95"] == pytest.approx(50 / latency_scale)
    assert latency["e2e_norm_intvty"]["mean"] == pytest.approx(100 / latency_scale)


@given(scale=st.integers(1, 1000), invalid_turn=st.one_of(st.integers(-1000, -1), st.integers(2, 1000)),
       subagent=st.booleans())
def test_expected_tokens_ignore_turns_outside_the_trace(scale, invalid_turn, subagent):
    conversation = "session::sa:child" if subagent else "session"
    records = [{"metadata": {"conversation_id": conversation, "turn_index": index}}
               for index in (0, 1, invalid_turn)]
    trace = {"id": "session", "requests": [{"type": "n", "output_length": 10 * scale},
                                           {"type": "s", "output_length": 20 * scale}]}
    _, metrics = compute_request_metrics(records, traces=[trace])
    assert metrics["tokens"]["output_expected"]["mean"] == 15 * scale
    assert metrics["tokens"]["output_expected"]["std"] == 5 * scale
