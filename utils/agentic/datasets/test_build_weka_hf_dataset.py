from __future__ import annotations

from utils.agentic.datasets.build_weka_hf_dataset import _filter_trace_256k


def _request(t: float, api_time: float, tokens: int = 1) -> dict:
    return {
        "t": t,
        "type": "n",
        "model": "test-model",
        "in": tokens,
        "out": 0,
        "hash_ids": [],
        "api_time": api_time,
    }


def _subagent(agent_id: str, requests: list[dict]) -> dict:
    first_t = min(req["t"] for req in requests)
    last_end = max(req["t"] + req["api_time"] for req in requests)
    return {
        "t": first_t,
        "type": "subagent",
        "agent_id": agent_id,
        "duration_ms": int(round((last_end - first_t) * 1000)),
        "total_tokens": sum(req["in"] + req["out"] for req in requests),
        "requests": requests,
    }


def test_filter_without_drops_preserves_overlapping_subagents_exactly() -> None:
    trace = {
        "id": "overlap",
        "requests": [
            _request(0.0, 1.0),
            _subagent("child-a", [_request(2.0, 5.0)]),
            _subagent("child-b", [_request(3.0, 5.0)]),
        ],
    }

    filtered = _filter_trace_256k(trace, cap=256_000)

    assert filtered == trace
    child_a, child_b = filtered["requests"][1:]
    assert child_b["t"] < child_a["t"] + child_a["duration_ms"] / 1000.0


def test_filter_uses_uniform_shift_and_preserves_overlap() -> None:
    trace = {
        "id": "shifted-overlap",
        "requests": [
            _request(0.0, 10.0, tokens=300_000),
            _subagent("child-a", [_request(4.0, 5.0)]),
            _subagent("child-b", [_request(5.0, 5.0)]),
        ],
    }

    filtered = _filter_trace_256k(trace, cap=256_000)

    child_a, child_b = filtered["requests"]
    assert child_a["t"] == 0.0
    assert child_b["t"] == 1.0
    assert child_a["requests"][0]["t"] == 0.0
    assert child_b["requests"][0]["t"] == 1.0
    assert child_b["t"] < child_a["t"] + child_a["duration_ms"] / 1000.0


def test_partial_subagent_filter_recomputes_group_metadata() -> None:
    trace = {
        "id": "partial",
        "requests": [
            _request(0.0, 1.0),
            _subagent(
                "child",
                [
                    _request(2.0, 3.0, tokens=300_000),
                    _request(4.0, 2.0, tokens=100),
                    _request(8.0, 1.0, tokens=200),
                ],
            ),
        ],
    }

    filtered = _filter_trace_256k(trace, cap=256_000)

    child = filtered["requests"][1]
    assert [req["t"] for req in child["requests"]] == [4.0, 8.0]
    assert child["t"] == 4.0
    assert child["duration_ms"] == 5_000
    assert child["total_tokens"] == 300
