"""Tests for descriptive AgentX fixed-window diagnostics."""

from __future__ import annotations

import pytest

from .stability_metrics import compute_stability_metrics


def _record(
    *,
    second: float,
    source: str,
    ttft_ms: float,
    e2e_ms: float,
    itl_ms: float,
) -> dict:
    return {
        "metadata": {
            "credit_issued_ns": int(second * 1e9),
            "request_start_ns": int(second * 1e9),
            "source_trace_id": source,
            "conversation_id": f"replay-{source}",
        },
        "metrics": {
            "time_to_first_token": {"value": ttft_ms},
            "request_latency": {"value": e2e_ms},
            "inter_token_latency": {"value": itl_ms},
        },
    }


def test_computes_non_overlapping_ranges_and_root_coverage() -> None:
    records = [
        _record(second=0, source="root-a", ttft_ms=100, e2e_ms=1000, itl_ms=20),
        _record(second=10, source="root-a", ttft_ms=200, e2e_ms=2000, itl_ms=40),
        _record(second=610, source="root-b", ttft_ms=300, e2e_ms=3000, itl_ms=50),
        _record(second=620, source="root-b", ttft_ms=400, e2e_ms=4000, itl_ms=100),
    ]

    result = compute_stability_metrics(records, benchmark_duration_s=1200)

    assert result["window_seconds"] == 600
    assert result["expected_window_count"] == 2
    assert result["observed_window_count"] == 2
    assert result["min_window_requests"] == 2
    assert result["root_trajectory_count"] == 2
    assert result["root_trajectory_kish_effective_count"] == pytest.approx(2)
    assert result["root_trajectory_largest_share"] == pytest.approx(0.5)
    assert result["observed_ranges"]["ttft"]["p90"] == pytest.approx(
        {"min": 0.19, "max": 0.39}
    )
    assert result["observed_ranges"]["e2el"]["p75"] == pytest.approx(
        {"min": 1.75, "max": 3.75}
    )
    # Per-window p90 ITL is 38 ms and 95 ms. Interactivity inverts each
    # latency estimate, so the slower window is the lower bound.
    assert result["observed_ranges"]["intvty"]["p90"] == pytest.approx(
        {"min": 1 / 0.095, "max": 1 / 0.038}
    )


def test_subagent_requests_share_their_root_trajectory_cluster() -> None:
    records = [
        _record(second=0, source="root-a", ttft_ms=100, e2e_ms=1000, itl_ms=20),
        _record(second=1, source="root-a", ttft_ms=100, e2e_ms=1000, itl_ms=20),
        _record(second=2, source="root-a", ttft_ms=100, e2e_ms=1000, itl_ms=20),
        _record(second=601, source="root-b", ttft_ms=100, e2e_ms=1000, itl_ms=20),
    ]

    result = compute_stability_metrics(records, benchmark_duration_s=1200)

    assert result["root_trajectory_count"] == 2
    assert result["root_trajectory_kish_effective_count"] == pytest.approx(16 / 10)
    assert result["root_trajectory_largest_share"] == pytest.approx(0.75)


def test_omits_ranges_when_run_has_fewer_than_two_observed_windows() -> None:
    records = [
        _record(second=0, source="root-a", ttft_ms=100, e2e_ms=1000, itl_ms=20),
        _record(second=10, source="root-a", ttft_ms=200, e2e_ms=2000, itl_ms=40),
    ]

    result = compute_stability_metrics(records, benchmark_duration_s=600)

    assert result["expected_window_count"] == 1
    assert result["observed_window_count"] == 1
    assert result["observed_ranges"] == {}


def test_excludes_records_beyond_the_configured_measurement_windows() -> None:
    records = [
        _record(second=0, source="root-a", ttft_ms=100, e2e_ms=1000, itl_ms=20),
        _record(second=601, source="root-b", ttft_ms=200, e2e_ms=2000, itl_ms=40),
        _record(second=1201, source="root-c", ttft_ms=9999, e2e_ms=9999, itl_ms=9999),
    ]

    result = compute_stability_metrics(records, benchmark_duration_s=1200)

    assert result["expected_window_count"] == 2
    assert result["observed_window_count"] == 2
    assert result["observed_ranges"]["ttft"]["p90"]["max"] == pytest.approx(0.2)


def test_falls_back_to_request_start_when_credit_timestamp_is_null() -> None:
    first = _record(second=0, source="root-a", ttft_ms=100, e2e_ms=1000, itl_ms=20)
    second = _record(second=601, source="root-b", ttft_ms=200, e2e_ms=2000, itl_ms=40)
    first["metadata"]["credit_issued_ns"] = None
    second["metadata"]["credit_issued_ns"] = None

    result = compute_stability_metrics([first, second], benchmark_duration_s=1200)

    assert result["observed_window_count"] == 2
    assert result["observed_ranges"]["ttft"]["p90"] == pytest.approx(
        {"min": 0.1, "max": 0.2}
    )


def test_rejects_non_positive_window_size() -> None:
    with pytest.raises(ValueError, match="window_seconds must be positive"):
        compute_stability_metrics([], benchmark_duration_s=3600, window_seconds=0)
