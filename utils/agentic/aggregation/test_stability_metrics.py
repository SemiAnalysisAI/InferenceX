"""Tests for descriptive AgentX fixed-window diagnostics."""

from __future__ import annotations

import pytest

from .stability_metrics import _convergence_summary, compute_stability_metrics


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
    assert result["convergence"] == {
        "checkpoint_seconds": 300,
        "tolerance_ratio": 0.05,
        "min_confirmation_seconds": 1200,
        "horizon_seconds": 1200,
        "metrics": {
            "ttft": {"p75": None, "p90": None},
            "e2el": {"p75": None, "p90": None},
            "intvty": {"p75": None, "p90": None},
        },
    }


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


def test_reports_insufficient_duration_without_stabilization() -> None:
    result = compute_stability_metrics(
        [_record(second=0, source="root-a", ttft_ms=100, e2e_ms=1000, itl_ms=20)],
        benchmark_duration_s=1200,
    )

    assert result["convergence"]["horizon_seconds"] == 1200
    assert result["convergence"]["metrics"]["ttft"]["p90"] is None


def test_no_checkpoint_qualifies_when_a_late_prefix_leaves_the_band() -> None:
    checkpoints = [
        {"seconds": 300, "requests": 10, "value": 100.0},
        {"seconds": 600, "requests": 20, "value": 100.0},
        {"seconds": 900, "requests": 30, "value": 100.0},
        {"seconds": 1200, "requests": 40, "value": 100.0},
        {"seconds": 1500, "requests": 50, "value": 106.0},
        {"seconds": 1800, "requests": 60, "value": 100.0},
    ]

    assert (
        _convergence_summary(
            checkpoints,
            tolerance_ratio=0.05,
            min_confirmation_seconds=1200,
        )
        is None
    )


@pytest.mark.parametrize("boundary_value", [105.0, 100.0 / 1.05])
def test_exact_symmetric_five_percent_boundary_qualifies(boundary_value: float) -> None:
    checkpoints = [
        {"seconds": 300, "requests": 10, "value": boundary_value},
        {"seconds": 600, "requests": 20, "value": 100.0},
        {"seconds": 900, "requests": 30, "value": 100.0},
        {"seconds": 1200, "requests": 40, "value": 100.0},
        {"seconds": 1500, "requests": 50, "value": 100.0},
    ]

    result = _convergence_summary(
        checkpoints,
        tolerance_ratio=0.05,
        min_confirmation_seconds=1200,
    )

    assert result is not None
    assert result["time_seconds"] == 300


def test_confirmation_period_rejects_an_otherwise_stable_late_checkpoint() -> None:
    checkpoints = [
        {"seconds": 300, "requests": 10, "value": 80.0},
        {"seconds": 600, "requests": 20, "value": 100.0},
        {"seconds": 900, "requests": 30, "value": 100.0},
        {"seconds": 1200, "requests": 40, "value": 100.0},
        {"seconds": 1500, "requests": 50, "value": 100.0},
    ]

    assert (
        _convergence_summary(
            checkpoints,
            tolerance_ratio=0.05,
            min_confirmation_seconds=1200,
        )
        is None
    )


def test_reciprocal_series_stabilizes_at_the_same_checkpoint() -> None:
    latency = [
        {"seconds": 300, "requests": 10, "value": 0.08},
        {"seconds": 600, "requests": 20, "value": 0.105},
        {"seconds": 900, "requests": 30, "value": 0.102},
        {"seconds": 1200, "requests": 40, "value": 0.101},
        {"seconds": 1500, "requests": 50, "value": 0.1},
        {"seconds": 1800, "requests": 60, "value": 0.1},
    ]
    interactivity = [{**item, "value": 1 / float(item["value"])} for item in latency]

    latency_result = _convergence_summary(
        latency, tolerance_ratio=0.05, min_confirmation_seconds=1200
    )
    interactivity_result = _convergence_summary(
        interactivity, tolerance_ratio=0.05, min_confirmation_seconds=1200
    )

    assert latency_result is not None
    assert interactivity_result is not None
    assert latency_result["time_seconds"] == interactivity_result["time_seconds"] == 600
    assert interactivity_result["min"] == pytest.approx(
        1 / float(latency_result["max"])
    )
    assert interactivity_result["max"] == pytest.approx(
        1 / float(latency_result["min"])
    )


def test_excludes_records_at_and_beyond_the_complete_convergence_horizon() -> None:
    records = [
        _record(second=0, source="root-a", ttft_ms=100, e2e_ms=1000, itl_ms=20),
        _record(second=299, source="root-a", ttft_ms=100, e2e_ms=1000, itl_ms=20),
        _record(second=300, source="root-b", ttft_ms=100, e2e_ms=1000, itl_ms=20),
        _record(second=1500, source="root-c", ttft_ms=9999, e2e_ms=9999, itl_ms=9999),
        _record(second=1600, source="root-d", ttft_ms=9999, e2e_ms=9999, itl_ms=9999),
    ]

    result = compute_stability_metrics(records, benchmark_duration_s=1799)
    p90_ttft = result["convergence"]["metrics"]["ttft"]["p90"]

    assert result["convergence"]["horizon_seconds"] == 1500
    assert p90_ttft is not None
    assert p90_ttft["time_seconds"] == 300
    assert p90_ttft["requests"] == 2
    assert p90_ttft["min"] == pytest.approx(0.1)
    assert p90_ttft["max"] == pytest.approx(0.1)


def test_convergence_request_count_only_includes_metric_samples() -> None:
    records = [
        _record(second=0, source="root-a", ttft_ms=100, e2e_ms=1000, itl_ms=20),
        _record(second=10, source="root-a", ttft_ms=100, e2e_ms=1000, itl_ms=0),
        _record(second=300, source="root-b", ttft_ms=100, e2e_ms=1000, itl_ms=20),
        _record(second=600, source="root-c", ttft_ms=100, e2e_ms=1000, itl_ms=20),
        _record(second=900, source="root-d", ttft_ms=100, e2e_ms=1000, itl_ms=20),
        _record(second=1200, source="root-e", ttft_ms=100, e2e_ms=1000, itl_ms=20),
    ]

    result = compute_stability_metrics(records, benchmark_duration_s=1500)
    p90_intvty = result["convergence"]["metrics"]["intvty"]["p90"]

    assert p90_intvty is not None
    assert p90_intvty["time_seconds"] == 300
    assert p90_intvty["requests"] == 1


def test_rejects_invalid_convergence_parameters() -> None:
    with pytest.raises(ValueError, match="checkpoint_seconds must be positive"):
        compute_stability_metrics([], benchmark_duration_s=3600, checkpoint_seconds=0)
    with pytest.raises(ValueError, match="tolerance_ratio must be non-negative"):
        compute_stability_metrics([], benchmark_duration_s=3600, tolerance_ratio=-0.01)
    with pytest.raises(
        ValueError, match="min_confirmation_seconds must be non-negative"
    ):
        compute_stability_metrics(
            [], benchmark_duration_s=3600, min_confirmation_seconds=-1
        )
