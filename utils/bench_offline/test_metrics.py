from datetime import timedelta
from types import SimpleNamespace

import pytest

from metrics import (
    aggregate_passes,
    huawei_comparison,
    summarize_pass,
)


def request_output(
    *,
    token_tpot_s: float = 0.02,
    first_iter: int = 10,
    last_iter: int = 266,
    accepted: int = 300,
    drafted: int = 600,
):
    output_tokens = 625
    decode_window = token_tpot_s * (output_tokens - 1)
    timing = SimpleNamespace(
        arrival_time=timedelta(seconds=0),
        first_scheduled_time=timedelta(seconds=1),
        first_token_time=timedelta(seconds=5),
        last_token_time=timedelta(seconds=5 + decode_window),
    )
    speculative = SimpleNamespace(
        total_accepted_draft_tokens=accepted,
        total_draft_tokens=drafted,
    )
    perf = SimpleNamespace(
        timing_metrics=timing,
        first_iter=first_iter,
        last_iter=last_iter,
        speculative_decoding=speculative,
    )
    completion = SimpleNamespace(
        token_ids=list(range(output_tokens)),
        request_perf_metrics=perf,
    )
    return SimpleNamespace(outputs=[completion])


def test_token_tpot_and_derived_throughput():
    measured = summarize_pass(
        [request_output() for _ in range(8)],
        wall_seconds=20.0,
        expected_output_tokens=625,
        num_gpus=8,
        max_draft_tokens=3,
    )
    aggregate = measured["aggregate"]
    assert aggregate["mean_token_tpot_ms"] == pytest.approx(20.0)
    assert aggregate["mean_step_tpot_ms"] == pytest.approx(48.75)
    assert aggregate["derived_output_tput_per_gpu"] == pytest.approx(50.0)
    assert aggregate["derived_step_tput_per_gpu"] == pytest.approx(
        1 / 0.04875
    )
    assert aggregate["wall_output_tput_per_gpu"] == pytest.approx(31.25)
    assert aggregate["acceptance_rate"] == pytest.approx(0.5)
    assert aggregate["raw_speculative_metrics_available"] is True
    assert aggregate["observed_tokens_per_step"] == pytest.approx(
        624 / 256
    )
    assert aggregate["effective_accepted_drafts_per_step"] == pytest.approx(
        624 / 256 - 1
    )
    assert aggregate["effective_acceptance_rate"] == pytest.approx(
        (624 / 256 - 1) / 3
    )
    assert len(aggregate["output_sequence_sha256"]) == 64


def test_zero_trt_speculative_counters_are_unavailable():
    measured = summarize_pass(
        [request_output(accepted=0, drafted=0)],
        wall_seconds=10.0,
        expected_output_tokens=625,
        num_gpus=8,
        max_draft_tokens=3,
    )
    aggregate = measured["aggregate"]
    assert aggregate["raw_speculative_metrics_available"] is False
    assert aggregate["acceptance_rate"] is None
    assert aggregate["accepted_drafts_per_step"] is None
    assert aggregate["effective_acceptance_rate"] == pytest.approx(
        (624 / 256 - 1) / 3
    )


def test_pooling_passes_does_not_multiply_active_concurrency():
    passes = [
        summarize_pass(
            [request_output() for _ in range(8)],
            wall_seconds=20.0,
            expected_output_tokens=625,
            num_gpus=8,
            max_draft_tokens=3,
        )
        for _ in range(3)
    ]
    aggregate = aggregate_passes(passes, num_gpus=8)
    assert aggregate["concurrency"] == 8
    assert aggregate["request_samples"] == 24
    assert aggregate["pass_count"] == 3
    assert aggregate["derived_output_tput_per_gpu"] == pytest.approx(50.0)
    assert aggregate["wall_output_tput_per_gpu"] == pytest.approx(31.25)
    assert len(set(aggregate["per_pass_output_sequence_sha256"])) == 1


def test_huawei_conversion_uses_observed_tokens_per_step():
    comparison = huawei_comparison(
        concurrency=16,
        b300_output_tput_per_gpu=200.0,
        b300_step_tput_per_gpu=80.0,
        observed_tokens_per_step=2.5,
        mtp_draft_tokens=3,
    )
    assert comparison is not None
    assert comparison["estimated_token_tput_per_chip"] == pytest.approx(
        56.70 * 2.5
    )
    assert comparison["published_dataset_token_tput_per_chip"] == (
        pytest.approx(56.70 * 2.44)
    )
    assert comparison["published_acceptance_rate"] == pytest.approx(0.48)
    assert comparison["conversion"].endswith(
        "trt_observed_tokens_per_step"
    )
    assert comparison["b300_to_huawei_published_output_ratio"] == (
        pytest.approx(200.0 / (56.70 * 2.44))
    )
    assert comparison["b300_to_huawei_step_rate_ratio"] == pytest.approx(
        80.0 / 56.70
    )


def test_huawei_ratio_is_not_claimed_for_different_mtp_depth():
    comparison = huawei_comparison(
        concurrency=16,
        b300_output_tput_per_gpu=200.0,
        b300_step_tput_per_gpu=80.0,
        observed_tokens_per_step=1.8,
        mtp_draft_tokens=2,
    )
    assert comparison is not None
    assert comparison["comparable"] is False
    assert comparison["b300_to_huawei_published_output_ratio"] is None
    assert comparison["b300_to_huawei_step_rate_ratio"] is None


def test_huawei_ratio_is_reported_for_exact_batch_tp4():
    comparison = huawei_comparison(
        concurrency=16,
        b300_output_tput_per_gpu=200.0,
        b300_step_tput_per_gpu=80.0,
        observed_tokens_per_step=3.0,
        mtp_draft_tokens=3,
        effective_parallelism="TP4",
        active_gpu_count=4,
    )
    assert comparison is not None
    assert comparison["global_batch_match"] is True
    assert comparison["device_count_match"] is False
    assert comparison["hardware_topology_match"] is False
    assert comparison["b300_active_gpu_count"] == 4
    assert comparison["huawei_gate_passed"] is True


def test_huawei_ratio_is_not_reported_for_unpublished_global_batch():
    assert (
        huawei_comparison(
            concurrency=32,
            b300_output_tput_per_gpu=400.0,
            b300_step_tput_per_gpu=120.0,
            observed_tokens_per_step=3.0,
            mtp_draft_tokens=3,
            effective_parallelism="TP4",
            active_gpu_count=4,
        )
        is None
    )
