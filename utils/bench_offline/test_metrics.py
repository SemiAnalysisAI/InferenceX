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
    )
    aggregate = measured["aggregate"]
    assert aggregate["mean_token_tpot_ms"] == pytest.approx(20.0)
    assert aggregate["derived_output_tput_per_gpu"] == pytest.approx(50.0)
    assert aggregate["wall_output_tput_per_gpu"] == pytest.approx(31.25)
    assert aggregate["acceptance_rate"] == pytest.approx(0.5)
    assert aggregate["observed_tokens_per_step"] == pytest.approx(
        624 / 256
    )


def test_pooling_passes_does_not_multiply_active_concurrency():
    passes = [
        summarize_pass(
            [request_output() for _ in range(8)],
            wall_seconds=20.0,
            expected_output_tokens=625,
            num_gpus=8,
        )
        for _ in range(3)
    ]
    aggregate = aggregate_passes(passes, num_gpus=8)
    assert aggregate["concurrency"] == 8
    assert aggregate["request_samples"] == 24
    assert aggregate["pass_count"] == 3
    assert aggregate["derived_output_tput_per_gpu"] == pytest.approx(50.0)
    assert aggregate["wall_output_tput_per_gpu"] == pytest.approx(31.25)


def test_huawei_conversion_uses_observed_tokens_per_step():
    comparison = huawei_comparison(
        concurrency=8,
        b300_output_tput_per_gpu=200.0,
        observed_tokens_per_step=2.5,
    )
    assert comparison is not None
    assert comparison["estimated_token_tput_per_chip"] == pytest.approx(
        56.70 * 2.5
    )
    assert comparison["conversion"].endswith(
        "trt_observed_tokens_per_step"
    )
