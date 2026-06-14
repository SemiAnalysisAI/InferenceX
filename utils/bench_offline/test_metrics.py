from datetime import timedelta
from types import SimpleNamespace

import pytest

from metrics import (
    huawei_comparison,
    huawei_filter_round_latencies,
    select_full_batch_decode_rounds,
    summarize_decode_rounds,
    summarize_requests,
)


def request_output(
    *,
    output_tokens: int = 1025,
    first_iter: int = 10,
    last_iter: int = 350,
):
    timing = SimpleNamespace(
        arrival_time=timedelta(seconds=0),
        first_scheduled_time=timedelta(seconds=1),
        first_token_time=timedelta(seconds=5),
        last_token_time=timedelta(seconds=25),
    )
    perf = SimpleNamespace(
        timing_metrics=timing,
        first_iter=first_iter,
        last_iter=last_iter,
    )
    completion = SimpleNamespace(
        token_ids=list(range(output_tokens)),
        request_perf_metrics=perf,
    )
    return SimpleNamespace(outputs=[completion])


def iteration(
    index: int,
    *,
    latency_ms: float = 20.0,
    active: int = 8,
    queued: int = 0,
    scheduled: int = 8,
    context: int = 0,
    generation: int = 8,
    paused: int = 0,
    drafted: int = 24,
    accepted: int = 12,
):
    return {
        "iter": index,
        "iterLatencyMS": latency_ms,
        "numActiveRequests": active,
        "numQueuedRequests": queued,
        "inflightBatchingStats": {
            "numScheduledRequests": scheduled,
            "numContextRequests": context,
            "numGenRequests": generation,
            "numPausedRequests": paused,
        },
        "specDecodingStats": {
            "numDraftTokens": drafted,
            "numAcceptedTokens": accepted,
            "numRequestsWithDraftTokens": generation,
            "acceptanceLength": (
                1.0 + accepted / generation if generation else 0.0
            ),
        },
    }


def fixed_batch_stats(local_batch_size=8):
    stats = [
        iteration(
            1,
            latency_ms=50.0,
            active=local_batch_size,
            scheduled=local_batch_size,
            context=local_batch_size,
            generation=0,
            drafted=0,
            accepted=0,
        )
    ]
    stats.extend(
        iteration(
            index,
            latency_ms=100.0 if index in {2, 257} else 20.0,
            active=local_batch_size,
            scheduled=local_batch_size,
            generation=local_batch_size,
            drafted=local_batch_size * 3,
            accepted=local_batch_size * 3 // 2,
        )
        for index in range(2, 258)
    )
    return stats


def test_request_summary_keeps_wall_and_latency_diagnostics():
    summary = summarize_requests(
        [request_output() for _ in range(64)],
        wall_seconds=100.0,
        expected_output_tokens=1025,
        num_gpus=8,
    )
    aggregate = summary["aggregate"]
    assert aggregate["request_samples"] == 64
    assert aggregate["wall_output_tput_per_gpu"] == pytest.approx(82.0)
    assert aggregate["mean_ttft_ms"] == pytest.approx(5000.0)
    assert aggregate["overall_observed_tokens_per_step"] == pytest.approx(
        1024 / 340
    )
    assert len(aggregate["output_sequence_sha256"]) == 64


def test_huawei_filter_skips_first_and_drops_only_upper_outlier():
    filtered = huawei_filter_round_latencies(
        [100.0] + [20.0] * 254 + [100.0]
    )
    assert filtered["retained_rounds"] == 254
    assert filtered["outlier_rounds"] == 1
    assert filtered["mean_ms"] == pytest.approx(20.0)


def test_fixed_batch_decode_round_summary_matches_huawei_arithmetic():
    summary = summarize_decode_rounds(
        fixed_batch_stats(),
        global_batch_size=64,
        local_batch_size=8,
        num_gpus=8,
    )
    assert summary["measured_decode_rounds"] == 256
    assert summary["decode_round_tpot_ms"] == pytest.approx(20.0)
    assert summary["decode_step_tput_per_gpu"] == pytest.approx(400.0)
    assert summary["observed_tokens_per_step"] == pytest.approx(2.5)
    assert summary["raw_acceptance_rate"] == pytest.approx(0.5)
    assert summary["effective_acceptance_rate"] == pytest.approx(0.5)
    assert summary["output_tput_per_gpu"] == pytest.approx(1000.0)
    assert summary["equivalent_output_tpot_ms"] == pytest.approx(8.0)
    assert summary["filter"]["retained_rounds"] == 254
    assert summary["schedule_validation"]["selected_first_iter"] == 2
    assert summary["schedule_validation"]["selected_last_iter"] == 257


def test_decode_summary_requires_same_window_speculative_counters():
    stats = fixed_batch_stats()
    for item in stats:
        item["specDecodingStats"] = {
            "numDraftTokens": 0,
            "numAcceptedTokens": 0,
        }
    with pytest.raises(RuntimeError, match="omitted speculative counters"):
        summarize_decode_rounds(
            stats,
            global_batch_size=64,
            local_batch_size=8,
            num_gpus=8,
        )


def test_decode_validation_rejects_staggered_first_generation_round():
    stats = [
        iteration(
            1,
            active=8,
            scheduled=8,
            context=8,
            generation=0,
            drafted=0,
            accepted=0,
        ),
        iteration(
            2,
            active=4,
            queued=4,
            scheduled=4,
            generation=4,
        )
    ]
    stats.extend(iteration(index) for index in range(3, 260))
    with pytest.raises(RuntimeError, match="fixed local batch"):
        select_full_batch_decode_rounds(
            stats,
            local_batch_size=8,
            required_rounds=256,
        )


def test_decode_validation_ignores_only_inactive_prior_pass_tail():
    stats = [
        iteration(
            0,
            active=0,
            scheduled=8,
            context=0,
            generation=8,
        ),
        *fixed_batch_stats(),
    ]
    selected, diagnostics = select_full_batch_decode_rounds(
        stats,
        local_batch_size=8,
        required_rounds=256,
    )
    assert len(selected) == 256
    assert selected[0]["iter"] == 2
    assert diagnostics["leading_inactive_iterations_ignored"] == 1
    assert diagnostics["leading_inactive_first_iter"] == 0
    assert diagnostics["leading_inactive_last_iter"] == 0
    assert diagnostics["prefill_iter"] == 1


def test_decode_validation_rejects_active_decode_before_prefill():
    stats = [
        iteration(
            0,
            active=8,
            scheduled=8,
            context=0,
            generation=8,
        ),
        *fixed_batch_stats(),
    ]
    with pytest.raises(RuntimeError, match="full-local-batch prefill"):
        select_full_batch_decode_rounds(
            stats,
            local_batch_size=8,
            required_rounds=256,
        )


def test_decode_validation_rejects_staggered_prefill():
    stats = [
        iteration(
            1,
            active=4,
            queued=4,
            scheduled=4,
            context=4,
            generation=0,
        ),
        iteration(
            2,
            active=8,
            scheduled=4,
            context=4,
            generation=0,
        ),
    ]
    stats.extend(iteration(index) for index in range(3, 260))
    with pytest.raises(RuntimeError, match="full-local-batch prefill"):
        select_full_batch_decode_rounds(
            stats,
            local_batch_size=8,
            required_rounds=256,
        )


def test_decode_validation_rejects_mixed_prefill_and_decode():
    stats = [
        iteration(
            1,
            scheduled=8,
            context=4,
            generation=4,
        )
    ]
    stats.extend(iteration(index) for index in range(2, 260))
    with pytest.raises(RuntimeError, match="mixed prefill and decode"):
        select_full_batch_decode_rounds(
            stats,
            local_batch_size=8,
            required_rounds=256,
        )


def test_huawei_comparison_uses_raw_step_rate_and_separate_yield():
    decode = summarize_decode_rounds(
        fixed_batch_stats(),
        global_batch_size=64,
        local_batch_size=8,
        num_gpus=8,
    )
    comparison = huawei_comparison(64, decode)
    assert comparison["global_batch_match"] is True
    assert comparison["device_count_match"] is False
    assert comparison["huawei_local_batch_size"] == 4
    assert comparison["b300_to_huawei_decode_step_ratio"] == pytest.approx(
        400.0 / 210.16
    )
    assert comparison["published_output_tput_per_chip"] == pytest.approx(
        210.16 * 2.44
    )


def test_huawei_comparison_names_gb300_and_matches_device_count():
    decode = summarize_decode_rounds(
        fixed_batch_stats(local_batch_size=4),
        global_batch_size=64,
        local_batch_size=4,
        num_gpus=16,
    )
    comparison = huawei_comparison(
        64,
        decode,
        hardware_key="gb300",
        hardware_label="GB300 NVL16",
    )
    assert comparison["device_count_match"] is True
    assert comparison["gb300_active_gpu_count"] == 16
    assert comparison[
        "gb300_to_huawei_decode_step_ratio"
    ] == pytest.approx(200.0 / 210.16)
    assert comparison["hardware_to_huawei_decode_step_ratio"] == pytest.approx(
        200.0 / 210.16
    )
