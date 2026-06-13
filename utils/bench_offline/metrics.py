"""Metric extraction for TensorRT-LLM request performance telemetry."""

from __future__ import annotations

import math
from datetime import timedelta
from statistics import fmean
from typing import Any, Iterable


HUAWEI_REFERENCE: dict[int, dict[str, float | int]] = {
    8: {
        "global_batch_size": 16,
        "chips": 16,
        "step_tpot_ms": 17.64,
        "step_tput_per_chip": 56.70,
    },
    32: {
        "global_batch_size": 64,
        "chips": 16,
        "step_tpot_ms": 19.03,
        "step_tput_per_chip": 210.16,
    },
    64: {
        "global_batch_size": 128,
        "chips": 16,
        "step_tpot_ms": 20.61,
        "step_tput_per_chip": 388.23,
    },
}


def _seconds(value: Any) -> float:
    if isinstance(value, timedelta):
        return value.total_seconds()
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    return float(value)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def extract_request_metrics(
    request_output: Any,
    expected_output_tokens: int,
) -> dict[str, Any]:
    completions = getattr(request_output, "outputs", None)
    if not completions:
        raise RuntimeError("TRT request returned no completion outputs")
    completion = completions[0]
    token_ids = getattr(completion, "token_ids", None)
    if token_ids is None:
        raise RuntimeError("TRT completion did not return token IDs")
    output_tokens = len(token_ids)
    if output_tokens != expected_output_tokens:
        raise RuntimeError(
            f"TRT generated {output_tokens} tokens; expected "
            f"{expected_output_tokens}"
        )

    perf = getattr(completion, "request_perf_metrics", None)
    if perf is None:
        raise RuntimeError("TRT completion did not return request perf metrics")
    timing = perf.timing_metrics
    arrival = _seconds(timing.arrival_time)
    first_token = _seconds(timing.first_token_time)
    last_token = _seconds(timing.last_token_time)
    first_scheduled = _seconds(timing.first_scheduled_time)

    decode_window = last_token - first_token
    decode_tokens = output_tokens - 1
    if decode_window <= 0 or decode_tokens <= 0:
        raise RuntimeError(
            f"Invalid decode telemetry: window={decode_window}, "
            f"decode_tokens={decode_tokens}"
        )

    first_iter = perf.first_iter
    last_iter = perf.last_iter
    if first_iter is None or last_iter is None:
        raise RuntimeError("TRT request perf metrics omitted first/last iter")
    decode_iterations = int(last_iter) - int(first_iter)
    if decode_iterations <= 0:
        raise RuntimeError(
            f"Invalid TRT decode iteration count: {decode_iterations}"
        )

    speculative = perf.speculative_decoding
    accepted = int(speculative.total_accepted_draft_tokens)
    drafted = int(speculative.total_draft_tokens)
    return {
        "output_tokens": output_tokens,
        "decode_tokens": decode_tokens,
        "decode_iterations": decode_iterations,
        "ttft_s": first_token - arrival,
        "queue_s": first_scheduled - arrival,
        "decode_window_s": decode_window,
        "e2e_s": last_token - arrival,
        "token_tpot_s": decode_window / decode_tokens,
        "observed_tokens_per_step": decode_tokens / decode_iterations,
        "accepted_draft_tokens": accepted,
        "proposed_draft_tokens": drafted,
        "acceptance_rate": accepted / drafted if drafted else 0.0,
    }


def aggregate_requests(
    request_metrics: list[dict[str, Any]],
    wall_seconds: float,
    num_gpus: int,
    concurrency: int | None = None,
) -> dict[str, Any]:
    if not request_metrics:
        raise ValueError("No request metrics to aggregate")
    if wall_seconds <= 0:
        raise ValueError(f"Invalid measured wall time: {wall_seconds}")

    token_tpots = [float(item["token_tpot_s"]) for item in request_metrics]
    ttfts = [float(item["ttft_s"]) for item in request_metrics]
    e2els = [float(item["e2e_s"]) for item in request_metrics]
    mean_tpot = fmean(token_tpots)
    total_output_tokens = sum(
        int(item["output_tokens"]) for item in request_metrics
    )
    total_decode_tokens = sum(
        int(item["decode_tokens"]) for item in request_metrics
    )
    total_decode_iterations = sum(
        int(item["decode_iterations"]) for item in request_metrics
    )
    total_accepted = sum(
        int(item["accepted_draft_tokens"]) for item in request_metrics
    )
    total_drafted = sum(
        int(item["proposed_draft_tokens"]) for item in request_metrics
    )
    active_concurrency = concurrency or len(request_metrics)
    return {
        "request_samples": len(request_metrics),
        "concurrency": active_concurrency,
        "wall_seconds": wall_seconds,
        "mean_token_tpot_ms": mean_tpot * 1000.0,
        "median_token_tpot_ms": _percentile(token_tpots, 50) * 1000.0,
        "p90_token_tpot_ms": _percentile(token_tpots, 90) * 1000.0,
        "p99_token_tpot_ms": _percentile(token_tpots, 99) * 1000.0,
        "mean_ttft_ms": fmean(ttfts) * 1000.0,
        "p99_ttft_ms": _percentile(ttfts, 99) * 1000.0,
        "mean_e2e_ms": fmean(e2els) * 1000.0,
        "p99_e2e_ms": _percentile(e2els, 99) * 1000.0,
        "derived_output_tput_per_gpu": (
            active_concurrency / mean_tpot / num_gpus
        ),
        "wall_output_tput_per_gpu": (
            total_output_tokens / wall_seconds / num_gpus
        ),
        "acceptance_rate": (
            total_accepted / total_drafted if total_drafted else 0.0
        ),
        "accepted_drafts_per_step": (
            total_accepted / total_decode_iterations
        ),
        "observed_tokens_per_step": (
            total_decode_tokens / total_decode_iterations
        ),
        "total_output_tokens": total_output_tokens,
        "total_decode_tokens": total_decode_tokens,
        "total_decode_iterations": total_decode_iterations,
        "total_accepted_draft_tokens": total_accepted,
        "total_proposed_draft_tokens": total_drafted,
    }


def summarize_pass(
    outputs: Iterable[Any],
    wall_seconds: float,
    expected_output_tokens: int,
    num_gpus: int,
) -> dict[str, Any]:
    requests = [
        extract_request_metrics(output, expected_output_tokens)
        for output in outputs
    ]
    return {
        "requests": requests,
        "aggregate": aggregate_requests(requests, wall_seconds, num_gpus),
    }


def aggregate_passes(
    passes: list[dict[str, Any]],
    num_gpus: int,
) -> dict[str, Any]:
    if not passes:
        raise ValueError("No measured passes to aggregate")
    concurrency = len(passes[0]["requests"])
    requests: list[dict[str, Any]] = []
    total_wall = 0.0
    for measured_pass in passes:
        if len(measured_pass["requests"]) != concurrency:
            raise ValueError("Measured passes have different request counts")
        requests.extend(measured_pass["requests"])
        total_wall += float(measured_pass["aggregate"]["wall_seconds"])
    aggregate = aggregate_requests(
        requests,
        total_wall,
        num_gpus,
        concurrency=concurrency,
    )
    aggregate["pass_count"] = len(passes)
    aggregate["per_pass_derived_output_tput_per_gpu"] = [
        item["aggregate"]["derived_output_tput_per_gpu"] for item in passes
    ]
    aggregate["per_pass_wall_output_tput_per_gpu"] = [
        item["aggregate"]["wall_output_tput_per_gpu"] for item in passes
    ]
    return aggregate


def huawei_comparison(
    concurrency: int,
    b300_output_tput_per_gpu: float,
    observed_tokens_per_step: float,
) -> dict[str, Any] | None:
    reference = HUAWEI_REFERENCE.get(concurrency)
    if reference is None:
        return None
    converted = (
        float(reference["step_tput_per_chip"]) * observed_tokens_per_step
    )
    return {
        **reference,
        "conversion": (
            "published_step_tput_per_chip * "
            "trt_observed_tokens_per_step"
        ),
        "estimated_token_tput_per_chip": converted,
        "b300_to_huawei_ratio": (
            b300_output_tput_per_gpu / converted if converted else None
        ),
        "normalized_estimate": True,
    }
