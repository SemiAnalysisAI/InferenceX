"""Huawei-style decode-round and request diagnostics for TRT offline runs."""

from __future__ import annotations

import hashlib
import math
import re
from datetime import timedelta
from statistics import fmean
from typing import Any, Iterable

from trt_config import (
    HUAWEI_MEASURED_DECODE_ROUNDS,
    MTP_DRAFT_TOKENS,
)


HUAWEI_REFERENCE: dict[int, dict[str, float | int]] = {
    16: {
        "global_batch_size": 16,
        "chips": 16,
        "decode_round_tpot_ms": 17.64,
        "decode_step_tput_per_chip": 56.70,
        "published_accepted_drafts_per_step": 1.44,
        "published_mtp_draft_tokens": 3,
    },
    64: {
        "global_batch_size": 64,
        "chips": 16,
        "decode_round_tpot_ms": 19.03,
        "decode_step_tput_per_chip": 210.16,
        "published_accepted_drafts_per_step": 1.44,
        "published_mtp_draft_tokens": 3,
    },
    128: {
        "global_batch_size": 128,
        "chips": 16,
        "decode_round_tpot_ms": 20.61,
        "decode_step_tput_per_chip": 388.23,
        "published_accepted_drafts_per_step": 1.44,
        "published_mtp_draft_tokens": 3,
    },
}
TRT_ITER_LOG_PATTERN = re.compile(
    r"\[RANK\s+(?P<rank>\d+)\].*?"
    r"\biter\s*=\s*(?P<iter>\d+).*?"
    r"host_step_time\s*=\s*(?P<host>[0-9.eE+-]+)ms,\s*"
    r"prev_device_step_time\s*=\s*(?P<device>[0-9.eE+-]+)ms"
)


def _seconds(value: Any) -> float:
    if isinstance(value, timedelta):
        return value.total_seconds()
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    return float(value)


def percentile(values: list[float], value: float) -> float:
    """Match NumPy's default linear percentile interpolation."""
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _token_sha256(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        if token_id < 0 or token_id > 0xFFFFFFFF:
            raise ValueError("Generated token IDs must fit in uint32")
        digest.update(token_id.to_bytes(4, "little"))
    return digest.hexdigest()


def _sequence_sha256(requests: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for request in requests:
        digest.update(bytes.fromhex(str(request["output_token_sha256"])))
    return digest.hexdigest()


def extract_request_metrics(
    request_output: Any,
    expected_output_tokens: int,
) -> dict[str, Any]:
    completions = getattr(request_output, "outputs", None)
    if not completions:
        raise RuntimeError("TRT request returned no completion outputs")
    completion = completions[0]
    raw_token_ids = getattr(completion, "token_ids", None)
    if raw_token_ids is None:
        raise RuntimeError("TRT completion did not return token IDs")
    token_ids = [int(token_id) for token_id in raw_token_ids]
    if len(token_ids) != expected_output_tokens:
        raise RuntimeError(
            f"TRT generated {len(token_ids)} tokens; expected "
            f"{expected_output_tokens}"
        )

    perf = getattr(completion, "request_perf_metrics", None)
    if perf is None:
        raise RuntimeError("TRT completion did not return request metrics")
    timing = perf.timing_metrics
    arrival = _seconds(timing.arrival_time)
    first_scheduled = _seconds(timing.first_scheduled_time)
    first_token = _seconds(timing.first_token_time)
    last_token = _seconds(timing.last_token_time)
    first_iter = int(perf.first_iter)
    last_iter = int(perf.last_iter)
    decode_iterations = last_iter - first_iter
    decode_tokens = len(token_ids) - 1
    if decode_iterations <= 0 or decode_tokens <= 0:
        raise RuntimeError(
            "Invalid TRT request telemetry: "
            f"decode_iterations={decode_iterations}, "
            f"decode_tokens={decode_tokens}"
        )
    return {
        "output_tokens": len(token_ids),
        "output_token_sha256": _token_sha256(token_ids),
        "decode_tokens": decode_tokens,
        "first_iter": first_iter,
        "last_iter": last_iter,
        "decode_iterations": decode_iterations,
        "ttft_s": first_token - arrival,
        "queue_s": first_scheduled - arrival,
        "decode_window_s": last_token - first_token,
        "e2e_s": last_token - arrival,
        "overall_observed_tokens_per_step": (
            decode_tokens / decode_iterations
        ),
    }


def summarize_requests(
    outputs: Iterable[Any],
    *,
    wall_seconds: float,
    expected_output_tokens: int,
    num_gpus: int,
) -> dict[str, Any]:
    if wall_seconds <= 0:
        raise ValueError(f"Invalid measured wall time: {wall_seconds}")
    requests = [
        extract_request_metrics(output, expected_output_tokens)
        for output in outputs
    ]
    if not requests:
        raise ValueError("No request outputs to summarize")
    ttfts = [float(item["ttft_s"]) for item in requests]
    e2els = [float(item["e2e_s"]) for item in requests]
    total_output_tokens = sum(int(item["output_tokens"]) for item in requests)
    total_decode_tokens = sum(int(item["decode_tokens"]) for item in requests)
    total_decode_iterations = sum(
        int(item["decode_iterations"]) for item in requests
    )
    return {
        "requests": requests,
        "aggregate": {
            "request_samples": len(requests),
            "output_sequence_sha256": _sequence_sha256(requests),
            "wall_seconds": wall_seconds,
            "wall_output_tput_per_gpu": (
                total_output_tokens / wall_seconds / num_gpus
            ),
            "mean_ttft_ms": fmean(ttfts) * 1000.0,
            "median_ttft_ms": percentile(ttfts, 50) * 1000.0,
            "p90_ttft_ms": percentile(ttfts, 90) * 1000.0,
            "p99_ttft_ms": percentile(ttfts, 99) * 1000.0,
            "mean_e2e_ms": fmean(e2els) * 1000.0,
            "median_e2e_ms": percentile(e2els, 50) * 1000.0,
            "p90_e2e_ms": percentile(e2els, 90) * 1000.0,
            "p99_e2e_ms": percentile(e2els, 99) * 1000.0,
            "overall_observed_tokens_per_step": (
                total_decode_tokens / total_decode_iterations
            ),
            "total_output_tokens": total_output_tokens,
            "total_decode_tokens": total_decode_tokens,
            "total_decode_iterations": total_decode_iterations,
        },
    }


def summarize_outputs(
    outputs: Iterable[Any],
    *,
    wall_seconds: float,
    expected_output_tokens: int,
    num_gpus: int,
) -> dict[str, Any]:
    """Summarize fixed-length outputs without request performance telemetry."""
    if wall_seconds <= 0:
        raise ValueError(f"Invalid measured wall time: {wall_seconds}")
    requests: list[dict[str, Any]] = []
    for request_output in outputs:
        completions = getattr(request_output, "outputs", None)
        if not completions:
            raise RuntimeError("TRT request returned no completion outputs")
        raw_token_ids = getattr(completions[0], "token_ids", None)
        if raw_token_ids is None:
            raise RuntimeError("TRT completion did not return token IDs")
        token_ids = [int(token_id) for token_id in raw_token_ids]
        if len(token_ids) != expected_output_tokens:
            raise RuntimeError(
                f"TRT generated {len(token_ids)} tokens; expected "
                f"{expected_output_tokens}"
            )
        requests.append(
            {
                "output_tokens": len(token_ids),
                "output_token_sha256": _token_sha256(token_ids),
            }
        )
    if not requests:
        raise ValueError("No request outputs to summarize")
    total_output_tokens = sum(
        int(item["output_tokens"]) for item in requests
    )
    return {
        "requests": requests,
        "aggregate": {
            "request_samples": len(requests),
            "output_sequence_sha256": _sequence_sha256(requests),
            "wall_seconds": wall_seconds,
            "wall_output_tput_per_gpu": (
                total_output_tokens / wall_seconds / num_gpus
            ),
            "total_output_tokens": total_output_tokens,
            "request_perf_metrics_collected": False,
        },
    }


def _iteration_fields(stat: dict[str, Any]) -> dict[str, Any]:
    inflight = stat.get("inflightBatchingStats") or {}
    spec = stat.get("specDecodingStats") or {}
    return {
        "iter": int(stat.get("iter", -1)),
        "latency_ms": float(stat.get("iterLatencyMS", 0.0)),
        "active": int(stat.get("numActiveRequests", 0)),
        "queued": int(stat.get("numQueuedRequests", 0)),
        "scheduled": int(inflight.get("numScheduledRequests", 0)),
        "context": int(inflight.get("numContextRequests", 0)),
        "generation": int(inflight.get("numGenRequests", 0)),
        "paused": int(inflight.get("numPausedRequests", 0)),
        "drafted": int(spec.get("numDraftTokens", 0)),
        "accepted": int(spec.get("numAcceptedTokens", 0)),
        "requests_with_drafts": int(
            spec.get("numRequestsWithDraftTokens", 0)
        ),
        "acceptance_length": float(spec.get("acceptanceLength", 0.0)),
    }


def _is_full_batch_decode(
    item: dict[str, Any],
    local_batch_size: int,
) -> bool:
    return (
        item["latency_ms"] > 0
        and item["context"] == 0
        and item["generation"] == local_batch_size
        and item["scheduled"] == local_batch_size
        and item["active"] == local_batch_size
        and item["queued"] == 0
        and item["paused"] == 0
    )


def _is_full_batch_prefill(
    item: dict[str, Any],
    local_batch_size: int,
) -> bool:
    return (
        item["latency_ms"] > 0
        and item["context"] == local_batch_size
        and item["generation"] == 0
        and item["scheduled"] == local_batch_size
        and item["active"] == local_batch_size
        and item["queued"] == 0
        and item["paused"] == 0
    )


def _is_inactive_prior_pass_tail(
    item: dict[str, Any],
    local_batch_size: int,
) -> bool:
    return (
        item["latency_ms"] > 0
        and item["context"] == 0
        and item["generation"] == local_batch_size
        and item["scheduled"] == local_batch_size
        and item["active"] == 0
        and item["queued"] == 0
        and item["paused"] == 0
    )


def select_full_batch_decode_rounds(
    iteration_stats: list[dict[str, Any]],
    *,
    local_batch_size: int,
    required_rounds: int,
    allow_staged_prefill: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one consecutive fixed-batch decode window or fail."""
    normalized = [_iteration_fields(stat) for stat in iteration_stats]
    raw_scheduled = [item for item in normalized if item["scheduled"] > 0]
    leading_inactive: list[dict[str, Any]] = []
    for item in raw_scheduled:
        if not _is_inactive_prior_pass_tail(item, local_batch_size):
            break
        leading_inactive.append(item)
    scheduled = raw_scheduled[len(leading_inactive) :]
    if allow_staged_prefill:
        runs: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        previous_iter: int | None = None
        for item in scheduled:
            is_consecutive = (
                previous_iter is None
                or item["iter"] == previous_iter + 1
            )
            if _is_full_batch_decode(item, local_batch_size):
                if not is_consecutive:
                    if current:
                        runs.append(current)
                    current = []
                current.append(item)
            elif current:
                runs.append(current)
                current = []
            previous_iter = item["iter"]
        if current:
            runs.append(current)
        usable = next(
            (run for run in runs if len(run) >= required_rounds),
            None,
        )
        if usable is None:
            longest = max((len(run) for run in runs), default=0)
            raise RuntimeError(
                "TRT did not produce enough consecutive full-batch "
                f"decode rounds after staged prefill: "
                f"{longest} < {required_rounds}"
            )
        selected = usable[:required_rounds]
        selected_start = scheduled.index(selected[0])
        setup = scheduled[:selected_start]
        first_generation = next(
            (
                item["iter"]
                for item in scheduled
                if item["generation"] > 0
            ),
            None,
        )
        diagnostics = {
            "schedule_mode": "staged_prefill_then_full_decode",
            "iteration_stats_count": len(iteration_stats),
            "scheduled_iterations": len(raw_scheduled),
            "pass_scheduled_iterations": len(scheduled),
            "leading_inactive_iterations_ignored": len(leading_inactive),
            "leading_inactive_first_iter": (
                leading_inactive[0]["iter"]
                if leading_inactive
                else None
            ),
            "leading_inactive_last_iter": (
                leading_inactive[-1]["iter"]
                if leading_inactive
                else None
            ),
            "setup_iterations": len(setup),
            "context_only_iterations": sum(
                item["context"] > 0 and item["generation"] == 0
                for item in setup
            ),
            "mixed_context_generation_iterations": sum(
                item["context"] > 0 and item["generation"] > 0
                for item in setup
            ),
            "partial_decode_iterations": sum(
                item["generation"] > 0
                and not _is_full_batch_decode(item, local_batch_size)
                for item in setup
            ),
            "full_batch_prefill_iterations": sum(
                _is_full_batch_prefill(item, local_batch_size)
                for item in setup
            ),
            "prefill_iter": None,
            "prefill_local_batch": None,
            "first_generation_iter": first_generation,
            "full_batch_decode_rounds_available": len(usable),
            "selected_first_iter": selected[0]["iter"],
            "selected_last_iter": selected[-1]["iter"],
            "selected_local_batch_min": min(
                item["generation"] for item in selected
            ),
            "selected_local_batch_max": max(
                item["generation"] for item in selected
            ),
        }
        return selected, diagnostics

    mixed = [
        item
        for item in scheduled
        if item["context"] > 0 and item["generation"] > 0
    ]
    if mixed:
        raise RuntimeError(
            "TRT mixed prefill and decode in the same iteration; the "
            "benchmark did not establish Huawei's prefill/decode barrier"
        )

    first_generation_index = next(
        (
            index
            for index, item in enumerate(scheduled)
            if item["generation"] > 0
        ),
        None,
    )
    if first_generation_index is None:
        raise RuntimeError("TRT iteration stats contain no decode rounds")
    prefill = scheduled[:first_generation_index]
    if (
        len(prefill) != 1
        or not _is_full_batch_prefill(prefill[0], local_batch_size)
    ):
        raise RuntimeError(
            "TRT did not execute one full-local-batch prefill iteration "
            f"before decode: {prefill}"
        )
    first_generation = scheduled[first_generation_index]
    if not _is_full_batch_decode(first_generation, local_batch_size):
        raise RuntimeError(
            "TRT began decode before the fixed local batch was active: "
            f"{first_generation}"
        )

    consecutive: list[dict[str, Any]] = []
    previous_iter: int | None = None
    for item in scheduled[first_generation_index:]:
        if not _is_full_batch_decode(item, local_batch_size):
            break
        if previous_iter is not None and item["iter"] != previous_iter + 1:
            break
        consecutive.append(item)
        previous_iter = item["iter"]
    if len(consecutive) < required_rounds:
        raise RuntimeError(
            "TRT did not produce enough consecutive full-batch decode "
            f"rounds: {len(consecutive)} < {required_rounds}"
        )

    selected = consecutive[:required_rounds]
    diagnostics = {
        "schedule_mode": "huawei_prefill_barrier",
        "iteration_stats_count": len(iteration_stats),
        "scheduled_iterations": len(raw_scheduled),
        "pass_scheduled_iterations": len(scheduled),
        "leading_inactive_iterations_ignored": len(leading_inactive),
        "leading_inactive_first_iter": (
            leading_inactive[0]["iter"] if leading_inactive else None
        ),
        "leading_inactive_last_iter": (
            leading_inactive[-1]["iter"] if leading_inactive else None
        ),
        "context_only_iterations": sum(
            item["context"] > 0 and item["generation"] == 0
            for item in scheduled
        ),
        "full_batch_prefill_iterations": 1,
        "prefill_iter": prefill[0]["iter"],
        "prefill_local_batch": prefill[0]["context"],
        "mixed_context_generation_iterations": len(mixed),
        "full_batch_decode_rounds_available": len(consecutive),
        "selected_first_iter": selected[0]["iter"],
        "selected_last_iter": selected[-1]["iter"],
        "selected_local_batch_min": min(
            item["generation"] for item in selected
        ),
        "selected_local_batch_max": max(
            item["generation"] for item in selected
        ),
    }
    return selected, diagnostics


def huawei_filter_round_latencies(
    latencies_ms: list[float],
) -> dict[str, Any]:
    """Apply CANN's first-round skip and upper-IQR outlier filter."""
    return filter_round_latencies(latencies_ms, skip_rounds=1)


def filter_round_latencies(
    latencies_ms: list[float],
    *,
    skip_rounds: int,
) -> dict[str, Any]:
    """Skip startup rounds, then remove only upper-IQR outliers."""
    if skip_rounds < 0:
        raise ValueError("Round skip count cannot be negative")
    if len(latencies_ms) <= skip_rounds:
        raise ValueError(
            f"Timing needs more than {skip_rounds} decode rounds"
        )
    after_skip = latencies_ms[skip_rounds:]
    q1 = percentile(after_skip, 25)
    q3 = percentile(after_skip, 75)
    upper_fence = q3 + 1.5 * (q3 - q1)
    retained = [
        latency
        for latency in after_skip
        if latency <= upper_fence
    ]
    if not retained:
        raise RuntimeError("Upper-IQR filtering removed every round")
    return {
        "first_round_skipped": skip_rounds > 0,
        "rounds_skipped": skip_rounds,
        "q1_ms": q1,
        "q3_ms": q3,
        "upper_iqr_fence_ms": upper_fence,
        "retained_rounds": len(retained),
        "outlier_rounds": len(after_skip) - len(retained),
        "retained_latencies_ms": retained,
        "mean_ms": fmean(retained),
        "median_ms": percentile(retained, 50),
        "p90_ms": percentile(retained, 90),
        "p99_ms": percentile(retained, 99),
    }


def summarize_decode_rounds(
    iteration_stats: list[dict[str, Any]],
    *,
    global_batch_size: int,
    local_batch_size: int,
    num_gpus: int,
    required_rounds: int = HUAWEI_MEASURED_DECODE_ROUNDS,
    mtp_draft_tokens: int = MTP_DRAFT_TOKENS,
    allow_staged_prefill: bool = False,
    latency_rounds_to_skip: int = 1,
    timing_source: str = "iter_latency_ms",
) -> dict[str, Any]:
    selected, diagnostics = select_full_batch_decode_rounds(
        iteration_stats,
        local_batch_size=local_batch_size,
        required_rounds=required_rounds,
        allow_staged_prefill=allow_staged_prefill,
    )
    latencies = [float(item["latency_ms"]) for item in selected]
    filtered = filter_round_latencies(
        latencies,
        skip_rounds=latency_rounds_to_skip,
    )
    decode_round_tpot_ms = float(filtered["mean_ms"])
    decode_step_tput_per_gpu = (
        global_batch_size / (decode_round_tpot_ms / 1000.0) / num_gpus
    )

    drafted = sum(int(item["drafted"]) for item in selected)
    accepted = sum(int(item["accepted"]) for item in selected)
    generation_slots = sum(int(item["generation"]) for item in selected)
    if drafted <= 0 or generation_slots <= 0:
        raise RuntimeError(
            "TRT iteration stats omitted speculative counters for the "
            "validated 256-round window"
        )
    accepted_drafts_per_step = accepted / generation_slots
    tokens_per_step = 1.0 + accepted_drafts_per_step
    acceptance_rate = accepted / drafted
    if not 1.0 <= tokens_per_step <= mtp_draft_tokens + 1.0:
        raise RuntimeError(
            f"Observed MTP token yield is invalid: {tokens_per_step}"
        )

    output_tput_per_gpu = decode_step_tput_per_gpu * tokens_per_step
    equivalent_output_tpot_ms = decode_round_tpot_ms / tokens_per_step
    return {
        "global_batch_size": global_batch_size,
        "local_batch_size": local_batch_size,
        "active_gpu_count": num_gpus,
        "measured_decode_rounds": required_rounds,
        "decode_round_tpot_ms": decode_round_tpot_ms,
        "median_decode_round_tpot_ms": float(filtered["median_ms"]),
        "p90_decode_round_tpot_ms": float(filtered["p90_ms"]),
        "p99_decode_round_tpot_ms": float(filtered["p99_ms"]),
        "decode_step_tput_per_gpu": decode_step_tput_per_gpu,
        "decode_step_tput_total": decode_step_tput_per_gpu * num_gpus,
        "observed_tokens_per_step": tokens_per_step,
        "accepted_drafts_per_step": accepted_drafts_per_step,
        "effective_acceptance_rate": (
            accepted_drafts_per_step / mtp_draft_tokens
        ),
        "raw_acceptance_rate": acceptance_rate,
        "raw_accepted_draft_tokens": accepted,
        "raw_proposed_draft_tokens": drafted,
        "raw_generation_slots": generation_slots,
        "token_yield_source": "iteration_spec_decoding_stats",
        "equivalent_output_tpot_ms": equivalent_output_tpot_ms,
        "output_tput_per_gpu": output_tput_per_gpu,
        "timing_source": timing_source,
        "filter": {
            key: value
            for key, value in filtered.items()
            if key != "retained_latencies_ms"
        },
        "schedule_validation": diagnostics,
        "selected_round_latencies_ms": latencies,
    }


def parse_trt_iteration_log(log_text: str) -> dict[int, dict[str, float]]:
    """Parse rank-0 TRT iteration timing rows."""
    parsed: dict[int, dict[str, float]] = {}
    for match in TRT_ITER_LOG_PATTERN.finditer(log_text):
        if int(match.group("rank")) != 0:
            continue
        iteration = int(match.group("iter"))
        row = {
            "host_step_time_ms": float(match.group("host")),
            "previous_device_step_time_ms": float(
                match.group("device")
            ),
        }
        # TRT resets iteration IDs for the measured executor pass after
        # engine warmup. The final occurrence is therefore authoritative.
        parsed[iteration] = row
    return parsed


def apply_trt_host_step_timing(
    aggregate: dict[str, Any],
    log_text: str,
    *,
    global_batch_size: int,
    num_gpus: int,
    skip_rounds: int,
) -> dict[str, Any]:
    """Replace overlap-invalid iterLatencyMS with TRT host-step timing."""
    schedule = aggregate["schedule_validation"]
    first_iter = int(schedule["selected_first_iter"])
    last_iter = int(schedule["selected_last_iter"])
    iteration_ids = list(range(first_iter, last_iter + 1))
    parsed = parse_trt_iteration_log(log_text)
    missing = [
        iteration
        for iteration in iteration_ids
        if iteration not in parsed
    ]
    if missing:
        raise RuntimeError(
            "TRT iteration log is missing selected host-step rows: "
            f"{missing[:12]}"
        )
    host_latencies = [
        parsed[iteration]["host_step_time_ms"]
        for iteration in iteration_ids
    ]
    filtered = filter_round_latencies(
        host_latencies,
        skip_rounds=skip_rounds,
    )
    decode_round_tpot_ms = float(filtered["mean_ms"])
    decode_step_tput_per_gpu = (
        global_batch_size / (decode_round_tpot_ms / 1000.0) / num_gpus
    )
    tokens_per_step = float(aggregate["observed_tokens_per_step"])
    updated = dict(aggregate)
    updated["stats_iter_latency_diagnostic"] = {
        "decode_round_tpot_ms": aggregate["decode_round_tpot_ms"],
        "median_decode_round_tpot_ms": aggregate[
            "median_decode_round_tpot_ms"
        ],
        "p90_decode_round_tpot_ms": aggregate[
            "p90_decode_round_tpot_ms"
        ],
        "p99_decode_round_tpot_ms": aggregate[
            "p99_decode_round_tpot_ms"
        ],
        "decode_step_tput_per_gpu": aggregate[
            "decode_step_tput_per_gpu"
        ],
        "timing_source": aggregate.get("timing_source"),
    }
    updated.update(
        {
            "decode_round_tpot_ms": decode_round_tpot_ms,
            "median_decode_round_tpot_ms": float(
                filtered["median_ms"]
            ),
            "p90_decode_round_tpot_ms": float(filtered["p90_ms"]),
            "p99_decode_round_tpot_ms": float(filtered["p99_ms"]),
            "decode_step_tput_per_gpu": decode_step_tput_per_gpu,
            "decode_step_tput_total": (
                decode_step_tput_per_gpu * num_gpus
            ),
            "equivalent_output_tpot_ms": (
                decode_round_tpot_ms / tokens_per_step
            ),
            "output_tput_per_gpu": (
                decode_step_tput_per_gpu * tokens_per_step
            ),
            "filter": {
                key: value
                for key, value in filtered.items()
                if key != "retained_latencies_ms"
            },
            "selected_round_latencies_ms": host_latencies,
            "timing_source": "trt_print_iter_log_host_step_time",
        }
    )
    device_latencies = [
        parsed[iteration + 1]["previous_device_step_time_ms"]
        for iteration in iteration_ids
        if iteration + 1 in parsed
    ]
    if len(device_latencies) > skip_rounds:
        device = filter_round_latencies(
            device_latencies,
            skip_rounds=skip_rounds,
        )
        updated["device_forward_timing_diagnostic"] = {
            "samples": len(device_latencies),
            "mean_ms": float(device["mean_ms"]),
            "median_ms": float(device["median_ms"]),
            "p90_ms": float(device["p90_ms"]),
            "p99_ms": float(device["p99_ms"]),
        }
    return updated


def aggregate_replicated_decode_rounds(
    replica_aggregates: list[dict[str, Any]],
    *,
    global_batch_size: int,
    num_gpus: int,
    skip_rounds: int,
    mtp_draft_tokens: int,
) -> dict[str, Any]:
    """Aggregate synchronized replica rounds as one fixed rack batch."""
    if not replica_aggregates:
        raise ValueError("At least one replica aggregate is required")
    replica_count = len(replica_aggregates)
    if global_batch_size % replica_count != 0:
        raise ValueError(
            f"Rack global batch {global_batch_size} is not divisible by "
            f"{replica_count} replicas"
        )
    if global_batch_size % num_gpus != 0:
        raise ValueError(
            f"Rack global batch {global_batch_size} is not divisible by "
            f"{num_gpus} GPUs"
        )

    round_vectors = [
        [float(value) for value in aggregate["selected_round_latencies_ms"]]
        for aggregate in replica_aggregates
    ]
    measured_rounds = int(
        replica_aggregates[0]["measured_decode_rounds"]
    )
    if any(len(values) != measured_rounds for values in round_vectors):
        raise RuntimeError(
            "Replica timing vectors do not all contain the validated "
            f"{measured_rounds} decode rounds"
        )
    if any(
        int(aggregate["measured_decode_rounds"]) != measured_rounds
        for aggregate in replica_aggregates
    ):
        raise RuntimeError("Replica measured-round counts do not match")

    rack_latencies = [
        max(values)
        for values in zip(*round_vectors, strict=True)
    ]
    filtered = filter_round_latencies(
        rack_latencies,
        skip_rounds=skip_rounds,
    )
    decode_round_tpot_ms = float(filtered["mean_ms"])
    decode_step_tput_per_gpu = (
        global_batch_size / (decode_round_tpot_ms / 1000.0) / num_gpus
    )

    drafted = sum(
        int(aggregate["raw_proposed_draft_tokens"])
        for aggregate in replica_aggregates
    )
    accepted = sum(
        int(aggregate["raw_accepted_draft_tokens"])
        for aggregate in replica_aggregates
    )
    generation_slots = sum(
        int(aggregate["raw_generation_slots"])
        for aggregate in replica_aggregates
    )
    if drafted <= 0 or generation_slots <= 0:
        raise RuntimeError(
            "Replica aggregates omitted speculative counters for the "
            "validated rack window"
        )
    accepted_drafts_per_step = accepted / generation_slots
    tokens_per_step = 1.0 + accepted_drafts_per_step
    if not 1.0 <= tokens_per_step <= mtp_draft_tokens + 1.0:
        raise RuntimeError(
            f"Observed rack MTP token yield is invalid: {tokens_per_step}"
        )
    output_tput_per_gpu = decode_step_tput_per_gpu * tokens_per_step

    return {
        "global_batch_size": global_batch_size,
        "engine_global_batch_size": global_batch_size // replica_count,
        "local_batch_size": global_batch_size // num_gpus,
        "active_gpu_count": num_gpus,
        "replica_count": replica_count,
        "measured_decode_rounds": measured_rounds,
        "decode_round_tpot_ms": decode_round_tpot_ms,
        "median_decode_round_tpot_ms": float(filtered["median_ms"]),
        "p90_decode_round_tpot_ms": float(filtered["p90_ms"]),
        "p99_decode_round_tpot_ms": float(filtered["p99_ms"]),
        "decode_step_tput_per_gpu": decode_step_tput_per_gpu,
        "decode_step_tput_total": decode_step_tput_per_gpu * num_gpus,
        "observed_tokens_per_step": tokens_per_step,
        "accepted_drafts_per_step": accepted_drafts_per_step,
        "effective_acceptance_rate": (
            accepted_drafts_per_step / mtp_draft_tokens
        ),
        "raw_acceptance_rate": accepted / drafted,
        "raw_accepted_draft_tokens": accepted,
        "raw_proposed_draft_tokens": drafted,
        "raw_generation_slots": generation_slots,
        "token_yield_source": (
            "summed_iteration_spec_decoding_stats_across_replicas"
        ),
        "equivalent_output_tpot_ms": (
            decode_round_tpot_ms / tokens_per_step
        ),
        "output_tput_per_gpu": output_tput_per_gpu,
        "timing_source": (
            "slowest_replica_trt_print_iter_log_host_step_time"
        ),
        "logical_round_aggregation": "maximum_replica_host_step_time",
        "filter": {
            key: value
            for key, value in filtered.items()
            if key != "retained_latencies_ms"
        },
        "selected_round_latencies_ms": rack_latencies,
        "replica_timing_summary": [
            {
                "decode_round_tpot_ms": float(
                    aggregate["decode_round_tpot_ms"]
                ),
                "output_tput_per_gpu": float(
                    aggregate["output_tput_per_gpu"]
                ),
                "observed_tokens_per_step": float(
                    aggregate["observed_tokens_per_step"]
                ),
            }
            for aggregate in replica_aggregates
        ],
        "schedule_validation": {
            "mode": "synchronized_replicated_fixed_global_batch",
            "replica_count": replica_count,
            "measured_decode_rounds": measured_rounds,
            "round_completion_rule": "slowest_replica",
            "replica_windows": [
                aggregate["schedule_validation"]
                for aggregate in replica_aggregates
            ],
        },
    }


def pr_reference_comparison(
    decode_rounds: dict[str, Any],
    *,
    profile_name: str,
    reference_concurrency: int,
    reference_active_global_batch: int,
    reference_prefill_gpu_count: int,
    reference_output_tput_per_decode_gpu: float,
    reference_output_tput_per_total_gpu: float,
    reference_recipe_url: str,
) -> dict[str, Any]:
    decode_gpus = int(decode_rounds["active_gpu_count"])
    measured = float(decode_rounds["output_tput_per_gpu"])
    total_output = measured * decode_gpus
    reference_total_gpus = decode_gpus + reference_prefill_gpu_count
    normalized_total = total_output / reference_total_gpus
    return {
        "profile": profile_name,
        "mode": "offline_decode_saturation_vs_disaggregated_serving",
        "reference_concurrency": reference_concurrency,
        "reference_active_global_batch": reference_active_global_batch,
        "reference_prefill_gpu_count": reference_prefill_gpu_count,
        "reference_decode_gpu_count": decode_gpus,
        "reference_total_gpu_count": reference_total_gpus,
        "reference_output_tput_per_decode_gpu": (
            reference_output_tput_per_decode_gpu
        ),
        "reference_output_tput_per_total_gpu": (
            reference_output_tput_per_total_gpu
        ),
        "measured_output_tput_per_decode_gpu": measured,
        "measured_output_tput_total": total_output,
        "measured_output_tput_per_reference_total_gpu": normalized_total,
        "offline_to_reference_decode_gpu_ratio": (
            measured / reference_output_tput_per_decode_gpu
        ),
        "offline_to_reference_total_gpu_ratio": (
            normalized_total / reference_output_tput_per_total_gpu
        ),
        "reference_recipe_url": reference_recipe_url,
        "comparison_note": (
            "The offline run copies the PR decode topology and kernels, "
            "but performs 8K prefill on the same decode GPUs before timing "
            "a saturated decode-only window. The total-fleet normalization "
            "uses the PR's prefill GPU count only as a comparison denominator."
        ),
    }


def huawei_comparison(
    global_batch_size: int,
    decode_rounds: dict[str, Any],
    *,
    hardware_key: str = "b300",
    hardware_label: str = "B300",
) -> dict[str, Any]:
    reference = HUAWEI_REFERENCE[global_batch_size]
    device_count_match = (
        int(decode_rounds["active_gpu_count"])
        == int(reference["chips"])
    )
    huawei_tokens_per_step = 1.0 + float(
        reference["published_accepted_drafts_per_step"]
    )
    huawei_output_tput = (
        float(reference["decode_step_tput_per_chip"])
        * huawei_tokens_per_step
    )
    step_tput = float(decode_rounds["decode_step_tput_per_gpu"])
    output_tput = float(decode_rounds["output_tput_per_gpu"])
    measured_tokens_per_step = float(
        decode_rounds["observed_tokens_per_step"]
    )
    huawei_same_yield_output_tput = (
        float(reference["decode_step_tput_per_chip"])
        * measured_tokens_per_step
    )
    active_gpu_count = int(decode_rounds["active_gpu_count"])
    local_batch = int(decode_rounds["local_batch_size"])
    prefix = hardware_key.lower().replace("-", "_")
    comparison = {
        **reference,
        "mode": "fixed_global_batch_offline_decode",
        "global_batch_match": True,
        "device_count_match": device_count_match,
        "hardware_topology_match": False,
        "hardware_key": hardware_key,
        "hardware_label": hardware_label,
        "active_gpu_count": active_gpu_count,
        "local_batch_size": local_batch,
        "huawei_local_batch_size": (
            global_batch_size / int(reference["chips"])
        ),
        "published_tokens_per_step": huawei_tokens_per_step,
        "published_output_tput_per_chip": huawei_output_tput,
        "decode_round_tpot_ms_measured": float(
            decode_rounds["decode_round_tpot_ms"]
        ),
        "decode_step_tput_per_gpu_measured": step_tput,
        "observed_tokens_per_step_measured": measured_tokens_per_step,
        "output_tput_per_gpu_measured": output_tput,
        "huawei_output_tput_per_chip_at_measured_tokens_per_step": (
            huawei_same_yield_output_tput
        ),
        "hardware_to_huawei_decode_step_ratio": (
            step_tput
            / float(reference["decode_step_tput_per_chip"])
        ),
        "hardware_to_huawei_same_yield_output_ratio": (
            output_tput / huawei_same_yield_output_tput
        ),
        "hardware_to_huawei_output_ratio": (
            output_tput / huawei_output_tput
        ),
        "comparison_note": (
            "The global batch, sequence length, MTP depth, warmup count, "
            "decode-round count, and timing filter match the Huawei code. "
            f"The measured hardware is {active_gpu_count} {hardware_label} "
            "GPUs with FP4, while Huawei publishes 16 950DT chips with "
            "hybrid MXFP8/MXFP4."
        ),
    }
    comparison.update(
        {
            f"{prefix}_active_gpu_count": active_gpu_count,
            f"{prefix}_local_batch_size": local_batch,
            f"{prefix}_decode_round_tpot_ms": comparison[
                "decode_round_tpot_ms_measured"
            ],
            f"{prefix}_decode_step_tput_per_gpu": step_tput,
            f"{prefix}_observed_tokens_per_step": comparison[
                "observed_tokens_per_step_measured"
            ],
            f"{prefix}_output_tput_per_gpu": output_tput,
            f"{prefix}_to_huawei_decode_step_ratio": comparison[
                "hardware_to_huawei_decode_step_ratio"
            ],
            f"{prefix}_to_huawei_same_yield_output_ratio": comparison[
                "hardware_to_huawei_same_yield_output_ratio"
            ],
            f"{prefix}_to_huawei_output_ratio": comparison[
                "hardware_to_huawei_output_ratio"
            ],
        }
    )
    return comparison


def huawei_scaled_local_batch_comparison(
    rack_global_batch_size: int,
    decode_rounds: dict[str, Any],
    *,
    hardware_key: str,
    hardware_label: str,
) -> dict[str, Any]:
    """Compare a full rack at the same requests-per-accelerator as Huawei."""
    active_gpu_count = int(decode_rounds["active_gpu_count"])
    local_batch = int(decode_rounds["local_batch_size"])
    matching = [
        reference
        for reference in HUAWEI_REFERENCE.values()
        if (
            int(reference["global_batch_size"])
            // int(reference["chips"])
            == local_batch
        )
    ]
    if len(matching) != 1:
        raise ValueError(
            "Huawei has no unique reference row for local batch "
            f"{local_batch}"
        )
    reference = matching[0]
    reference_global_batch = int(reference["global_batch_size"])
    huawei_tokens_per_step = 1.0 + float(
        reference["published_accepted_drafts_per_step"]
    )
    huawei_output_tput = (
        float(reference["decode_step_tput_per_chip"])
        * huawei_tokens_per_step
    )
    step_tput = float(decode_rounds["decode_step_tput_per_gpu"])
    output_tput = float(decode_rounds["output_tput_per_gpu"])
    measured_tokens_per_step = float(
        decode_rounds["observed_tokens_per_step"]
    )
    huawei_same_yield_output_tput = (
        float(reference["decode_step_tput_per_chip"])
        * measured_tokens_per_step
    )
    return {
        **reference,
        "mode": "scaled_global_batch_same_local_batch_offline_decode",
        "rack_global_batch_size": rack_global_batch_size,
        "reference_global_batch_size": reference_global_batch,
        "global_batch_match": (
            rack_global_batch_size == reference_global_batch
        ),
        "scaled_global_batch": True,
        "device_count_match": active_gpu_count == int(reference["chips"]),
        "local_batch_match": True,
        "hardware_topology_match": False,
        "hardware_key": hardware_key,
        "hardware_label": hardware_label,
        "active_gpu_count": active_gpu_count,
        "local_batch_size": local_batch,
        "huawei_local_batch_size": local_batch,
        "published_tokens_per_step": huawei_tokens_per_step,
        "published_output_tput_per_chip": huawei_output_tput,
        "decode_round_tpot_ms_measured": float(
            decode_rounds["decode_round_tpot_ms"]
        ),
        "decode_step_tput_per_gpu_measured": step_tput,
        "observed_tokens_per_step_measured": measured_tokens_per_step,
        "output_tput_per_gpu_measured": output_tput,
        "huawei_output_tput_per_chip_at_measured_tokens_per_step": (
            huawei_same_yield_output_tput
        ),
        "hardware_to_huawei_decode_step_ratio": (
            step_tput
            / float(reference["decode_step_tput_per_chip"])
        ),
        "hardware_to_huawei_same_yield_output_ratio": (
            output_tput / huawei_same_yield_output_tput
        ),
        "hardware_to_huawei_output_ratio": (
            output_tput / huawei_output_tput
        ),
        "comparison_note": (
            f"The rack GBS {rack_global_batch_size} scales Huawei GBS "
            f"{reference_global_batch} from 16 chips to "
            f"{active_gpu_count} GPUs, preserving local batch "
            f"{local_batch}. Sequence length, fixed-batch decode-window "
            "selection, and upper-IQR filtering match. The copied GB300 "
            "recipe uses MTP1 while Huawei publishes MTP3, so raw decode "
            "steps and observed output-token yield are both reported."
        ),
    }
