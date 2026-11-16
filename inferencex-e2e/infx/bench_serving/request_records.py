"""Per-attempt records of one measured benchmark run, as equal-length columns."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .backend_request_func import RequestFuncOutput

KEY = "request_records"
SCHEMA_VERSION = 1
# The ``outcome`` column stores an index into this tuple; 0 is the only passing outcome.
OUTCOMES = (
    "completed",
    "http_status",
    "timeout",
    "connection",
    "stream_error",
    "empty_response",
    "other",
)
# Bits of the ``completion_evidence`` column: what a stream showed before it ended.
COMPLETION_EVIDENCE = {"done": 1, "finish_reason": 2, "usage": 4}
MS_DECIMALS = 3
TPOT_DEFINITION = "(e2el_ms - ttft_ms) / (output_len - 1); null when output_len < 2"
# time.perf_counter from just before the first measured dispatch until every attempt returned
DURATION_SOURCE = "client_perf_counter_first_dispatch_to_last_return"


def _ms(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds * 1000, MS_DECIMALS)


def _elapsed(output: RequestFuncOutput) -> float | None:
    if output.start_time is None or output.end_time is None:
        return None
    return output.end_time - output.start_time


def accounting_gaps(block: Mapping[str, Any]) -> list[str]:
    """Why ``block`` does not account for every requested attempt; empty when it does.

    Every column must hold one entry per dispatched attempt, every requested attempt must have
    been dispatched, and the counts must equal what the ``outcome`` column holds.
    """
    counts, columns = block["counts"], block["columns"]
    lengths = {len(values) for values in columns.values()}
    gaps = []
    if len(lengths) > 1:
        gaps.append("column_length_mismatch")
    if lengths != {counts["dispatched"]}:
        gaps.append("dispatched_count_mismatch")
    if counts["requested"] != counts["dispatched"]:
        gaps.append("requested_not_dispatched")
    by_outcome = Counter(columns["outcome"])
    if (
        counts["completed"] != by_outcome[0]
        or counts["completed"] + counts["failed"] != counts["dispatched"]
    ):
        gaps.append("outcome_count_mismatch")
    by_class = counts["failed_by_class"]
    if sum(by_class.values()) != counts["failed"] or any(
        by_class.get(name, 0) != by_outcome[index]
        for index, name in enumerate(block["outcomes"])
        if index
    ):
        gaps.append("failure_class_count_mismatch")
    if sum(counts["http_status"].values()) != by_class.get("http_status", 0):
        gaps.append("http_status_count_mismatch")
    return gaps


def build_request_records(
    outputs: Sequence[RequestFuncOutput],
    output_lens: Sequence[int],
    *,
    requested: int,
    warmups: int,
    max_concurrency: int | None,
    request_rate: float,
    benchmark_start_time: float,
    benchmark_start_time_unix: float,
    benchmark_end_time_unix: float,
    duration: float,
) -> dict[str, Any]:
    """One entry per measured attempt of a run, failed ones included, plus the run's accounting.

    ``output_lens`` holds each completed attempt's output token count. ``benchmark_start_time`` is
    the ``time.perf_counter`` reading that ``duration`` and each ``start_time`` are measured from.
    A failed attempt keeps the TTFT it observed, and its ``e2el_ms`` is the time until it failed.
    ``accounting`` states whether the block accounts for every requested attempt.
    """
    columns: dict[str, list[Any]] = {
        name: []
        for name in (
            "start_offset_ms",
            "outcome",
            "ttft_ms",
            "tpot_ms",
            "e2el_ms",
            "output_len",
            "input_len",
            "completion_evidence",
        )
    }
    http_status: Counter[str] = Counter()
    for index, output in enumerate(outputs):
        output_len = output_lens[index] if index < len(output_lens) else None
        tpot = None
        if output.success:
            outcome = 0
            e2el: float | None = output.latency
            if output_len is not None and output_len > 1:
                tpot = (output.latency - output.ttft) / (output_len - 1)
        else:
            failure = output.failure_class if output.failure_class in OUTCOMES[1:] else "other"
            outcome = OUTCOMES.index(failure)
            e2el = _elapsed(output)
            if output.http_status is not None:
                http_status[str(output.http_status)] += 1
        columns["start_offset_ms"].append(
            None if output.start_time is None else _ms(output.start_time - benchmark_start_time)
        )
        columns["outcome"].append(outcome)
        columns["ttft_ms"].append(_ms(output.ttft) if output.ttft > 0 else None)
        columns["tpot_ms"].append(_ms(tpot))
        columns["e2el_ms"].append(_ms(e2el))
        if index < len(output_lens):
            columns["output_len"].append(output_len if output.success else None)
        columns["input_len"].append(output.prompt_len)
        columns["completion_evidence"].append(output.completion_evidence)

    by_outcome = Counter(columns["outcome"])
    block = {
        "schema_version": SCHEMA_VERSION,
        "counts": {
            "requested": requested,
            "dispatched": len(outputs),
            "completed": by_outcome[0],
            "failed": len(outputs) - by_outcome[0],
            "failed_by_class": {
                name: by_outcome[index] for index, name in enumerate(OUTCOMES) if index
            },
            "http_status": dict(sorted(http_status.items())),
            "warmup_excluded": warmups,
            "retries": 0,
        },
        "accounting": {},
        "window": {
            "benchmark_start_time_unix": benchmark_start_time_unix,
            "benchmark_end_time_unix": benchmark_end_time_unix,
            "duration": duration,
            "duration_source": DURATION_SOURCE,
        },
        "load": {
            "max_concurrency": max_concurrency,
            "max_concurrency_kind": "request_concurrency_limit",
            "request_rate": request_rate if request_rate < float("inf") else "inf",
        },
        "tpot_definition": TPOT_DEFINITION,
        "ms_decimals": MS_DECIMALS,
        "outcomes": list(OUTCOMES),
        "completion_evidence_bits": dict(COMPLETION_EVIDENCE),
        "columns": columns,
    }
    gaps = accounting_gaps(block)
    block["accounting"].update(complete=not gaps, reasons=gaps)
    return block
