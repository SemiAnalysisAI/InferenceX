"""Closed-loop serving measurements using the existing H3 request records."""

from __future__ import annotations

import math
import statistics


def settings(concurrency: int | None, deadline_seconds: float | None = None) -> dict | None:
    if concurrency is None:
        if deadline_seconds is not None:
            raise ValueError("delivery deadline requires --serving-concurrency")
        return None
    if type(concurrency) is not int or not 1 <= concurrency <= 32:
        raise ValueError("serving concurrency must be an integer in [1, 32]")
    if deadline_seconds is not None and (
        type(deadline_seconds) not in (int, float) or not math.isfinite(deadline_seconds)
        or not 0 < deadline_seconds <= 86400
    ):
        raise ValueError("delivery deadline must be finite and in (0, 86400] seconds")
    return {"mode": "closed_loop", "concurrency": concurrency,
            "delivery_deadline_seconds": deadline_seconds}


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_window(run: dict) -> int:
    """Verify wall time and observed client concurrency from request intervals."""
    load = run["configuration"].get("serving")
    if not isinstance(load, dict) or load != settings(load.get("concurrency"), load.get("delivery_deadline_seconds")):
        raise ValueError("invalid serving configuration")
    window = run["measurement"]
    start, end, wall = (window.get(key) for key in ("start_monotonic_seconds", "end_monotonic_seconds", "wall_seconds"))
    if (window.get("boundary") != "submit_to_downloaded_media" or window.get("concurrency") != load["concurrency"]
            or not all(_finite(value) for value in (start, end, wall)) or end <= start
            or abs(end - start - wall) > 1e-6):
        raise ValueError("invalid serving measurement window")
    events = []
    if not isinstance(run.get("records"), list):
        raise ValueError("serving records must be a list")
    for record in run["records"]:
        if not isinstance(record, dict):
            raise ValueError("serving request must be an object")
        if record["phase"] != "measurement" or not record["attempted"]:
            continue
        timing = record.get("timing_window", {})
        if not isinstance(timing, dict):
            raise ValueError("serving request lacks a transport window")
        begin, finish = timing.get("start_monotonic_seconds"), timing.get("transport_end_monotonic_seconds")
        if (not all(_finite(value) for value in (begin, finish)) or not start <= begin < finish <= end
                or record.get("submit_to_media_seconds") is not None and (
                    not _finite(record["submit_to_media_seconds"]) or record["submit_to_media_seconds"] < 0
                    or begin + record["submit_to_media_seconds"] > finish + 1e-6)):
            raise ValueError("request transport falls outside serving window")
        events.extend(((begin, 1), (finish, -1)))
    active = peak = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    if peak > load["concurrency"]:
        raise ValueError("observed requests exceed declared serving concurrency")
    return peak


def summarize(run: dict) -> dict:
    load = run["configuration"]["serving"]
    records = [r for r in run["records"] if r["phase"] == "measurement"]
    valid = [r for r in records if r["status"] == "succeeded" and (r.get("media") or {}).get("valid") is True]
    values = [r.get("submit_to_media_seconds") for r in valid]
    complete = bool(values) and all(_finite(value) and value >= 0 for value in values)
    values = sorted(values) if complete else []
    wall = run["measurement"]["wall_seconds"]
    deadline = load["delivery_deadline_seconds"]
    on_time = sum(value <= deadline for value in values) if deadline is not None and (complete or not valid) else None
    durations = [(r.get("media") or {}).get("video", {}).get("duration_seconds") for r in valid]
    seconds = sum(durations) if all(_finite(value) and value > 0 for value in durations) else None
    return {
        **load, "capacity_qualified": False,
        "client_ready_latency_seconds": {
            "values": values, "sample_count": len(values), "valid_clip_count": len(valid),
            "population": "technically valid measured requests; failures remain in completion counts",
            "p50": statistics.median(values) if values else None,
            "p90": values[math.ceil(len(values) * .9) - 1] if len(values) >= 10 else None,
            "p95": values[math.ceil(len(values) * .95) - 1] if len(values) >= 20 else None,
            "quantile_method": "nearest_rank for P90/P95; sample floors 10/20 are not statistical qualification",
        },
        "submitted": sum(r["attempted"] for r in records),
        "outcomes": {name: sum(r.get("outcome") == name for r in records) for name in (
            "completed", "invalid_media", "provider_failed", "provider_cancelled", "timed_out",
            "transport_error", "validation_error", "interrupted", "not_started")},
        "observed_submission_rate_per_second": sum(r["attempted"] for r in records) / wall if wall > 0 else None,
        "offered_request_rate_per_second": None,
        "peak_client_in_flight": validate_window(run),
        "valid_video_seconds_per_second": seconds / wall if seconds is not None and wall > 0 else None,
        "deadline_met_valid_clips": on_time,
        "deadline_attainment_fraction": on_time / len(records) if on_time is not None and records else None,
        "deadline_goodput_clips_per_second": on_time / wall if on_time is not None and wall > 0 else None,
        "queue_delay_seconds": None, "server_ready_latency_seconds": None,
        "observed_batch_sizes": None,
        "limitations": ["Closed-loop delivery load, not a fixed arrival-rate or sustainable-capacity test.",
                        "Local validation follows delivery and is excluded from the throughput window.",
                        "Client polling observations are not server-side queue or execution timestamps.",
                        "Deadline goodput requires technical validity, not calibrated perceptual quality."],
    }
