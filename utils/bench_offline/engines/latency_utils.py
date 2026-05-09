"""Helpers for extracting per-request latency samples from engine outputs."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, List, Optional


def to_seconds(value: Any) -> Optional[float]:
    """Convert common timestamp/duration objects to seconds."""
    if value is None:
        return None
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, datetime):
        return value.timestamp()
    if hasattr(value, "total_seconds"):
        try:
            return float(value.total_seconds())
        except (TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_metric(obj: Any, *names: str) -> Any:
    """Read a metric from dicts, enum-keyed dicts, or attribute objects."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
        for key, value in obj.items():
            key_names = [str(key)]
            key_value = getattr(key, "value", None)
            key_name = getattr(key, "name", None)
            if key_value is not None:
                key_names.append(str(key_value))
            if key_name is not None:
                key_names.append(str(key_name).lower())
            if any(name in key_names for name in names):
                return value
        return None
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def append_latency_sample(
    metrics: Any,
    n_out: int,
    ttfts: List[float],
    e2els: List[float],
    tpots: List[float],
) -> bool:
    """Append TTFT/E2E/TPOT samples from a framework metric object.

    Supports direct latency values such as ``ttft``/``tpot`` as well as
    timestamp-style metrics from vLLM and TensorRT-LLM.
    """
    if metrics is None:
        return False

    before = (len(ttfts), len(e2els), len(tpots))

    ttft = to_seconds(get_metric(
        metrics,
        "ttft",
        "time_to_first_token",
        "first_token_latency",
    ))
    e2e = to_seconds(get_metric(metrics, "e2e", "e2e_latency"))
    tpot = to_seconds(get_metric(metrics, "tpot", "time_per_output_token"))

    arrival = to_seconds(get_metric(metrics, "arrival_time"))
    first = to_seconds(get_metric(
        metrics,
        "first_token_time",
        "first_token_ts",
    ))
    finished = to_seconds(get_metric(
        metrics,
        "finished_time",
        "last_token_time",
        "last_token_ts",
    ))

    if ttft is None and arrival is not None and first is not None and first >= arrival:
        ttft = first - arrival
    decode_latency = None
    if first is not None and finished is not None and finished >= first:
        decode_latency = finished - first
    if e2e is None and arrival is not None and finished is not None and finished >= arrival:
        e2e = finished - arrival
    if e2e is None and ttft is not None and decode_latency is not None:
        e2e = ttft + decode_latency
    if tpot is None and decode_latency is not None and n_out > 1:
        tpot = decode_latency / (n_out - 1)

    if ttft is not None and ttft >= 0:
        ttfts.append(ttft)
    if e2e is not None and e2e >= 0:
        e2els.append(e2e)
    if tpot is not None and tpot > 0:
        tpots.append(tpot)

    return before != (len(ttfts), len(e2els), len(tpots))
