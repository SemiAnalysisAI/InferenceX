"""Bounded public audit metadata from retained power-validation sidecars."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from .cpu_side import HEADLINE_PREFERENCE

_REASON_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_CPU_COUNT_FIELDS = ("expected_sockets", "observed_sockets", "sample_row_count")


def _bounded_reasons(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [
        reason for reason in values if isinstance(reason, str) and _REASON_CODE.fullmatch(reason)
    ][:32]


def _cpu_audit(cpu: Mapping[str, Any]) -> dict[str, Any]:
    """Project the CPU-side provenance: sensor kind, source, counts, reasons."""
    summary: dict[str, Any] = {}
    kind = cpu.get("sensor_kind")
    if kind in HEADLINE_PREFERENCE:
        summary["sensor_kind"] = kind
    source = cpu.get("source")
    if isinstance(source, str) and 0 < len(source) <= 32:
        summary["source"] = source
    for key in _CPU_COUNT_FIELDS:
        value = cpu.get(key)
        if type(value) is int and value >= 0:
            summary[key] = value
    summary["reason_codes"] = _bounded_reasons(cpu.get("reason_codes"))
    return summary


def audit_summary(validation: Mapping[str, Any], source: str) -> dict[str, Any]:
    """Project the shared app audit contract without publishing raw telemetry."""
    audit: dict[str, Any] = {"source": source}
    window = validation.get("benchmark_window") or {}
    producer = validation.get("producer") or {}
    fields = {
        "window_start_unix": window.get("start_time_unix"),
        "window_end_unix": window.get("end_time_unix"),
        "expected_gpu_count": validation.get("expected_gpu_count"),
        "observed_gpu_count": validation.get("observed_gpu_count"),
    }
    counts = validation.get("per_gpu_sample_counts") or {}
    if counts:
        fields["sample_count"] = sum(counts.values())
    gaps = validation.get("per_gpu_max_sample_gap_s") or {}
    finite_gaps = [
        value for value in gaps.values() if type(value) in (int, float) and math.isfinite(value)
    ]
    if finite_gaps:
        fields["max_sample_gap_s"] = max(finite_gaps)
    audit.update(
        {
            key: value
            for key, value in fields.items()
            if type(value) in (int, float) and math.isfinite(value) and value >= 0
        }
    )
    for target, original in (
        ("producer_sha", "producer_git_commit"),
        ("exporter_image_sha256", "exporter_image_sha256"),
    ):
        value = producer.get(original)
        if isinstance(value, str) and 0 < len(value) <= 128:
            audit[target] = value
    ids = validation.get("observed_gpu_ids")
    if ids is None:
        ids = list((validation.get("per_gpu_role") or {}).keys())
    if ids:
        audit["observed_gpu_ids"] = list(
            dict.fromkeys(str(value) for value in ids if 0 < len(str(value)) <= 128)
        )[:1024]
    cpu = validation.get("cpu")
    if isinstance(cpu, Mapping):
        audit["cpu"] = _cpu_audit(cpu)
    return {
        "power_invalid_reasons": _bounded_reasons(validation.get("reasons", [])),
        "power_audit": audit,
    }
