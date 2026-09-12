"""Bounded public audit metadata from retained power-validation sidecars."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any


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
    finite_gaps = [value for value in gaps.values()
                   if type(value) in (int, float) and math.isfinite(value)]
    if finite_gaps:
        fields["max_sample_gap_s"] = max(finite_gaps)
    for key, value in fields.items():
        if type(value) in (int, float) and math.isfinite(value) and value >= 0:
            audit[key] = value
    for target, original in (("producer_sha", "producer_git_commit"),
                             ("exporter_image_sha256", "exporter_image_sha256")):
        value = producer.get(original)
        if isinstance(value, str) and 0 < len(value) <= 128:
            audit[target] = value
    ids = validation.get("observed_gpu_ids")
    if ids is None:
        ids = list((validation.get("per_gpu_role") or {}).keys())
    if ids:
        audit["observed_gpu_ids"] = list(dict.fromkeys(
            str(value) for value in ids if 0 < len(str(value)) <= 128
        ))[:1024]
    return {
        "power_invalid_reasons": [reason for reason in validation.get("reasons", [])
                                  if isinstance(reason, str)
                                  and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", reason)][:32],
        "power_audit": audit,
    }
