from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .cache_validity import modern_profile_error

_STEMS = ("full_tokens", "cached_tokens", "offload_tokens", "requests")


def _series(metrics: dict[str, Any], stem: str) -> tuple[dict, set]:
    names = [
        f"{prefix}prefix_cache_{stem}{suffix}"
        for prefix in ("atom:", "atom_")
        for suffix in ("", "_total")
    ]
    found, invalid = {}, set()
    for name in names:
        if name not in metrics:
            continue
        entry = metrics[name]
        if (
            not isinstance(entry, dict)
            or entry.get("type") != "counter"
            or not isinstance(entry.get("series"), list)
        ):
            invalid.add(None)
            continue
        for series in entry["series"]:
            if not isinstance(series, dict):
                invalid.add(None)
                continue
            endpoint = series.get("endpoint_url")
            labels = series.get("labels")
            labels = {} if labels is None else labels
            if (
                not isinstance(endpoint, str)
                or not endpoint
                or not isinstance(labels, dict)
                or any(not isinstance(k, str) or not isinstance(v, str) for k, v in labels.items())
            ):
                invalid.add(endpoint if isinstance(endpoint, str) else None)
                continue
            key = (endpoint, tuple(sorted(labels.items())))
            if key in found:
                invalid.add(endpoint)
            found[key] = series
    return found, invalid


def _total(series: dict[str, Any]) -> float | None:
    stats = series.get("stats")
    value = stats.get("total") if isinstance(stats, dict) else None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or value != int(value)
    ):
        return None
    return value


def account_cache(
    metrics: dict[str, Any],
    *,
    endpoint_roles: Mapping[str, str] | None,
    disagg: bool | None,
    kv_offloading: str | None,
    collector_export: Mapping[str, Any],
    request_scope: Mapping[str, Any] | None,
) -> tuple[tuple[float | None, float | None, float | None] | None, dict[str, Any]]:
    detail: dict[str, Any] = {"schema": "atom-cache-validity/1", "pricing_eligible": False}
    counts = None

    def finish(status: str, reason: str | None = None) -> tuple:
        detail.update(status=status, reasons=[reason] if reason else [])
        return counts, detail

    if reason := modern_profile_error(collector_export):
        return finish("WINDOW_UNRESOLVED", reason)
    counts = (None, None, None)
    series, invalid = {}, {}
    for stem in _STEMS:
        series[stem], invalid[stem] = _series(metrics, stem)
    observed = {key[0] for stem in _STEMS[:3] for key in series[stem]}
    if endpoint_roles is None and disagg is False and len(observed) == 1:
        accounted = observed
    elif endpoint_roles and not observed - endpoint_roles.keys():
        accounted = {url for url, role in endpoint_roles.items() if role in {"prefill", "unified"}}
    else:
        accounted = set()
    if not accounted:
        return finish("ROLE_OR_COVERAGE_UNRESOLVED", "cache endpoint roles unresolved")
    if reason := modern_profile_error(collector_export, accounted):
        counts = None
        return finish("WINDOW_UNRESOLVED", reason)
    series = {
        stem: {k: s for k, s in group.items() if k[0] in accounted}
        for stem, group in series.items()
    }
    invalid = {stem: bool(bad & {*accounted, None}) for stem, bad in invalid.items()}
    if invalid["full_tokens"] or invalid["cached_tokens"]:
        return finish("INVALID_COUNTER", "duplicate or malformed full/cached series")
    keys = set(series["full_tokens"])
    if not keys or accounted - {key[0] for key in keys} or set(series["cached_tokens"]) != keys:
        return finish("MISSING_OBSERVATION", "full/cached series coverage differs")
    values = {stem: {k: _total(s) for k, s in group.items()} for stem, group in series.items()}
    totals = {
        stem: sum(group.values()) if group and None not in group.values() else None
        for stem, group in values.items()
    }
    full, cached, offload, admissions = (totals[stem] for stem in _STEMS)
    if (
        full is None
        or not math.isfinite(full)
        or full <= 0
        or cached is None
        or not math.isfinite(cached)
        or cached > full
        or any(values["cached_tokens"][k] > values["full_tokens"][k] for k in keys)
    ):
        return finish("INVALID_COUNTER", "full/cached tokens outside their valid range")
    cpu_complete = set(series["offload_tokens"]) == keys
    cpu_invalid = invalid["offload_tokens"] or (bool(series["offload_tokens"]) and offload is None)
    if cpu_complete and offload is not None:
        cpu_invalid |= not math.isfinite(offload) or any(
            values["cached_tokens"][k] + values["offload_tokens"][k] > values["full_tokens"][k]
            for k in keys
        )
    if not series["offload_tokens"] and not cpu_invalid and kv_offloading == "none":
        offload = 0.0
    elif not cpu_complete or cpu_invalid:
        offload = None
    counts = (full, cached, offload)
    scope = dict(request_scope or {})
    valid_scope = all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
        and value == int(value)
        for value in (
            scope.get(k)
            for k in (
                "profiling_requests",
                "profiling_prompt_tokens",
                "error_records",
                "cancelled_records",
            )
        )
    )
    if set(series["requests"]) != keys or invalid["requests"]:
        admissions = None
    conserved = (
        valid_scope
        and admissions == scope.get("profiling_requests")
        and full == scope.get("profiling_prompt_tokens")
        and scope.get("error_records") == scope.get("cancelled_records") == 0
    )
    detail["request_scope"] = {
        "server_window_admissions": admissions,
        "server_window_prompt_tokens": full,
        **scope,
        "conserved": conserved,
    }
    if cpu_invalid:
        return finish("INVALID_COUNTER", "offload counters invalid or incompatible")
    if offload is None:
        return finish(
            "PARTIAL_COMPONENT", "CPU reuse missing without confirmed absence of CPU tier"
        )
    if not conserved:
        return finish(
            "REQUEST_SCOPE_UNRESOLVED", "counters do not match successful profiling requests"
        )
    detail["pricing_eligible"] = True
    return finish("VALID_ZERO" if cached + offload == 0 else "VALID_COMPLETE")
