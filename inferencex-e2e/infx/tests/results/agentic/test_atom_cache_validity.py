from __future__ import annotations

import json
from copy import deepcopy

import pytest

from infx.results.agentic.artifacts import load_records_with_accounting
from infx.results.agentic.server_metrics import compute_server_metrics, request_scope_from_records

P, D = "http://prefill/metrics", "http://decode/metrics"
ROLES = {P: "prefill", D: "decode"}
SCOPE = {
    "profiling_requests": 5,
    "profiling_prompt_tokens": 1000,
    "error_records": 0,
    "cancelled_records": 0,
}


def _export():
    return {
        "schema_version": "1.1",
        "aiperf_version": "1.0.12",
        "metrics_phase": "profiling",
        "summary": {
            "phase_time_ranges": {"profiling": {"start_ns": 1000, "end_ns": 4000}},
            "endpoints_configured": [P, D],
            "endpoints_successful": [P, D],
        },
        "metrics": {
            f"atom:prefix_cache_{stem}": {
                "type": "counter",
                "series": [
                    {"endpoint_url": url, "stats": {"total": total}}
                    for url, total in ((P, value), (D, 1000 if stem == "full_tokens" else 0))
                ],
            }
            for stem, value in (
                ("full_tokens", 1000),
                ("cached_tokens", 600),
                ("offload_tokens", 100),
                ("requests", 5),
            )
        },
    }


def _run(export, *, scope=None, offload="dram", roles=ROLES):
    flat, nested, _ = compute_server_metrics(
        export,
        framework="mooncake-atom",
        records=[],
        endpoint_roles=roles,
        request_scope=SCOPE if scope is None else scope,
        deployment={"disagg": roles == ROLES, "kv_offloading": offload},
    )
    assert flat["server_gpu_cache_hit_rate"] == nested["cache"]["gpu_cache_hit_rate"]
    return nested


def _rates(result, gpu, cpu, overall):
    for key, expected in (("gpu", gpu), ("cpu", cpu), ("external", cpu), ("overall", overall)):
        assert result["cache"][f"{key}_cache_hit_rate"] == (
            pytest.approx(expected) if expected is not None else None
        )


@pytest.mark.parametrize("unified", [False, True])
def test_modern_pd_and_unified_totals_match_the_successful_cohort(unified):
    export = _export()
    export["metrics"]["atom:lmcache_loaded_tokens"] = {
        "type": "counter", "series": [{"endpoint_url": P, "stats": {}}],
    }
    if unified:
        for entry in export["metrics"].values():
            entry["series"] = entry["series"][:1]
    export["metrics"]["atom:prompt_tokens"] = {
        "series": [{"endpoint_url": url, "stats": {"total": 1500}} for url in (P, D)]
    }
    result = _run(export, roles={P: "unified"} if unified else ROLES)
    _rates(result, 0.6, 0.1, 0.7)
    assert result["tokens"]["prompt_total"] == 1000
    assert result["tokens"]["prompt_by_source"] == {
        "gpu_cache_hit": 600,
        "cpu_or_external_cache_hit": 100,
        "computed": 300,
        "raw": {},
    }
    core = result["cache_validity"]
    assert core["schema"] == "atom-cache-validity/1"
    assert core["core_schema"] == "infx-cache-validity-core/1"
    assert core["status_class"] == "VALID_COMPLETE"
    assert core["published"] is True and core["basis"] == "tiers"
    assert core["pricing_eligible"] is True and core["request_scope"]["conserved"] is True


@pytest.mark.parametrize(
    ("cached", "offload", "status"), [(0, 0, "VALID_ZERO"), (0, 100, "VALID_COMPLETE")]
)
def test_true_zero_and_cpu_only_are_eligible(cached, offload, status):
    export = _export()
    for stem, value in (("cached_tokens", cached), ("offload_tokens", offload)):
        export["metrics"][f"atom:prefix_cache_{stem}"]["series"][0]["stats"]["total"] = value
    result = _run(export)
    _rates(result, 0, offload / 1000, offload / 1000)
    assert result["cache_validity"]["status_class"] == status
    assert result["cache_validity"]["pricing_eligible"] is True


@pytest.mark.parametrize(
    "proof", ["legacy", "version", "future", "phase", "endpoint", "window", "negative_window"]
)
def test_unproven_exports_keep_raw_telemetry_without_pricing_qualification(proof):
    export = _export()
    if proof == "legacy":
        del export["metrics_phase"]
        for entry in export["metrics"].values():
            for series in entry["series"]:
                series["timeslices"] = [
                    {"start_ns": 1000, "end_ns": 4000, "total": series["stats"]["total"]}
                ]
    elif proof in {"version", "future"}:
        export["aiperf_version"] = "1.0.11" if proof == "version" else "1.0.13"
    elif proof == "phase":
        export["metrics_phase"] = "warmup"
    elif proof == "endpoint":
        export["summary"]["endpoints_successful"] = [D]
    elif proof == "negative_window":
        export["summary"]["phase_time_ranges"]["profiling"]["start_ns"] = -1
    else:
        export["summary"]["phase_time_ranges"] = {}
    result = _run(export)
    _rates(result, 0.6, 0.1, 0.7)
    assert result["cache_validity"]["status_class"] == "UNRESOLVED"
    assert result["cache_validity"]["pricing_eligible"] is False


def test_distinct_label_domains_are_summed_but_duplicate_aliases_are_rejected():
    export = _export()
    for entry in export["metrics"].values():
        first, decode = entry["series"]
        entry["series"] = [
            {
                **first,
                "labels": {"engine": str(i)},
                "stats": {"total": first["stats"]["total"] * weight},
            }
            for i, weight in enumerate((0.4, 0.6))
        ] + [decode]
    assert _run(export)["cache_validity"]["pricing_eligible"] is True
    export["metrics"]["atom_prefix_cache_full_tokens_total"] = deepcopy(
        export["metrics"]["atom:prefix_cache_full_tokens"]
    )
    result = _run(export)
    _rates(result, None, None, None)
    assert result["cache_validity"]["status_class"] == "INVALID"


@pytest.mark.parametrize(
    ("offload", "total", "cpu", "status"),
    [
        ("none", None, 0, "VALID_COMPLETE"),
        ("dram", None, None, "PARTIAL_COMPONENT"),
        ("none", -1, None, "INVALID_COUNTER"),
        ("none", 100, None, "INVALID_COUNTER"),
    ],
)
def test_only_confirmed_absence_of_cpu_can_supply_a_missing_zero(offload, total, cpu, status):
    export = _export()
    if total is not None:
        export["metrics"]["atom:prefix_cache_offload_tokens"]["series"][0]["stats"]["total"] = total
    else:
        del export["metrics"]["atom:prefix_cache_offload_tokens"]
    result = _run(export, offload=offload)
    _rates(result, 0.6, cpu, 0.6 if cpu == 0 else None)
    assert result["cache_validity"]["status"] == status
    assert result["cache_validity"]["pricing_eligible"] is (cpu == 0)


@pytest.mark.parametrize(
    ("offload", "engine", "cpu", "status"),
    [
        ("none", "0", 0, "VALID_COMPLETE"),
        ("dram", "0", None, "PARTIAL_COMPONENT"),
        ("none", "extra", None, "INVALID_COUNTER"),
    ],
)
def test_missing_cpu_domains_require_a_disabled_tier_and_matching_present_domains(
    offload, engine, cpu, status
):
    export = _export()
    for stem, values in (
        ("full_tokens", (400, 600)),
        ("cached_tokens", (240, 360)),
        ("requests", (2, 3)),
    ):
        export["metrics"][f"atom:prefix_cache_{stem}"]["series"] = [
            {"endpoint_url": P, "labels": {"engine": str(i)}, "stats": {"total": value}}
            for i, value in enumerate(values)
        ]
    export["metrics"]["atom:prefix_cache_offload_tokens"]["series"] = [
        {"endpoint_url": P, "labels": {"engine": engine}, "stats": {"total": 0}}
    ]
    result = _run(export, offload=offload)
    _rates(result, 0.6, cpu, 0.6 if cpu == 0 else None)
    assert result["cache_validity"]["status"] == status
    assert result["cache_validity"]["pricing_eligible"] is (cpu == 0)


@pytest.mark.parametrize(
    ("stem", "value", "gpu"),
    [
        ("cached_tokens", None, None),
        ("cached_tokens", True, None),
        ("cached_tokens", "600", None),
        ("cached_tokens", float("inf"), None),
        ("offload_tokens", 500, 0.6),
    ],
)
def test_invalid_modern_counter_is_never_published_as_a_valid_rate(stem, value, gpu):
    export = _export()
    export["metrics"][f"atom:prefix_cache_{stem}"]["series"][0]["stats"]["total"] = value
    result = _run(export)
    _rates(result, gpu, None, None)
    assert result["cache_validity"]["status_class"] == "INVALID"
    assert result["cache_validity"]["pricing_eligible"] is False


@pytest.mark.parametrize(
    "problem", ["missing_requests", "requests", "tokens", "error", "cancel", "missing_cancel"]
)
def test_different_or_incomplete_cohorts_are_not_pricing_input(problem):
    export, scope = _export(), dict(SCOPE)
    if problem == "missing_requests":
        del export["metrics"]["atom:prefix_cache_requests"]
    elif problem == "missing_cancel":
        del scope["cancelled_records"]
    else:
        key, value = {
            "requests": ("profiling_requests", 4),
            "tokens": ("profiling_prompt_tokens", 999),
            "error": ("error_records", 1),
            "cancel": ("cancelled_records", 1),
        }[problem]
        scope[key] = value
    result = _run(export, scope=scope)
    _rates(result, 0.6, 0.1, 0.7)
    assert result["cache_validity"]["status"] == "REQUEST_SCOPE_UNRESOLVED"
    assert result["cache_validity"]["pricing_eligible"] is False


def test_warmup_errors_do_not_block_a_complete_profiling_cohort(tmp_path):
    records = [
        {"metadata": {"benchmark_phase": "warmup"}, "error": {"type": "Timeout"}},
        {
            "metadata": {"benchmark_phase": "profiling"},
            "metrics": {"usage_prompt_tokens": {"value": 1000}},
        },
    ]
    path = tmp_path / "profile_export.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records))
    successful, accounting = load_records_with_accounting(path)
    scope = request_scope_from_records(successful, accounting)
    export = _export()
    export["metrics"]["atom:prefix_cache_requests"]["series"][0]["stats"]["total"] = 1
    assert accounting["records_error_dropped"] == 1
    assert scope["error_records"] == 0
    assert _run(export, scope=scope)["cache_validity"]["pricing_eligible"] is True


@pytest.mark.parametrize(
    "usage", [(-10, 1010), (True, 999), (1.9, 999), (float("inf"), 1000), (None, 1000)]
)
def test_invalid_usage_cannot_become_a_successful_prompt_cohort(usage):
    records = [{"metrics": {"usage_prompt_tokens": {"value": value}}} for value in usage]
    scope = request_scope_from_records(records, {"records_profiling_error_dropped": 0})
    assert scope["profiling_prompt_tokens"] is None
    assert _run(_export(), scope=scope)["cache_validity"]["pricing_eligible"] is False


def test_conflicting_offload_metadata_cannot_prove_no_cpu_tier():
    export = _export()
    del export["metrics"]["atom:prefix_cache_offload_tokens"]
    _, result, _ = compute_server_metrics(
        export,
        framework="mooncake-atom",
        records=[],
        request_scope=SCOPE,
        env={
            "SRT_PREFILL_ENDPOINTS": "prefill",
            "SRT_DECODE_ENDPOINTS": "decode",
            "DISAGG": "true",
            "KV_OFFLOADING": " NONE ",
            "KV_OFFLOAD_BACKEND": "mooncake",
        },
    )
    _rates(result, 0.6, None, None)
    assert result["cache_validity"]["pricing_eligible"] is False


@pytest.mark.parametrize(("cached", "offload", "gpu"), [(150, 0, None), (50, 60, 0.25)])
def test_a_paired_counter_cannot_hide_in_another_label_domain(cached, offload, gpu):
    export = _export()
    for stem, values in (
        ("full_tokens", (100, 100)),
        ("cached_tokens", (cached, 0)),
        ("offload_tokens", (offload, 0)),
        ("requests", (1, 1)),
    ):
        export["metrics"][f"atom:prefix_cache_{stem}"]["series"] = [
            {"endpoint_url": P, "labels": {"engine": str(i)}, "stats": {"total": value}}
            for i, value in enumerate(values)
        ]
    result = _run(export, scope={**SCOPE, "profiling_requests": 2, "profiling_prompt_tokens": 200})
    _rates(result, gpu, None, None)
    assert result["cache_validity"]["status_class"] == "INVALID"


@pytest.mark.parametrize("labels", [None, [], {"rank": 0}])
def test_null_labels_are_unlabelled_but_malformed_labels_are_invalid(labels):
    export = _export()
    for entry in export["metrics"].values():
        entry["series"][0]["labels"] = labels
    result = _run(export)
    _rates(
        result,
        0.6 if labels is None else None,
        0.1 if labels is None else None,
        0.7 if labels is None else None,
    )
    assert result["cache_validity"]["pricing_eligible"] is (labels is None)


def test_fractional_admissions_cannot_cancel_into_an_integer_cohort():
    export = _export()
    for stem, values in (
        ("full_tokens", (500, 500)),
        ("cached_tokens", (300, 300)),
        ("offload_tokens", (50, 50)),
        ("requests", (2.5, 2.5)),
    ):
        export["metrics"][f"atom:prefix_cache_{stem}"]["series"] = [
            {"endpoint_url": P, "labels": {"engine": str(i)}, "stats": {"total": value}}
            for i, value in enumerate(values)
        ]
    result = _run(export)
    _rates(result, 0.6, 0.1, 0.7)
    assert result["cache_validity"]["status"] == "REQUEST_SCOPE_UNRESOLVED"
    assert result["cache_validity"]["pricing_eligible"] is False


def test_absent_metrics_publish_a_missing_ineligible_core():
    _, nested, _ = compute_server_metrics({}, framework="atom", records=[])
    core = nested["cache_validity"]
    assert core["schema"] == core["core_schema"] == "infx-cache-validity-core/1"
    assert core["status_class"] == "MISSING" and core["basis"] is None
    assert core["published"] is False and core["pricing_eligible"] is False


@pytest.mark.parametrize("problem", ["gauge", "malformed_series", "partial_requests"])
def test_non_counter_or_dropped_series_cannot_qualify_a_complete_export(problem):
    export = _export()
    if problem == "gauge":
        export["metrics"]["atom:prefix_cache_full_tokens"]["type"] = "gauge"
    elif problem == "malformed_series":
        export["metrics"]["atom:prefix_cache_full_tokens"]["series"].append(None)
    else:
        export["metrics"]["atom:prefix_cache_requests"]["series"].append(None)
    result = _run(export)
    assert result["cache_validity"]["pricing_eligible"] is False
    assert result["cache_validity"]["status"] == (
        "REQUEST_SCOPE_UNRESOLVED" if problem == "partial_requests" else "INVALID_COUNTER"
    )
