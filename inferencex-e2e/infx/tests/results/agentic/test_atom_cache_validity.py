from __future__ import annotations

import json
from copy import deepcopy

import pytest

from infx.results.agentic.artifacts import load_records_with_accounting
from infx.results.agentic.server_metrics import compute_server_metrics, request_scope_from_records

P = "http://prefill/metrics"
D = "http://decode/metrics"
ROLES = {P: "prefill", D: "decode"}
SCOPE = {
    "profiling_requests": 5,
    "profiling_prompt_tokens": 1000,
    "error_records": 0,
    "cancelled_records": 0,
}


def _export(*, modern=False):
    metrics = {}
    for stem, value in (
        ("prefix_cache_full_tokens", 1000),
        ("prefix_cache_cached_tokens", 600),
        ("prefix_cache_offload_tokens", 100),
        ("prefix_cache_requests", 5),
        ("lmcache_loaded_tokens", 900),
    ):
        metrics[f"atom:{stem}"] = {
            "series": [
                {
                    "endpoint_url": url,
                    "stats": {"total": total},
                    "timeslices": [
                        {"start_ns": 0, "end_ns": 1000, "total": 0},
                        {"start_ns": 1000, "end_ns": 4000, "total": total},
                    ],
                }
                for url, total in (
                    (P, value),
                    (D, 1000 if stem == "prefix_cache_full_tokens" else 0),
                )
            ]
        }
    export = {
        "metrics": metrics,
        "summary": {
            "phase_time_ranges": {"profiling": {"start_ns": 1000, "end_ns": 4000}},
            "endpoints_configured": [P, D],
            "endpoints_successful": [P, D],
        },
    }
    if modern:
        export.update(schema_version="1.1", aiperf_version="1.0.12", metrics_phase="profiling")
        for entry in metrics.values():
            for series in entry["series"]:
                del series["timeslices"]
    return export


def _run(export, *, scope=None, offload="dram"):
    flat, nested, _ = compute_server_metrics(
        export,
        framework="mooncake-atom",
        records=[],
        endpoint_roles=ROLES,
        request_scope=SCOPE if scope is None else scope,
        deployment={"disagg": True, "kv_offloading": offload},
    )
    assert flat["server_gpu_cache_hit_rate"] == nested["cache"]["gpu_cache_hit_rate"]
    return nested


def _rates(result, gpu, cpu, overall):
    for key, expected in (("gpu", gpu), ("cpu", cpu), ("external", cpu), ("overall", overall)):
        assert result["cache"][f"{key}_cache_hit_rate"] == (
            pytest.approx(expected) if expected is not None else None
        )


@pytest.mark.parametrize("modern", [False, True])
def test_qualified_pd_preserves_the_old_detail_and_prompt_sources(modern):
    export = _export(modern=modern)
    export["metrics"]["atom:prompt_tokens"] = {
        "series": [{"endpoint_url": url, "stats": {"total": 1500}} for url in (P, D)]
    }
    result = _run(export)
    _rates(result, 0.6, 0.1, 0.7)
    assert result["tokens"]["prompt_total"] == 1000
    detail = result["cache_validity"]
    assert detail["schema"] == "atom-cache-validity/1"
    assert detail["status"] == "VALID_COMPLETE" and detail["pricing_eligible"] is True
    assert detail["accounted_endpoints"] == [P]
    assert detail["request_scope"]["conserved"] is True
    assert result["tokens"]["prompt_by_source"] == {
        "gpu_cache_hit": 600,
        "cpu_or_external_cache_hit": 100,
        "computed": 300,
        "raw": {
            "atom:prefix_cache_full_tokens": 1000,
            "atom:prefix_cache_cached_tokens": 600,
            "atom:prefix_cache_offload_tokens": 100,
            "atom:lmcache_loaded_tokens": 900,
            "atom:prefix_cache_requests": 5,
        },
    }


@pytest.mark.parametrize(
    ("cached", "offload", "status"), [(0, 0, "VALID_ZERO"), (0, 100, "VALID_COMPLETE")]
)
def test_true_zero_and_cpu_only_are_eligible(cached, offload, status):
    export = _export(modern=True)
    for stem, value in (("cached", cached), ("offload", offload)):
        export["metrics"][f"atom:prefix_cache_{stem}_tokens"]["series"][0]["stats"]["total"] = value
    result = _run(export)
    _rates(result, 0, offload / 1000, offload / 1000)
    assert result["cache_validity"]["status"] == status
    assert result["cache_validity"]["pricing_eligible"] is True


@pytest.mark.parametrize("missing", ["window", "slices", "gap"])
def test_unproven_legacy_window_keeps_raw_telemetry_but_cannot_be_priced(missing):
    export = _export()
    series = export["metrics"]["atom:prefix_cache_full_tokens"]["series"][0]
    if missing == "window":
        del export["summary"]
    elif missing == "slices":
        del series["timeslices"]
    else:
        series["timeslices"][1]["start_ns"] = 1001
    result = _run(export)
    _rates(result, 0.6, 0.1, 0.7)
    assert result["cache_validity"]["status"] == "WINDOW_UNRESOLVED"
    assert result["cache_validity"]["observed_scope"] is None
    assert result["cache_validity"]["pricing_eligible"] is False


@pytest.mark.parametrize("proof", ["version", "phase", "endpoint", "window"])
def test_modern_header_alone_does_not_prove_a_window(proof):
    export = _export(modern=True)
    if proof == "version":
        export["aiperf_version"] = "1.0.11"
    elif proof == "phase":
        export["metrics_phase"] = "warmup"
    elif proof == "endpoint":
        export["summary"]["endpoints_successful"] = [D]
    else:
        export["summary"]["phase_time_ranges"] = {}
    result = _run(export)
    _rates(result, 0.6, 0.1, 0.7)
    assert result["cache_validity"]["status"] == "WINDOW_UNRESOLVED"
    assert result["cache_validity"]["pricing_eligible"] is False


@pytest.mark.parametrize("warmup_proof", [False, True])
def test_positive_start_slice_requires_matching_warmup_counters(warmup_proof):
    export = _export()
    warmup = {}
    for stem, value in (("full", 100), ("cached", 50), ("offload", 10)):
        name = f"atom:prefix_cache_{stem}_tokens"
        series = export["metrics"][name]["series"][0]
        series["stats"]["total"] += value
        series["timeslices"][0].update(end_ns=1500, total=value)
        series["timeslices"][1]["start_ns"] = 1500
        warmup[name] = {"series": [{"endpoint_url": P, "stats": {"total": value}}]}
    if warmup_proof:
        export["warmup_metrics"] = warmup
    result = _run(export)
    _rates(result, 0.6, 0.1, 0.7)
    assert result["cache_validity"]["pricing_eligible"] is warmup_proof
    assert result["cache_validity"]["status"] == (
        "VALID_COMPLETE" if warmup_proof else "WINDOW_UNRESOLVED"
    )


@pytest.mark.parametrize("problem", ["missing", "gap", "invalid", "duplicate"])
def test_transfer_diagnostics_never_veto_complete_cache_counters(problem):
    export = _export()
    entry = export["metrics"]["atom:lmcache_loaded_tokens"]
    if problem == "missing":
        del export["metrics"]["atom:lmcache_loaded_tokens"]
    elif problem == "gap":
        entry["series"][0]["timeslices"][1]["start_ns"] = 1001
    elif problem == "invalid":
        entry["series"][0]["stats"]["total"] = -1
    else:
        export["metrics"]["atom_lmcache_loaded_tokens_total"] = deepcopy(entry)
    result = _run(export)
    _rates(result, 0.6, 0.1, 0.7)
    assert result["cache_validity"]["pricing_eligible"] is True
    assert "atom:lmcache_loaded_tokens" not in result["tokens"]["prompt_by_source"]["raw"]


def test_distinct_label_domains_are_summed_but_duplicate_aliases_are_rejected():
    export = _export(modern=True)
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
    assert result["cache_validity"]["status"] == "INVALID_COUNTER"


@pytest.mark.parametrize("modern", [False, True])
def test_confirmed_no_cpu_matches_measured_zero_but_invalid_cpu_is_not_zero(modern):
    export = _export(modern=modern)
    entry = export["metrics"].pop("atom:prefix_cache_offload_tokens")
    result = _run(export, offload="none")
    _rates(result, 0.6, 0, 0.6)
    assert result["cache_validity"]["cpu_component"] == "NO_CPU_TIER_CONFIRMED"
    assert result["cache_validity"]["pricing_eligible"] is True
    entry["series"][0]["stats"]["total"] = -1
    export["metrics"]["atom:prefix_cache_offload_tokens"] = entry
    result = _run(export, offload="none")
    _rates(result, 0.6, None, None)
    assert result["cache_validity"]["status"] == "INVALID_COUNTER"


@pytest.mark.parametrize(
    "problem", ["missing_total", "overflow", "missing_requests", "scope", "error", "cancel"]
)
def test_invalid_or_different_cohorts_never_become_pricing_input(problem):
    export = _export(modern=True)
    scope = dict(SCOPE)
    if problem == "missing_total":
        del export["metrics"]["atom:prefix_cache_cached_tokens"]["series"][0]["stats"]["total"]
    elif problem == "overflow":
        export["metrics"]["atom:prefix_cache_offload_tokens"]["series"][0]["stats"]["total"] = 500
    elif problem == "missing_requests":
        del export["metrics"]["atom:prefix_cache_requests"]
    elif problem == "scope":
        scope["profiling_prompt_tokens"] = 999
    else:
        scope["error_records" if problem == "error" else "cancelled_records"] = 1
    result = _run(export, scope=scope)
    _rates(
        result,
        None if problem == "missing_total" else 0.6,
        None if problem in {"missing_total", "overflow"} else 0.1,
        None if problem in {"missing_total", "overflow"} else 0.7,
    )
    assert result["cache_validity"]["pricing_eligible"] is False
    assert result["cache_validity"]["status"] == (
        "INVALID_COUNTER"
        if problem in {"missing_total", "overflow"}
        else "REQUEST_SCOPE_UNRESOLVED"
    )


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
    assert accounting["records_error_dropped"] == 1
    assert accounting["records_profiling_error_dropped"] == 0
    assert scope == {
        "profiling_requests": 1,
        "profiling_prompt_tokens": 1000,
        "error_records": 0,
        "cancelled_records": 0,
    }
    successful[0]["metrics"] = {}
    assert request_scope_from_records(successful, accounting)["profiling_prompt_tokens"] is None


@pytest.mark.parametrize(
    "usage", [(-10, 1010), (True, 999), (1.9, 999), (float("inf"), 1000), (None, 1000)]
)
def test_invalid_usage_cannot_cancel_out_or_be_truncated_into_a_pricing_cohort(usage):
    records = [{"metrics": {"usage_prompt_tokens": {"value": value}}} for value in usage]
    scope = request_scope_from_records(records, {"records_profiling_error_dropped": 0})
    assert scope["profiling_prompt_tokens"] is None
    export = _export(modern=True)
    export["metrics"]["atom:prefix_cache_requests"]["series"][0]["stats"]["total"] = 2
    result = _run(export, scope=scope)
    _rates(result, 0.6, 0.1, 0.7)
    assert result["cache_validity"]["status"] == "REQUEST_SCOPE_UNRESOLVED"
    assert result["cache_validity"]["pricing_eligible"] is False


def test_no_cpu_confirmation_cannot_bypass_conflicting_backend_metadata():
    export = _export(modern=True)
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
    assert result["cache_validity"]["cpu_component"] == "MISSING"
    assert result["cache_validity"]["pricing_eligible"] is False


@pytest.mark.parametrize(("cached", "offload", "gpu"), [(150, 0, None), (50, 60, 0.25)])
def test_a_paired_counter_cannot_hide_behind_another_label_domain(cached, offload, gpu):
    export = _export(modern=True)
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
    assert result["cache_validity"]["status"] == "INVALID_COUNTER"
    assert result["cache_validity"]["pricing_eligible"] is False


@pytest.mark.parametrize("modern", [False, True])
def test_only_legacy_callers_may_omit_the_cancelled_count(modern):
    scope = {key: value for key, value in SCOPE.items() if key != "cancelled_records"}
    result = _run(_export(modern=modern), scope=scope)
    assert result["cache_validity"]["pricing_eligible"] is (not modern)
    for key, value in (
        ("profiling_requests", True),
        ("error_records", False),
        ("cancelled_records", None),
        ("profiling_prompt_tokens", 1000.1),
    ):
        result = _run(_export(modern=modern), scope={**SCOPE, key: value})
        assert result["cache_validity"]["status"] == "REQUEST_SCOPE_UNRESOLVED"


@pytest.mark.parametrize("labels", [None, []])
def test_null_labels_are_unlabelled_but_malformed_labels_remain_invalid(labels):
    export = _export(modern=True)
    export["metrics"]["atom:prefix_cache_cached_tokens"]["series"][0]["labels"] = labels
    result = _run(export)
    if labels is None:
        _rates(result, 0.6, 0.1, 0.7)
        assert result["cache_validity"]["pricing_eligible"] is True
    else:
        _rates(result, None, None, None)
        assert result["cache_validity"]["status"] == "INVALID_COUNTER"
        assert result["cache_validity"]["pricing_eligible"] is False
