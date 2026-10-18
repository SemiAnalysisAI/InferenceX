from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from infx.bench.agentic.replay import _server_metrics_urls
from infx.results.agentic.server_metrics import compute_server_metrics, endpoint_roles_from_env
from infx.tests.results.agentic.test_process_agentic_result import (
    _assert_stable_server_metrics_schema,
    _run_processor,
    _write_fixture,
)

PREFILL = "http://10.0.0.1:6100/metrics"
DECODE = "http://10.0.0.2:6100/metrics"
ROLES = {PREFILL: "prefill", DECODE: "decode"}
WINDOW = (1_000, 4_000)
# warmup | profiling | profiling | tail: no slice straddles the window edges.
EDGES = ((0, 1_000), (1_000, 2_500), (2_500, 4_000), (4_000, 5_000))
ZERO = (0.0, 0.0, 0.0, 0.0)
SCOPE = {"profiling_requests": 5, "profiling_prompt_tokens": 1000, "error_records": 0}
PD = {"disagg": True, "kv_offloading": "dram"}
RATE_KEYS = (
    "gpu_cache_hit_rate",
    "cpu_cache_hit_rate",
    "external_cache_hit_rate",
    "overall_cache_hit_rate",
)
NO_RATES = (None, None, None, None)


def _series(url: str, values: tuple, edges: tuple = EDGES) -> dict:
    slices = [{"start_ns": lo, "end_ns": hi, "total": v} for (lo, hi), v in zip(edges, values)]
    return {"endpoint_url": url, "stats": {"total": sum(values)}, "timeslices": slices}


def _export(counters: dict[str, dict[str, tuple]], edges: tuple = EDGES, window=WINDOW) -> dict:
    metrics = {
        f"atom:{stem}": {"type": "counter", "series": [_series(u, v, edges) for u, v in ep.items()]}
        for stem, ep in counters.items()
    }
    export: dict = {"metrics": metrics}
    if window is not None:
        profiling = {"start_ns": window[0], "end_ns": window[1]}
        export["summary"] = {"phase_time_ranges": {"profiling": profiling}}
    return export


def _pd(offload: tuple | None = (0, 50, 50, 0), **override: tuple) -> dict:
    """Interior prefill: full 1000, cached 600, offload 100, loaded 150, 5 admissions."""
    prefill = {
        "prefix_cache_full_tokens": (0, 400, 600, 0),
        "prefix_cache_cached_tokens": (0, 200, 400, 0),
        "prefix_cache_offload_tokens": offload,
        "lmcache_loaded_tokens": (0, 70, 80, 0),
        "prefix_cache_requests": (0, 3, 2, 0),
    } | override
    decode = {"prefix_cache_full_tokens": (0, 500, 650, 0), "prefix_cache_requests": (0, 3, 2, 0)}
    return _export(
        {
            stem: {PREFILL: values, DECODE: decode.get(stem, ZERO)}
            for stem, values in prefill.items()
            if values is not None
        }
    )


def _run(export: dict, *, roles=ROLES, scope=SCOPE, deployment=PD, framework="atom-disagg"):
    _, nested, warnings = compute_server_metrics(
        export,
        framework=framework,
        records=[],
        endpoint_roles=roles,
        request_scope=scope,
        deployment=deployment,
    )
    return nested, warnings


def _rates(nested: dict) -> tuple:
    return tuple(nested["cache"][key] for key in RATE_KEYS)


def test_complete_pd_observation_is_eligible_and_ignores_decode_and_load_volume() -> None:
    nested, warnings = _run(_pd())
    validity = nested["cache_validity"]

    assert _rates(nested) == (0.6, 0.1, 0.1, pytest.approx(0.7))
    assert nested["cache"]["prefix_cache_queries"] == 1000
    assert nested["tokens"]["prompt_by_source"]["raw"]["atom:lmcache_loaded_tokens"] == 150
    assert validity["status"] == "VALID_COMPLETE"
    assert validity["pricing_eligible"] is True
    assert validity["observed_scope"] == "profiling_window_interior_timeslices"
    assert validity["accounted_endpoints"] == [PREFILL]
    assert validity["request_scope"]["conserved"] is True
    assert warnings == []


@pytest.mark.parametrize(
    ("kv_offloading", "status", "cpu", "eligible"),
    [
        ("dram", "PARTIAL_COMPONENT", None, False),
        (None, "PARTIAL_COMPONENT", None, False),
        ("none", "VALID_COMPLETE", 0.0, True),
    ],
)
def test_missing_cpu_counter_is_partial_unless_no_cpu_tier_is_configured(
    kv_offloading: str | None, status: str, cpu: float | None, eligible: bool
) -> None:
    nested, _ = _run(_pd(offload=None), deployment={"disagg": True, "kv_offloading": kv_offloading})

    assert nested["cache"]["gpu_cache_hit_rate"] == 0.6
    assert nested["cache"]["cpu_cache_hit_rate"] == cpu
    assert nested["cache_validity"]["status"] == status
    assert nested["cache_validity"]["pricing_eligible"] is eligible


@pytest.mark.parametrize(
    ("offload", "kv_offloading"),
    [(None, "none"), ((0, 0, 0, 0), "dram")],
)
def test_cached_above_full_is_refused_on_every_cpu_branch(offload, kv_offloading) -> None:
    deployment = {"disagg": True, "kv_offloading": kv_offloading}
    export = _pd(offload=offload, prefix_cache_cached_tokens=(0, 500, 700, 0))

    nested, warnings = _run(export, deployment=deployment)

    assert _rates(nested) == NO_RATES
    assert nested["cache_validity"]["status"] == "INVALID_COUNTER"
    assert nested["cache_validity"]["pricing_eligible"] is False
    assert any("cached" in w for w in warnings)


def test_missing_counter_contract_zero_matches_an_explicit_zero() -> None:
    no_cpu_tier = {"disagg": True, "kv_offloading": "none"}

    explicit, _ = _run(_pd(offload=(0, 0, 0, 0)), deployment=no_cpu_tier)
    missing, _ = _run(_pd(offload=None), deployment=no_cpu_tier)

    assert _rates(explicit) == _rates(missing) == (0.6, 0.0, 0.0, 0.6)
    assert explicit["cache_validity"]["cpu_component"] == "MEASURED"
    assert missing["cache_validity"]["cpu_component"] == "NO_CPU_TIER_CONFIRMED"
    assert explicit["cache_validity"]["pricing_eligible"] is True
    assert missing["cache_validity"]["pricing_eligible"] is True


@pytest.mark.parametrize(
    ("cached", "offload", "status", "rates"),
    [
        ((0, 0, 0, 0), (0, 0, 0, 0), "VALID_ZERO", (0.0, 0.0, 0.0, 0.0)),
        ((0, 0, 0, 0), (0, 100, 150, 0), "VALID_COMPLETE", (0.0, 0.25, 0.25, 0.25)),
    ],
)
def test_zero_and_cpu_only_reuse_are_eligible_measurements(cached, offload, status, rates) -> None:
    nested, _ = _run(_pd(offload=offload, prefix_cache_cached_tokens=cached))

    assert _rates(nested) == rates
    assert nested["cache_validity"]["status"] == status
    assert nested["cache_validity"]["pricing_eligible"] is True


@pytest.mark.parametrize(
    ("roles", "deployment", "reason"),
    [
        (None, PD, "no launcher endpoint roles"),
        (None, {"disagg": None}, "no launcher endpoint roles"),
        ({DECODE: "decode"}, PD, f"unmapped=['{PREFILL}']"),
        ({**ROLES, "http://10.0.0.3:6100/metrics": "prefill"}, PD, "missing=['http://10.0.0.3"),
        ({PREFILL: "decode", DECODE: "decode"}, PD, "unmapped=[], missing=[]"),
    ],
)
def test_unresolved_roles_publish_no_rates(roles, deployment, reason) -> None:
    nested, warnings = _run(_pd(), roles=roles, deployment=deployment)

    assert _rates(nested) == NO_RATES
    assert nested["cache_validity"]["status"] == "ROLE_OR_COVERAGE_UNRESOLVED"
    assert nested["cache_validity"]["pricing_eligible"] is False
    assert any(reason in w for w in warnings)


def test_unified_fallback_needs_a_declared_non_disagg_single_endpoint() -> None:
    single = _export({stem: {PREFILL: v} for stem, v in _pd_values().items()})
    two = _export({stem: {PREFILL: v, DECODE: v} for stem, v in _pd_values().items()})
    unified = {"disagg": False, "kv_offloading": "dram"}

    one, _ = _run(single, roles=None, deployment=unified, framework="atom")
    many, _ = _run(two, roles=None, deployment=unified, framework="atom")

    assert one["cache_validity"]["roles_source"] == "single_endpoint_non_disagg"
    assert one["cache_validity"]["pricing_eligible"] is True
    assert many["cache_validity"]["status"] == "ROLE_OR_COVERAGE_UNRESOLVED"


def _pd_values() -> dict[str, tuple]:
    return {
        "prefix_cache_full_tokens": (0, 400, 600, 0),
        "prefix_cache_cached_tokens": (0, 200, 400, 0),
        "prefix_cache_offload_tokens": (0, 50, 50, 0),
        "prefix_cache_requests": (0, 3, 2, 0),
    }


@pytest.mark.parametrize(
    ("override", "status", "reason"),
    [
        ({"prefix_cache_cached_tokens": (0, -1, 400, 0)}, "WINDOW_UNRESOLVED", "invalid"),
        ({"prefix_cache_full_tokens": (0, math.nan, 600, 0)}, "WINDOW_UNRESOLVED", "invalid"),
        (
            {"prefix_cache_full_tokens": ZERO, "prefix_cache_cached_tokens": ZERO},
            "INVALID_COUNTER",
            "no prompt tokens",
        ),
        ({"prefix_cache_cached_tokens": (0, 400, 550, 0)}, "INVALID_COUNTER", "exceeds full"),
    ],
)
def test_invalid_counters_are_refused_not_clamped(override, status, reason) -> None:
    nested, warnings = _run(_pd(**override))

    assert _rates(nested) == NO_RATES
    assert nested["cache_validity"]["status"] == status
    assert any(reason in w for w in warnings)


def test_missing_timeslices_never_fall_back_to_stats_total() -> None:
    export = _pd()
    for entry in export["metrics"].values():
        for series in entry["series"]:
            del series["timeslices"]

    nested, _ = _run(export)

    assert _rates(nested) == NO_RATES
    assert nested["cache_validity"]["status"] == "WINDOW_UNRESOLVED"
    cumulative = nested["cache_validity"]["diagnostics"]["cumulative_stats_total"]
    assert cumulative["atom:prefix_cache_full_tokens"] == 1000


def test_missing_profiling_window_is_unresolved() -> None:
    export = _pd()
    del export["summary"]

    nested, _ = _run(export)

    assert _rates(nested) == NO_RATES
    assert nested["cache_validity"]["status"] == "WINDOW_UNRESOLVED"


# Window (1_200, 4_000): (1_000, 1_400) straddles the start, (3_600, 4_200) the end.
STRADDLE = ((0, 1_000), (1_000, 1_400), (1_400, 2_600), (2_600, 3_600), (3_600, 4_200))


def _unified(full: tuple, cached: tuple, warmup: dict | None = None) -> dict:
    export = _export(
        {
            "prefix_cache_full_tokens": {PREFILL: full},
            "prefix_cache_cached_tokens": {PREFILL: cached},
            "prefix_cache_offload_tokens": {PREFILL: (0, 0, 0, 0, 0)},
            "prefix_cache_requests": {PREFILL: (0, 0, 3, 2, 0)},
        },
        edges=STRADDLE,
        window=(1_200, 4_000),
    )
    if warmup is not None:
        export["warmup_metrics"] = {
            f"atom:{stem}": {"series": [{"endpoint_url": PREFILL, "stats": {"total": total}}]}
            for stem, total in warmup.items()
        }
    return export


@pytest.mark.parametrize(
    ("full", "cached", "warmup", "status"),
    [
        ((0, 0, 400, 600, 0), (0, 0, 200, 400, 0), None, "VALID_COMPLETE"),
        ((0, 0, 400, 600, 9), (0, 0, 200, 400, 0), None, "WINDOW_UNRESOLVED"),
        ((300, 90, 400, 600, 0), (100, 30, 200, 400, 0), None, "WINDOW_UNRESOLVED"),
        (
            (300, 90, 400, 600, 0),
            (100, 30, 200, 400, 0),
            {"prefix_cache_full_tokens": 390, "prefix_cache_cached_tokens": 130},
            "VALID_COMPLETE",
        ),
        (
            (300, 90, 400, 600, 0),
            (100, 30, 200, 400, 0),
            {"prefix_cache_full_tokens": 300, "prefix_cache_cached_tokens": 100},
            "WINDOW_UNRESOLVED",
        ),
    ],
)
def test_boundary_slice_activity_must_be_proven_warmup_only(full, cached, warmup, status) -> None:
    unified = {"disagg": False, "kv_offloading": "dram"}
    nested, _ = _run(_unified(full, cached, warmup), roles=None, deployment=unified)

    assert nested["cache_validity"]["status"] == status
    assert nested["cache_validity"]["pricing_eligible"] is (status == "VALID_COMPLETE")
    assert nested["cache"]["gpu_cache_hit_rate"] == 0.6


def test_sampling_gap_inside_the_window_is_unresolved() -> None:
    gappy = ((0, 1_000), (1_000, 2_000), (2_200, 4_000), (4_000, 5_000))
    export = _export(
        {stem: {PREFILL: values} for stem, values in _pd_values().items()}, edges=gappy
    )

    nested, warnings = _run(export, roles=None, deployment={"disagg": False})

    assert nested["cache_validity"]["status"] == "WINDOW_UNRESOLVED"
    assert any("sampling gap" in w for w in warnings)


def test_duplicate_alias_series_for_one_endpoint_is_invalid() -> None:
    export = _pd()
    export["metrics"]["atom_prefix_cache_cached_tokens_total"] = export["metrics"][
        "atom:prefix_cache_cached_tokens"
    ]

    nested, _ = _run(export)

    assert _rates(nested) == NO_RATES
    assert nested["cache_validity"]["status"] == "INVALID_COUNTER"


@pytest.mark.parametrize(
    "scope",
    [
        {**SCOPE, "profiling_requests": 4, "profiling_prompt_tokens": 900},
        {**SCOPE, "error_records": 1},
        {**SCOPE, "profiling_prompt_tokens": None},
        None,
    ],
)
def test_request_scope_mismatch_keeps_window_telemetry_but_blocks_pricing(scope) -> None:
    nested, _ = _run(_pd(), scope=scope)

    assert _rates(nested) == (0.6, 0.1, 0.1, pytest.approx(0.7))
    assert nested["cache_validity"]["status"] == "REQUEST_SCOPE_UNRESOLVED"
    assert nested["cache_validity"]["pricing_eligible"] is False


def test_unrelated_router_endpoint_is_not_a_missing_prefill() -> None:
    export = _pd()
    router = "http://10.0.0.9:8000/metrics"
    export["metrics"]["atomesh:http_requests_total"] = {
        "type": "counter",
        "series": [_series(router, (0, 5, 5, 0))],
    }

    nested, _ = _run(export, roles={**ROLES, router: "router"})

    assert nested["cache_validity"]["status"] == "VALID_COMPLETE"


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        (
            {"SRT_PREFILL_ENDPOINTS": "10.0.0.1:6100", "SRT_DECODE_ENDPOINTS": "10.0.0.2:6100,"},
            ROLES,
        ),
        (
            {"SRT_AGG_ENDPOINTS": "a:1,b:1", "SRT_PREFILL_ENDPOINTS": "c:1"},
            {"http://a:1/metrics": "unified", "http://b:1/metrics": "unified"},
        ),
    ],
)
def test_endpoint_roles_cover_launcher_scrape_urls(
    env: dict[str, str], expected: dict[str, str]
) -> None:
    roles = endpoint_roles_from_env(env)

    assert roles == expected
    assert set(roles) == set(_server_metrics_urls(env))


@pytest.mark.parametrize(
    "env",
    [
        {"AIPERF_SERVER_METRICS_URLS": "http://x/metrics", "SRT_PREFILL_ENDPOINTS": "a:1"},
        {"SRTCTL_FRONTEND_TYPE": "dynamo", "SRT_PREFILL_ENDPOINTS": "a:1"},
        {"SRT_PREFILL_ENDPOINTS": "a:1", "SRT_DECODE_ENDPOINTS": "a:1"},
        {},
    ],
)
def test_endpoint_roles_unknown_without_launcher_metadata(env: dict[str, str]) -> None:
    assert endpoint_roles_from_env(env) is None


DISAGG_ENV = {
    "FRAMEWORK": "atom-disagg",
    "IS_MULTINODE": "true",
    "DISAGG": "true",
    **{
        f"{side}_{key}": value
        for side in ("PREFILL", "DECODE")
        for key, value in {
            "NUM_WORKERS": "1",
            "TP": "8",
            "PP_SIZE": "1",
            "DCP_SIZE": "1",
            "PCP_SIZE": "1",
            "EP": "1",
            "DP_ATTN": "false",
            "HARDWARE": "mi355x",
        }.items()
    },
}
ROLE_ENV = {"SRT_PREFILL_ENDPOINTS": "10.0.0.1:6100", "SRT_DECODE_ENDPOINTS": "10.0.0.2:6100"}


def _atom_result_dir(tmp_path: Path) -> tuple[Path, list[dict]]:
    result_dir = _write_fixture(tmp_path)
    artifacts = result_dir / "aiperf_artifacts"
    (artifacts / "server_metrics_export.json").write_text(json.dumps(_pd()))
    jsonl = artifacts / "profile_export.jsonl"
    records = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    for record in records:
        record["metrics"]["usage_prompt_tokens"] = {"value": 1000 // len(records), "unit": "tokens"}
    return result_dir, records


def _write_records(result_dir: Path, records: list[dict]) -> None:
    jsonl = result_dir / "aiperf_artifacts" / "profile_export.jsonl"
    jsonl.write_text("".join(json.dumps(record) + "\n" for record in records))


def _failed(phase: str | None) -> dict:
    metadata = {} if phase is None else {"benchmark_phase": phase}
    return {"metadata": metadata, "error": {"type": "Timeout", "message": "request timed out"}}


def test_processor_entry_point_carries_validity_through_the_serializer(tmp_path: Path) -> None:
    result_dir, records = _atom_result_dir(tmp_path)
    _write_records(result_dir, records)

    agg = _run_processor(result_dir, tmp_path / "out", env_overrides={**DISAGG_ENV, **ROLE_ENV})
    unresolved = _run_processor(result_dir, tmp_path / "unresolved", env_overrides=DISAGG_ENV)

    _assert_stable_server_metrics_schema(agg)
    assert len(records) == 5
    assert _rates(agg["server_metrics"]) == (0.6, 0.1, 0.1, 0.7)
    assert agg["server_metrics"]["cache_validity"]["status"] == "VALID_COMPLETE"
    assert agg["server_metrics"]["cache_validity"]["pricing_eligible"] is True
    assert _rates(unresolved["server_metrics"]) == NO_RATES
    assert unresolved["server_metrics"]["cache_validity"]["pricing_eligible"] is False
    assert (
        agg["server_metrics"]["tokens"]["prompt_total"]
        == (unresolved["server_metrics"]["tokens"]["prompt_total"])
    )


@pytest.mark.parametrize(
    ("failed_phases", "legacy", "status", "profiling_errors"),
    [
        ((), False, "VALID_COMPLETE", 0),
        (("warmup",), False, "VALID_COMPLETE", 0),
        (("profiling",), False, "REQUEST_SCOPE_UNRESOLVED", 1),
        (("warmup", "profiling"), False, "REQUEST_SCOPE_UNRESOLVED", 1),
        ((), True, "VALID_COMPLETE", 0),
        ((None,), True, "REQUEST_SCOPE_UNRESOLVED", 1),
    ],
    ids=["clean", "warmup-error", "profiling-error", "both-phases", "legacy-clean", "legacy-error"],
)
def test_only_profiling_errors_from_the_loader_block_pricing(
    tmp_path: Path,
    failed_phases: tuple[str | None, ...],
    legacy: bool,
    status: str,
    profiling_errors: int,
) -> None:
    result_dir, records = _atom_result_dir(tmp_path)
    if legacy:
        for record in records:
            del record["metadata"]["benchmark_phase"]
    _write_records(result_dir, [_failed(phase) for phase in failed_phases] + records)

    agg = _run_processor(result_dir, tmp_path / "out", env_overrides={**DISAGG_ENV, **ROLE_ENV})
    validity = agg["server_metrics"]["cache_validity"]
    accounting = agg["request_accounting"]

    assert _rates(agg["server_metrics"]) == (0.6, 0.1, 0.1, 0.7)
    assert validity["status"] == status
    assert validity["pricing_eligible"] is (status == "VALID_COMPLETE")
    assert validity["request_scope"]["error_records"] == profiling_errors
    assert accounting["records_profiled"] == 5
    assert accounting["records_error_dropped"] == len(failed_phases)
    assert accounting["records_profiling_error_dropped"] == profiling_errors
    assert accounting["records_warmup_dropped"] == failed_phases.count("warmup")
    expected_categories = {"Timeout": len(failed_phases)} if failed_phases else {}
    assert accounting["error_categories"] == expected_categories
