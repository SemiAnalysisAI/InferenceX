from __future__ import annotations

import copy

import pytest

from infx.results.agentic.server_metrics import compute_server_metrics

PREFILL = "http://prefill/metrics"
DECODE = "http://decode/metrics"
PROMPT = "trtllm_prompt_tokens"
CACHED = "trtllm_prompt_cached_tokens"


def _series(endpoint: str, value: float, mode: str = "prefill") -> dict:
    return {
        "endpoint_url": endpoint,
        "labels": {"dynamo_component": "backend", "disaggregation_mode": mode, "model_name": "m"},
        "stats": {"total": value},
    }


def _pair(cached: float = 600, suffix: str = "", mode: str = "prefill") -> dict:
    return {
        "schema_version": "1.1",
        "aiperf_version": "1.0.12",
        "metrics_phase": "profiling",
        "summary": {
            "phase_time_ranges": {"profiling": {"start_ns": 1, "end_ns": 100}},
            "endpoints_configured": [PREFILL, DECODE],
            "endpoints_successful": [PREFILL, DECODE],
        },
        "metrics": {
            name + suffix: {"type": "counter", "series": [_series(PREFILL, value, mode)]}
            for name, value in ((PROMPT, 1000), (CACHED, cached))
        },
    }


def _run(export: dict, disaggregated: bool | None = True, **kwargs) -> dict:
    return compute_server_metrics(
        export, framework="dynamo-trt", records=[], disaggregated=disaggregated, **kwargs
    )[1]


@pytest.mark.parametrize("suffix", ["", "_total", "both"])
def test_prefill_counter_aliases_exclude_decode_and_publish_only_combined_interval(suffix):
    export = _pair(suffix="_total" if suffix == "_total" else "")
    for name, entry in list(export["metrics"].items()):
        entry["series"].append(_series(DECODE, 1000, "decode"))
        if suffix == "both":
            export["metrics"][name + "_total"] = copy.deepcopy(entry)
    nested = _run(export)

    assert nested["cache"]["overall_cache_hit_rate"] == pytest.approx(0.6)
    assert nested["cache"]["gpu_cache_hit_rate"] is None
    assert nested["cache"]["cpu_cache_hit_rate"] is None
    assert nested["cache"]["cached_tokens_by_source"] == {"combined": 600}
    assert nested["tokens"]["prompt_total"] == 1000
    assert nested["tokens"]["prompt_by_source"]["computed"] == 400
    core = nested["cache_validity"]
    assert (core["status_class"], core["basis"], core["published"], core["pricing_eligible"]) == (
        "VALID_INTERVAL",
        "combined",
        True,
        False,
    )
    source = next(source for source in nested["sources"] if source["role"] == "prefill")
    assert source["prefix_cache_hit_rate"] == pytest.approx(0.6)


@pytest.mark.parametrize("cached", [0, 1000])
def test_zero_and_full_reuse_remain_publishable(cached):
    nested = _run(_pair(cached))
    assert nested["cache"]["overall_cache_hit_rate"] == (0 if cached == 0 else 1)
    assert nested["cache_validity"]["pricing_eligible"] is False


@pytest.mark.parametrize("labelled", [False, True])
def test_aggregated_deployment_accepts_unlabelled_or_explicit_unified_counter(labelled):
    export = _pair(mode="aggregated")
    if not labelled:
        for entry in export["metrics"].values():
            entry["series"][0]["labels"] = None
    nested = _run(export, False)
    assert nested["cache"]["overall_cache_hit_rate"] == pytest.approx(0.6)


@pytest.mark.parametrize("bad", [None, True, -1, 0.5, float("nan"), float("inf"), "600"])
def test_invalid_counter_total_blocks_headline_and_source_fallback(bad):
    nested = _run(_pair(bad))
    assert nested["cache"]["overall_cache_hit_rate"] is None
    assert nested["cache_validity"]["pricing_eligible"] is False
    assert nested["sources"][0]["prefix_cache_hit_rate"] is None


@pytest.mark.parametrize(
    "fault", ["unpaired", "rank", "duplicate", "alias", "bounds", "labels", "type", "empty"]
)
def test_counter_domains_must_pair_without_conflicts_or_compensating_totals(fault):
    export = _pair()
    prompt = export["metrics"][PROMPT]["series"]
    cached = export["metrics"][CACHED]["series"]
    if fault == "unpaired":
        cached.append(_series(DECODE, 1))
    elif fault == "rank":
        cached[0]["labels"]["rank"] = "1"
    elif fault == "duplicate":
        cached.append(copy.deepcopy(cached[0]))
    elif fault == "alias":
        export["metrics"][CACHED + "_total"] = {
            "type": "counter",
            "series": [_series(PREFILL, 601)],
        }
    elif fault == "bounds":
        prompt[0]["stats"]["total"] = 100
        prompt.append(_series(DECODE, 1000))
        cached.append(_series(DECODE, 0))
    elif fault == "labels":
        cached[0]["labels"] = []
    elif fault == "type":
        export["metrics"][CACHED]["type"] = "gauge"
    else:
        prompt[0]["stats"]["total"] = cached[0]["stats"]["total"] = 0
    nested = _run(export)
    assert nested["cache"]["overall_cache_hit_rate"] is None
    assert all(source["prefix_cache_hit_rate"] is None for source in nested["sources"])


@pytest.mark.parametrize("fault", ["unknown", "decode_only", "conflict", "deployment", "declared"])
def test_unproven_or_contradictory_roles_do_not_publish(fault):
    export = _pair()
    kwargs = {}
    disaggregated = None
    if fault == "unknown":
        for entry in export["metrics"].values():
            entry["series"].append({**_series(DECODE, 1), "labels": {}})
    elif fault == "decode_only":
        for entry in export["metrics"].values():
            entry["series"][0]["labels"]["disaggregation_mode"] = "decode"
    elif fault == "conflict":
        export["metrics"]["trtllm_requests"] = {"series": [_series(PREFILL, 1, "decode")]}
    elif fault == "deployment":
        disaggregated = False
    else:
        kwargs["endpoint_roles"] = {PREFILL: "decode"}
    nested = _run(export, disaggregated, **kwargs)
    assert nested["cache"]["overall_cache_hit_rate"] is None
    assert nested["cache_validity"]["published"] is False


@pytest.mark.parametrize("fault", ["old", "phase", "window", "unscraped", "unconfigured"])
def test_only_modern_profiling_totals_with_accounted_endpoint_coverage_publish(fault):
    export = _pair()
    if fault == "old":
        export["aiperf_version"] = "1.0.11"
    elif fault == "phase":
        export["metrics_phase"] = "warmup"
    elif fault == "window":
        export["summary"]["phase_time_ranges"]["profiling"]["end_ns"] = 0
    else:
        key = "endpoints_successful" if fault == "unscraped" else "endpoints_configured"
        export["summary"][key].remove(PREFILL)
    nested = _run(export)
    assert nested["cache"]["overall_cache_hit_rate"] is None
    assert nested["cache_validity"]["pricing_eligible"] is False


def test_gauge_remains_source_diagnostic_and_never_overrides_counter_rate():
    export = _pair()
    export["metrics"]["trtllm_kv_cache_hit_rate"] = {
        "type": "gauge",
        "series": [{"endpoint_url": PREFILL, "stats": {"avg": 0.9}}],
    }
    nested = _run(export)
    assert nested["cache"]["overall_cache_hit_rate"] == pytest.approx(0.6)
    assert nested["sources"][0]["prefix_cache_hit_rate"] == pytest.approx(0.9)


def test_idle_domain_and_series_order_do_not_change_weighted_rate():
    export = _pair()
    for name, value in ((PROMPT, 3000), (CACHED, 900)):
        series = export["metrics"][name]["series"]
        series.append(
            {**_series(PREFILL, value), "labels": {"rank": "1", "disaggregation_mode": "prefill"}}
        )
        series.append(_series(DECODE, 0))
    export["metrics"][CACHED]["series"].reverse()
    nested = _run(export)
    assert nested["cache"]["overall_cache_hit_rate"] == pytest.approx(0.375)
    assert nested["tokens"]["prompt_total"] == 4000


@pytest.mark.parametrize("mode", [None, "", "router"])
def test_explicit_invalid_role_cannot_fall_back_to_aggregated_deployment(mode):
    export = _pair()
    for entry in export["metrics"].values():
        entry["series"][0]["labels"]["disaggregation_mode"] = mode
    assert _run(export, False)["cache"]["overall_cache_hit_rate"] is None


def test_conflicts_on_excluded_decode_do_not_block_prefill_measurement():
    export = _pair()
    for entry in export["metrics"].values():
        entry["series"].append(_series(DECODE, 1000, "decode"))
    export["metrics"]["trtllm_other_metric"] = {"series": [_series(DECODE, 1, "aggregated")]}
    assert _run(export)["cache"]["overall_cache_hit_rate"] == pytest.approx(0.6)
