from __future__ import annotations

import math

import pytest

from infx.results.agentic.server_metrics import compute_server_metrics


def _export(full: dict, cached: dict, offload: dict) -> dict:
    return {
        "metrics": {
            f"atom:prefix_cache_{stem}_tokens": {
                "series": [
                    {"endpoint_url": f"http://{endpoint}/metrics", "stats": {"total": value}}
                    for endpoint, value in values.items()
                ]
            }
            for stem, values in (("full", full), ("cached", cached), ("offload", offload))
        }
    }


def _compute(export: dict, env: dict) -> tuple[dict, dict]:
    flat, nested, _ = compute_server_metrics(export, framework="mooncake-atom", records=[], env=env)
    return flat, nested


def _assert_rates(
    flat: dict, nested: dict, gpu: float | None, cpu: float | None, overall: float | None
):
    for name, expected in (("gpu", gpu), ("cpu", cpu), ("external", cpu), ("overall", overall)):
        actual = flat[f"server_{name}_cache_hit_rate"]
        assert actual == (pytest.approx(expected) if expected is not None else None)
        assert nested["cache"][f"{name}_cache_hit_rate"] == actual


@pytest.mark.parametrize("decode_cached", [800, math.nan])
def test_prefill_counters_exclude_decode_and_transfer_volume(decode_cached: float) -> None:
    export = _export({"p": 1000, "d": 1000}, {"p": 600, "d": decode_cached}, {"p": 100, "d": -1})
    export["metrics"]["atom:lmcache_loaded_tokens"] = {
        "series": [{"endpoint_url": "http://p/metrics", "stats": {"total": 900}}]
    }
    flat, nested = _compute(
        export, {"SRT_PREFILL_ENDPOINTS": "p", "SRT_DECODE_ENDPOINTS": "d", "DISAGG": "true"}
    )
    _assert_rates(flat, nested, 0.6, 0.1, 0.7)
    assert nested["cache"]["prefix_cache_queries"] == 1000
    assert nested["cache"]["external_prefix_cache_hits"] == 100
    assert nested["cache"]["external_prefix_cache_queries"] == 1000


@pytest.mark.parametrize(
    ("env", "offload", "expected"),
    [
        ({"SRT_AGG_ENDPOINTS": "worker", "SRT_PREFILL_ENDPOINTS": "unused"}, 50, 0.5),
        ({"DISAGG": "false"}, 0, 0),
    ],
)
def test_unified_zero_gpu_and_offload_counter_alias(
    env: dict, offload: int, expected: float
) -> None:
    export = _export({"worker": 100}, {"worker": 0}, {"worker": offload})
    export["metrics"] = {
        name.replace("atom:", "atom_") + "_total": value
        for name, value in export["metrics"].items()
    }
    flat, nested = _compute(export, env)
    _assert_rates(flat, nested, 0, expected, expected)


def test_multiple_prefills_and_labelled_series_are_token_weighted() -> None:
    export = _export(
        {"p": 100, "q": 900, "idle": 0},
        {"p": 80, "q": 20, "idle": 0},
        {"p": 10, "q": 180, "idle": 0},
    )
    for stem, values in (("full", (400, 500)), ("cached", (5, 15)), ("offload", (80, 100))):
        series = export["metrics"][f"atom:prefix_cache_{stem}_tokens"]["series"]
        series[:] = [s for s in series if s["endpoint_url"] != "http://q/metrics"]
        series.extend(
            {
                "endpoint_url": "http://q/metrics",
                "labels": {"engine": str(i)},
                "stats": {"total": v},
            }
            for i, v in enumerate(values)
        )
    flat, nested = _compute(export, {"SRT_PREFILL_ENDPOINTS": "p,q,idle"})
    _assert_rates(flat, nested, 0.1, 0.19, 0.29)


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"DISAGG": "true"},
        {"SRT_PREFILL_ENDPOINTS": "p", "SRT_DECODE_ENDPOINTS": "p,d"},
        {"SRT_PREFILL_ENDPOINTS": "p,missing", "SRT_DECODE_ENDPOINTS": "d"},
        {"SRT_PREFILL_ENDPOINTS": "p"},
        {
            "SRT_PREFILL_ENDPOINTS": "p",
            "SRT_DECODE_ENDPOINTS": "d",
            "AIPERF_SERVER_METRICS_URLS": "custom",
        },
        {
            "SRT_PREFILL_ENDPOINTS": "p",
            "SRT_DECODE_ENDPOINTS": "d",
            "SRTCTL_FRONTEND_TYPE": "dynamo",
        },
    ],
)
def test_unknown_conflicting_or_incomplete_roles_do_not_mix_endpoints(env: dict) -> None:
    flat, nested = _compute(
        _export({"p": 100, "d": 100}, {"p": 60, "d": 0}, {"p": 10, "d": 0}), env
    )
    _assert_rates(flat, nested, None, None, None)
    assert nested["cache_validity"]["status"] == "ROLE_OR_COVERAGE_UNRESOLVED"


def test_single_endpoint_without_explicit_unified_topology_stays_unknown() -> None:
    flat, nested = _compute(_export({"worker": 100}, {"worker": 60}, {"worker": 10}), {})
    _assert_rates(flat, nested, None, None, None)


@pytest.mark.parametrize(
    ("offload", "cpu", "overall"),
    [({"p": 0, "q": 0}, 0, 0.2), ({}, None, None), ({"p": 10}, None, None)],
)
def test_missing_or_partial_offload_is_distinct_from_zero(
    offload: dict, cpu: float | None, overall: float | None
) -> None:
    flat, nested = _compute(
        _export({"p": 100, "q": 100}, {"p": 20, "q": 20}, offload),
        {"SRT_PREFILL_ENDPOINTS": "p,q"},
    )
    _assert_rates(flat, nested, 0.2, cpu, overall)
    assert nested["cache"]["external_prefix_cache_queries"] == (200 if cpu is not None else None)


@pytest.mark.parametrize(
    ("stem", "value"),
    [
        ("full", 0),
        ("cached", -1),
        ("cached", 101),
        ("cached", math.nan),
        ("cached", math.inf),
        ("cached", None),
    ],
)
def test_invalid_counted_counter_does_not_produce_rates(stem: str, value: object) -> None:
    export = _export({"worker": 100}, {"worker": 20}, {"worker": 10})
    export["metrics"][f"atom:prefix_cache_{stem}_tokens"]["series"][0]["stats"]["total"] = value
    flat, nested = _compute(export, {"DISAGG": "false"})
    _assert_rates(flat, nested, None, None, None)


def test_unreadable_series_does_not_get_silently_skipped() -> None:
    export = _export({"worker": 100}, {"worker": 20}, {"worker": 10})
    export["metrics"]["atom:prefix_cache_cached_tokens"]["series"].append(
        {
            "endpoint_url": "http://worker/metrics",
            "labels": {"engine": "other"},
            "stats": {"max": 5},
        }
    )
    flat, nested = _compute(export, {"DISAGG": "false"})
    _assert_rates(flat, nested, None, None, None)


@pytest.mark.parametrize("offload", [30, math.nan])
def test_invalid_worker_cpu_count_cannot_hide_in_another_workers_denominator(
    offload: float,
) -> None:
    flat, nested = _compute(
        _export({"p": 100, "q": 900}, {"p": 80, "q": 20}, {"p": offload, "q": 0}),
        {"SRT_PREFILL_ENDPOINTS": "p,q"},
    )
    _assert_rates(flat, nested, 0.1, None, None)


def test_unknown_cache_roles_preserve_other_atom_telemetry() -> None:
    export = _export({"p": 100, "d": 100}, {"p": 60, "d": 0}, {"p": 10, "d": 0})
    export["metrics"].update(
        {
            "atom:prompt_tokens": {"series": [{"stats": {"total": 200}}]},
            "atom:generation_tokens": {"series": [{"stats": {"total": 30}}]},
            "atom:kv_cache_usage_ratio": {"series": [{"stats": {"max": 0.75}}]},
        }
    )
    flat, nested = _compute(export, {})
    _assert_rates(flat, nested, None, None, None)
    assert flat["total_prompt_tokens"] == nested["tokens"]["prompt_total"] == 200
    assert flat["total_generation_tokens"] == nested["tokens"]["generation_total"] == 30
    assert flat["gpu_kv_cache_usage_pct"] == nested["kv_cache"]["gpu_usage_pct"] == 0.75
