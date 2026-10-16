"""Phase validity of AIPerf server-metric counters and the TensorRT-LLM reuse rate.

Fixtures follow the AIPerf 0.12 ``server_metrics_export.json`` shape: counter
names without the Prometheus ``_total`` suffix, 1 s timeslices per series and
``summary.phase_time_ranges``. The P/D topology mirrors the Dynamo TRT runs
(``prefill`` and ``backend`` components); decode workers report every
transferred prompt token as cached.
"""

from __future__ import annotations

import copy
import math

import pytest

from infx.results.agentic.server_metrics import compute_server_metrics

NS = 1_000_000_000
P0 = 1_000 * NS
PROFILING = (P0, P0 + 60 * NS)
WARMUP = (P0 - 120 * NS, P0 + int(0.3 * NS))
FRONTEND = "http://10.0.0.1:8000/metrics"
PREFILL = ["http://10.0.0.2:7500/metrics", "http://10.0.0.3:7501/metrics"]
DECODE = ["http://10.0.0.4:7502/metrics", "http://10.0.0.5:7503/metrics"]
PROMPT = {PREFILL[0]: 1000.0, PREFILL[1]: 2000.0, DECODE[0]: 1000.0, DECODE[1]: 2000.0}
CACHED = {PREFILL[0]: 900.0, PREFILL[1]: 1800.0, DECODE[0]: 1000.0, DECODE[1]: 2000.0}


def _role(endpoint: str) -> str:
    return "prefill" if endpoint in PREFILL else "backend"


def _slices(start_s: float, end_s: float, total: float) -> list[dict]:
    count = int(round(end_s - start_s))
    first = P0 + round(start_s * NS)
    per = total / count
    return [
        {
            "start_ns": first + i * NS,
            "end_ns": first + (i + 1) * NS,
            "total": per,
            "rate": per,
        }
        for i in range(count)
    ]


def _counter(
    endpoint: str,
    total: float,
    *,
    start_s: float = 0.0,
    end_s: float = 60.0,
    slice_sum: float | None = None,
    timeslices: bool = True,
) -> dict:
    series = {
        "endpoint_url": endpoint,
        "labels": {"dynamo_component": _role(endpoint), "model_name": "m"},
        "stats": {"total": total},
    }
    if timeslices:
        series["timeslices"] = _slices(
            start_s, end_s, total if slice_sum is None else slice_sum
        )
    return series


def _histogram(endpoint: str, total: float, count: int, *, start_s: float = 0.0):
    slices = []
    items = _slices(start_s, 60.0, total)
    per_count = count / len(items)
    for item in items:
        slices.append(
            {
                "start_ns": item["start_ns"],
                "end_ns": item["end_ns"],
                "count": per_count,
                "avg": item["total"] / per_count,
            }
        )
    return {
        "endpoint_url": endpoint,
        "labels": {"dynamo_namespace": "dynamo"},
        "stats": {"sum": total, "count": count},
        "timeslices": slices,
    }


def _export(
    *,
    names: tuple[str, ...] = ("",),
    prompt: dict[str, float] | None = None,
    cached: dict[str, float] | None = None,
    gauge: bool = False,
    phase_time_ranges: bool = True,
) -> dict:
    prompt = PROMPT if prompt is None else prompt
    cached = CACHED if cached is None else cached
    metrics: dict = {
        "dynamo_frontend_input_sequence_tokens": {
            "type": "histogram",
            "series": [_histogram(FRONTEND, 3000.0, 60)],
        }
    }
    for suffix in names:
        metrics[f"trtllm_prompt_tokens{suffix}"] = {
            "type": "counter",
            "series": [_counter(e, v) for e, v in prompt.items()],
        }
        metrics[f"trtllm_prompt_cached_tokens{suffix}"] = {
            "type": "counter",
            "series": [_counter(e, v) for e, v in cached.items()],
        }
    if gauge:
        metrics["trtllm_kv_cache_hit_rate"] = {
            "type": "gauge",
            "series": [
                {
                    "endpoint_url": e,
                    "labels": {"dynamo_component": _role(e)},
                    "stats": {"avg": 0.93 if e in PREFILL else 0.0},
                }
                for e in PREFILL + DECODE
            ],
        }
    endpoints = [FRONTEND, *PREFILL, *DECODE]
    summary: dict = {
        "endpoints_configured": endpoints,
        "endpoints_successful": list(endpoints),
    }
    if phase_time_ranges:
        summary["phase_time_ranges"] = {
            "profiling": {"start_ns": PROFILING[0], "end_ns": PROFILING[1]},
            "warmup": {"start_ns": WARMUP[0], "end_ns": WARMUP[1]},
        }
    return {"summary": summary, "metrics": metrics}


def _run(export: dict, framework: str = "dynamo-trt") -> tuple[dict, dict]:
    flat, nested, _ = compute_server_metrics(export, framework=framework, records=[])
    return flat, nested


def _cache_rate(nested: dict) -> dict:
    return nested["validity"]["cache_rate"]


class TestTrtCounterNames:
    def test_stripped_counter_names_feed_the_prefill_reuse_rate(self) -> None:
        _, nested = _run(_export())

        assert nested["cache"]["overall_cache_hit_rate"] == pytest.approx(0.9)
        assert _cache_rate(nested)["status"] == "CONDITIONAL_INTERVAL"
        assert _cache_rate(nested)["role_scope"] == "prefill"

    def test_native_total_names_are_still_accepted(self) -> None:
        _, nested = _run(_export(names=("_total",)))

        assert nested["cache"]["overall_cache_hit_rate"] == pytest.approx(0.9)

    def test_alias_pairs_with_equal_values_count_once(self) -> None:
        _, nested = _run(_export(names=("", "_total")))

        assert nested["cache"]["overall_cache_hit_rate"] == pytest.approx(0.9)
        assert _cache_rate(nested)["cached_tokens"] == pytest.approx(2700.0)
        assert _cache_rate(nested)["prompt_tokens"] == pytest.approx(3000.0)

    def test_alias_conflict_rejects_the_rate(self) -> None:
        export = _export(names=("", "_total"))
        native = export["metrics"]["trtllm_prompt_cached_tokens_total"]["series"][0]
        native["stats"]["total"] = 950.0

        _, nested = _run(export)

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert _cache_rate(nested)["status"] == "REJECTED"
        assert "alias_conflict" in _cache_rate(nested)["reasons"]


class TestTrtRoleAndTier:
    def test_decode_transfer_is_excluded_and_reported(self) -> None:
        _, nested = _run(_export())

        excluded = {
            (item["metric"], item["endpoint_url"], item["reason"])
            for item in _cache_rate(nested)["excluded"]
        }
        assert excluded == {
            (metric, endpoint, "decode_role_excluded")
            for metric in ("trtllm_prompt_tokens", "trtllm_prompt_cached_tokens")
            for endpoint in DECODE
        }

    def test_role_mixed_gauge_is_diagnostic_only(self) -> None:
        _, nested = _run(_export(gauge=True))

        assert nested["cache"]["gpu_cache_hit_rate"] is None
        diagnostic = nested["validity"]["diagnostics"]["trtllm_kv_cache_hit_rate"]
        assert diagnostic["status"] == "GAUGE_DERIVED"
        assert diagnostic["all_series_avg"] == pytest.approx(0.465)
        assert diagnostic["by_role"] == pytest.approx({"prefill": 0.93, "decode": 0.0})
        assert nested["cache"]["overall_cache_hit_rate"] == pytest.approx(0.9)

    def test_combined_cached_is_not_presented_as_gpu_tier(self) -> None:
        _, nested = _run(_export())

        assert nested["cache"]["cached_tokens_by_source"] == {"combined": 2700.0}
        assert _cache_rate(nested)["tier"] == "combined"
        prompt_by_source = nested["tokens"]["prompt_by_source"]
        assert prompt_by_source["gpu_cache_hit"] is None
        assert prompt_by_source["computed"] == pytest.approx(300.0)
        assert prompt_by_source["raw"] == {"combined": 2700.0}

    def test_rate_is_never_marked_exact_without_request_binding(self) -> None:
        _, nested = _run(_export())

        assert _cache_rate(nested)["eligible_exact_actual"] is False
        assert {"request_scope_not_bound", "expected_instances_not_verified"} <= set(
            _cache_rate(nested)["reasons"]
        )


class TestTrtPhaseValidity:
    def test_contaminated_cached_series_blocks_rate_and_keeps_telemetry(self) -> None:
        export = _export()
        export["metrics"]["trtllm_prompt_cached_tokens"]["series"][0] = _counter(
            PREFILL[0], 4900.0, start_s=-120.0
        )

        _, nested = _run(export)

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert nested["cache"]["cached_tokens_by_source"] == {}
        rate = _cache_rate(nested)
        assert rate["status"] == "REJECTED"
        assert {"activity_before_profiling", "cached_exceeds_prompt"} <= set(rate["reasons"])
        assert rate["cached_tokens"] == pytest.approx(6700.0)
        assert rate["raw_rate"] == pytest.approx(6700.0 / 3000.0)
        contaminated = [s for s in rate["series"] if s["status"] == "PHASE_CONTAMINATED"]
        assert [(s["endpoint_url"], s["reasons"]) for s in contaminated] == [
            (PREFILL[0], ["activity_before_profiling"])
        ]
        assert contaminated[0]["first_slice_offset_s"] == pytest.approx(-120.0)

    def test_total_above_slice_sum_is_a_boundary_residual(self) -> None:
        export = _export()
        export["metrics"]["trtllm_prompt_tokens"]["series"][0] = _counter(
            PREFILL[0], 1098.0, slice_sum=1000.0
        )

        _, nested = _run(export)

        assert nested["cache"]["overall_cache_hit_rate"] is None
        rate = _cache_rate(nested)
        assert rate["status"] == "BOUNDARY_RESIDUAL"
        residual = [s for s in rate["series"] if s["status"] == "BOUNDARY_RESIDUAL"]
        assert residual[0]["residual"] == pytest.approx(98.0)
        assert residual[0]["reasons"] == ["total_exceeds_slice_sum"]

    def test_slice_sum_above_total_is_a_suspected_reset(self) -> None:
        export = _export()
        export["metrics"]["trtllm_prompt_cached_tokens"]["series"][1] = _counter(
            PREFILL[1], 0.0, slice_sum=1800.0
        )

        _, nested = _run(export)

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert _cache_rate(nested)["status"] == "REJECTED"
        assert "slice_sum_exceeds_total" in _cache_rate(nested)["reasons"]

    @pytest.mark.parametrize(
        ("total", "reason"),
        [(-5.0, "negative_total"), (math.nan, "nonfinite_total"), (math.inf, "nonfinite_total")],
    )
    def test_negative_and_nonfinite_counters_are_rejected(self, total, reason) -> None:
        export = _export()
        series = export["metrics"]["trtllm_prompt_cached_tokens"]["series"][0]
        series["stats"]["total"] = total

        _, nested = _run(export)

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert _cache_rate(nested)["status"] == "REJECTED"
        assert reason in _cache_rate(nested)["reasons"]

    def test_missing_counter_total_is_unknown_not_rejected(self) -> None:
        export = _export()
        del export["metrics"]["trtllm_prompt_tokens"]["series"][1]["stats"]["total"]

        _, nested = _run(export)

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert _cache_rate(nested)["status"] == "UNKNOWN"
        assert "missing_total" in _cache_rate(nested)["reasons"]

    def test_missing_phase_window_is_unknown(self) -> None:
        _, nested = _run(_export(phase_time_ranges=False))

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert _cache_rate(nested)["status"] == "UNKNOWN"
        assert "phase_window_missing" in _cache_rate(nested)["reasons"]
        assert nested["validity"]["phase_windows"] is None

    def test_series_without_timeslices_is_unknown(self) -> None:
        export = _export()
        export["metrics"]["trtllm_prompt_tokens"]["series"][1] = _counter(
            PREFILL[1], 2000.0, timeslices=False
        )

        _, nested = _run(export)

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert _cache_rate(nested)["status"] == "UNKNOWN"
        assert "no_timeslices" in _cache_rate(nested)["reasons"]

    def test_late_created_series_is_partial_coverage(self) -> None:
        export = _export()
        export["metrics"]["trtllm_prompt_tokens"]["series"][1] = _counter(
            PREFILL[1], 2000.0, start_s=40.0
        )

        _, nested = _run(export)

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert _cache_rate(nested)["status"] == "PARTIAL_COVERAGE"
        assert "window_starts_after_profiling" in _cache_rate(nested)["reasons"]

    def test_prefill_endpoint_missing_its_cached_series_is_partial_coverage(self) -> None:
        cached = {k: v for k, v in CACHED.items() if k != PREFILL[1]}

        _, nested = _run(_export(cached=cached))

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert _cache_rate(nested)["status"] == "PARTIAL_COVERAGE"
        assert "missing_pair" in _cache_rate(nested)["reasons"]

    def test_unscraped_configured_endpoint_is_partial_coverage(self) -> None:
        prompt = {k: v for k, v in PROMPT.items() if k != PREFILL[1]}
        cached = {k: v for k, v in CACHED.items() if k != PREFILL[1]}
        export = _export(prompt=prompt, cached=cached)
        export["summary"]["endpoints_successful"].remove(PREFILL[1])

        _, nested = _run(export)

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert _cache_rate(nested)["status"] == "PARTIAL_COVERAGE"
        assert "endpoint_not_scraped" in _cache_rate(nested)["reasons"]


class TestSharedPhaseSummary:
    def test_histograms_pass_through_the_same_phase_check(self) -> None:
        export = _export()
        export["metrics"]["dynamo_frontend_cached_tokens"] = {
            "type": "histogram",
            "series": [_histogram(FRONTEND, 2700.0, 60, start_s=-119.9)],
        }
        gapped = _histogram(FRONTEND, 3000.0, 60)
        del gapped["timeslices"][30]
        export["metrics"]["dynamo_frontend_output_tokens"] = {
            "type": "histogram",
            "series": [gapped],
        }

        _, nested = _run(export)

        by_status = nested["validity"]["metrics_by_status"]
        assert "dynamo_frontend_cached_tokens" in by_status["PHASE_CONTAMINATED"]
        assert "dynamo_frontend_output_tokens" in by_status["REJECTED"]
        assert "dynamo_frontend_input_sequence_tokens" not in {
            name for names in by_status.values() for name in names
        }

    def test_empty_histogram_is_unknown_not_rejected(self) -> None:
        export = _export()
        export["metrics"]["dynamo_component_router_shared_cache_hit_rate"] = {
            "type": "histogram",
            "series": [{"endpoint_url": FRONTEND, "labels": {}, "stats": {"count": 0}}],
        }

        _, nested = _run(export)

        by_status = nested["validity"]["metrics_by_status"]
        assert by_status == {"UNKNOWN": ["dynamo_component_router_shared_cache_hit_rate"]}

    def test_other_backends_keep_published_values_and_are_not_evaluated(self) -> None:
        def vllm_export(with_phase_data: bool) -> dict:
            hits = _counter(PREFILL[0], 900.0)
            queries = _counter(PREFILL[0], 1000.0)
            if not with_phase_data:
                hits.pop("timeslices")
                queries.pop("timeslices")
            export = {
                "metrics": {
                    "vllm:prefix_cache_hits": {"type": "counter", "series": [hits]},
                    "vllm:prefix_cache_queries": {"type": "counter", "series": [queries]},
                }
            }
            if with_phase_data:
                export["summary"] = copy.deepcopy(_export()["summary"])
            return export

        _, with_phase = _run(vllm_export(True), framework="vllm")
        _, without_phase = _run(vllm_export(False), framework="vllm")

        assert with_phase["cache"] == without_phase["cache"]
        assert with_phase["cache"]["gpu_cache_hit_rate"] == pytest.approx(0.9)
        assert with_phase["validity"]["cache_rate"]["status"] == "NOT_EVALUATED"
        assert with_phase["validity"]["cache_rate"]["eligible_exact_actual"] is False
