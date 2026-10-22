"""The cache validity core is filled for every adapter, so one consumer rule covers every producer."""

from __future__ import annotations

from infx.results.agentic.server_metrics import compute_server_metrics
from infx.tests.results.agentic.test_atom_cache_validity import _export as _atom_export
from infx.tests.results.agentic.test_atom_cache_validity import _run as _run_atom
from infx.tests.results.agentic.test_server_metrics_closeout import _pair
from infx.tests.results.agentic.test_server_metrics_closeout import _run as _run_trt


def _counter(value: float) -> dict:
    return {
        "type": "counter",
        "series": [
            {
                "endpoint_url": "http://10.0.0.5:8000/metrics",
                "labels": {},
                "stats": {"total": value},
            }
        ],
    }


class TestCacheValidityCore:
    def test_atom_detail_maps_complete_partial_and_invalid_states(self) -> None:
        complete = _run_atom(_atom_export())["cache_validity"]
        export = _atom_export()
        del export["metrics"]["atom:prefix_cache_offload_tokens"]
        partial = _run_atom(export)["cache_validity"]
        export = _atom_export()
        cached = export["metrics"]["atom:prefix_cache_cached_tokens"]["series"][0]
        cached["stats"]["total"] = 1300
        cached["timeslices"][1]["total"] = 1300
        invalid = _run_atom(export)["cache_validity"]

        assert complete["status_class"] == "VALID_COMPLETE"
        assert complete["published"] is True
        assert complete["basis"] == "tiers"
        assert partial["status_class"] == "PARTIAL"
        assert partial["published"] is True
        assert partial["pricing_eligible"] is False
        assert invalid["status_class"] == "INVALID"
        assert invalid["published"] is False

    def test_trt_interval_telemetry_is_published_but_not_exact(self) -> None:
        core = _run_trt(_pair())["cache_validity"]

        assert core["status"] == "CONDITIONAL_INTERVAL"
        assert core["status_class"] == "VALID_INTERVAL"
        assert core["published"] is True
        assert core["pricing_eligible"] is False
        assert core["basis"] == "combined"

    def test_trt_rejected_rate_is_not_published(self) -> None:
        core = _run_trt(_pair(cached=1200.0))["cache_validity"]

        assert core["status_class"] == "INVALID"
        assert core["published"] is False
        assert core["basis"] is None
        assert "cached_exceeds_prompt" in core["reasons"]

    def test_adapter_without_assessment_is_not_evaluated_on_its_tiers(self) -> None:
        export = {
            "metrics": {
                "vllm:prefix_cache_hits": _counter(600.0),
                "vllm:prefix_cache_queries": _counter(1000.0),
            }
        }
        _, nested, _ = compute_server_metrics(export, framework="vllm", records=[])
        core = nested["cache_validity"]

        assert core["status_class"] == "NOT_EVALUATED"
        assert core["published"] is True
        assert core["pricing_eligible"] is False
        assert core["basis"] == "tiers"

    def test_missing_server_metrics_is_missing(self) -> None:
        _, nested, _ = compute_server_metrics({}, framework="vllm", records=[])
        core = nested["cache_validity"]

        assert core["status_class"] == "MISSING"
        assert core["published"] is False
        assert core["reasons"] == ["server_metrics_missing"]
