"""The cache validity core is filled for every adapter, so one consumer rule covers every producer."""

from __future__ import annotations

from infx.results.agentic.cache_validity import STATUS_CLASSES
from infx.results.agentic.server_metrics import compute_server_metrics
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

    def test_every_class_is_from_the_shared_vocabulary(self) -> None:
        for export in (_pair(), _pair(cached=1200.0), {}):
            _, nested, _ = compute_server_metrics(export, framework="dynamo-trt", records=[])
            assert nested["cache_validity"]["status_class"] in STATUS_CLASSES
