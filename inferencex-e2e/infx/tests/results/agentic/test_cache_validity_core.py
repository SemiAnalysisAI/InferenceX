from __future__ import annotations

import pytest

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


class TestDisaggregatedPromptTotals:
    @pytest.mark.parametrize(
        "framework, metric",
        [
            ("vllm", "vllm:prompt_tokens"),
            ("sglang", "sglang:prompt_tokens"),
            ("trtllm", "trtllm_prompt_tokens"),
        ],
    )
    def test_complete_profile_records_replace_duplicate_worker_prompts(self, framework, metric):
        export = {"metrics": {metric: _counter(2000.0)}}
        records = [{"metrics": {"input_sequence_length": {"value": n}}} for n in (800, 200)]
        flat, nested, _ = compute_server_metrics(
            export, framework=framework, records=records, disaggregated=True
        )

        assert flat["total_prompt_tokens"] == nested["tokens"]["prompt_total"] == 1000
        assert nested["cache_validity"]["pricing_eligible"] is False

    @pytest.mark.parametrize("value", [None, True, -1, 1.5, float("inf")])
    def test_partial_or_invalid_profile_records_do_not_supply_a_total(self, value):
        records = [{"metrics": {"input_sequence_length": n}} for n in (800, value)]
        flat, nested, _ = compute_server_metrics(
            {"metrics": {"vllm:prompt_tokens": _counter(2000.0)}},
            framework="vllm",
            records=records,
            disaggregated=True,
        )

        assert flat["total_prompt_tokens"] is None
        assert nested["tokens"]["prompt_total"] is None

    def test_missing_metrics_does_not_publish_a_partial_client_total(self):
        records = [{"metrics": {"input_sequence_length": 800}}, {"metrics": {}}]
        flat, nested, _ = compute_server_metrics(
            {}, framework="vllm", records=records, disaggregated=True
        )

        assert flat["total_prompt_tokens"] is None
        assert nested["tokens"]["prompt_total"] is None
        assert nested["cache_validity"]["status_class"] == "MISSING"

    @pytest.mark.parametrize("totals", [[1234, None], [-10, 1244], [True], [float("inf")]])
    def test_incomplete_or_invalid_frontend_cannot_supply_a_prompt_total(self, totals):
        counter = _counter(0)
        counter["series"] = [{"stats": {"total": value}} for value in totals]
        flat, nested, _ = compute_server_metrics(
            {"metrics": {"dynamo_frontend_input_sequence_tokens": counter}},
            framework="dynamo-trt",
            records=[],
            disaggregated=True,
        )

        assert flat["total_prompt_tokens"] is None
        assert nested["tokens"]["prompt_total"] is None

    def test_frontend_total_remains_available_without_client_usage(self):
        flat, nested, _ = compute_server_metrics(
            {"metrics": {"dynamo_frontend_input_sequence_tokens": _counter(1234.0)}},
            framework="dynamo-trt",
            records=[],
            disaggregated=True,
        )

        assert flat["total_prompt_tokens"] == nested["tokens"]["prompt_total"] == 1234
