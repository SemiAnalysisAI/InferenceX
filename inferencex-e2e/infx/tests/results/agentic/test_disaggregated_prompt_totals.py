from __future__ import annotations

import pytest

from infx.results.agentic.server_metrics import compute_server_metrics


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

    @pytest.mark.parametrize("metric_type, key", [("counter", "total"), ("histogram", "sum")])
    def test_frontend_total_remains_available_without_client_usage(self, metric_type, key):
        export = {
            "metrics": {
                name: {"type": metric_type, "series": [{"stats": {key: value, "count": 10}}]}
                for name, value in (
                    ("dynamo_frontend_input_sequence_tokens", 1000.0),
                    ("dynamo_frontend_cached_tokens", 800.0),
                )
            }
        }
        flat, nested, _ = compute_server_metrics(
            export, framework="dynamo-vllm", records=[], disaggregated=True
        )

        assert flat["total_prompt_tokens"] == nested["tokens"]["prompt_total"] == 1000
        assert flat["server_overall_cache_hit_rate"] == pytest.approx(0.8)
        assert nested["cache"]["frontend_cache_hit_rate"] == pytest.approx(0.8)

    def test_replacing_worker_scope_clears_normalized_tiers_and_keeps_raw_diagnostics(self):
        export = {
            "metrics": {
                "sglang:prompt_tokens": {
                    "type": "counter",
                    "series": [
                        {"endpoint_url": endpoint, "stats": {"total": 1000}}
                        for endpoint in ("http://prefill/metrics", "http://decode/metrics")
                    ],
                },
                "sglang:cached_tokens": {
                    "type": "counter",
                    "series": [
                        {
                            "endpoint_url": endpoint,
                            "labels": {"cache_source": source},
                            "stats": {"total": total},
                        }
                        for source, total in (("device", 900), ("host", 100))
                        for endpoint in ("http://prefill/metrics", "http://decode/metrics")
                    ],
                },
            }
        }
        flat, nested, _ = compute_server_metrics(
            export,
            framework="sglang",
            records=[{"metrics": {"input_sequence_length": 1000}}],
            disaggregated=True,
        )

        assert flat["total_prompt_tokens"] == nested["tokens"]["prompt_total"] == 1000
        assert nested["tokens"]["prompt_by_source"] == {
            "gpu_cache_hit": None,
            "cpu_or_external_cache_hit": None,
            "computed": None,
            "raw": {"device": 1800, "host": 200},
        }
        assert nested["cache"]["overall_cache_hit_rate"] == pytest.approx(1.0)
        assert nested["cache_validity"]["pricing_eligible"] is False
