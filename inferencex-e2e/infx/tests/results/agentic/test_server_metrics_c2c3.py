from __future__ import annotations

import pytest

from infx.results.agentic.server_metrics import compute_server_metrics

FE = "http://10.0.0.1:8000/metrics"
PRE = "http://10.0.0.2:7500/metrics"
DEC = "http://10.0.0.3:7501/metrics"


def _counter(rows: list[tuple[str, dict, float]]) -> dict:
    return {
        "type": "counter",
        "series": [
            {"endpoint_url": e, "labels": labels, "stats": {"total": float(v)}}
            for e, labels, v in rows
        ],
    }


def _histogram(endpoint: str, total: float, count: float = 10.0) -> dict:
    return {
        "type": "histogram",
        "series": [
            {"endpoint_url": endpoint, "labels": {}, "stats": {"sum": float(total), "count": count}}
        ],
    }


def _gauge(endpoint: str, value: float) -> dict:
    return {
        "type": "gauge",
        "series": [
            {
                "endpoint_url": endpoint,
                "labels": {},
                "stats": {"avg": float(value), "max": float(value)},
            }
        ],
    }


def _run(metrics: dict, framework: str) -> tuple[dict, dict]:
    flat, nested, _ = compute_server_metrics({"metrics": metrics}, framework=framework, records=[])
    return flat, nested


def _vllm_role(component: str, source: str | None = None) -> dict:
    return {"dynamo_component": component, **({"source": source} if source else {})}


class TestDynamoFrontendOverall:
    def test_frontend_total_reaches_nested_overall(self) -> None:
        metrics = {
            "dynamo_frontend_cached_tokens": _histogram(FE, 800),
            "dynamo_frontend_input_sequence_tokens": _histogram(FE, 1000),
        }
        flat, nested = _run(metrics, "dynamo-vllm")
        assert flat["server_overall_cache_hit_rate"] == pytest.approx(0.8)
        assert nested["cache"]["overall_cache_hit_rate"] == pytest.approx(0.8)
        assert nested["cache"]["overall_source"] == "frontend"

    def test_measured_zero_frontend_rate_is_kept(self) -> None:
        metrics = {
            "dynamo_frontend_cached_tokens": _histogram(FE, 0),
            "dynamo_frontend_input_sequence_tokens": _histogram(FE, 1000),
            "dynamo_component_router_shared_cache_hit_rate": _gauge(FE, 0.7),
        }
        flat, nested = _run(metrics, "dynamo-vllm")
        assert flat["server_overall_cache_hit_rate"] == 0.0
        assert nested["cache"]["overall_cache_hit_rate"] == 0.0
        assert nested["cache"]["overall_source"] == "frontend"

    def test_router_estimate_is_labelled_as_an_estimate(self) -> None:
        metrics = {"dynamo_component_router_shared_cache_hit_rate": _gauge(FE, 0.7)}
        flat, nested = _run(metrics, "dynamo-vllm")
        assert flat["server_overall_cache_hit_rate"] == pytest.approx(0.7)
        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert nested["cache"]["overall_source"] == "router_estimate"
        assert nested["cache_validity"]["published"] is False
        assert nested["cache_validity"]["basis"] is None
        assert nested["cache_validity"]["pricing_eligible"] is False

    def test_worker_counters_take_precedence_over_the_frontend(self) -> None:
        metrics = {
            "vllm:prompt_tokens_by_source": _counter(
                [
                    (PRE, _vllm_role("prefill", "local_cache_hit"), 600),
                    (PRE, _vllm_role("prefill", "local_compute"), 400),
                ]
            ),
            "dynamo_frontend_cached_tokens": _histogram(FE, 800),
            "dynamo_frontend_input_sequence_tokens": _histogram(FE, 1000),
        }
        _, nested = _run(metrics, "dynamo-vllm")
        assert nested["cache"]["overall_cache_hit_rate"] == pytest.approx(0.6)
        assert nested["cache"]["overall_source"] == "worker_counters"
        assert nested["cache"]["frontend_cache_hit_rate"] == pytest.approx(0.8)


class TestVllmPrefillDecodeRoles:
    def test_decode_kv_transfer_is_not_a_cache_hit(self) -> None:
        metrics = {
            "vllm:prompt_tokens_by_source": _counter(
                [
                    (DEC, _vllm_role("backend", "external_kv_transfer"), 1000),
                    (DEC, _vllm_role("backend", "local_cache_hit"), 0),
                    (DEC, _vllm_role("backend", "local_compute"), 0),
                    (PRE, _vllm_role("prefill", "external_kv_transfer"), 500),
                    (PRE, _vllm_role("prefill", "local_cache_hit"), 200),
                    (PRE, _vllm_role("prefill", "local_compute"), 300),
                ]
            )
        }
        _, nested = _run(metrics, "dynamo-vllm")
        assert nested["cache"]["gpu_cache_hit_rate"] == pytest.approx(0.2)
        assert nested["cache"]["external_cache_hit_rate"] == pytest.approx(0.5)
        assert nested["cache"]["overall_cache_hit_rate"] == pytest.approx(0.7)
        assert nested["cache"]["role_scope"] == "prefill"
        assert nested["tokens"]["prompt_by_source"]["cpu_or_external_cache_hit"] == pytest.approx(
            500
        )

    def test_prefix_cache_counters_use_the_prefill_role(self) -> None:
        metrics = {
            "vllm:prefix_cache_hits": _counter(
                [(PRE, _vllm_role("prefill"), 600), (DEC, _vllm_role("backend"), 0)]
            ),
            "vllm:prefix_cache_queries": _counter(
                [(PRE, _vllm_role("prefill"), 1000), (DEC, _vllm_role("backend"), 1000)]
            ),
        }
        _, nested = _run(metrics, "dynamo-vllm")
        assert nested["cache"]["gpu_cache_hit_rate"] == pytest.approx(0.6)
        assert nested["cache"]["prefix_cache_hits"] == pytest.approx(600)
        assert nested["cache"]["prefix_cache_queries"] == pytest.approx(1000)

    def test_unlabelled_engines_keep_the_combined_scope(self) -> None:
        metrics = {
            "vllm:prompt_tokens_by_source": _counter(
                [(PRE, {"source": "local_cache_hit"}, 300), (PRE, {"source": "local_compute"}, 700)]
            )
        }
        _, nested = _run(metrics, "vllm")
        assert nested["cache"]["overall_cache_hit_rate"] == pytest.approx(0.3)
        assert nested["cache"]["role_scope"] == "combined"

    def test_mixed_roles_exclude_known_decode_transfers(self) -> None:
        metrics = {
            "vllm:prompt_tokens_by_source": _counter(
                [
                    (PRE, _vllm_role("prefill", "local_cache_hit"), 600),
                    (PRE, _vllm_role("prefill", "local_compute"), 400),
                    (DEC, _vllm_role("backend", "external_kv_transfer"), 1000),
                    (FE, {"source": "local_compute"}, 100),
                ]
            )
        }
        _, nested = _run(metrics, "dynamo-vllm")
        assert nested["cache"]["overall_cache_hit_rate"] == pytest.approx(600 / 1100)
        assert nested["cache"]["role_scope"] == "mixed"
        assert nested["tokens"]["prompt_by_source"]["cpu_or_external_cache_hit"] is None
        assert nested["cache_validity"]["pricing_eligible"] is False

    def test_decode_only_has_no_reuse_measurement(self) -> None:
        metrics = {
            "vllm:prompt_tokens_by_source": _counter(
                [(DEC, _vllm_role("backend", "external_kv_transfer"), 1000)]
            )
        }
        _, nested, _ = compute_server_metrics(
            {"metrics": metrics}, framework="dynamo-vllm", records=[], disaggregated=True
        )
        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert nested["cache"]["role_scope"] == "unresolved"
        assert nested["cache_validity"]["published"] is False

    @pytest.mark.parametrize("disaggregated", [False, True])
    def test_backend_component_respects_deployment_mode(self, disaggregated: bool) -> None:
        metrics = {
            "vllm:prompt_tokens_by_source": _counter(
                [
                    (PRE, _vllm_role("backend", "local_cache_hit"), 300),
                    (PRE, _vllm_role("backend", "local_compute"), 700),
                ]
            )
        }
        _, nested, _ = compute_server_metrics(
            {"metrics": metrics},
            framework="dynamo-vllm",
            records=[],
            disaggregated=disaggregated,
        )
        if disaggregated:
            assert nested["cache"]["overall_cache_hit_rate"] is None
            assert nested["cache"]["role_scope"] == "unresolved"
        else:
            assert nested["cache"]["overall_cache_hit_rate"] == pytest.approx(0.3)
            assert nested["cache"]["role_scope"] == "combined"

    @pytest.mark.parametrize("mode", ["garbage", None])
    def test_explicit_unknown_mode_does_not_become_combined(self, mode: str | None) -> None:
        metrics = {
            "vllm:prompt_tokens_by_source": _counter(
                [(PRE, {"disaggregation_mode": mode, "source": "external_kv_transfer"}, 1000)]
            )
        }
        _, nested, _ = compute_server_metrics(
            {"metrics": metrics}, framework="vllm", records=[], disaggregated=False
        )
        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert nested["cache"]["role_scope"] == "unresolved"
        assert nested["cache_validity"]["published"] is False


class TestSglangNonDeviceTiers:
    @staticmethod
    def _metrics(cached: dict[str, float], prompt: float = 1000.0) -> dict:
        return {
            "sglang:prompt_tokens": _counter([(PRE, {}, prompt)]),
            "sglang:cached_tokens": _counter(
                [(PRE, {"cache_source": k}, v) for k, v in cached.items()]
            ),
        }

    def test_storage_tier_is_counted_with_the_other_non_device_tiers(self) -> None:
        _, nested = _run(self._metrics({"device": 600, "host": 200, "storage": 100}), "sglang")
        cache = nested["cache"]
        assert cache["gpu_cache_hit_rate"] == pytest.approx(0.6)
        assert cache["cpu_cache_hit_rate"] == pytest.approx(0.2)
        assert cache["external_cache_hit_rate"] == pytest.approx(0.3)
        assert cache["overall_cache_hit_rate"] == pytest.approx(0.9)
        assert cache["gpu_cache_hit_rate"] + cache["external_cache_hit_rate"] == pytest.approx(
            cache["overall_cache_hit_rate"]
        )

    def test_device_and_host_only_keep_external_equal_to_host(self) -> None:
        _, nested = _run(self._metrics({"device": 600, "host": 200}), "sglang")
        assert nested["cache"]["external_cache_hit_rate"] == pytest.approx(0.2)
        assert nested["cache"]["cpu_cache_hit_rate"] == pytest.approx(0.2)

    def test_symmetric_prefill_and_decode_sum_keeps_the_ratio(self) -> None:
        roles = {"prefill": {"engine_type": "prefill"}, "decode": {"engine_type": "decode"}}
        metrics = {
            "sglang:prompt_tokens": _counter(
                [(PRE, roles["prefill"], 1000), (DEC, roles["decode"], 1000)]
            ),
            "sglang:cached_tokens": _counter(
                [
                    (PRE, {**roles["prefill"], "cache_source": "device"}, 900),
                    (DEC, {**roles["decode"], "cache_source": "device"}, 900),
                ]
            ),
        }
        _, nested = _run(metrics, "sglang")
        assert nested["cache"]["overall_cache_hit_rate"] == pytest.approx(0.9)


def test_frontend_only_total_reaches_the_cache_validity_core() -> None:
    metrics = {
        "dynamo_frontend_cached_tokens": _histogram(FE, 800),
        "dynamo_frontend_input_sequence_tokens": _histogram(FE, 1000),
    }
    _, nested = _run(metrics, "dynamo-vllm")
    core = nested["cache_validity"]
    assert core["status_class"] == "NOT_EVALUATED"
    assert core["published"] is True
    assert core["basis"] == "combined"


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
@pytest.mark.parametrize("records", [[], [{"metrics": {"input_sequence_length": 1000}}]])
def test_disaggregated_prompt_total_uses_the_profile_cohort(
    framework: str, records: list[dict]
) -> None:
    metrics = {f"{framework}:prompt_tokens": _counter([(PRE, {}, 1000), (DEC, {}, 1000)])}
    flat, nested, _ = compute_server_metrics(
        {"metrics": metrics}, framework=framework, records=records, disaggregated=True
    )
    expected = 1000 if records else None
    assert flat["total_prompt_tokens"] == expected
    assert nested["tokens"]["prompt_total"] == expected
    assert nested["cache_validity"]["pricing_eligible"] is False
