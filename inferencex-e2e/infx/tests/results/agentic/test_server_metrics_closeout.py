from __future__ import annotations

import copy

import pytest

from infx.results.agentic.server_metrics import compute_server_metrics

NS = 1_000_000_000
START = 1_000 * NS
END = START + 60 * NS
PRE = ["http://10.0.0.2:7500/metrics", "http://10.0.0.3:7501/metrics"]
DEC = "http://10.0.0.4:7502/metrics"
UNLABELLED = "http://10.0.0.9:7509/metrics"


def _slices(total: float, start_ns: int = START, end_ns: int = END) -> list[dict]:
    bounds = list(range(start_ns, end_ns, NS))
    per = total / len(bounds)
    return [
        {"start_ns": s, "end_ns": min(s + NS, end_ns), "total": per, "rate": per}
        for s in bounds
    ]


def _series(
    endpoint: str,
    role: str | None,
    total: float,
    *,
    start_ns: int = START,
    end_ns: int = END,
    labels: dict | None = None,
    slices: list[dict] | None = None,
) -> dict:
    series_labels = {"model_name": "m", **(labels or {})}
    if role is not None:
        series_labels["dynamo_component"] = role
    return {
        "endpoint_url": endpoint,
        "labels": series_labels,
        "stats": {"total": total},
        "timeslices": slices if slices is not None else _slices(total, start_ns, end_ns),
    }


def _export(prompt: list[dict], cached: list[dict], *, endpoints: bool = True) -> dict:
    summary: dict = {
        "phase_time_ranges": {
            "profiling": {"start_ns": START, "end_ns": END},
            "warmup": {"start_ns": START - 120 * NS, "end_ns": START},
        }
    }
    if endpoints:
        urls = sorted({s["endpoint_url"] for s in prompt + cached})
        summary["endpoints_configured"] = urls
        summary["endpoints_successful"] = list(urls)
    return {
        "summary": summary,
        "metrics": {
            "trtllm_prompt_tokens": {"type": "counter", "series": prompt},
            "trtllm_prompt_cached_tokens": {"type": "counter", "series": cached},
        },
    }


def _with_warmup(export: dict, total: float) -> dict:
    """Add the same series to ``warmup_metrics`` with the given warmup-phase total."""
    export["warmup_metrics"] = {
        name: {
            "type": "counter",
            "series": [
                _series(
                    s["endpoint_url"],
                    s["labels"].get("dynamo_component"),
                    total,
                    start_ns=START - 120 * NS,
                    end_ns=START,
                    labels={k: v for k, v in s["labels"].items() if k != "dynamo_component"},
                )
                for s in entry["series"]
            ],
        }
        for name, entry in export["metrics"].items()
    }
    return export


def _run(export: dict, **kwargs) -> dict:
    _, nested, _ = compute_server_metrics(
        export, framework="dynamo-trt", records=[], **kwargs
    )
    return nested


def _rate(nested: dict) -> dict:
    return nested["validity"]["cache_rate"]


def _pair(prompt: float = 1000.0, cached: float = 900.0, **kwargs) -> dict:
    return _export(
        [_series(PRE[0], "prefill", prompt, **kwargs)],
        [_series(PRE[0], "prefill", cached, **kwargs)],
    )


class TestPairedCounterBounds:
    def test_cached_above_prompt_is_rejected_and_not_published(self) -> None:
        nested = _run(_pair(cached=1200.0))

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert nested["tokens"]["prompt_by_source"]["computed"] is None
        assert _rate(nested)["status"] == "REJECTED"
        assert "cached_exceeds_prompt" in _rate(nested)["reasons"]
        assert nested["sources"][0]["prefix_cache_hit_rate"] is None
        assert _rate(nested)["raw_rate"] == pytest.approx(1.2)

    def test_one_domain_above_prompt_is_not_offset_by_the_sum(self) -> None:
        export = _export(
            [_series(PRE[0], "prefill", 1000.0), _series(PRE[1], "prefill", 3000.0)],
            [_series(PRE[0], "prefill", 1200.0), _series(PRE[1], "prefill", 1000.0)],
        )

        nested = _run(export)

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert "cached_exceeds_prompt" in _rate(nested)["reasons"]

    def test_cached_in_a_domain_without_prompt_is_rejected(self) -> None:
        export = _export(
            [_series(PRE[0], "prefill", 1000.0), _series(PRE[1], "prefill", 0.0)],
            [_series(PRE[0], "prefill", 900.0), _series(PRE[1], "prefill", 5.0)],
        )

        nested = _run(export)

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert "cached_exceeds_prompt" in _rate(nested)["reasons"]

    @pytest.mark.parametrize(("cached", "expected"), [(0.0, 0.0), (1000.0, 1.0)])
    def test_zero_and_full_reuse_stay_publishable(self, cached, expected) -> None:
        nested = _run(_pair(cached=cached))

        assert nested["cache"]["overall_cache_hit_rate"] == pytest.approx(expected)
        assert _rate(nested)["status"] == "CONDITIONAL_INTERVAL"
        assert nested["tokens"]["prompt_by_source"]["computed"] == pytest.approx(
            1000.0 - cached
        )

    def test_idle_prefill_domain_does_not_block(self) -> None:
        export = _export(
            [_series(PRE[0], "prefill", 1000.0), _series(PRE[1], "prefill", 0.0)],
            [_series(PRE[0], "prefill", 900.0), _series(PRE[1], "prefill", 0.0)],
        )

        nested = _run(export)

        assert nested["cache"]["overall_cache_hit_rate"] == pytest.approx(0.9)


class TestDeploymentRolesAndCounterPairs:
    @pytest.mark.parametrize(
        "explicit_mode, expected", [(None, 0.9), ("decode", None), ("garbage", None)]
    )
    def test_aggregated_backend_component_does_not_override_explicit_mode(
        self, explicit_mode, expected
    ) -> None:
        labels = {"disaggregation_mode": explicit_mode} if explicit_mode else {}
        export = _export(
            [_series(PRE[0], "backend", 1000.0, labels=labels)],
            [_series(PRE[0], "backend", 900.0, labels=labels)],
        )
        nested = _run(export, disaggregated=False)

        assert nested["cache"]["overall_cache_hit_rate"] == expected
        assert nested["cache_validity"]["pricing_eligible"] is False
        assert nested["sources"][0]["prefix_cache_hit_rate"] == expected

    def test_non_role_components_do_not_conflict_with_explicit_worker_roles(self) -> None:
        export = _export(
            [
                _series(PRE[0], None, 1000.0, labels={"disaggregation_mode": "prefill"}),
                _series(DEC, None, 1000.0, labels={"disaggregation_mode": "decode"}),
            ],
            [
                _series(PRE[0], None, 900.0, labels={"disaggregation_mode": "prefill"}),
                _series(DEC, None, 1000.0, labels={"disaggregation_mode": "decode"}),
            ],
        )
        export["metrics"]["other_metric"] = {
            "type": "gauge",
            "series": [_series(PRE[0], "tensorrt_llm", 1.0), _series(DEC, "tensorrt_llm", 1.0)],
        }
        nested = _run(export, disaggregated=True)

        assert nested["cache"]["overall_cache_hit_rate"] == pytest.approx(0.9)
        assert _rate(nested)["role_scope"] == "prefill"
        assert {s["endpoint_url"] for s in _rate(nested)["excluded"]} == {DEC}

    def test_decode_only_is_not_treated_as_combined(self) -> None:
        export = _export(
            [_series(DEC, "backend", 1000.0)], [_series(DEC, "backend", 1000.0)]
        )

        nested = _run(export)

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert _rate(nested)["role_scope"] != "combined"
        assert "decode_only_no_prefill_scope" in _rate(nested)["reasons"]

    def test_unknown_role_beside_prefill_blocks_instead_of_being_excluded(self) -> None:
        export = _export(
            [_series(PRE[0], "prefill", 1000.0), _series(UNLABELLED, None, 1000.0)],
            [_series(PRE[0], "prefill", 900.0), _series(UNLABELLED, None, 0.0)],
        )

        nested = _run(export)

        rate = _rate(nested)
        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert "role_unknown" in rate["reasons"]
        assert {s["endpoint_url"] for s in rate["unresolved"]} == {UNLABELLED}
        assert UNLABELLED not in {s["endpoint_url"] for s in rate["excluded"]}

    def test_unlabelled_series_need_an_authoritative_deployment_mode(self) -> None:
        export = _export(
            [_series(PRE[0], None, 1000.0)], [_series(PRE[0], None, 900.0)]
        )

        unknown = _run(copy.deepcopy(export))
        unified = _run(copy.deepcopy(export), disaggregated=False)

        assert unknown["cache"]["overall_cache_hit_rate"] is None
        assert "deployment_mode_unknown" in _rate(unknown)["reasons"]
        assert unified["cache"]["overall_cache_hit_rate"] == pytest.approx(0.9)
        assert _rate(unified)["role_scope"] == "combined"

    def test_disaggregated_deployment_without_prefill_series_publishes_nothing(
        self,
    ) -> None:
        export = _export(
            [_series(PRE[0], None, 1000.0)], [_series(PRE[0], None, 900.0)]
        )

        nested = _run(export, disaggregated=True)

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert "no_prefill_scope" in _rate(nested)["reasons"]

    def test_role_labels_contradicting_the_deployment_block(self) -> None:
        nested = _run(_pair(), disaggregated=False)

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert "role_conflicts_with_deployment" in _rate(nested)["reasons"]

    def test_same_endpoint_with_different_rank_is_not_a_pair(self) -> None:
        export = _export(
            [_series(PRE[0], "prefill", 1000.0, labels={"rank": "0"})],
            [_series(PRE[0], "prefill", 900.0, labels={"rank": "1"})],
        )

        nested = _run(export)

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert "missing_pair" in _rate(nested)["reasons"]

    def test_instance_order_does_not_change_the_result(self) -> None:
        prompt = [
            _series(PRE[0], "prefill", 1000.0),
            _series(PRE[1], "prefill", 3000.0),
            _series(DEC, "backend", 4000.0),
        ]
        cached = [
            _series(PRE[0], "prefill", 900.0),
            _series(PRE[1], "prefill", 2100.0),
            _series(DEC, "backend", 4000.0),
        ]

        forward = _run(_export(prompt, cached))
        backward = _run(_export(prompt[::-1], cached[::-1]))

        assert forward["cache"] == backward["cache"]
        assert forward["cache"]["overall_cache_hit_rate"] == pytest.approx(0.75)
        assert _rate(forward)["status"] == _rate(backward)["status"]

    def test_missing_scrape_metadata_is_not_full_coverage(self) -> None:
        export = _export(
            [_series(PRE[0], "prefill", 1000.0)],
            [_series(PRE[0], "prefill", 900.0)],
            endpoints=False,
        )

        nested = _run(export)

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert "scrape_coverage_unknown" in _rate(nested)["reasons"]


class TestCounterAliases:
    def test_equal_total_aliases_with_different_windows_conflict_in_either_order(
        self,
    ) -> None:
        clean = _series(PRE[0], "prefill", 1000.0)
        early = _series(PRE[0], "prefill", 1000.0, start_ns=START - 120 * NS)
        outcomes = []
        for first, second in ((clean, early), (early, clean)):
            export = _export([copy.deepcopy(first)], [_series(PRE[0], "prefill", 900.0)])
            export["metrics"]["trtllm_prompt_tokens_total"] = {
                "type": "counter",
                "series": [copy.deepcopy(second)],
            }
            nested = _run(export)
            outcomes.append(
                (
                    nested["cache"]["overall_cache_hit_rate"],
                    _rate(nested)["status"],
                    "alias_conflict" in _rate(nested)["reasons"],
                )
            )

        assert outcomes[0] == outcomes[1] == (None, "REJECTED", True)

    def test_duplicate_series_under_one_name_is_not_an_alias(self) -> None:
        export = _export(
            [_series(PRE[0], "prefill", 1000.0), _series(PRE[0], "prefill", 1000.0)],
            [_series(PRE[0], "prefill", 900.0)],
        )

        nested = _run(export)

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert "duplicate_series" in _rate(nested)["reasons"]
        assert "alias_conflict" not in _rate(nested)["reasons"]


class TestProfilingWindowEvidence:
    def test_activity_after_profiling_end_is_not_valid(self) -> None:
        nested = _run(_pair(end_ns=END + 100 * NS))

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert _rate(nested)["status"] == "PHASE_CONTAMINATED"
        assert "activity_after_profiling" in _rate(nested)["reasons"]

    def test_nonpositive_duration_slice_is_rejected(self) -> None:
        def backward(total: float) -> list[dict]:
            return [
                {"start_ns": START, "end_ns": START - NS, "total": total / 2},
                {"start_ns": START - NS, "end_ns": END, "total": total / 2},
            ]

        export = _export(
            [_series(PRE[0], "prefill", 1000.0, slices=backward(1000.0))],
            [_series(PRE[0], "prefill", 900.0, slices=backward(900.0))],
        )

        nested = _run(export)

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert _rate(nested)["status"] == "REJECTED"
        assert "slice_nonpositive_duration" in _rate(nested)["reasons"]

    def test_activity_one_second_before_profiling_is_not_valid(self) -> None:
        nested = _run(_pair(start_ns=START - NS))

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert _rate(nested)["status"] == "PHASE_CONTAMINATED"
        assert "activity_before_profiling" in _rate(nested)["reasons"]

    def test_activity_in_a_slice_straddling_the_start_is_uncertain(self) -> None:
        nested = _run(_pair(start_ns=START - NS // 2))

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert _rate(nested)["status"] == "BOUNDARY_UNCERTAIN"
        assert "activity_straddles_profiling_start" in _rate(nested)["reasons"]

    def test_idle_slices_outside_the_window_need_zero_warmup_evidence(self) -> None:
        def padded(total: float) -> list[dict]:
            inside = _slices(total)
            before = [
                {"start_ns": START - k * NS, "end_ns": START - (k - 1) * NS, "total": 0.0}
                for k in (2, 1)
            ]
            after = [
                {"start_ns": END + k * NS, "end_ns": END + (k + 1) * NS, "total": 0.0}
                for k in (0, 1)
            ]
            return before + inside + after

        export = _export(
            [_series(PRE[0], "prefill", 1000.0, slices=padded(1000.0))],
            [_series(PRE[0], "prefill", 900.0, slices=padded(900.0))],
        )
        with_zero_warmup = _with_warmup(copy.deepcopy(export), 0.0)

        without = _run(export)
        idle_warmup = _run(with_zero_warmup)

        assert without["cache"]["overall_cache_hit_rate"] is None
        assert "window_opens_before_profiling" in _rate(without)["reasons"]
        assert idle_warmup["cache"]["overall_cache_hit_rate"] == pytest.approx(0.9)
        assert _rate(idle_warmup)["status"] == "CONDITIONAL_INTERVAL"

    def test_warmup_jump_landing_after_the_start_is_not_hidden(self) -> None:
        def early_with_jump(profiling_total: float, warmup_total: float) -> list[dict]:
            flat = [
                {"start_ns": START - k * NS, "end_ns": START - (k - 1) * NS, "total": 0.0}
                for k in range(120, 0, -1)
            ]
            inside = _slices(profiling_total)
            inside[1]["total"] += warmup_total
            return flat + inside

        export = _export(
            [_series(PRE[0], "prefill", 1500.0, slices=early_with_jump(1000.0, 500.0))],
            [_series(PRE[0], "prefill", 1350.0, slices=early_with_jump(900.0, 450.0))],
        )

        nested = _run(_with_warmup(export, 500.0))

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert _rate(nested)["status"] == "PHASE_CONTAMINATED"
        assert "window_opens_before_profiling" in _rate(nested)["reasons"]

    def test_window_opening_within_one_slice_needs_zero_warmup_evidence(self) -> None:
        def early_by_half(total: float) -> list[dict]:
            first = {"start_ns": START - NS // 2, "end_ns": START + NS // 2, "total": 0.0}
            rest = _slices(total, START + NS // 2, END)
            return [first] + rest

        export = _export(
            [_series(PRE[0], "prefill", 1000.0, slices=early_by_half(1000.0))],
            [_series(PRE[0], "prefill", 900.0, slices=early_by_half(900.0))],
        )

        busy_warmup = _run(_with_warmup(copy.deepcopy(export), 50.0))
        idle_warmup = _run(_with_warmup(copy.deepcopy(export), 0.0))

        assert busy_warmup["cache"]["overall_cache_hit_rate"] is None
        assert _rate(busy_warmup)["status"] == "BOUNDARY_UNCERTAIN"
        assert "window_opens_before_profiling" in _rate(busy_warmup)["reasons"]
        assert idle_warmup["cache"]["overall_cache_hit_rate"] == pytest.approx(0.9)

    def test_late_start_needs_warmup_reference_evidence(self) -> None:
        late = _pair(start_ns=START + NS // 5)
        with_reference = _with_warmup(copy.deepcopy(late), 50.0)

        without = _run(late)
        with_ref = _run(with_reference)

        assert without["cache"]["overall_cache_hit_rate"] is None
        assert "window_starts_after_profiling" in _rate(without)["reasons"]
        assert with_ref["cache"]["overall_cache_hit_rate"] == pytest.approx(0.9)

    def test_numerator_and_denominator_need_the_same_window(self) -> None:
        export = _export(
            [_series(PRE[0], "prefill", 1000.0)],
            [_series(PRE[0], "prefill", 900.0, end_ns=END - 30 * NS)],
        )

        nested = _run(export)

        assert nested["cache"]["overall_cache_hit_rate"] is None
        assert {"window_ends_before_profiling", "window_support_mismatch"} <= set(
            _rate(nested)["reasons"]
        )


class TestTelemetryVersusEligibility:
    @pytest.mark.parametrize(
        ("export", "status", "published"),
        [
            (_pair(), "CONDITIONAL_INTERVAL", True),
            (_pair(cached=1200.0), "REJECTED", False),
        ],
    )
    def test_publication_follows_status_and_exact_stays_false(
        self, export, status, published
    ) -> None:
        nested = _run(copy.deepcopy(export))

        rate = _rate(nested)
        assert (rate["status"], rate["published"]) == (status, published)
        assert rate["eligible_exact_actual"] is False
        assert rate["cached_tokens"] is not None and rate["prompt_tokens"] is not None
