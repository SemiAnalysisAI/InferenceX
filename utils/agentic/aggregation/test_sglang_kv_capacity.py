from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from infx.results.agentic.server_metrics import compute_server_metrics


# Capacity-only rows from run 34738529222's c4 and c256 CSV exports.
FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures/sglang-kv-capacity-pr2823.json").read_text()
)


def capacity(metrics, logs=()):
    _, nested, _ = compute_server_metrics(
        {"metrics": metrics}, framework="sglang", records=[], server_logs=logs
    )
    return nested["kv_cache"]["gpu_total_tokens"]


@pytest.mark.parametrize("concurrency,expected", [("4", 6_338_048), ("256", 241_231_872)])
def test_capacity_counts_logical_pools_in_real_exports(concurrency, expected):
    assert capacity(FIXTURES[concurrency]) == expected


def test_complete_metrics_take_precedence_over_incomplete_role_logs():
    assert capacity(FIXTURES["4"], ["max_total_num_tokens=3211776, dp_size=1"]) == 6_338_048


@pytest.mark.parametrize("mutation", ["missing_endpoint", "invalid_rank", "missing_rank", "missing_stats", "changing", "disagreement", "nan"])
def test_ambiguous_ranked_capacity_is_not_partially_summed(mutation):
    metrics = deepcopy(FIXTURES["4"])
    series = metrics["sglang:max_total_num_tokens"]["series"][0]
    if mutation == "missing_endpoint":
        del series["endpoint_url"]
    elif mutation == "invalid_rank":
        series["labels"]["tp_rank"] = "unknown"
    elif mutation == "missing_rank":
        del series["labels"]["tp_rank"]
    elif mutation == "missing_stats":
        del series["stats"]
    elif mutation == "changing":
        series["stats"]["min"] = 123
    elif mutation == "disagreement":
        series["stats"] = {"min": 123, "max": 123, "avg": 123}
    else:
        series["stats"]["max"] = float("nan")
    assert capacity(metrics) is None


def test_worker_identity_retains_pools_sharing_one_endpoint():
    metrics = deepcopy(FIXTURES["4"])
    for series in metrics["sglang:max_total_num_tokens"]["series"]:
        series["labels"]["worker_id"] = series["endpoint_url"]
        series["endpoint_url"] = "http://shared-metrics/metrics"
        series["labels"]["engine_type"] = "aggregated"
    assert capacity(metrics) == 6_338_048


def test_legacy_unranked_series_remain_independent():
    assert capacity({"sglang:max_total_num_tokens": {"series": [
        {"stats": {"max": 1000}}, {"stats": {"max": 1200}},
    ]}}) == 2200
