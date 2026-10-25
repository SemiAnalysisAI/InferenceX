"""Tests for the adaptive-search sweep result writer (infx.results.sweep)."""

import json
from pathlib import Path
from typing import Any

from infx.serving.sweep_result import build_sweep_result, cache_check, probe_metrics, summarize_search

FIX = Path(__file__).parent / "fixtures" / "sweep"


def hist(fm, im, reason, boundary_type=None, n=7):
    bs = {"swept_dim_path": "phases.profiling.concurrency", "feasible_max": fm, "infeasible_min": im}
    if boundary_type:
        bs["boundary_type"] = boundary_type
        bs["binding_constraint"] = "time_to_first_token:avg"
    return {"boundary_summary": bs, "convergence_reason": reason, "iterations": [{}] * n}


def test_ok():
    out = summarize_search(hist({"value": 24}, {"value": 25}, "smooth_isotonic_precision_reached", "smooth"))
    assert (out["state"], out["max_conc"], out["infeasible_min"], out["iterations"]) == ("ok", 24, 25, 7)
    assert out["binding_constraint"] == "time_to_first_token:avg"


def test_no_pass_in_range_keeps_row_with_null_conc():
    out = summarize_search(hist(None, {"value": 1}, "smooth_isotonic_no_pass_in_range"))
    assert out["state"] == "no_pass_in_range" and out["max_conc"] is None


def test_no_failure_in_range_means_ceiling_hit():
    out = summarize_search(hist({"value": 512}, None, "smooth_isotonic_no_failure_in_range"))
    assert out["state"] == "no_failure_in_range" and out["max_conc"] == 512


def test_cliff_and_max_iterations():
    assert summarize_search(hist({"value": 40}, {"value": 64}, "smooth_isotonic_cliff_precision_reached", "cliff"))["state"] == "cliff"
    assert summarize_search(hist({"value": 40}, {"value": 44}, "max_iterations", "smooth"))["state"] == "max_iterations"


def test_null_boundary_summary_is_reported_not_raised():
    out = summarize_search({"boundary_summary": None, "convergence_reason": None, "iterations": []})
    assert out["state"] == "no_data" and out["max_conc"] is None


def test_probe_metrics_from_captured_fixture():
    # Values below were read by hand from the committed fixture (Task 1 spike
    # capture, the winning probe at concurrency 256): request_throughput.avg,
    # output_token_throughput.avg, time_to_first_token.avg/p90,
    # inter_token_latency.avg/p90.
    probe = json.loads((FIX / "profile_export_aiperf_probe_concurrency256.json").read_text())
    got = probe_metrics(probe, conc=256)
    assert got["rps"] == 9.681655869773321
    assert got["output_tps"] == 4957.00780532394
    assert got["tpm"] == round(got["output_tps"] * 60)
    assert got["ttft_avg_ms"] == 3388.5959436499998
    assert got["ttft_p90_ms"] == 7759.276425499999
    assert got["itl_avg_ms"] == 43.99354467741878
    assert got["itl_p90_ms"] == 48.15925213209393


def test_probe_metrics_missing_metric_raises_keyerror():
    import pytest

    with pytest.raises(KeyError):
        probe_metrics({"request_throughput": {"avg": 1.0}}, conc=256)


def test_cache_mismatch_flag():
    assert cache_check(nominal_pct=90, measured=0.5) == {
        "nominal_cache_hit": 0.9,
        "measured_cache_hit": 0.5,
        "cache_mismatch": True,
    }
    assert cache_check(nominal_pct=90, measured=0.85)["cache_mismatch"] is False
    assert cache_check(nominal_pct=0, measured=None)["cache_mismatch"] is None


def _env(**overrides: str) -> dict[str, str]:
    env: dict[str, str] = {
        "MODEL": "Qwen/Qwen3.8-27B",
        "TP": "2",
        "IS_MULTINODE": "false",
        "SERVING_SCENARIO_LABEL": "sweep-isl8k-osl512-cache50",
        "RESULT_FILENAME": "sweep_test",
    }
    env.update(overrides)
    return env


def _fixture_history() -> dict[str, Any]:
    history = json.loads((FIX / "search_history.json").read_text())
    # Trim iterations so the canned count is asserted, not the fixture's 16.
    history["iterations"] = [{}] * 16
    return history


def test_build_sweep_result_merges_probe_and_identity():
    probe = json.loads((FIX / "profile_export_aiperf_probe_concurrency256.json").read_text())
    out = build_sweep_result(_fixture_history(), probe, _env(), measured_cache_hit=0.44)
    assert out["state"] == "ok"
    assert out["max_conc"] == 256 and out["conc"] == 256
    assert out["infeasible_min"] == 264
    assert out["iterations"] == 16
    assert out["rps"] == 9.681655869773321
    assert out["tpm"] == 297420
    assert out["scenario"] == "sweep-isl8k-osl512-cache50"
    assert out["slo"] == [
        {"metric_tag": "time_to_first_token", "stat": "avg", "op": "lt", "threshold": 8000.0},
        {"metric_tag": "inter_token_latency", "stat": "p90", "op": "lt", "threshold": 50.0},
    ]
    # cacheNN suffix of the scenario: nominal 50% vs measured 44% -> no mismatch.
    assert out["nominal_cache_hit"] == 0.5
    assert out["measured_cache_hit"] == 0.44
    assert out["cache_mismatch"] is False
    # Identity fields copied the same way build_result copies them from env.
    assert out["model"] == "Qwen/Qwen3.8-27B"
    assert out["num_gpus"] == 2 and out["tp"] == 2
    assert out["kv_offloading"] is None


def test_build_sweep_result_without_winner_needs_no_probe_export():
    history = hist(None, {"value": 1}, "smooth_isotonic_no_pass_in_range")
    out = build_sweep_result(history, {}, _env(SERVING_SCENARIO_LABEL="sweep-no-cache"))
    assert out["state"] == "no_pass_in_range"
    assert "rps" not in out
    # No cacheNN suffix -> no cache axis in the result.
    assert "cache_mismatch" not in out
