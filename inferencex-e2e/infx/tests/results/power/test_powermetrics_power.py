"""Joining a Mac's powermetrics samples to one AgentX window."""

from __future__ import annotations

import json

import pytest

from infx.results.power import WHOLE_METRIC_KEYS, powermetrics

START = 1_000.0
END = 1_060.0


@pytest.fixture
def run(tmp_path):
    """Write the samples, window and aggregate, then validate them; returns the outcome."""

    def validate(rows, window=None):
        samples = tmp_path / "samples.csv"
        if rows is not None:
            samples.write_text("timestamp_unix,combined_w,cpu_w,gpu_w,ane_w\n" + "".join(rows))
        bench = tmp_path / "agentic_power_concurrency_1.json"
        bench.write_text(json.dumps(window or {
            "benchmark_start_time_unix": START, "benchmark_end_time_unix": END, "duration": END - START,
            "completed": 10, "total_input_tokens": 5_000, "total_output_tokens": 1_000,
        }))  # fmt: skip
        aggregate = tmp_path / "point.json"
        aggregate.write_text('{"model": "m", "avg_power_w": 1.0}')
        sidecar = tmp_path / "power_validation_point.json"
        valid = powermetrics.validate(
            samples_path=samples, bench_result=bench, agg_result=aggregate,
            validation_result=sidecar, audit_source=sidecar.name,
        )  # fmt: skip
        return valid, json.loads(aggregate.read_text()), json.loads(sidecar.read_text())

    return validate


def steady(start=START - 2, end=END + 2, watts=100.0, step=1.0):
    count = int((end - start) / step) + 1
    return [f"{start + step * i},0,0,{watts},0\n" for i in range(count)]


def test_steady_samples_give_the_windows_energy_per_query_and_per_token(run):
    valid, aggregate, sidecar = run(steady())

    assert valid is True
    assert aggregate["model"] == "m"
    assert aggregate["power_valid"] == 1
    assert aggregate["avg_power_w"] == aggregate["avg_total_gpu_power_w"] == 100.0
    assert aggregate["p75_power_w"] == pytest.approx(100.0)
    assert aggregate["p90_total_gpu_power_w"] == pytest.approx(100.0)
    assert aggregate["total_gpu_energy_j"] == 6_000.0
    assert aggregate["joules_per_successful_query"] == 600.0
    assert aggregate["joules_per_input_token"] == 1.2
    assert aggregate["joules_per_output_token"] == 6.0
    assert aggregate["joules_per_total_token"] == 1.0
    assert aggregate["power_audit"]["source"] == "power_validation_point.json"
    assert aggregate["power_audit"]["sample_count"] == 61
    assert sidecar["power_valid"] is True
    assert sidecar["reasons"] == []
    assert sidecar["power_scope"] == "gpu"
    assert sidecar["benchmark_window"]["start_time_unix"] == START
    assert sidecar["metrics"]["total_gpu_energy_j"] == 6_000.0


def test_power_is_time_weighted_over_the_window_alone(run):
    before = f"{START - 1},0,0,500,0\n"
    after = [f"{END},0,0,300,0\n", f"{END + 1},0,0,900,0\n"]

    valid, aggregate, _ = run([before, *steady(START, END - 1), *after])

    assert valid is True
    assert aggregate["total_gpu_energy_j"] == pytest.approx(100.0 * 59 + 200.0)


@pytest.mark.parametrize(("rows", "window", "reason"), [
    (None, None, "samples_csv_missing"),
    ([], None, "measurement_window_not_bracketed"),
    (["not,a,row\n"], None, "samples_csv_malformed"),
    (["nan,0,0,1,0\n"], None, "samples_csv_malformed"),
    ([f"{START},0,0,1,0\n", f"{START},0,0,1,0\n", f"{END},0,0,1,0\n"], None, "samples_csv_malformed"),
    (steady(start=START + 1), None, "measurement_window_not_bracketed"),
    (steady(end=END - 1), None, "measurement_window_not_bracketed"),
    ([f"{START},0,0,1,0\n", f"{START + 3.5},0,0,1,0\n", f"{END},0,0,1,0\n"], None, "sample_gap_exceeded"),
    (steady(), {"benchmark_start_time_unix": START}, "invalid_benchmark_window"),
], ids=["missing", "empty", "malformed", "not-finite", "not-increasing", "starts-late", "ends-early", "gap", "bad-window"])  # fmt: skip
def test_power_that_cannot_be_trusted_is_invalid_and_publishes_no_metric(run, rows, window, reason):
    valid, aggregate, sidecar = run(rows, window)

    assert valid is False
    assert aggregate["power_valid"] == 0
    assert not set(aggregate) & set(WHOLE_METRIC_KEYS)
    assert sidecar["power_valid"] is False
    assert reason in sidecar["reasons"]
    assert sidecar["metrics"] == {}
