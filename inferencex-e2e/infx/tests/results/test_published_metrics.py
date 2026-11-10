"""A published row's derived interactivity, frontier, line, GPUs, tok/$ and engine source."""

import argparse

import pytest

from infx.results import published_metrics as metrics


def point(id, itl, tput, full=True):
    key = "p90_full_response_itl" if full else "p90_itl"
    return {"id": id, "benchmark_type": "agentic_traces", "metrics": {key: itl, "tput_per_gpu": tput}}


def test_interactivity_is_one_over_p90_itl_full_response_first():
    assert metrics.interactivity(point("a", 0.004, 1)) == 250
    assert metrics.interactivity(point("a", 0.005, 1, full=False)) == 200
    partial = {"benchmark_type": "agentic_traces",
               "metrics": {"p90_itl": 0.005, "p75_full_response_itl": 0.004}}
    assert metrics.interactivity(partial) is None
    assert metrics.interactivity({"benchmark_type": "agentic_traces", "metrics": {"p90_itl": 0}}) is None


def test_a_fixed_sequence_row_plots_its_published_p90_interactivity():
    row = {"benchmark_type": "single_turn", "metrics": {"p90_itl": 0.004, "p90_intvty": 310.5}}

    assert metrics.interactivity(row) == 310.5
    assert metrics.interactivity({**row, "metrics": {"p90_itl": 0.004}}) is None


def test_frontier_keeps_rows_no_other_beats_on_interactivity_and_throughput():
    rows = [
        point("slow-big", 0.010, 900), point("mid", 0.005, 500), point("dominated", 0.005, 400),
        point("tie-low", 0.004, 100), point("tie-high", 0.004, 300), point("fast", 0.002, 50),
        point("beaten", 0.008, 450), {"id": "no-metrics", "metrics": {}},
    ]

    assert [row["id"] for row in metrics.frontier(rows)] == ["slow-big", "mid", "tie-high", "fast"]


def test_frontier_takes_a_lower_better_latency_axis():
    rows = [
        {"id": id, "metrics": {"p90_e2el": e2el, "tput_per_gpu": tput}}
        for id, e2el, tput in [("fast", 10, 100), ("dominated", 20, 90), ("big", 30, 400),
                               ("no-e2el", None, 999)]
    ]

    assert [row["id"] for row in metrics.frontier(rows, ("p90_e2el", "tput_per_gpu"))] == [
        "big", "fast"]
    assert not metrics.plotted(rows[3], ("p90_e2el", "tput_per_gpu"))


def test_frontier_axes_name_known_metrics_or_fail_listing_them():
    assert metrics.frontier_axes("p90_ttft, tok_per_usd") == ("p90_ttft", "tok_per_usd")
    with pytest.raises(argparse.ArgumentTypeError, match="median_intvty.*p90_e2el.*tok_per_usd"):
        metrics.frontier_axes("p90_ttft,dollars")


def test_lines_split_a_curve_by_date_and_a_fixed_sequence_by_spec_method():
    agentic = {"hardware": "b300", "framework": "sglang", "precision": "fp4",
               "benchmark_type": "agentic_traces", "spec_method": "mtp", "curve_date": "2026-10-12"}
    fixed = {**agentic, "benchmark_type": "single_turn", "isl": 8192, "osl": 1024}

    assert metrics.line_of(agentic) == metrics.line_of({**agentic, "spec_method": "none"})
    assert metrics.line_of(agentic) != metrics.line_of({**agentic, "curve_date": "2026-10-29"})
    assert metrics.line_of(fixed) != metrics.line_of({**fixed, "spec_method": "none"})


@pytest.mark.parametrize("fields,gpus", [
    ({"disagg": False, "prefill_tp": 8, "prefill_ep": 8, "num_prefill_gpu": 64}, 8),
    ({"disagg": False, "prefill_tp": 4, "prefill_ep": 1, "num_prefill_gpu": 4}, 4),
    ({"disagg": False, "prefill_tp": 4, "prefill_ep": 4, "num_prefill_gpu": 4}, 4),
    ({"disagg": True, "num_prefill_gpu": 8, "num_decode_gpu": 16}, 24),
    ({"disagg": True, "num_prefill_gpu": 8}, None),
])
def test_physical_gpus_counts_shared_tp_and_ep_gpus_once(fields, gpus):
    assert metrics.physical_gpus(fields) == gpus


def test_published_adds_curve_gpus_price_and_engine():
    row = {
        "id": "1", "hardware": "h100", "framework": "dynamo-sglang", "precision": "fp4",
        "benchmark_type": "agentic_traces", "disagg": False, "prefill_tp": 8, "prefill_ep": 8,
        "num_prefill_gpu": 64, "image": "lmsysorg/sglang:nightly-dev-20261016-a93f16f0@sha256:00",
        "metrics": {"tput_per_gpu": 100.0, "total_tput_tps": 800.0, "p90_itl": 0.01},
    }

    entry = metrics.published(row, {"h100": {"h": 2.0, "r": 4.0}})

    assert entry["num_prefill_gpu"] == 64 and entry["physical_gpus"] == 8
    assert entry["tput_per_physical_gpu"] == 100.0
    assert (entry["tok_per_usd"], entry["price_key"]) == (180000.0, "h")
    assert entry["curve"]["hardware"] == "h100"
    assert entry["engine"] == {"repository": "sgl-project/sglang", "commit": "a93f16f0"}
    assert metrics.published(row, {})["tok_per_usd"] is None


@pytest.mark.parametrize("framework,image,source", [
    ("dynamo-trt", "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc28", ("NVIDIA/TensorRT-LLM", None)),
    ("vllm", "vllm/vllm-openai:nightly-20261019", ("vllm-project/vllm", None)),
    ("atom", "rocm/atom:v1-0123abc", (None, "0123abc")),
])
def test_engine_source_reads_a_commit_but_not_a_date_from_the_image_tag(framework, image, source):
    found = metrics.engine_source({"framework": framework, "image": image})
    assert (found["repository"], found["commit"]) == source
