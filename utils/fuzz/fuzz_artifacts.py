import copy
import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import example, given, strategies as st

from infx.results.agentic.artifacts import load_records_with_accounting
from infx.results.agentic.server_metrics import compute_server_metrics
from infx.results.fixed_sequence import build_result
from infx.results.result_filename import point_filename, result_stem

from cases import TEXT


@pytest.mark.parametrize("layout,parallelism,expected", [
    ("single", {"TP": "8"}, (90, 60, 30)),
    ("single", {"TP": "4", "PP_SIZE": "2", "PCP_SIZE": "3"}, (30, 20, 10)),
    ("disaggregate", {"PREFILL_GPUS": "8", "DECODE_GPUS": "16"}, (30, 30, 30)),
    ("aggregate", {"AGGREGATE_GPUS": "16"}, (45, 30, 15)),
])
@given(scale=st.integers(1, 10_000), latency=st.integers(1, 10_000), dcp=st.integers(1, 16),
       identity=TEXT)
def test_result_units_and_gpu_denominators(layout, parallelism, expected, scale, latency, dcp, identity):
    env = {"RUNNER_TYPE": "fixture", "FRAMEWORK": "sglang", "PRECISION": "fp8",
           "SPEC_DECODING": "none", "ISL": "8192", "OSL": "1024", "MODEL_PREFIX": "fixture",
           "IMAGE": identity, "RECIPE_FINGERPRINT": identity, "DISAGG": str(layout == "disaggregate"),
           "TP": "8", "EP_SIZE": "1", "DP_ATTENTION": "false", "DCP_SIZE": str(dcp)}
    if layout != "single":
        env.update(IS_MULTINODE="true", PREFILL_GPUS="0", DECODE_GPUS="0")
        for role in ("PREFILL", "DECODE"):
            env.update({f"{role}_NUM_WORKERS": "1", f"{role}_TP": "8", f"{role}_EP": "1",
                        f"{role}_DP_ATTN": "false", f"{role}_DCP_SIZE": str(dcp)})
    env.update(parallelism)
    benchmark = {"max_concurrency": 16, "model_id": "example/model", "total_token_throughput": 720 * scale,
                 "output_throughput": 480 * scale, "mean_ttft_ms": 800 * latency, "mean_tpot_ms": 20 * latency}
    original = copy.deepcopy((env, benchmark))
    result = build_result(benchmark, env)
    for name, value in zip(("tput_per_gpu", "output_tput_per_gpu", "input_tput_per_gpu"), expected):
        assert result[name] == pytest.approx(value * scale)
    assert result["mean_ttft"] == pytest.approx(0.8 * latency)
    assert result["mean_intvty"] == pytest.approx(50 / latency)
    assert result["image"] == result["recipe_fingerprint"] == identity
    assert (env, benchmark) == original


@given(base=TEXT, config=TEXT, fingerprint=TEXT, conc=st.integers(1, 10**9),
       gpus=st.integers(1, 100_000), context=st.integers(1, 10**9), generation=st.integers(1, 10**9))
@example(base="模型" * 100, config="config" * 100, fingerprint="a" * 64,
         conc=1024, gpus=128, context=131072, generation=16384)
def test_artifact_names_survive_filesystem_and_retain_point_identity(base, config, fingerprint, conc, gpus, context, generation):
    stem = result_stem(base, fingerprint)
    filename = point_filename(stem, config, str(conc), str(gpus), str(context), str(generation))
    assert filename.endswith(f"_conc{conc}_gpus_{gpus}_ctx_{context}_gen_{generation}.json")
    assert len(stem.encode()) <= 120
    wrapped = "power_validation_" + filename + ".tmp"
    assert len(wrapped.encode()) <= 255
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / wrapped
        path.write_text("{}")
    assert result_stem(base, fingerprint + "x") != stem


@given(good=st.integers(0, 20), warmup=st.integers(0, 20), failed=st.integers(1, 20),
       warmup_failed=st.integers(0, 20), whitespace=st.text(alphabet=" \t\r\n", min_size=1, max_size=30),
       wrapped=st.booleans(), legacy=st.booleans())
def test_agentic_accounting_drops_failed_and_warmup_records(good, warmup, failed, warmup_failed, whitespace, wrapped, legacy):
    accepted = [{"id": i, **({} if legacy else {"metadata": {"benchmark_phase": "profiling"}})}
                for i in range(good)]
    error = {"message": whitespace} if wrapped else whitespace
    records = [*accepted, *[{"metadata": {"benchmark_phase": "warmup"}} for _ in range(warmup)],
               *[{"error": error} for _ in range(failed)],
               *[{"metadata": {"benchmark_phase": "warmup"}, "error": error} for _ in range(warmup_failed)]]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profile_export.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in reversed(records)) + "\n\n")
        actual, accounting = load_records_with_accounting(path)
    assert actual == list(reversed(accepted))
    assert accounting == {"records_total": good + warmup + failed + warmup_failed,
                          "records_profiled": good, "records_dropped_total": warmup + failed + warmup_failed,
                          "records_warmup_dropped": warmup + warmup_failed,
                          "records_error_dropped": failed + warmup_failed,
                          "error_categories": {"unknown": failed + warmup_failed}}


@pytest.mark.parametrize("framework", ["sglang", "vllm", "dynamo-vllm"])
@given(capacities=st.lists(st.integers(1, 1_000_000), min_size=1, max_size=8),
       spacing=st.text(alphabet=" \t", min_size=1, max_size=6), multiple_logs=st.booleans())
def test_server_capacity_counts_latest_rank_once_per_log(framework, capacities, spacing, multiple_logs):
    logs = []
    for factor in ([1, 2] if multiple_logs else [1]):
        lines = ["INFO unrelated startup message"]
        for multiplier, separator in ((2, " "), (1, spacing)):
            for rank, capacity in enumerate(capacities):
                tokens = capacity * factor * multiplier
                if framework == "sglang":
                    tag = separator.join((f"DP{rank}", "TP0", "EP0"))
                    lines.append(f"[2026-01-01 00:00:00 {tag}] max_total_num_tokens={tokens}")
                else:
                    lines.append(f"INFO{separator}(EngineCore_DP{rank} pid=123) GPU KV cache size: {tokens:,} tokens")
        logs.append("\n".join(lines))
    _, metrics, _ = compute_server_metrics({}, framework=framework, records=[], server_logs=logs)
    assert metrics["kv_cache"]["gpu_total_tokens"] == sum(capacities) * (3 if multiple_logs else 1)
