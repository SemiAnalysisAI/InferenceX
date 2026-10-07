"""Published-row contract: producer stamps, collector quarantine, and committed JSON Schemas."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from infx.results import agentic, fixed_sequence
from infx.results.agentic.common import round_floats
from infx.results.evals import build_row
from infx.results.power import ALL_POWER_METRIC_KEYS, POWER_METRIC_SCHEMA_VERSION, with_power_metrics
from infx.results.power.audit import audit_summary
from infx.results.schema.models import AGENTX_ROW, EVAL_ROW, FIXED_SEQUENCE_ROW, RUN_STATS_ROW
from infx.results.schema.quarantine import row_errors

PROJECT = Path(__file__).resolve().parents[3]

BENCHMARK = {
    "model_id": "deepseek-ai/DeepSeek-R1-0528",
    "max_concurrency": 64,
    "num_prompts": 640,
    "completed": 640,
    "benchmark_outcome": {
        "status": "passed", "requested": 640, "completed": 640, "failed": 0,
        "max_failure_rate": 0.05,
    },
    "total_token_throughput": 16000.0,
    "output_throughput": 8000.0,
    "mean_ttft_ms": 120.0,
    "p99.9_ttft_ms": 350.0,
    "mean_tpot_ms": 20.0,
    "std_tpot_ms": 2.0,
    "median_e2el_ms": 5000.0,
}
FIXED_ENV = {
    "RUNNER_TYPE": "b200", "FRAMEWORK": "sglang", "PRECISION": "fp8", "SPEC_DECODING": "none",
    "ISL": "1024", "OSL": "1024", "DISAGG": "false", "MODEL_PREFIX": "dsr1",
    "IMAGE": "lmsysorg/sglang:v0.5", "RECIPE_FINGERPRINT": "a" * 64,
    "TP": "8", "EP_SIZE": "1", "DP_ATTENTION": "false",
}
MULTINODE_ENV = {
    **FIXED_ENV, "RUNNER_TYPE": "gb200", "FRAMEWORK": "dynamo-sglang", "IS_MULTINODE": "true",
    "DISAGG": "true", "PREFILL_GPUS": "8", "DECODE_GPUS": "16", "PREFILL_NUM_WORKERS": "2",
    "PREFILL_TP": "4", "PREFILL_EP": "4", "PREFILL_DP_ATTN": "true", "DECODE_NUM_WORKERS": "1",
    "DECODE_TP": "16", "DECODE_EP": "16", "DECODE_DP_ATTN": "false",
    "ROUTER_METADATA": '{"name": "dynamo-router", "version": "1.5.0"}', "KV_P2P_TRANSFER": "nixl",
}
# A shared-worker point: no decode GPUs, so decode TP/EP are zero.
AGGREGATE_ENV = {
    **MULTINODE_ENV, "DISAGG": "false", "PREFILL_GPUS": "0", "DECODE_GPUS": "0",
    "AGGREGATE_GPUS": "8", "DECODE_NUM_WORKERS": "0",
}
AGENTX_ENV = {
    "RUNNER_TYPE": "cluster:b200-nv", "CONC": "8", "IMAGE": "vllm/vllm-openai:v0.11",
    "RECIPE_FINGERPRINT": "b" * 64, "MODEL": "deepseek-ai/DeepSeek-V4-Pro", "MODEL_PREFIX": "dsv4",
    "FRAMEWORK": "vllm", "PRECISION": "fp4", "SPEC_DECODING": "mtp", "DISAGG": "false",
    "IS_MULTINODE": "false", "TP": "8", "EP_SIZE": "8", "DP_ATTENTION": "true",
    "KV_OFFLOADING": "dram", "KV_OFFLOAD_BACKEND": "lmcache",
    "KV_OFFLOAD_BACKEND_METADATA": '{"name": "lmcache"}', "TOTAL_CPU_DRAM_GB": "512",
}
AGENTX_MULTINODE_ENV = {
    **AGENTX_ENV, "FRAMEWORK": "dynamo-sglang", "IS_MULTINODE": "true", "DISAGG": "true",
    "PREFILL_NUM_WORKERS": "1", "PREFILL_TP": "8", "PREFILL_EP": "8", "PREFILL_DP_ATTN": "true",
    "DECODE_NUM_WORKERS": "1", "DECODE_TP": "16", "DECODE_EP": "16", "DECODE_DP_ATTN": "true",
    "ROUTER_METADATA": '{"name": "dynamo-router", "version": "1.5.0"}',
    "KV_P2P_TRANSFER": "mooncake",
}


def fixed_row(env=FIXED_ENV):
    return fixed_sequence.build_result(BENCHMARK, env)


def multinode_row():
    row = with_power_metrics(
        fixed_row(MULTINODE_ENV), metric_keys=ALL_POWER_METRIC_KEYS,
        schema_version=POWER_METRIC_SCHEMA_VERSION, power_valid=True,
        metrics={"avg_power_w": 700.0, "p90_total_gpu_power_w": 16800.0,
                 "total_gpu_energy_j": 1.2e6, "prefill_avg_power_w": 650.0,
                 "decode_joules_per_output_token": 0.5},
    )
    validation = {"reasons": [], "expected_gpu_count": 24, "observed_gpu_count": 24,
                  "per_gpu_role": {"gpu-0": "prefill"}}
    return row | audit_summary(validation, "power_validation_point.json")


def record(conversation, turn, start_s):
    return {
        "metadata": {
            "conversation_id": conversation, "turn_index": turn, "benchmark_phase": "profiling",
            "request_start_ns": int(start_s * 1e9), "request_end_ns": int((start_s + 1) * 1e9),
        },
        "metrics": {
            "input_sequence_length": {"value": 100 + turn, "unit": "tokens"},
            "output_sequence_length": {"value": 50, "unit": "tokens"},
            "time_to_first_token": {"value": 30.0, "unit": "ms"},
            "request_latency": {"value": 1000.0, "unit": "ms"},
            "inter_token_latency": {"value": 18.0, "unit": "ms"},
        },
        "error": None,
    }


def agentx_row(env=AGENTX_ENV):
    records = [record("a", 0, 1.0), record("a", 1, 2.5), record("b", 0, 1.5)]
    aggregate = {"metadata": {"dataset": {"hf_dataset_name": "semianalysisai/cc-traces"}}}
    return round_floats(agentic.build_result(records, aggregate, {}, env))


def agentx_multinode_row():
    return with_power_metrics(
        agentx_row(AGENTX_MULTINODE_ENV), metric_keys=ALL_POWER_METRIC_KEYS,
        schema_version=POWER_METRIC_SCHEMA_VERSION, power_valid=False, metrics={},
    )


EVAL_META = {
    "infmax_model_prefix": "dsr1", "hw": "b200", "framework": "sglang", "precision": "fp8",
    "spec_decoding": "none", "isl": "1024", "osl": "1024", "tp": 8, "ep": 1,
    "dp_attention": False, "conc": 64, "eval_suite": "gsm8k",
}


def eval_row():
    return build_row(EVAL_META, {"task": "gsm8k", "strict": 0.9, "strict_se": 0.01, "n_eff": 1319,
                                 "source": "eval_results/eval_a/results.json"})


def without(row, key):
    return {name: value for name, value in row.items() if name != key}


def run_module(module, *args, cwd):
    env = {**os.environ, "PYTHONPATH": str(PROJECT)}
    return subprocess.run([sys.executable, "-P", "-m", module, *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=60)


def field(error):
    """The top-level field an error names, skipping the topology variant tag."""
    return next((part for part in error["loc"] if part not in ("single-node", "multinode")), None)


def rejection_summary(path):
    return {
        (entry["source"], field(error), error["type"])
        for entry in json.loads(path.read_text())
        for error in entry["errors"]
    }


def test_benchmark_collector_publishes_valid_rows_and_quarantines_malformed_ones(tmp_path):
    valid = {
        "bmk_single/agg_single.json": fixed_row(),
        "bmk_multinode/agg_multinode.json": multinode_row(),
        "bmk_aggregate/agg_aggregate.json": fixed_row(AGGREGATE_ENV),
        "bmk_agentic_single/single.json": agentx_row(),
        "bmk_agentic_multinode/multinode_conc8.json": agentx_multinode_row(),
    }
    missing_isl = fixed_row()
    del missing_isl["isl"]
    string_conc = agentx_row() | {"conc": "8"}
    not_finite = multinode_row() | {"tput_per_gpu": float("nan")}
    malformed = {
        "bmk_missing_isl/agg.json": missing_isl,
        "bmk_agentic_string_conc/point.json": string_conc,
        "bmk_nan/agg.json": not_finite,
    }
    for name, row in (valid | malformed).items():
        path = tmp_path / "results" / name
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(row))

    result = run_module("infx.results.collect_results", "results", "bmk", cwd=tmp_path)

    assert result.returncode == 1
    published = json.loads((tmp_path / "agg_bmk.json").read_text())
    assert sorted(published, key=json.dumps) == sorted(valid.values(), key=json.dumps)
    assert rejection_summary(tmp_path / "rejected_rows.json") == {
        ("bmk_missing_isl/agg.json", "isl", "missing"),
        ("bmk_agentic_string_conc/point.json", "conc", "int_type"),
        ("bmk_nan/agg.json", "tput_per_gpu", "finite_number"),
    }
    rejected = {entry["source"]: entry["row"] for entry in
                json.loads((tmp_path / "rejected_rows.json").read_text())}
    assert rejected["bmk_agentic_string_conc/point.json"] == string_conc
    assert math.isnan(rejected["bmk_nan/agg.json"]["tput_per_gpu"])
    annotations = [line for line in result.stderr.splitlines() if line.startswith("::error")]
    sources = [line.removeprefix("::error title=Rejected result row::").split(": ")[0]
               for line in annotations]
    assert sorted(sources) == sorted(malformed)


def write_eval_set(root, name, meta, results):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "meta_env.json").write_text(json.dumps(meta))
    (directory / "results.json").write_text(json.dumps({
        "lm_eval_version": "0.4.9",
        "results": {"gsm8k": results},
        "configs": {"gsm8k": {"metric_list": [{"metric": "exact_match"}],
                              "filter_list": [{"name": "strict-match"},
                                              {"name": "flexible-extract"}]}},
        "n-samples": {"gsm8k": {"effective": 1319}},
    }))


def test_eval_collector_publishes_valid_rows_and_quarantines_malformed_ones(tmp_path):
    scores = {"exact_match,strict-match": 0.9, "exact_match_stderr,strict-match": 0.01,
              "exact_match,flexible-extract": 0.88, "exact_match_stderr,flexible-extract": "N/A"}
    roots = tmp_path / "evals"
    write_eval_set(roots, "eval_valid", EVAL_META, scores)
    write_eval_set(roots, "eval_nan", EVAL_META | {"conc": 32},
                   scores | {"exact_match,flexible-extract": float("nan")})
    write_eval_set(roots, "eval_no_conc", without(EVAL_META, "conc"), scores)

    result = run_module("infx.results.collect_eval_results", "evals", "all", cwd=tmp_path)

    assert result.returncode == 1
    published = json.loads((tmp_path / "agg_eval_all.json").read_text())
    assert [(row["conc"], row["score"], row["em_flexible_se"]) for row in published] == [
        (64, 0.9, "N/A"),
    ]
    assert rejection_summary(tmp_path / "rejected_rows.json") == {
        ("evals/eval_nan/results.json [gsm8k]", "em_flexible", "finite_number"),
        ("evals/eval_no_conc/results.json [gsm8k]", "conc", "greater_than"),
    }
    assert len([line for line in result.stderr.splitlines() if line.startswith("::error")]) == 2


@pytest.mark.parametrize("schema,row,expected", [
    (FIXED_SEQUENCE_ROW, lambda: fixed_row() | {"ttft_p95": 0.2},
     {("ttft_p95", "string_pattern_mismatch")}),
    (FIXED_SEQUENCE_ROW, lambda: fixed_row() | {"p95_ttft": True}, {("p95_ttft", "float_type")}),
    (FIXED_SEQUENCE_ROW, lambda: without(fixed_row(), "result_schema_version"),
     {("result_schema_version", "missing")}),
    (FIXED_SEQUENCE_ROW, lambda: fixed_row() | {"result_schema_version": True},
     {("result_schema_version", "int_type")}),
    (FIXED_SEQUENCE_ROW, lambda: fixed_row() | {"dp_attention": False},
     {("dp_attention", "literal_error")}),
    (FIXED_SEQUENCE_ROW, lambda: fixed_row() | {"is_multinode": "false"}, {(None, "topology")}),
    # AgentX latency lives in request_metrics; only power metrics are top-level extras.
    (AGENTX_ROW, lambda: agentx_row() | {"avg_power_w": 650.0}, set()),
    (AGENTX_ROW, lambda: agentx_row() | {"mean_ttft": 0.03},
     {("mean_ttft", "string_pattern_mismatch")}),
    (AGENTX_ROW, lambda: agentx_row() | {"request_metrics": {"qps": {"mean": float("inf")}}},
     {("request_metrics", "finite_number")}),
    (EVAL_ROW, lambda: eval_row() | {"eval_suite": None}, {("eval_suite", "string_type")}),
    (RUN_STATS_ROW, lambda: {"result_schema_version": 1, "n_success": 3, "total": 2},
     {(None, "value_error")}),
])
def test_contract_rules(schema, row, expected):
    assert {(field(error), error["type"]) for error in row_errors(schema, row())} == expected


def test_fixed_sequence_producer_stamps_version_without_pydantic(tmp_path):
    (tmp_path / "point.json").write_text(json.dumps(BENCHMARK))
    # Runner result processing uses a bare interpreter without the validation dependency.
    script = (
        "import runpy, sys\n"
        "class NoPydantic:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.partition('.')[0] in ('pydantic', 'pydantic_core'):\n"
        "            raise ImportError(name)\n"
        "sys.meta_path.insert(0, NoPydantic())\n"
        "runpy.run_module('infx.results.fixed_sequence', run_name='__main__')\n"
    )
    env = {**os.environ, **FIXED_ENV, "RESULT_FILENAME": "point", "PYTHONPATH": str(PROJECT)}
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "agg_point.json").read_text())["result_schema_version"] == 1


def test_committed_json_schemas_match_the_models(tmp_path):
    result = run_module("infx.results.schema", "export", str(tmp_path), cwd=PROJECT)

    assert result.returncode == 0, result.stderr
    exported = {path.name: path.read_text() for path in tmp_path.glob("*.json")}
    committed = {path.name: path.read_text() for path in (PROJECT / "schemas").glob("*.json")}
    assert exported == committed, "regenerate: python -m infx.results.schema export schemas"
