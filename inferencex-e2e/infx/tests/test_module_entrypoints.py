"""Exercise module entrypoints across checkout boundaries."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def invoke(tmp_path):
    def run(module, *args, **environment):
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        command = ["-P", "-m", module]
        env["PYTHONPATH"] = str(ROOT)
        return subprocess.run(
            [sys.executable, *command, *args], cwd=tmp_path,
            env={**env, **environment}, capture_output=True, text=True, timeout=10,
        )
    return run


@pytest.mark.parametrize("payload,expected", [
    (None, []),
    ('{"value": 3}', [{"value": 3}]),
    ('[{"value": 3}, {"value": 4}]', [[{"value": 3}, {"value": 4}]]),
])
def test_collector_preserves_nested_json_and_empty_inputs(invoke, tmp_path, payload, expected):
    inputs = tmp_path / "inputs" / "nested"
    inputs.mkdir(parents=True)
    if payload is not None:
        (inputs / "result.json").write_text(payload)
    (inputs / "ignored.txt").write_text("not JSON")

    result = invoke("infx.results.collect_results", "inputs", "test")

    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ""
    assert json.loads((tmp_path / "agg_test.json").read_text()) == expected


@pytest.mark.parametrize("published", [None, b'[{"previous": true}]\n'])
def test_collector_does_not_publish_partial_output_on_invalid_json(invoke, tmp_path, published):
    inputs = tmp_path / "inputs"
    (inputs / "nested").mkdir(parents=True)
    (inputs / "valid.json").write_text('{"value": 3}')
    (inputs / "nested/invalid.json").write_text("{")
    output = tmp_path / "agg_test.json"
    if published is not None:
        output.write_bytes(published)

    result = invoke("infx.results.collect_results", "inputs", "test")

    assert result.returncode != 0
    assert "JSONDecodeError" in result.stderr
    if published is None:
        assert not output.exists()
    else:
        assert output.read_bytes() == published


def test_collector_excludes_failed_benchmarks_and_retains_diagnostics(invoke, tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    rows = {
        "legacy": {"conc": 4},
        "passed": {"conc": 8, "benchmark_outcome": {
            "status": "passed", "attempt_id": "good", "finalized": True,
        }},
        "failed": {"conc": 16, "benchmark_outcome": {
            "status": "failed", "attempt_id": "partial", "finalized": False,
        }},
        "fixed_seq": {"conc": 32, "benchmark_outcome": {
            "status": "passed", "requested": 100, "completed": 99,
            "failed": 1, "max_failure_rate": 0.05,
        }},
    }
    for name, row in rows.items():
        (inputs / f"{name}.json").write_text(json.dumps(row))
    before = {path.name: path.read_bytes() for path in inputs.iterdir()}

    result = invoke("infx.results.collect_results", "inputs", "bmk")

    assert result.returncode == 0, result.stderr
    assert sorted(json.loads((tmp_path / "agg_bmk.json").read_text()), key=lambda r: r["conc"]) == [
        rows["legacy"], rows["passed"], rows["fixed_seq"],
    ]
    assert {path.name: path.read_bytes() for path in inputs.iterdir()} == before


@pytest.mark.parametrize("outcome", [
    None, {"status": "unknown"},
    {"status": "passed", "attempt_id": "partial", "finalized": False},
    {"status": "passed", "attempt_id": "partial"},
    {"status": "passed", "attempt_id": "partial", "finalized": 1},
    {"status": "passed", "finalized": True},
    {"status": "failed", "attempt_id": "", "finalized": False},
])
def test_collector_refuses_malformed_present_outcomes(invoke, tmp_path, outcome):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "result.json").write_text(json.dumps({"benchmark_outcome": outcome}))
    output = tmp_path / "agg_bmk.json"
    output.write_text("previous aggregate")

    result = invoke("infx.results.collect_results", "inputs", "bmk")

    assert result.returncode != 0
    assert "benchmark_outcome" in result.stderr
    assert output.read_text() == "previous aggregate"


def test_collector_preserves_eval_payloads_without_benchmark_filtering(invoke, tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    payload = {"results": {"task": {"acc": 0.5}}, "benchmark_outcome": {"status": "other"}}
    (inputs / "eval.json").write_text(json.dumps(payload))

    result = invoke("infx.results.collect_results", "inputs", "eval")

    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "agg_eval.json").read_text()) == [payload]


def test_filename_entrypoint_retains_environment_and_point_arguments(invoke):
    result = invoke("infx.results.result_filename",
                    RESULT_FILENAME_BASE="model_tp8", RECIPE_FINGERPRINT="abc")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "model_tp8_recipe-abc\n"

    point = invoke("infx.results.result_filename",
                   "--point", "model_tp8", "config", "4", "8", "1024", "512")
    assert point.returncode == 0, point.stderr
    assert point.stdout == "model_tp8_config_conc4_gpus_8_ctx_1024_gen_512.json\n"

    invalid = invoke("infx.results.result_filename",
                     "--point", "model_tp8", "config", "four", "8", "", "")
    assert invalid.returncode != 0
    assert "Expected numeric point identity" in invalid.stderr


def test_run_statistics_reject_invalid_run_id_without_publishing(invoke, tmp_path):
    result = invoke(
        "infx.workflows.calc_success_rate", "stats",
        GITHUB_RUN_ID="not-an-integer", GITHUB_REPOSITORY="example/project", GITHUB_TOKEN="unused",
    )
    assert result.returncode != 0
    assert "ValueError" in result.stderr
    assert not (tmp_path / "stats.json").exists()
