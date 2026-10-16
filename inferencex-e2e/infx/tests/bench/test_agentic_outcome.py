from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from infx.bench.agentic.run import Plan, _score, execute
from infx.bench.env import BenchError
from infx.tests.bench.stubs import executable
from infx.tests.bench.test_agentic_command import _point, _runtime


@pytest.fixture
def point(tmp_path):
    env = _point(
        tmp_path,
        AGENTIC_ATTEMPT_ID="12345678-1234-4234-8234-123456789012",
        GITHUB_RUN_ID="123",
        GITHUB_RUN_ATTEMPT="2",
        GITHUB_JOB="benchmark",
        SLURM_JOB_ID="456",
    )
    artifacts = tmp_path / "results/aiperf_artifacts"
    artifacts.mkdir(parents=True)
    records = [{
        "metadata": {
            "benchmark_phase": phase,
            "conversation_id": "a",
            "turn_index": 0,
            "request_start_ns": 1_000_000_000,
            "request_end_ns": 2_000_000_000,
        },
        "metrics": {
            "input_sequence_length": {"value": 100},
            "output_sequence_length": {"value": 10},
        },
        "error": None,
    } for phase in ("warmup", "profiling")]
    (artifacts / "profile_export.jsonl").write_text("\n".join(map(json.dumps, records)))
    (artifacts / "profile_export_aiperf.json").write_text(json.dumps({
        "request_count": {"avg": 9},
        "error_request_count": {"avg": 1},
        "completed_request_count": {"avg": 10},
    }))
    return env, artifacts, tmp_path / "agentx.json"


@pytest.mark.parametrize("replay_rc,invalid_profile,power,require_power,prefix,expected", [
    (0, False, "0", "0", None, 0),
    (7, True, "1", "1", "vllm:", 7),
    (0, True, "0", "0", None, 1),
    (0, False, "1", "1", "vllm:", 1),
    (0, False, "1", "0", None, 0),
    (0, False, "0", "0", "vllm:", 1),
])
def test_score_persists_actual_gate_decisions(
    point, replay_rc, invalid_profile, power, require_power, prefix, expected,
):
    env, artifacts, output = point
    env.update(ENABLE_AGENTX_POWER=power, REQUIRE_POWER=require_power)
    if prefix:
        env["AIPERF_REQUIRED_SERVER_METRIC_PREFIX"] = prefix
    if invalid_profile:
        (artifacts / "profile_export_aiperf.json").write_text(json.dumps({
            "request_count": {"avg": 1}, "error_request_count": {"avg": 9},
        }))
    plan = Plan.from_env(env)
    if prefix and not (replay_rc or invalid_profile or require_power == "1"):
        with pytest.raises(BenchError, match="required AIPerf server metrics"):
            _score(plan, sys.executable, env, replay_rc)
    else:
        assert _score(plan, sys.executable, env, replay_rc) == expected

    aggregate = json.loads(output.read_text())
    outcome = aggregate["benchmark_outcome"]
    assert outcome["status"] == ("failed" if expected else "passed")
    assert outcome["finalized"] is True
    assert outcome["exit_code"] == expected
    assert outcome["replay_exit_code"] == replay_rc
    assert outcome["aggregate_exit_code"] == 0
    assert outcome["attempt_id"] == env["AGENTIC_ATTEMPT_ID"]
    assert outcome["result_filename"] == "agentx.json"
    assert outcome["provenance"] == {
        "github_run_id": "123", "github_run_attempt": "2",
        "github_job": "benchmark", "slurm_job_id": "456",
    }
    validation = outcome["validation"]
    assert validation["failed_request_threshold"] == 0.1
    assert (validation["successes"], validation["errors"], validation["completed"]) == (
        (1, 9, 10) if invalid_profile else (9, 1, 10)
    )
    assert bool(validation["failures"]) == invalid_profile
    assert outcome["power"] == {
        "required": require_power == "1",
        "mode": "missing" if power == "1" else "off",
        "checked": power == "1",
        "exit_code": int(power == require_power == "1"),
    }
    assert outcome["server_metrics"]["status"] == (
        "not_required" if prefix is None else
        "skipped" if replay_rc or invalid_profile or require_power == "1" else "failed"
    )
    assert (aggregate["num_requests_total"], aggregate["num_requests_successful"]) == (2, 1)


@pytest.mark.parametrize("bootstrap,overrides,expected", [
    (False, {"AGGREGATE_RC": "1"}, 1),
    (False, {"AGGREGATE_RC": "1", "REPLAY_RC": "7"}, 7),
    (False, {"HF_RC": "7"}, 7),
    (True, {}, 6),
])
def test_failed_retry_cannot_publish_prior_attempt(tmp_path, bootstrap, overrides, expected):
    env = _point(tmp_path)
    plan = Plan.from_env(env)
    runtime = _runtime(tmp_path)
    output = tmp_path / "agentx.json"
    unrelated = tmp_path / "other-point.json"
    unrelated.write_text('{"conc": 4}')
    assert execute(plan, runtime, env) == 0
    previous = output.read_bytes()
    assert json.loads(previous)["benchmark_outcome"]["status"] == "passed"

    if bootstrap:
        executable(tmp_path / "uv", "#!/bin/sh\nexit 6\n")
        retry = subprocess.run(
            [sys.executable, "-m", "infx.bench", "agentic"],
            env={
                **env,
                "AIPERF_RUNTIME_DIR": str(runtime.root),
                "PATH": f"{tmp_path}{os.pathsep}{env['PATH']}",
            },
            capture_output=True, text=True, check=False,
        )
        assert retry.returncode == expected, retry.stderr
    else:
        assert execute(plan, runtime, {**env, **overrides}) == expected

    assert not output.exists()
    assert [path.read_bytes() for path in tmp_path.glob("agentx.json.superseded-*")] == [previous]
    assert unrelated.read_text() == '{"conc": 4}'


@pytest.mark.parametrize("multinode,filename", [
    ("false", "agentx.json"), ("true", "agentx_conc8.json"),
])
def test_invalid_retry_cannot_publish_prior_attempt(tmp_path, multinode, filename):
    env = _point(tmp_path, IS_MULTINODE=multinode)
    del env["AIPERF_EXPERIMENTAL_FAST"]
    output = tmp_path / filename
    previous = json.dumps({"benchmark_outcome": {
        "status": "passed", "attempt_id": "previous", "finalized": True,
    }})
    output.write_text(previous)

    retry = subprocess.run(
        [sys.executable, "-m", "infx.bench", "agentic"],
        env=env, capture_output=True, text=True, check=False,
    )
    assert retry.returncode == 1
    assert "AIPERF_EXPERIMENTAL_FAST" in retry.stderr
    assert not output.exists()
    assert [path.read_text() for path in tmp_path.glob(f"{filename}.superseded-*")] == [previous]


def test_interrupted_after_aggregation_leaves_failed_diagnostic(point, tmp_path):
    env, _, output = point
    result = subprocess.run(
        [sys.executable, "-m", "infx.results.agentic.process_agentic_result"],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    diagnostic = output.read_bytes()
    outcome = json.loads(diagnostic)["benchmark_outcome"]
    assert outcome["status"] == "failed"
    assert outcome["finalized"] is False
    assert outcome["attempt_id"] == env["AGENTIC_ATTEMPT_ID"]
    results = tmp_path / "publishable"
    results.mkdir()
    output.rename(results / output.name)
    collected = subprocess.run(
        [sys.executable, "-m", "infx.results.collect_results", str(results), "bmk"],
        env=env, cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert collected.returncode == 0, collected.stderr
    assert json.loads((tmp_path / "agg_bmk.json").read_text()) == []
    assert (results / output.name).read_bytes() == diagnostic
