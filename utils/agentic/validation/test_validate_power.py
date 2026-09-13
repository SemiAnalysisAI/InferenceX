"""Required AgentX power must survive the producer-to-workflow boundary."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
VALID = {
    "power_metric_schema_version": 2,
    "power_valid": 1,
    "avg_power_w": 500,
    "avg_total_gpu_power_w": 1000,
    "total_gpu_energy_j": 3000,
    "joules_per_output_token": 20,
}
ROLES = {
    "prefill_gpu_energy_j": 1000,
    "decode_gpu_energy_j": 2000,
    "prefill_joules_per_input_token": 10,
    "decode_joules_per_output_token": 15,
}


def _check(*paths, disagg=False):
    return subprocess.run(
        [sys.executable, "-m", "infx.results.agentic.validate_power",
         *(["--disagg"] if disagg else []), *map(str, paths)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
    )


def test_accepts_all_valid_results_without_rewriting_them(tmp_path):
    paths = [tmp_path / f"result_conc{conc}.json" for conc in (1, 4)]
    for path in paths:
        path.write_text(json.dumps(VALID))
    before = [path.read_bytes() for path in paths]
    result = _check(*paths)
    assert result.returncode == 0, result.stderr
    assert [path.read_bytes() for path in paths] == before


@pytest.mark.parametrize("field,value", [
    ("power_metric_schema_version", 1), ("power_metric_schema_version", True),
    ("power_valid", 0), ("power_valid", True), ("power_valid", "1"),
    ("avg_power_w", 0), ("avg_power_w", True), ("avg_power_w", "500"),
    ("avg_total_gpu_power_w", float("inf")),
    ("total_gpu_energy_j", -1), ("total_gpu_energy_j", float("nan")),
    ("total_gpu_energy_j", None),
    ("joules_per_output_token", 0), ("joules_per_output_token", None),
])
def test_rejects_bad_power_even_after_a_valid_result(tmp_path, field, value):
    valid = tmp_path / "result_conc1.json"
    invalid = tmp_path / "result_conc4.json"
    valid.write_text(json.dumps(VALID))
    invalid.write_text(json.dumps({**VALID, field: value}))
    before = invalid.read_bytes()
    result = _check(valid, invalid)
    assert result.returncode == 1
    assert str(invalid) in result.stderr and field in result.stderr
    assert invalid.read_bytes() == before


@pytest.mark.parametrize("content", [None, "{", "[]", "{}", "true"])
def test_missing_or_unprocessed_result_fails(tmp_path, content):
    path = tmp_path / "result.json"
    if content is not None:
        path.write_text(content)
    result = _check(path)
    assert result.returncode == 1
    assert str(path) in result.stderr


def test_no_results_fails():
    assert _check().returncode != 0


@pytest.mark.parametrize("missing_role", [None, *ROLES])
def test_disaggregated_results_require_positive_role_energy(tmp_path, missing_role):
    path = tmp_path / "split.json"
    result = {**VALID, **ROLES}
    if missing_role:
        result[missing_role] = 0
    path.write_text(json.dumps(result))
    checked = _check(path, disagg=True)
    assert checked.returncode == int(missing_role is not None), checked.stderr
    if missing_role:
        assert missing_role in checked.stderr


@pytest.mark.parametrize("topology", ["single", "aggregate", "split"])
@pytest.mark.parametrize("bad_result", [False, True, "missing", "whole-only"])
def test_shipped_workflow_checks_each_returned_aggregate(tmp_path, topology, bad_result):
    multinode = topology != "single"
    template = "benchmark-multinode-tmpl.yml" if multinode else "benchmark-tmpl.yml"
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows" / template).read_text())
    step = next(step for job in workflow["jobs"].values() for step in job.get("steps", [])
                if step.get("name") == "Validate required AgentX power")
    if bad_result != "missing":
        paths = [tmp_path / "result.json"]
        if multinode:
            paths = [tmp_path / "result_conc1.json", tmp_path / "result_conc4.json"]
        for path in paths:
            path.write_text(json.dumps({**VALID, **(ROLES if topology == "split" else {})}))
        if bad_result is True:
            paths[-1].write_text(json.dumps({**VALID, "power_valid": 0}))
        elif bad_result == "whole-only":
            paths[-1].write_text(json.dumps(VALID))
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", step["run"]], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT), "RESULT_FILENAME": "result",
             "DISAGG": "true" if topology == "split" else "false",
             "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True, text=True, timeout=10,
    )
    should_fail = bad_result is True or bad_result == "missing" or (bad_result == "whole-only" and topology == "split")
    assert result.returncode == int(should_fail), result.stderr
