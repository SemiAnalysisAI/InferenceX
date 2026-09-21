"""Behavioral checks for binding a matrix point to a native SRT recipe."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from infx.srt_slurm.single_node import runtime_arguments, submission_fields
from infx.srt_slurm.synthetic_acceptance import plan_commands

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils/srt-slurm/src"))
from srtctl.core.overrides import apply_overrides_to_recipe, parse_overrides


@pytest.fixture
def point(tmp_path):
    recipe = {
        "engine": "sglang",
        "model": {"path": "hf:test/model", "container": "test:tag", "precision": "fp8"},
        "roles": {"agg": {
            "nodes": 1, "workers": 1, "gpus": 4,
            "args": {"tensor-parallel-size": 4, "data-parallel-size": 1, "max-running-requests": 32},
        }},
        "benchmark": {"type": "custom", "env": {
            "MODEL": "test/model", "ISL": "256", "OSL": "64", "RANDOM_RANGE_RATIO": "0.5",
        }},
    }
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump({"base": recipe, "zip_override_conc": {
        "benchmark": {"env": {"CONC": ["2", "4"]}},
    }}))
    env = {
        "FRAMEWORK": "sglang", "MODEL": "test/model", "IMAGE": "test:tag", "PRECISION": "fp8",
        "TP": "4", "GPU_COUNT": "4", "PP_SIZE": "1", "DCP_SIZE": "1", "PCP_SIZE": "1",
        "EP_SIZE": "1", "DP_ATTENTION": "false", "SPEC_DECODING": "none", "IS_AGENTIC": "0",
        "RUN_EVAL": "false", "EVAL_ONLY": "false", "ISL": "256", "OSL": "64",
        "RANDOM_RANGE_RATIO": "0.5", "CONC": "2", "RESULT_FILENAME": "point-identity",
        "GPU_MONITOR_INTERVAL": "3", "MODEL_PREFIX": "test",
    }
    return path, recipe, env


def test_native_binding_submits_one_point_and_keeps_server_settings(point):
    path, recipe, env = point
    argv = runtime_arguments(f"{path}:base", env)
    overrides = parse_overrides(argv[1::2], [])
    actual = copy.deepcopy(recipe)
    apply_overrides_to_recipe(actual, overrides)
    assert actual["benchmark"]["env"] == {
        "MODEL": "test/model", "ISL": "256", "OSL": "64", "RANDOM_RANGE_RATIO": "0.5",
        "CONC": "2", "RESULT_FILENAME": "point-identity", "GPU_MONITOR_INTERVAL": "3",
        "RUN_EVAL": "false", "EVAL_ONLY": "false", "RESULT_DIR": "/logs",
    }
    assert actual["roles"]["agg"]["args"] == {
        "tensor-parallel-size": 4, "data-parallel-size": 1, "max-running-requests": 32,
    }
    commands = plan_commands(f"{path}:base", "sglang", ["--json", "--yes", *argv], env)
    assert commands == [["srtctl", "apply", "--json", "--yes", *argv, "--file", f"{path}:base"]]


@pytest.mark.parametrize("field,value,message", [
    ("TP", "2", "tensor-parallel-size"), ("IMAGE", "other:tag", "image"),
    ("ISL", "128", "ISL"), ("RUN_EVAL", "true", "RUN_EVAL"),
    ("PP_SIZE", "2", "PP_SIZE"), ("RESULT_FILENAME", "", "Missing runtime input"),
])
def test_mismatched_point_fails_before_submission(point, field, value, message):
    path, _, env = point
    with pytest.raises(ValueError, match=message):
        runtime_arguments(f"{path}:base", {**env, field: value})


def test_multi_variant_selection_is_rejected(point):
    path, _, env = point
    with pytest.raises(ValueError, match="exactly one"):
        runtime_arguments(str(path), env)


@pytest.mark.parametrize("record,expected", [
    ({"status": "submitted", "slurm_job_id": "42", "output_dir": "/shared/42"}, ("42", "/shared/42")),
    ({"status": "error"}, None),
    ({"status": "submitted", "slurm_job_id": "42;43", "output_dir": "/shared/42"}, None),
    ({"status": "submitted", "slurm_job_id": "42", "output_dir": "relative"}, None),
])
def test_submission_manifest(tmp_path, record, expected):
    path = tmp_path / "submission.json"
    path.write_text(json.dumps(record))
    if expected is None:
        with pytest.raises(ValueError):
            submission_fields(path)
    else:
        assert submission_fields(path) == expected


@pytest.mark.parametrize("failure", ["none", "allocation", "submission"])
def test_pool_launcher_stages_artifacts_and_propagates_failure(point, tmp_path, failure):
    path, _, point_env = point
    binaries = tmp_path / "bin"
    binaries.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (tmp_path / "benchmarks").symlink_to(ROOT / "benchmarks", target_is_directory=True)
    capture = tmp_path / "cancelled"
    # Only external executables are stubbed; run the real pool launcher, shared
    # setup/profile/acceptance helpers, binder, and artifact collection.
    scripts = {
        "git": 'if [[ "$1" == clone ]]; then mkdir -p "${@: -1}/configs"; else echo test-commit; fi',
        "uv": 'if [[ "$1" == venv ]]; then mkdir -p .venv/bin; echo ":" > .venv/bin/activate; fi',
        "unsquashfs": "exit 0",
        "squeue": '[[ "$TEST_FAILURE" == submission ]] && echo "42 RUNNING"; exit 0',
        "sacct": 'if [[ "$TEST_FAILURE" == allocation ]]; then echo "FAILED|1:0"; else echo "COMPLETED|0:0"; fi',
        "scancel": 'printf "%s\\n" "$@" >> "$CANCEL_CAPTURE"',
        "tail": 'exit 0',
    }
    for name, script in scripts.items():
        binary = binaries / name
        binary.write_text(f"#!/usr/bin/env bash\n{script}\n")
        binary.chmod(0o755)
    srtctl = binaries / "srtctl"
    srtctl.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1]) / '42'\n"
        "logs = output / 'logs'\n"
        "logs.mkdir(parents=True)\n"
        "(logs / 'sweep_42.log').write_text('benchmark complete\\n')\n"
        "(logs / (os.environ['RESULT_FILENAME'] + '.json')).write_text('{\"completed\":2}')\n"
        "(logs / 'gpu_metrics.csv').write_text('gpu,power\\n0,300\\n')\n"
        "(logs / 'gpu_metrics_context.json').write_text('{\"device_count\":4}')\n"
        "print(json.dumps({'status':'submitted', 'slurm_job_id':'42', 'output_dir':str(output)}))\n"
        "sys.exit(7 if os.environ['TEST_FAILURE'] == 'submission' else 0)\n"
    )
    srtctl.chmod(0o755)
    env = {
        **os.environ, **point_env,
        "PATH": f"{binaries}:{Path(sys.executable).parent}:{os.environ['PATH']}",
        "PYTHONPATH": f"{ROOT}:{ROOT / 'utils/srt-slurm/src'}",
        "GITHUB_WORKSPACE": str(tmp_path), "SRT_RECIPE": f"{path.name}:base",
        "IS_MULTINODE": "false", "REQUIRE_POWER": "1", "SALLOC_TIME_LIMIT": "10",
        "HF_HUB_CACHE_MOUNT": str(tmp_path), "AIPERF_MMAP_CACHE_HOST_PATH": str(tmp_path),
        "HF_HUB_CACHE": "/hf", "DSR1_FP8_MODEL_PATH": str(model), "MODEL_PREFIX": "dsr1",
        "INFERENCEX_RUNTIME_ENV_VARS": "REQUIRE_POWER", "AIPERF_DRAIN_TIMEOUT_SECONDS": "1",
        "AIPERF_DRAIN_POLL_SECONDS": "1", "TEST_FAILURE": failure, "CANCEL_CAPTURE": str(capture),
    }
    result = subprocess.run(
        ["bash", str(ROOT / "runners/launch_h200-dgxc-slurm.sh")], cwd=tmp_path,
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == {"none": 0, "allocation": 1, "submission": 7}[failure], result.stderr
    assert json.loads((tmp_path / "point-identity.json").read_text()) == {"completed": 2}
    assert (tmp_path / "gpu_metrics.csv").read_text() == "gpu,power\n0,300\n"
    assert json.loads((tmp_path / "gpu_metrics_context.json").read_text()) == {"device_count": 4}
    assert (tmp_path / "srt-single-node-logs.tar.gz").stat().st_size > 0
    assert (capture.read_text() if capture.exists() else "") == ("42\n" if failure == "submission" else "")
