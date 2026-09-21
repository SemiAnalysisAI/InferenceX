"""Behavioral checks for binding a matrix point to a native SRT recipe."""

import copy
import json
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
