"""Behavioral checks for binding a matrix point to a native SRT recipe."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from infx.srt_slurm.single_node import runtime_arguments, select_recipe, submission_fields
from infx.srt_slurm.synthetic_acceptance import plan_commands, selected_recipes

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
            "USE_CHAT_TEMPLATE": "false",
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
        "USE_CHAT_TEMPLATE": "false",
        "CONC": "2", "RESULT_FILENAME": "point-identity", "GPU_MONITOR_INTERVAL": "3",
        "RUN_EVAL": "false", "EVAL_ONLY": "false", "RESULT_DIR": "/logs",
        "FRAMEWORK": "sglang",
    }
    assert actual["roles"]["agg"]["args"] == {
        "tensor-parallel-size": 4, "data-parallel-size": 1, "max-running-requests": 32,
    }
    commands = plan_commands(f"{path}:base", "sglang", ["--json", "--yes", *argv], env)
    assert commands == [["srtctl", "apply", "--json", "--yes", *argv, "--file", f"{path}:base"]]


@pytest.mark.parametrize("field,value,message", [
    ("TP", "2", "tensor-parallel-size"), ("IMAGE", "other:tag", "image"),
    ("ISL", "128", "ISL"), ("RUN_EVAL", "yes", "RUN_EVAL"),
    ("PP_SIZE", "2", "PP_SIZE"), ("RESULT_FILENAME", "", "Missing runtime input"),
    ("EP_SIZE", "2", "expert-parallel-size"), ("SPEC_DECODING", "mtp", "SPEC_DECODING"),
])
def test_mismatched_point_fails_before_submission(point, field, value, message):
    path, _, env = point
    with pytest.raises(ValueError, match=message):
        runtime_arguments(f"{path}:base", {**env, field: value})


def test_native_variants_select_only_the_matching_matrix_point(point):
    path, _, env = point
    config, recipe = select_recipe(str(path), {**env, "CONC": "4"})
    assert config == f"{path}:zip_override_conc[1]"
    assert recipe["benchmark"]["env"]["CONC"] == "4"
    argv = runtime_arguments(config, {**env, "CONC": "4"})
    assert plan_commands(config, "sglang", ["--json", *argv], env) == [[
        "srtctl", "apply", "--json", *argv, "--file", f"{path}:zip_override_conc[1]",
    ]]
    with pytest.raises(ValueError, match="exactly one"):
        select_recipe(str(path), {**env, "CONC": "8"})


def test_ambiguous_native_variants_are_rejected(point):
    path, recipe, env = point
    path.write_text(yaml.safe_dump({"base": recipe, "override_first": {}, "override_second": {}}))
    with pytest.raises(ValueError, match="exactly one"):
        runtime_arguments(str(path), env)


def test_mtp_binding_uses_real_verification_and_preserves_expert_parallelism(point):
    path, recipe, env = point
    recipe["roles"]["agg"]["args"].update({
        "expert-parallel-size": 4, "speculative-algorithm": "EAGLE",
        "speculative-num-steps": 2, "speculative-num-draft-tokens": 3,
    })
    recipe["roles"]["agg"]["env"] = {"SGLANG_SIMULATE_ACC_LEN": "2.5"}
    recipe["benchmark"]["env"]["USE_CHAT_TEMPLATE"] = "true"
    path.write_text(yaml.safe_dump({"base": recipe}))
    env = {**env, "EP_SIZE": "4", "SPEC_DECODING": "mtp"}
    argv = runtime_arguments(f"{path}:base", env)
    commands = plan_commands(f"{path}:base", "sglang", ["--json", *argv], env)
    assert commands == [[
        "srtctl", "apply", "--json", *argv, "--file", f"{path}:base",
        "--unset", "roles.agg.env.SGLANG_SIMULATE_ACC_LEN",
    ]]
    recipe["benchmark"]["env"]["USE_CHAT_TEMPLATE"] = "false"
    path.write_text(yaml.safe_dump({"base": recipe}))
    with pytest.raises(ValueError, match="USE_CHAT_TEMPLATE"):
        runtime_arguments(f"{path}:base", env)


def test_concurrency_selector_keeps_graph_capture_coupled_to_client(point):
    path, recipe, env = point
    path.write_text(yaml.safe_dump({"base": recipe, "zip_override_conc": {
        "roles": {"agg": {"args": {"cuda-graph-max-bs": [2, 4]}}},
        "benchmark": {"env": {"CONC": ["2", "4"]}},
    }}))
    argv = runtime_arguments(f"{path}:zip_override_conc[1]", {**env, "CONC": "4"})
    actual = selected_recipes(yaml.safe_load(path.read_text()), "zip_override_conc[1]")[0][1]
    apply_overrides_to_recipe(actual, parse_overrides(argv[1::2], []))
    assert actual["roles"]["agg"]["args"]["cuda-graph-max-bs"] == 4
    assert actual["benchmark"]["env"]["CONC"] == "4"
    with pytest.raises(ValueError, match="CONC"):
        runtime_arguments(f"{path}:zip_override_conc[1]", env)


def test_eval_binding_changes_context_without_changing_selected_concurrency(point):
    path, recipe, env = point
    recipe["roles"]["agg"]["args"]["context-length"] = 512
    path.write_text(yaml.safe_dump({"base": recipe, "zip_override_conc": {
        "benchmark": {"env": {"CONC": ["2", "4"]}},
    }}))
    env = {**env, "EVAL_ONLY": "true", "RUN_EVAL": "true", "CONC": "4", "MAX_MODEL_LEN": "1024"}
    config, _ = select_recipe(str(path), env)
    argv = runtime_arguments(config, env)
    raw = yaml.safe_load(path.read_text())
    apply_overrides_to_recipe(raw, parse_overrides(argv[1::2], []))
    actual = selected_recipes(raw, "zip_override_conc[1]")[0][1]
    assert actual["roles"]["agg"]["args"]["context-length"] == 1024
    assert actual["benchmark"]["env"]["CONC"] == "4"
    assert len(plan_commands(config, "sglang", ["--json", *argv], env)) == 1


def test_dp_attention_is_validated_without_replacing_recipe_topology(point):
    path, recipe, env = point
    recipe["roles"]["agg"]["args"].update({
        "data-parallel-size": 4, "expert-parallel-size": 4, "enable-dp-attention": True,
    })
    path.write_text(yaml.safe_dump({"base": recipe}))
    env = {**env, "DP_ATTENTION": "true", "EP_SIZE": "4"}
    actual = copy.deepcopy(recipe)
    argv = runtime_arguments(f"{path}:base", env)
    apply_overrides_to_recipe(actual, parse_overrides(argv[1::2], []))
    assert actual["roles"]["agg"]["args"] == {
        "tensor-parallel-size": 4, "data-parallel-size": 4,
        "max-running-requests": 32, "expert-parallel-size": 4, "enable-dp-attention": True,
    }
    with pytest.raises(ValueError, match="data-parallel-size|DP_ATTENTION"):
        runtime_arguments(f"{path}:base", {**env, "DP_ATTENTION": "false"})


def test_trt_binding_keeps_engine_options_and_sets_eval_token_budget(point):
    path, recipe, env = point
    recipe["engine"] = {"type": "trtllm", "served_model_name": "test/model"}
    recipe["roles"]["agg"]["args"] = {
        "tensor_parallel_size": 4, "moe_expert_parallel_size": 4,
        "enable_attention_dp": True, "max_seq_len": 512, "max_num_tokens": 256,
        "speculative_config": {"decoding_type": "MTP", "num_nextn_predict_layers": 3},
        "cuda_graph_config": {"batch_sizes": [1, 2, 4]},
    }
    recipe["roles"]["agg"]["env"] = {"TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS": "3"}
    recipe["benchmark"]["env"]["USE_CHAT_TEMPLATE"] = "true"
    path.write_text(yaml.safe_dump({"base": recipe}))
    env = {**env, "FRAMEWORK": "trt", "EP_SIZE": "4", "DP_ATTENTION": "true",
           "SPEC_DECODING": "mtp", "EVAL_ONLY": "true", "MAX_MODEL_LEN": "1024"}
    argv = runtime_arguments(f"{path}:base", env)
    actual = copy.deepcopy(recipe)
    apply_overrides_to_recipe(actual, parse_overrides(argv[1::2], []))
    assert actual["roles"]["agg"]["args"] == {
        "tensor_parallel_size": 4, "moe_expert_parallel_size": 4,
        "enable_attention_dp": True, "max_seq_len": 1024, "max_num_tokens": 1024,
        "speculative_config": {"decoding_type": "MTP", "num_nextn_predict_layers": 3},
        "cuda_graph_config": {"batch_sizes": [1, 2, 4]},
    }
    assert plan_commands(f"{path}:base", "trt", ["--json", *argv], env) == [[
        "srtctl", "apply", "--json", *argv, "--file", f"{path}:base",
        "--unset", "roles.agg.env.TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS",
    ]]
    with pytest.raises(ValueError, match="moe_expert_parallel_size"):
        runtime_arguments(f"{path}:base", {**env, "EP_SIZE": "1"})


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


@pytest.mark.parametrize("pool,failure", [
    ("h200-dgxc-slurm", "none"), ("h200-dgxc-slurm", "allocation"),
    ("h200-dgxc-slurm", "submission"), ("h200-dgxc-slurm", "bootstrap"),
    ("h200-cw", "none"), ("h100-cw", "none"), ("h100-dgxc-slurm", "none"),
    ("b200-cw", "none"), ("b200-nb", "none"), ("b200-nscale-slurm", "none"),
    ("b300-dsxe", "none"),
])
def test_pool_launcher_stages_artifacts_and_propagates_failure(point, tmp_path, pool, failure):
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
        "make": '[[ "$TEST_FAILURE" == bootstrap ]] && exit 13; mkdir -p bin; touch bin/uv',
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
        "assert pathlib.Path('bin/uv').is_file(), 'native bootstrap was skipped'\n"
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
        "HF_HUB_CACHE": "/hf", "SRT_MODEL_PATH": str(model), "MODEL_PREFIX": "dsr1",
        "SLURM_ACCOUNT": "fixture", "SLURM_PARTITION": "fixture",
        "B200_SQUASH_DIR": str(tmp_path), "B300_HF_CACHE_HOST_DIR": str(tmp_path),
        "B300_HF_CACHE_CONTAINER_DIR": "/hf", "ENROOT_IMPORT_TIME_LIMIT": "10",
        "INFERENCEX_RUNTIME_ENV_VARS": "REQUIRE_POWER",
        "TEST_FAILURE": failure, "CANCEL_CAPTURE": str(capture),
    }
    env.pop("AIPERF_DRAIN_TIMEOUT_SECONDS", None)
    env.pop("AIPERF_DRAIN_POLL_SECONDS", None)
    result = subprocess.run(
        ["bash", str(ROOT / f"runners/launch_{pool}.sh")], cwd=tmp_path,
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == {"none": 0, "allocation": 1, "submission": 7, "bootstrap": 13}[failure], result.stderr
    if failure == "bootstrap":
        assert not (tmp_path / "srt-single-node-submission.json").exists()
        assert not capture.exists()
        return
    assert json.loads((tmp_path / "point-identity.json").read_text()) == {"completed": 2}
    assert (tmp_path / "gpu_metrics.csv").read_text() == "gpu,power\n0,300\n"
    assert json.loads((tmp_path / "gpu_metrics_context.json").read_text()) == {"device_count": 4}
    assert (tmp_path / "srt-single-node-logs.tar.gz").stat().st_size > 0
    cluster_config = yaml.safe_load(next(tmp_path.glob("srt-single.*/checkout/srtslurm.yaml")).read_text())
    assert cluster_config["containers"]["test:tag"] == "test:tag"
    assert cluster_config["use_exclusive_sbatch_directive"] is True
    assert (capture.read_text() if capture.exists() else "") == ("42\n" if failure == "submission" else "")
