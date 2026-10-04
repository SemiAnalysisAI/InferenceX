"""Behavioral checks for binding a matrix point to a native SRT recipe."""

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

from infx.srt_slurm.single_node import runtime_arguments, select_recipe, submission_fields
from infx.srt_slurm.synthetic_acceptance import plan_commands, selected_recipes

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "utils/srt-slurm/src"))
from srtctl.core.overrides import apply_overrides_to_recipe, parse_overrides


@pytest.mark.parametrize("sku,variant", [("gb200", "override_tp4"), ("gb300", "override_c1")])
def test_qwen_aggregate_power_shape_matches_serving_gpus(sku, variant):
    path = ROOT / (
        "benchmarks/multi_node/srt-slurm-recipes/qwen3.5/sglang/"
        f"{sku}-fp4/agentx/agg-variants.yaml"
    )
    [(_, recipe)] = selected_recipes(yaml.safe_load(path.read_text()), variant)
    env = recipe["benchmark"]["env"]
    agg = recipe["roles"]["agg"]

    assert env["IS_MULTINODE"] == "false"
    assert int(env["TP"]) == agg["args"]["tensor-parallel-size"]
    assert int(env["TP"]) * int(env["PP_SIZE"]) * int(env["PCP_SIZE"]) == agg["gpus"]


@pytest.fixture
def point(tmp_path):
    recipe = {
        "engine": "sglang",
        "resources": {"gpus_per_node": 8},
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
    assert actual["srun_options"] == {"gpus-per-node": "4"}
    assert actual["benchmark"]["env"] == {
        "MODEL": "test/model", "ISL": "256", "OSL": "64", "RANDOM_RANGE_RATIO": "0.5",
        "USE_CHAT_TEMPLATE": "false",
        "CONC": "2", "RESULT_FILENAME": "point-identity",
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


def test_atom_binding_uses_allocation_tp_and_native_mtp_arguments(point):
    path, recipe, env = point
    recipe["engine"] = "atom"
    recipe["roles"]["agg"]["args"] = {
        "method": "mtp", "num-speculative-tokens": 3, "kv_cache_dtype": "fp8",
        "enable-expert-parallel": True, "enable-dp-attention": True,
    }
    recipe["benchmark"]["env"]["USE_CHAT_TEMPLATE"] = "true"
    path.write_text(yaml.safe_dump({"base": recipe}))
    env = {**env, "FRAMEWORK": "atom", "EP_SIZE": "4", "DP_ATTENTION": "true",
           "SPEC_DECODING": "mtp", "EVAL_ONLY": "true", "MAX_MODEL_LEN": "2048"}
    argv = runtime_arguments(f"{path}:base", env)
    actual = copy.deepcopy(recipe)
    apply_overrides_to_recipe(actual, parse_overrides(argv[1::2], []))
    assert actual["roles"]["agg"]["args"] == {
        "method": "mtp", "num-speculative-tokens": 3, "kv_cache_dtype": "fp8",
        "enable-expert-parallel": True, "enable-dp-attention": True, "max-model-len": 2048,
    }
    assert plan_commands(f"{path}:base", "atom", ["--json", *argv], env) == [[
        "srtctl", "apply", "--json", *argv, "--file", f"{path}:base",
    ]]
    for changes, error in [
        ({"EP_SIZE": "2"}, "expert parallelism"),
        ({"EP_SIZE": "1"}, "enable-expert-parallel"),
        ({"TP": "8", "EP_SIZE": "8"}, "ATOM TP"),
        ({"DP_ATTENTION": "false"}, "DP_ATTENTION"),
        ({"DCP_SIZE": "8"}, "DCP_SIZE"),
    ]:
        with pytest.raises(ValueError, match=error):
            runtime_arguments(f"{path}:base", {**env, **changes})
    recipe["roles"]["agg"]["args"]["decode-context-parallel-size"] = 8
    path.write_text(yaml.safe_dump({"base": recipe}))
    runtime_arguments(f"{path}:base", {**env, "DCP_SIZE": "8"})
    with pytest.raises(ValueError, match="DCP_SIZE"):
        runtime_arguments(f"{path}:base", env)


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


def test_runtime_container_options_remain_native_mapping(point):
    path, recipe, env = point
    env = {**env, "SRT_SRUN_OPTIONS": json.dumps({
        "container-remap-root": "", "container-writable": "", "container-workdir": "/custom",
    })}
    argv = runtime_arguments(f"{path}:base", env)
    actual = copy.deepcopy(recipe)
    apply_overrides_to_recipe(actual, parse_overrides(argv[1::2], []))
    assert actual['srun_options'] == {
        'gpus-per-node': '4', 'container-remap-root': '', 'container-writable': '',
        'container-workdir': '/custom',
    }
    with pytest.raises(ValueError, match='must map option names to string values'):
        runtime_arguments(f"{path}:base", {**env, 'SRT_SRUN_OPTIONS': '{"container-remap-root": true}'})


@pytest.mark.parametrize("recipe_required,requested", [(True, "0"), (False, "1")])
def test_binding_keeps_either_required_power_policy(point, recipe_required, requested):
    path, recipe, env = point
    recipe["telemetry"] = {"required": recipe_required}
    path.write_text(yaml.safe_dump({"base": recipe}))
    argv = runtime_arguments(f"{path}:base", {**env, "REQUIRE_POWER": requested})
    actual = copy.deepcopy(recipe)
    apply_overrides_to_recipe(actual, parse_overrides(argv[1::2], []))
    assert actual["telemetry"]["required"] is True
    assert actual["benchmark"]["concurrencies"] == [2]
