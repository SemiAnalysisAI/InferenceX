"""Behavioral checks for binding a matrix point to a native SRT recipe."""

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

from infx.srt_slurm.single_node import main, runtime_arguments, select_recipe, submission_fields
from infx.srt_slurm.synthetic_acceptance import plan_commands

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "utils/srt-slurm/src"))
from srtctl.core.overrides import apply_overrides_to_recipe, parse_overrides


@pytest.fixture
def point(tmp_path, monkeypatch):
    """A fixed-sequence fragment, the point binding its first variant, and the shared blocks."""
    for workload, client in (("fixed-sequence", "client.sh"), ("agentic", "srt_agentic.sh")):
        shared = tmp_path / f"configs/srt-recipes/{workload}-single.yaml"
        shared.parent.mkdir(parents=True, exist_ok=True)
        shared.write_text(yaml.safe_dump({"benchmark": {"type": "custom", "command": f"bash {client}"}}))
    monkeypatch.setenv("INFERENCEX_REPOSITORY_ROOT", str(tmp_path))
    recipe = {
        "engine": "sglang",
        "resources": {"gpus_per_node": 8},
        "roles": {"agg": {
            "nodes": 1, "workers": 1, "gpus": 4,
            "args": {"tensor-parallel-size": 4, "data-parallel-size": 1, "max-running-requests": 32},
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
        "MODEL_PREFIX": "test",
    }
    return path, recipe, env


def test_native_binding_submits_one_point_and_keeps_server_settings(point):
    path, _, env = point
    _, actual = select_recipe(f"{path}:base", env)
    argv = runtime_arguments(f"{path}:base", env)
    apply_overrides_to_recipe(actual, parse_overrides(argv[1::2], []))
    assert actual["model"] == {"path": "hf:test/model", "container": "test:tag", "precision": "fp8"}
    assert actual["srun_options"] == {"gpus-per-node": "4"}
    assert actual["benchmark"] == {"type": "custom", "command": "bash client.sh", "env": {
        "MODEL": "test/model", "ISL": "256", "OSL": "64", "RANDOM_RANGE_RATIO": "0.5",
        "USE_CHAT_TEMPLATE": "false",
        "CONC": "2", "RESULT_FILENAME": "point-identity",
        "RUN_EVAL": "false", "EVAL_ONLY": "false", "RESULT_DIR": "/logs",
        "FRAMEWORK": "sglang",
    }}
    assert actual["roles"]["agg"]["args"] == {
        "tensor-parallel-size": 4, "data-parallel-size": 1, "max-running-requests": 32,
    }


@pytest.mark.parametrize("field,value,message", [
    ("TP", "2", "tensor-parallel-size"), ("RUN_EVAL", "yes", "RUN_EVAL"),
    ("PP_SIZE", "2", "PP_SIZE"), ("RESULT_FILENAME", "", "Missing runtime input"),
    ("EP_SIZE", "2", "expert-parallel-size"), ("SPEC_DECODING", "mtp", "SPEC_DECODING"),
])
def test_mismatched_point_fails_before_submission(point, field, value, message):
    path, _, env = point
    with pytest.raises(ValueError, match=message):
        runtime_arguments(f"{path}:base", {**env, field: value})


def test_agentx_fragments_are_composed_and_bound_to_the_point(point):
    path, recipe, env = point
    path.write_text(yaml.safe_dump({"base": recipe, "override_c2": {"benchmark": {"env": {"CONC": "2"}}}}))
    agentic = {**env, "IS_AGENTIC": "1", "IMAGE": "other:tag", "KV_OFFLOADING": "none"}
    selected, bound = select_recipe(str(path), agentic)
    assert selected == f"{path}:override_c2"
    assert bound["model"] == {"path": "hf:test/model", "container": "other:tag", "precision": "fp8"}
    assert bound["benchmark"] == {"type": "custom", "command": "bash srt_agentic.sh", "env": {
        "CONC": "2", "MODEL": "test/model", "KV_OFFLOADING": "none",
    }}  # fmt: skip
    path.write_text(yaml.safe_dump({"base": {**recipe, "model": {"container": "stale:tag"}}}))
    with pytest.raises(ValueError, match=r"remove base\.model\.container"):
        select_recipe(f"{path}:base", agentic)


def test_kv_offloading_selects_the_dram_variant_and_prepare_sizes_it_from_the_budget(
    point, monkeypatch, tmp_path
):
    path, recipe, env = point
    path.write_text(yaml.safe_dump({
        "base": recipe,
        "override_c2": {"benchmark": {"env": {"CONC": "2", "KV_OFFLOADING": "none"}}},
        "override_c2_dram": {
            "roles": {"agg": {"args": {"hicache-size": "@dram.per-gpu-gb"}}},
            "benchmark": {"env": {"CONC": "2", "KV_OFFLOADING": "dram"}},
        },
    }))  # fmt: skip
    # TP4 on 865 GB: 216.25 GB per GPU.
    dram = {**env, "IS_AGENTIC": "1", "KV_OFFLOADING": "dram", "TOTAL_CPU_DRAM_GB": "865",
            "DURATION": "600"}  # fmt: skip
    for name, value in dram.items():
        monkeypatch.setenv(name, value)
    output = tmp_path / "prepared"
    output.mkdir()
    monkeypatch.setattr(sys, "argv", ["single_node", "prepare", str(path), str(output / "args")])

    main()

    bound = yaml.safe_load((output / "recipe.yaml").read_text())
    assert bound["roles"]["agg"]["args"]["hicache-size"] == 216
    assert bound["benchmark"]["env"]["TOTAL_CPU_DRAM_GB"] == "865"
    none = {**dram, "KV_OFFLOADING": "none", "TOTAL_CPU_DRAM_GB": "0"}
    with pytest.raises(ValueError, match="KV_OFFLOADING: recipe dram != point none"):
        select_recipe(f"{path}:override_c2_dram", none)


def test_ambiguous_native_variants_are_rejected(point):
    path, recipe, env = point
    path.write_text(yaml.safe_dump({"base": recipe, "override_first": {}, "override_second": {}}))
    with pytest.raises(ValueError, match="exactly one"):
        runtime_arguments(str(path), env)


def test_mtp_binding_uses_real_verification_and_the_chat_template(point, tmp_path):
    path, recipe, env = point
    recipe["roles"]["agg"]["args"].update({
        "expert-parallel-size": 4, "speculative-algorithm": "EAGLE",
        "speculative-num-steps": 2, "speculative-num-draft-tokens": 3,
    })
    recipe["roles"]["agg"]["env"] = {"SGLANG_SIMULATE_ACC_LEN": "2.5"}
    path.write_text(yaml.safe_dump({"base": recipe}))
    env = {**env, "EP_SIZE": "4", "SPEC_DECODING": "mtp"}
    _, bound = select_recipe(f"{path}:base", env)
    assert bound["benchmark"]["env"]["USE_CHAT_TEMPLATE"] == "true"
    concrete = tmp_path / "bound.yaml"
    concrete.write_text(yaml.safe_dump(bound))
    argv = runtime_arguments(f"{path}:base", env)
    assert plan_commands(str(concrete), "sglang", ["--json", *argv], env) == [[
        "srtctl", "apply", "--json", *argv, "--file", str(concrete),
        "--unset", "roles.agg.env.SGLANG_SIMULATE_ACC_LEN",
    ]]


def test_concurrency_selects_its_variant_before_binding(point):
    path, recipe, env = point
    path.write_text(yaml.safe_dump({"base": recipe, "zip_override_conc": {
        "roles": {"agg": {"args": {"cuda-graph-max-bs": [2, 4]}}},
        "benchmark": {"env": {"CONC": ["2", "4"]}},
    }}))
    config, bound = select_recipe(str(path), {**env, "CONC": "4"})
    assert config == f"{path}:zip_override_conc[1]"
    assert bound["roles"]["agg"]["args"]["cuda-graph-max-bs"] == 4
    assert bound["benchmark"]["env"]["CONC"] == "4"
    with pytest.raises(ValueError, match="CONC: recipe 4 != point 2"):
        select_recipe(f"{path}:zip_override_conc[1]", env)
    with pytest.raises(ValueError, match="exactly one"):
        select_recipe(str(path), {**env, "CONC": "8"})


def test_eval_binding_changes_context_without_changing_selected_concurrency(point):
    path, recipe, env = point
    recipe["roles"]["agg"]["args"]["context-length"] = 512
    path.write_text(yaml.safe_dump({"base": recipe, "zip_override_conc": {
        "benchmark": {"env": {"CONC": ["2", "4"]}},
    }}))
    env = {**env, "EVAL_ONLY": "true", "RUN_EVAL": "true", "CONC": "4", "MAX_MODEL_LEN": "1024"}
    config, actual = select_recipe(str(path), env)
    argv = runtime_arguments(config, env)
    apply_overrides_to_recipe(actual, parse_overrides(argv[1::2], []))
    assert actual["roles"]["agg"]["args"]["context-length"] == 1024
    assert actual["benchmark"]["env"]["CONC"] == "4"


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
