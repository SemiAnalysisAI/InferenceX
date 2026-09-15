"""Exercise acceptance settings through the pinned native srtctl overrides."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils/srt-slurm/src"))

from infx.recipes.synthetic_acceptance import (
    build_overrides,
    plan_commands,
    selected_recipes,
)
from srtctl.core.overrides import apply_overrides_to_recipe, parse_overrides


def apply_native(recipe, overrides):
    result = copy.deepcopy(recipe)
    sets = [item.argv()[1] for item in overrides if not item.unset]
    unsets = [item.argv()[1] for item in overrides if item.unset]
    apply_overrides_to_recipe(result, parse_overrides(sets, unsets))
    return result


SYNTHETIC = {"SYNTHETIC_ACCEPTANCE": "true", "SYNTHETIC_ACCEPTANCE_LENGTH": "2.78"}


@pytest.mark.parametrize("framework", ["vllm", "dynamo-vllm"])
def test_vllm_preserves_per_role_json_and_restores_real_verification(framework):
    recipe = {
        "roles": {
            "prefill": {
                "args": {
                    "speculative-config": '{"method":"mtp","num_speculative_tokens":3,"custom":{"a":"x=y z"}}'
                }
            },
            "decode": {
                "args": {
                    "speculative-config": '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
                }
            },
        }
    }
    original = copy.deepcopy(recipe)
    synthetic = apply_native(recipe, build_overrides(recipe, framework, SYNTHETIC))
    assert json.loads(synthetic["roles"]["prefill"]["args"]["speculative-config"]) == {
        "method": "mtp",
        "num_speculative_tokens": 3,
        "custom": {"a": "x=y z"},
        "rejection_sample_method": "synthetic",
        "synthetic_acceptance_length": 2.78,
    }
    assert json.loads(synthetic["roles"]["decode"]["args"]["speculative-config"]) == {
        "method": "dspark",
        "num_speculative_tokens": 5,
        "draft_sample_method": "probabilistic",
        "rejection_sample_method": "synthetic",
        "synthetic_acceptance_length": 2.78,
    }
    restored = apply_native(
        synthetic,
        build_overrides(synthetic, framework, {**SYNTHETIC, "EVAL_ONLY": "true"}),
    )
    assert json.loads(restored["roles"]["decode"]["args"]["speculative-config"]) == {
        "method": "dspark",
        "num_speculative_tokens": 5,
        "draft_sample_method": "probabilistic",
        "rejection_sample_method": "block",
    }
    assert recipe == original


@pytest.mark.parametrize(
    "framework,key,value",
    [
        ("dynamo-sglang", "SGLANG_SIMULATE_ACC_LEN", "2.78"),
        ("trt", "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS", "1.78"),
        ("dynamo-trt", "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS", "1.78"),
    ],
)
def test_environment_overrides_target_workers_and_eval_removes_them(
    framework, key, value
):
    recipe = yaml.safe_load("""roles:
  prefill:
    env: &common
      KEEP: original
  decode:
    env: *common
  agg:
    nodes: 1
frontend:
  env:
    KEEP: frontend
benchmark:
  env:
    KEEP: client
""")
    rewritten = apply_native(recipe, build_overrides(recipe, framework, SYNTHETIC))
    assert [role["env"][key] for role in rewritten["roles"].values()] == [
        value,
        value,
        value,
    ]
    assert rewritten["roles"]["decode"]["env"]["KEEP"] == "original"
    assert rewritten["frontend"] == {"env": {"KEEP": "frontend"}}
    assert rewritten["benchmark"] == {"env": {"KEEP": "client"}}
    if framework == "dynamo-sglang":
        assert rewritten["roles"]["agg"]["env"] == {
            "SGLANG_SIMULATE_ACC_LEN": "2.78",
            "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
            "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
        }
    restored = apply_native(
        rewritten, build_overrides(rewritten, framework, {"EVAL_ONLY": "true"})
    )
    assert restored["roles"] == {
        "prefill": {"env": {"KEEP": "original"}},
        "decode": {"env": {"KEEP": "original"}},
        "agg": {"nodes": 1, "env": {}},
    }


def test_trt_replaces_existing_forced_acceptance():
    recipe = {
        "roles": {"agg": {"env": {"TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS": "9"}}}
    }
    rewritten = apply_native(recipe, build_overrides(recipe, "trt", SYNTHETIC))
    assert rewritten["roles"]["agg"]["env"] == {
        "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS": "1.78"
    }


@pytest.mark.parametrize(
    "reference,environment,expected",
    [
        ({"deepseek-v4-pro": [{3: 2.51}]}, {"MODEL_PREFIX": "dsv4"}, "2.51"),
        ({"custom": {3: 2.6}}, {"MODEL_PREFIX": "custom"}, "2.6"),
        (
            {"deepseek-v4-pro-0813": {"thinking_off": {5: 3.5}}},
            {
                "MODEL_PREFIX": "dsv4dspark",
                "NUM_SPEC_TOKENS": "5",
                "THINKING_MODE": "thinking_off",
            },
            "3.5",
        ),
    ],
)
def test_reference_lookup_preserves_model_aliases_and_token_selection(
    reference, environment, expected
):
    recipe = {"roles": {"decode": {"args": {"speculative-num-steps": 3}}}}
    env = {"SYNTHETIC_ACCEPTANCE": "true", **environment}
    rewritten = apply_native(
        recipe, build_overrides(recipe, "dynamo-sglang", env, reference=reference)
    )
    assert rewritten["roles"]["decode"]["env"]["SGLANG_SIMULATE_ACC_LEN"] == expected


@pytest.mark.parametrize(
    "environment,framework",
    [({}, "vllm"), ({"RUN_EVAL": "true"}, "vllm"), ({"EVAL_ONLY": "true"}, "tilert")],
)
def test_disabled_and_unsupported_eval_paths_do_not_read_recipe(environment, framework):
    assert plan_commands(
        "missing.yaml",
        framework,
        ["-f", "missing.yaml"],
        environment,
        enable_throughput=True,
    ) == [["srtctl", "apply", "-f", "missing.yaml"]]


def test_eval_only_launcher_mode_does_not_enable_throughput_injection():
    assert plan_commands(
        "missing.yaml",
        "vllm",
        ["-f", "missing.yaml"],
        SYNTHETIC,
        enable_throughput=False,
    ) == [["srtctl", "apply", "-f", "missing.yaml"]]


@pytest.mark.parametrize(
    "recipe,framework,environment,message",
    [
        ({}, "unknown", SYNTHETIC, "no synthetic-acceptance backend"),
        ({"roles": {"agg": {}}}, "vllm", SYNTHETIC, "no speculative-config"),
        (
            {"roles": {"agg": {"args": {"speculative-config": "[]"}}}},
            "vllm",
            SYNTHETIC,
            "JSON object",
        ),
        (
            {"roles": {"agg": {"env": {"SGLANG_SIMULATE_ACC_LEN": "1"}}}},
            "dynamo-sglang",
            SYNTHETIC,
            "already contains",
        ),
        (
            {"roles": {"agg": {}}},
            "trt",
            {**SYNTHETIC, "SYNTHETIC_ACCEPTANCE_LENGTH": "nan"},
            "finite",
        ),
    ],
)
def test_invalid_injection_fails_before_submission(
    recipe, framework, environment, message
):
    with pytest.raises(ValueError, match=message):
        build_overrides(recipe, framework, environment)


def test_eval_only_accepts_real_non_speculative_recipe():
    assert (
        build_overrides({"roles": {"agg": {"nodes": 1}}}, "vllm", {"EVAL_ONLY": "true"})
        == []
    )


def test_variant_plan_preserves_selected_json_and_native_caller_options(tmp_path):
    recipe = tmp_path / "recipe with spaces.yaml"
    original = """schema: 2
base:
  name: worker
  roles:
    decode:
      args:
        speculative-config: '{"method":"mtp","num_speculative_tokens":2}'
override_other:
  roles:
    decode:
      args:
        speculative-config: '{"method":"eagle3","num_speculative_tokens":4}'
zip_override_test:
  name: [first, second]
  roles:
    decode:
      args:
        speculative-config: ['{"method":"dspark","num_speculative_tokens":3}', '{"method":"mtp","num_speculative_tokens":5}']
"""
    recipe.write_text(original)
    commands = plan_commands(
        f"{recipe}:zip_override_test",
        "vllm",
        [
            "-f",
            f"{recipe}:zip_override_test",
            "--tags",
            "x y",
            "--set",
            'post_eval.command=["bash","eval.sh"]',
        ],
        SYNTHETIC,
        enable_throughput=True,
    )
    assert len(commands) == 2
    for index, command in enumerate(commands):
        assert command[:2] == ["srtctl", "apply"]
        assert command[command.index("--tags") + 1] == "x y"
        assert command[-4:-2] == ["--file", f"{recipe}:zip_override_test[{index}]"]
        native = yaml.safe_load(original)
        sets = [command[i + 1] for i, arg in enumerate(command[:-1]) if arg == "--set"]
        apply_overrides_to_recipe(native, parse_overrides(sets, []))
        resolved = selected_recipes(native, f"zip_override_test[{index}]")[0][1]
        assert json.loads(
            resolved["roles"]["decode"]["args"]["speculative-config"]
        ) == {
            "method": ["dspark", "mtp"][index],
            "num_speculative_tokens": [3, 5][index],
            "rejection_sample_method": "synthetic",
            "synthetic_acceptance_length": 2.78,
        }
        assert resolved["post_eval"]["command"] == ["bash", "eval.sh"]
    assert recipe.read_text() == original
    assert [
        selector for selector, _ in selected_recipes(yaml.safe_load(original), "*other")
    ] == ["override_other"]


def test_shell_adapter_forwards_arguments_and_failure_without_changing_recipe(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    original = 'schema: 2\nroles:\n  agg:\n    args:\n      speculative-config: \'{"method":"mtp","num_speculative_tokens":3}\'\n'
    recipe.write_text(original)
    binary = tmp_path / "srtctl"
    binary.write_text(
        f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\nsys.exit(7)\n"
    )
    binary.chmod(0o755)
    env = {
        **os.environ,
        **SYNTHETIC,
        "EVAL_ONLY": "false",
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "PYTHONPATH": str(ROOT / "utils/srt-slurm/src"),
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; apply_srt_recipe "$2" vllm throughput -f "$2" --tags "a b"',
            "bash",
            str(ROOT / "runners/slurm_utils.sh"),
            str(recipe),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 7, result.stderr
    argv = json.loads(result.stdout)
    assert argv[:5] == ["apply", "-f", str(recipe), "--tags", "a b"]
    assert json.loads(yaml.safe_load(argv[-1].split("=", 1)[1])) == {
        "method": "mtp",
        "num_speculative_tokens": 3,
        "rejection_sample_method": "synthetic",
        "synthetic_acceptance_length": 2.78,
    }
    assert recipe.read_text() == original


def test_caller_json_is_merged_and_materialized_by_native_srtctl(tmp_path):
    from srtctl.cli.submit import materialize_config_path

    recipe = tmp_path / "recipe.yaml"
    original = 'schema: 2\nroles:\n  agg:\n    args:\n      speculative-config: \'{"method":"mtp"}\'\n'
    recipe.write_text(original)
    commands = plan_commands(
        str(recipe),
        "vllm",
        [
            "-f",
            str(recipe),
            "--set",
            'roles.agg.args.speculative-config={"method":"dspark","num_speculative_tokens":5,"model":"draft model"}',
        ],
        SYNTHETIC,
        enable_throughput=True,
    )
    command = commands[0]
    sets = [command[i + 1] for i, arg in enumerate(command[:-1]) if arg == "--set"]
    with materialize_config_path(recipe, parse_overrides(sets, [])) as rendered:
        config = yaml.safe_load(rendered.read_text())
        assert json.loads(config["roles"]["agg"]["args"]["speculative-config"]) == {
            "method": "dspark",
            "num_speculative_tokens": 5,
            "model": "draft model",
            "rejection_sample_method": "synthetic",
            "synthetic_acceptance_length": 2.78,
        }
        assert recipe.read_text() == original
    assert not rendered.exists()


def test_plan_validates_zip_cardinality_before_any_submission(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("""schema: 2
base:
  name: worker
zip_override_tokens:
  roles:
    decode:
      args:
        speculative-config: ['{"method":"mtp","num_speculative_tokens":2}', '{"method":"mtp","num_speculative_tokens":3}']
""")
    # Native --set turns the only list dimension into a broadcast, so index 1
    # would cease to exist. Reject the entire plan rather than submit index 0.
    with pytest.raises(ValueError, match="out of range"):
        plan_commands(
            str(recipe), "vllm", ["-f", str(recipe)], SYNTHETIC, enable_throughput=True
        )


def test_caller_unset_cannot_silently_remove_generated_acceptance(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("schema: 2\nroles:\n  agg:\n    env: {}\n")
    with pytest.raises(ValueError, match="caller --unset roles.agg.env conflicts"):
        plan_commands(
            str(recipe),
            "dynamo-sglang",
            ["-f", str(recipe), "--unset", "roles.agg.env"],
            SYNTHETIC,
            enable_throughput=True,
        )


def test_invalid_recipe_cli_does_not_invoke_srtctl(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("[]\n")
    binary = tmp_path / "srtctl"
    binary.write_text("#!/bin/sh\nprintf unexpected-submission\n")
    binary.chmod(0o755)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "infx.recipes.synthetic_acceptance",
            str(recipe),
            "vllm",
            "throughput",
            "--",
            "-f",
            str(recipe),
        ],
        env={
            **os.environ,
            **SYNTHETIC,
            "EVAL_ONLY": "false",
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "PYTHONPATH": str(ROOT),
        },
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert "recipe must be a mapping" in result.stderr
