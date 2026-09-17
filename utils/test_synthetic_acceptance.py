"""Exercise automatic golden-AL selection through native srtctl overrides."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from infx.srt_slurm.synthetic_acceptance import (
    build_overrides,
    plan_commands,
    selected_recipes,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils/srt-slurm/src"))

# Exercise the pinned upstream checkout without installing its serving dependencies.
from srtctl.core.overrides import (  # noqa: E402
    apply_overrides_to_recipe,
    parse_overrides,
)

ENV = {
    "MODEL_PREFIX": "dsv4",
    "IS_AGENTIC": "1",
    "SPEC_DECODING": "mtp",
    "EVAL_ONLY": "false",
    "THINKING_MODE": "thinking_on",
}


@pytest.fixture
def golden_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "golden"
    directory.mkdir()
    for filename, model, acceptance in [
        ("dsv4_mtp.yaml", "deepseek-v4-pro", 2.4),
        ("dsv4-pro-0813-dspark.yaml", "deepseek-v4-pro-0813", 2.7),
        ("kimik3_dspark.yaml", "kimi-k3", 2.8),
        (
            "kimik3_dspark_probabilistic_sample_method_block_rejection_sample_method.yaml",
            "kimi-k3",
            2.9,
        ),
        ("minimaxm3_eagle3.yaml", "minimax-m3", 2.5),
        ("minimaxm3_eagle3_gqa.yaml", "minimax-m3", 2.6),
    ]:
        (directory / filename).write_text(
            yaml.safe_dump(
                {
                    model: {
                        "thinking_on": {2: 1.8, 3: acceptance},
                        "thinking_off": {3: 2.1},
                    }
                }
            )
        )
    return directory


def apply_native(recipe: dict[str, Any], argv: list[str]) -> dict[str, Any]:
    result = copy.deepcopy(recipe)
    sets = [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == "--set"]
    unsets = [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == "--unset"]
    apply_overrides_to_recipe(result, parse_overrides(sets, unsets))
    return result


def vllm_recipe(method: str = "mtp", **extra: Any) -> dict[str, Any]:
    return {
        "roles": {
            "agg": {
                "args": {
                    "speculative-config": json.dumps(
                        {"method": method, "num_speculative_tokens": 3, **extra}
                    )
                }
            }
        }
    }


@pytest.mark.parametrize(
    ("prefix", "method", "extra", "expected"),
    [
        ("dsv4", "mtp", {}, 2.4),
        ("dsv4", "dspark", {}, 2.7),
        ("kimik3", "dspark", {"draft_sample_method": "greedy"}, 2.8),
        ("kimik3", "dspark", {"draft_sample_method": "probabilistic"}, 2.9),
        ("minimaxm3", "eagle3", {"model": "Inferact/MiniMax-M3-EAGLE3"}, 2.5),
        ("minimaxm3", "eagle3", {"model": "Inferact/MiniMax-M3-EAGLE3-GQA"}, 2.6),
    ],
)
def test_automatic_curve_selection_preserves_json(
    golden_dir: Path, prefix: str, method: str, extra: dict[str, Any], expected: float
) -> None:
    recipe = vllm_recipe(method, custom={"literal": "x=y z"}, **extra)
    original = copy.deepcopy(recipe)
    env = {
        **ENV,
        "MODEL_PREFIX": prefix,
        "SPEC_DECODING": method,
        "RUN_EVAL": "true",
        "SYNTHETIC_ACCEPTANCE": "false",
        "SYNTHETIC_ACCEPTANCE_LENGTH": "99",
    }
    result = apply_native(recipe, build_overrides(recipe, "vllm", env, golden_dir=golden_dir))
    spec = json.loads(result["roles"]["agg"]["args"]["speculative-config"])
    assert spec == {
        "method": method,
        "num_speculative_tokens": 3,
        **extra,
        "custom": {"literal": "x=y z"},
        "rejection_sample_method": "synthetic",
        "synthetic_acceptance_length": expected,
    }
    assert recipe == original


def test_thinking_mode_and_decode_priority(golden_dir: Path) -> None:
    recipe = vllm_recipe(num_speculative_tokens=2)
    recipe["roles"]["decode"] = vllm_recipe()["roles"]["agg"]
    recipe["roles"]["prefill"] = {"args": {"tensor-parallel-size": 8}}
    result = apply_native(
        recipe,
        build_overrides(
            recipe,
            "dynamo-vllm",
            {**ENV, "THINKING_MODE": "thinking_off"},
            golden_dir=golden_dir,
        ),
    )
    assert (
        json.loads(result["roles"]["decode"]["args"]["speculative-config"])[
            "synthetic_acceptance_length"
        ]
        == 2.1
    )
    assert result["roles"]["prefill"] == {"args": {"tensor-parallel-size": 8}}


@pytest.mark.parametrize("sampling", [None, "unknown"])
def test_kimi_curve_requires_an_explicit_supported_sampler(
    golden_dir: Path, sampling: str | None
) -> None:
    fields = {} if sampling is None else {"draft_sample_method": sampling}
    with pytest.raises(ValueError, match="Kimi DSpark golden curve for draft sampling"):
        build_overrides(
            vllm_recipe("dspark", **fields),
            "vllm",
            {**ENV, "MODEL_PREFIX": "kimik3", "SPEC_DECODING": "dspark"},
            golden_dir=golden_dir,
        )


@pytest.mark.parametrize(
    ("framework", "args", "environment", "expected"),
    [
        (
            "dynamo-sglang",
            {
                "speculative-algorithm": "DSpark",
                "speculative-dspark-block-size": 3,
                "speculative-num-steps": 2,
            },
            {"SPEC_DECODING": "dspark"},
            {
                "SGLANG_SIMULATE_ACC_LEN": "2.7",
                "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
                "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
            },
        ),
        (
            "dynamo-sglang",
            {"speculative-algorithm": "EAGLE", "speculative-num-steps": 3},
            {},
            {
                "SGLANG_SIMULATE_ACC_LEN": "2.4",
                "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
                "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
            },
        ),
        (
            "trt",
            {
                "speculative_config": {
                    "decoding_type": "EAGLE3",
                    "max_draft_len": 3,
                    "speculative_model": "Inferact/MiniMax-M3-EAGLE3-GQA",
                }
            },
            {"MODEL_PREFIX": "minimaxm3", "SPEC_DECODING": "eagle3"},
            {"TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS": "1.6"},
        ),
    ],
)
def test_engine_token_selection_and_environment(
    golden_dir: Path,
    framework: str,
    args: dict[str, Any],
    environment: dict[str, str],
    expected: dict[str, str],
) -> None:
    recipe = {
        "roles": {
            "prefill": {"args": {}, "env": {"KEEP": "prefill"}},
            "decode": {"args": args, "env": {"KEEP": "worker"}},
        },
        "benchmark": {"env": {"KEEP": "client"}},
        "environment": {"KEEP_GLOBAL": "yes", **dict.fromkeys(expected, "99")},
    }
    result = apply_native(
        recipe,
        build_overrides(recipe, framework, {**ENV, **environment}, golden_dir=golden_dir),
    )
    assert result["roles"]["decode"]["env"] == {"KEEP": "worker", **expected}
    assert result["roles"]["prefill"]["env"] == {"KEEP": "prefill"}
    assert result["benchmark"] == {"env": {"KEEP": "client"}}
    assert result["environment"] == {"KEEP_GLOBAL": "yes"}


@pytest.mark.parametrize(
    "environment",
    [{"EVAL_ONLY": "true"}, {"IS_AGENTIC": "0"}, {"SPEC_DECODING": "none"}],
)
@pytest.mark.parametrize("framework", ["vllm", "dynamo-sglang", "trt"])
def test_real_runs_clear_synthetic_without_a_curve(
    tmp_path: Path, framework: str, environment: dict[str, str]
) -> None:
    recipe = vllm_recipe(
        "dspark", rejection_sample_method="synthetic", synthetic_acceptance_length=8
    )
    recipe["roles"]["agg"]["env"] = {
        "KEEP": "yes",
        "SGLANG_SIMULATE_ACC_LEN": "8",
        "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
        "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
        "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS": "7",
    }
    recipe["environment"] = copy.deepcopy(recipe["roles"]["agg"]["env"])
    env = {**ENV, **environment}
    del env["THINKING_MODE"]
    result = apply_native(
        recipe,
        build_overrides(recipe, framework, env, golden_dir=tmp_path / "absent"),
    )
    role = result["roles"]["agg"]
    if framework == "vllm":
        assert json.loads(role["args"]["speculative-config"]) == {
            "method": "dspark",
            "num_speculative_tokens": 3,
            "rejection_sample_method": "block",
        }
    elif framework == "trt":
        assert "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS" not in role["env"]
        assert "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS" not in result["environment"]
    else:
        assert not any(key.startswith("SGLANG_SIMULATE_ACC_") for key in role["env"])
        assert not any(key.startswith("SGLANG_SIMULATE_ACC_") for key in result["environment"])
    assert role["env"]["KEEP"] == "yes"
    assert result["environment"]["KEEP"] == "yes"


@pytest.mark.parametrize(
    "curve",
    [
        None,
        {"thinking_on": {2: 1.8}},
        {"thinking_off": {3: 2.1}},
        {"thinking_on": {3: float("nan")}},
    ],
)
def test_missing_or_invalid_golden_rejects_injection(
    tmp_path: Path, curve: dict[str, dict[int, float]] | None
) -> None:
    if curve is not None:
        (tmp_path / "dsv4_mtp.yaml").write_text(yaml.safe_dump({"deepseek-v4-pro": curve}))
    with pytest.raises(
        ValueError,
        match=r"(?:No committed golden curve|No golden acceptance|Invalid golden acceptance)",
    ):
        build_overrides(vllm_recipe(), "vllm", ENV, golden_dir=tmp_path)


def test_non_speculative_passthrough_and_malformed_spec(tmp_path: Path) -> None:
    assert (
        build_overrides(
            {"roles": {"agg": {}}},
            "vllm",
            {**ENV, "SPEC_DECODING": "none"},
            golden_dir=tmp_path,
        )
        == []
    )
    with pytest.raises(ValueError, match="JSON object"):
        build_overrides(
            {"roles": {"agg": {"args": {"speculative-config": "[]"}}}},
            "vllm",
            ENV,
            golden_dir=tmp_path,
        )


def test_variants_use_resolved_tokens_and_preserve_caller_arguments(
    tmp_path: Path, golden_dir: Path
) -> None:
    recipe = tmp_path / "recipe with spaces.yaml"
    raw = {
        "schema": 2,
        "base": {"name": "worker", **vllm_recipe()},
        "zip_override_test": {
            "name": ["first", "second"],
            "roles": {
                "agg": {
                    "args": {
                        "speculative-config": [
                            '{"method":"mtp","num_speculative_tokens":2}',
                            '{"method":"mtp","num_speculative_tokens":3}',
                        ]
                    }
                }
            },
        },
    }
    original = yaml.safe_dump(raw)
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
            'benchmark.env.KEEP="caller"',
        ],
        ENV,
        golden_dir=golden_dir,
    )
    assert len(commands) == 2
    for index, command in enumerate(commands):
        assert command[:2] == ["srtctl", "apply"]
        assert command[command.index("--tags") + 1] == "x y"
        selector = f"zip_override_test[{index}]"
        assert f"{recipe}:{selector}" in command
        resolved = selected_recipes(apply_native(raw, command), selector)[0][1]
        assert (
            json.loads(resolved["roles"]["agg"]["args"]["speculative-config"])[
                "synthetic_acceptance_length"
            ]
            == [1.8, 2.4][index]
        )
        assert resolved["benchmark"]["env"] == {"KEEP": "caller"}
    assert recipe.read_text() == original


def test_caller_json_is_merged_before_golden_selection(tmp_path: Path, golden_dir: Path) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump(vllm_recipe()))
    commands = plan_commands(
        str(recipe),
        "vllm",
        [
            "-f",
            str(recipe),
            "--set",
            'roles.agg.args.speculative-config={"method":"dspark","num_speculative_tokens":3,"model":"draft model"}',
        ],
        ENV,
        golden_dir=golden_dir,
    )
    result = apply_native(yaml.safe_load(recipe.read_text()), commands[0])
    assert json.loads(result["roles"]["agg"]["args"]["speculative-config"]) == {
        "method": "dspark",
        "num_speculative_tokens": 3,
        "model": "draft model",
        "rejection_sample_method": "synthetic",
        "synthetic_acceptance_length": 2.7,
    }


def test_shell_forwards_options_and_submission_failure(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        yaml.safe_dump(
            vllm_recipe(rejection_sample_method="synthetic", synthetic_acceptance_length=8)
        )
    )
    binary = tmp_path / "srtctl"
    binary.write_text(
        f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\nsys.exit(7)\n"
    )
    binary.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; apply_srt_recipe "$2" vllm -f "$2" --tags "a b"',
            "bash",
            str(ROOT / "runners/slurm_utils.sh"),
            str(recipe),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            **ENV,
            "EVAL_ONLY": "true",
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "utils/srt-slurm/src")]),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 7, result.stderr
    argv = json.loads(result.stdout)
    assert argv[:5] == ["apply", "-f", str(recipe), "--tags", "a b"]
    result_recipe = apply_native(yaml.safe_load(recipe.read_text()), argv)
    assert (
        json.loads(result_recipe["roles"]["agg"]["args"]["speculative-config"])[
            "rejection_sample_method"
        ]
        == "block"
    )


@pytest.mark.parametrize(
    "conflict",
    ["zip_cardinality", "roles.agg.env", 'roles."agg".env', " roles.agg.env "],
)
def test_invalid_plan_fails_before_any_submission(
    tmp_path: Path, golden_dir: Path, conflict: str
) -> None:
    recipe = tmp_path / "recipe.yaml"
    raw = {"schema": 2, **vllm_recipe()}
    arguments = ["-f", str(recipe)]
    framework = "vllm"
    if conflict == "zip_cardinality":
        raw = {
            "schema": 2,
            "base": {"name": "worker"},
            "zip_override_tokens": {
                "roles": {
                    "agg": {
                        "args": {
                            "speculative-config": [
                                '{"method":"mtp","num_speculative_tokens":2}',
                                '{"method":"mtp","num_speculative_tokens":3}',
                            ]
                        }
                    }
                }
            },
        }
    else:
        framework = "dynamo-sglang"
        raw = {
            "schema": 2,
            "roles": {
                "agg": {
                    "args": {
                        "speculative-algorithm": "EAGLE",
                        "speculative-num-steps": 3,
                    },
                    "env": {},
                }
            },
        }
        arguments += ["--unset", conflict]
    recipe.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=r"(?:zip_override|conflicts with golden acceptance)"):
        plan_commands(str(recipe), framework, arguments, ENV, golden_dir=golden_dir)


@pytest.mark.parametrize("canonical", [None, "eagle"])
def test_sglang_abbreviated_algorithm_selects_golden(
    golden_dir: Path, canonical: str | None
) -> None:
    args = {"speculative-algo": "EAGLE", "speculative-num-steps": 3}
    if canonical is not None:
        args["speculative-algorithm"] = canonical
    recipe = {"roles": {"decode": {"args": args}}}
    result = apply_native(
        recipe,
        build_overrides(recipe, "dynamo-sglang", ENV, golden_dir=golden_dir),
    )
    assert result["roles"]["decode"]["env"] == {
        "SGLANG_SIMULATE_ACC_LEN": "2.4",
        "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
        "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
    }


def test_sglang_conflicting_algorithm_aliases_fail(golden_dir: Path) -> None:
    recipe = {
        "roles": {
            "decode": {
                "args": {
                    "speculative-algo": "EAGLE",
                    "speculative-algorithm": "DSPARK",
                    "speculative-num-steps": 3,
                    "speculative-dspark-block-size": 3,
                }
            }
        }
    }
    with pytest.raises(ValueError, match="Conflicting speculative-algorithm and speculative-algo"):
        build_overrides(recipe, "dynamo-sglang", ENV, golden_dir=golden_dir)
