"""CPU-only checks for the llm-d recipe-to-vLLM contract."""

import json
from pathlib import Path
import shlex

import pytest
import yaml

from recipe import mooncake_config, role_assignments


RECIPES = Path(__file__).resolve().parents[1] / "llm-d-recipes/agentic"


@pytest.mark.parametrize("path", sorted(RECIPES.glob("*.yaml")), ids=lambda path: path.stem)
@pytest.mark.parametrize("eval_only", ["false", "true"])
def test_dspark_uses_role_golden_al_and_native_context(path, eval_only):
    recipe = yaml.safe_load(path.read_text())
    env = {
        "IS_AGENTIC": "1", "EVAL_ONLY": eval_only, "SPEC_DECODING": "mtp",
        "MODEL_NAME": "deepseek-ai/DeepSeek-V4-Pro-0813",
        "KV_OFFLOADING": "dram" if "mooncake" in recipe else "none",
        "KV_OFFLOAD_BACKEND": "mooncake" if "mooncake" in recipe else "",
    }
    for role in ("prefill", "decode"):
        if role not in recipe:
            continue
        assignment = role_assignments(recipe, role, env).splitlines()[0]
        args = shlex.split(assignment)[0].removeprefix("ROLE_EXTRA_ARGS=")
        start = args.index("--speculative-config ") + len("--speculative-config ")
        config, _ = json.JSONDecoder().raw_decode(args[start:])
        if eval_only == "true":
            assert "synthetic_acceptance_length" not in config
            assert "rejection_sample_method" not in config
        else:
            assert config["rejection_sample_method"] == "synthetic"
            assert config["synthetic_acceptance_length"] == {1: 1.84, 5: 3.61}[config["num_speculative_tokens"]]
            assert config["enable_adaptive_verification"] is False
        assert "--max-model-len" not in args
        assert "--max-num-seqs" not in args
        assert "--tool-call-parser deepseek_v4" in args
        assert "--reasoning-parser deepseek_v4" in args


def test_offload_metadata_and_runtime_budget_agree():
    recipe = yaml.safe_load((RECIPES / "agg-gb200-dep8-dspark-mooncake-agentic.yaml").read_text())
    env = {"IS_AGENTIC": "1", "KV_OFFLOADING": "none"}
    with pytest.raises(ValueError, match="KV_OFFLOADING=dram"):
        mooncake_config(recipe, env)
    env.update(KV_OFFLOADING="dram", KV_OFFLOAD_BACKEND="mooncake", TOTAL_CPU_DRAM_GB="1298", GPUS_PER_NODE="8")
    config = json.loads(mooncake_config(recipe, env))
    assert config["enable_offload"] is False
    assert config["global_segment_size"] * 8 == 1298 * 10**9


def test_no_offload_recipe_rejects_dram_metadata():
    with pytest.raises(ValueError, match="KV_OFFLOADING=none"):
        role_assignments({}, "prefill", {"IS_AGENTIC": "1", "KV_OFFLOADING": "dram"})


def test_fixed_length_recipe_is_not_rewritten():
    recipe = {"prefill": {"extra-args": '--speculative-config {"method":"dspark","num_speculative_tokens":5}'}}
    assignment = role_assignments(recipe, "prefill", {"IS_AGENTIC": "0"})
    assert "synthetic" not in assignment


def test_unregistered_golden_al_fails_closed():
    recipe = yaml.safe_load((RECIPES / "agg-gb200-tp8-dspark-agentic.yaml").read_text())
    env = {"IS_AGENTIC": "1", "KV_OFFLOADING": "none", "SPEC_DECODING": "mtp", "MODEL_NAME": "other"}
    with pytest.raises(ValueError, match="No registered DSpark golden AL"):
        role_assignments(recipe, "prefill", env)
