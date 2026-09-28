"""Behavioral checks for llm-d recipe acceptance-length selection."""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

E2E_ROOT = Path(__file__).resolve().parents[3]
RECIPE_MODULE = E2E_ROOT / "benchmarks/multi_node/llm-d/recipe.py"
RECIPE_DIR = E2E_ROOT / "benchmarks/multi_node/llm-d-recipes/agentic"


def _load_recipe_module():
    spec = importlib.util.spec_from_file_location("llm_d_recipe", RECIPE_MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(E2E_ROOT))
    spec.loader.exec_module(module)
    return module


def _spec_config(output: str) -> dict:
    match = re.search(r"--speculative-config (\{.*?\})(?:\s|$)", output)
    assert match is not None, output
    return json.loads(match.group(1))


@pytest.mark.parametrize(
    ("recipe_name", "role", "expect_golden", "kv_offloading"),
    [
        ("agg-gb200-tp8-dspark-agentic.yaml", "prefill", True, "none"),
        ("agg-gb200-dep8-dspark-agentic.yaml", "prefill", False, "none"),
        ("agg-gb200-dep8-dspark-mooncake-agentic.yaml", "prefill", False, "dram"),
        ("disagg-gb200-1p1d-dep8-dep8-dspark-agentic.yaml", "decode", False, "dram"),
    ],
)
def test_role_assignments_use_golden_al_only_for_tp8(
    recipe_name: str, role: str, expect_golden: bool, kv_offloading: str,
) -> None:
    recipe = _load_recipe_module()
    env = {
        "IS_AGENTIC": "1",
        "SPEC_DECODING": "mtp",
        "EVAL_ONLY": "false",
        "RUN_EVAL": "false",
        "MODEL_PREFIX": "dsv4",
        "THINKING_MODE": "thinking_on",
        "KV_OFFLOADING": kv_offloading,
    }
    if kv_offloading == "dram":
        env["KV_OFFLOAD_BACKEND"] = "mooncake"
    output = recipe.role_assignments(
        yaml.safe_load((RECIPE_DIR / recipe_name).read_text()),
        role,
        env,
    )
    config = _spec_config(output)
    if expect_golden:
        assert config["rejection_sample_method"] == "synthetic"
        assert config["synthetic_acceptance_length"] == 3.61
        assert config["enable_adaptive_verification"] is False
    else:
        assert config["enable_adaptive_verification"] is True
        assert "synthetic_acceptance_length" not in config
        assert "rejection_sample_method" not in config
