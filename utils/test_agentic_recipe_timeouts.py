"""vLLM's execute_model RPC timeout must outlast a Mooncake KV offload stall.

A recipe that offloads KV to Mooncake blocks inside `execute_model` while a
remote load completes. vLLM caps that RPC with
`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, which defaults to 300 s, and raising
`VLLM_RPC_TIMEOUT` does not cover this path. On the Kimi-K3 B300 sweep the
observed stall tail reached 240 s on a point that recovered, so the default
leaves under a minute of headroom and kills the engine with
`EngineDeadError: RPC call to sample_tokens timed out`.
"""

import re
from pathlib import Path

import pytest

AGENTIC_RECIPES = Path(__file__).resolve().parents[1] / "benchmarks/single_node/agentic"

# Minimum headroom over the longest stall observed on a passing point (240 s).
MIN_EXECUTE_MODEL_TIMEOUT_SECONDS = 600

# Recipes that still run on vLLM's 300 s default. Each needs its own
# perf-changelog entry and qualifying sweep, so they are not fixed here.
KNOWN_DEFAULT_TIMEOUT = frozenset(
    {
        "dsv4_fp4_b200_vllm_mtp.sh",
        "dsv4_fp4_b300_vllm_mtp.sh",
        "minimaxm3_fp8_h100_mtp.sh",
        "minimaxm3_fp8_h200_mtp.sh",
    }
)

EXPORT = re.compile(
    r"""^\s*export\s+VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=["']?(\d+)["']?\s*$""",
    re.MULTILINE,
)


def offloading_vllm_recipes():
    for recipe in sorted(AGENTIC_RECIPES.glob("*.sh")):
        text = recipe.read_text()
        if "vllm serve" in text and "MooncakeStoreConnector" in text:
            yield recipe, text


def test_offloading_recipes_are_discovered():
    """Guard the detection itself, so a rename cannot silently empty the suite."""
    names = {recipe.name for recipe, _ in offloading_vllm_recipes()}
    assert "kimik3_fp4_b300_vllm_mtp.sh" in names
    assert KNOWN_DEFAULT_TIMEOUT <= names


@pytest.mark.parametrize(
    "recipe,text",
    [pytest.param(r, t, id=r.name) for r, t in offloading_vllm_recipes()],
)
def test_mooncake_recipe_outlasts_an_offload_stall(recipe, text):
    if recipe.name in KNOWN_DEFAULT_TIMEOUT:
        pytest.skip("tracked separately; needs its own changelog entry and sweep")
    match = EXPORT.search(text)
    assert match, (
        f"{recipe.name} offloads KV to Mooncake but never exports "
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS, so execute_model keeps vLLM's 300 s default"
    )
    assert int(match.group(1)) >= MIN_EXECUTE_MODEL_TIMEOUT_SECONDS
