"""Tests for the framework-specific synthetic-acceptance injectors."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthetic_injectors import get_injector  # noqa: E402

TRTLLM_RECIPE = """name: dynamo-agg-gb200-tp4-c20-b1-eagle3
backend:
  type: trtllm
  aggregated_environment:
    HF_HUB_OFFLINE: '1'
    TRTLLM_ENABLE_PDL: '1'
  trtllm_config:
    aggregated:
      speculative_config:
        decoding_type: Eagle3
        max_draft_len: 3
        speculative_model: Inferact/MiniMax-M3-EAGLE3-GQA
frontend:
  type: dynamo
"""


def _noop(_msg):
    pass


def test_trtllm_registered_for_both_frameworks():
    assert get_injector("dynamo-trt") is not None
    assert get_injector("trt") is get_injector("dynamo-trt")


def test_trtllm_rewrite_injects_al_minus_one_into_environment():
    injector = get_injector("dynamo-trt")
    new, count = injector.rewrite(TRTLLM_RECIPE, 2.78, _noop)
    assert count == 1
    assert "    TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS: '1.78'\n" in new
    # Inserted directly under the environment header with the block's indentation.
    header = new.index("aggregated_environment:\n")
    assert new.index("TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS") > header
    assert new.count("TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS") == 1


def test_trtllm_rewrite_replaces_existing_value():
    injector = get_injector("dynamo-trt")
    recipe = TRTLLM_RECIPE.replace(
        "    HF_HUB_OFFLINE: '1'\n",
        "    HF_HUB_OFFLINE: '1'\n    TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS: '9'\n",
    )
    new, count = injector.rewrite(recipe, 3.02, _noop)
    assert count == 1
    assert "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS: '2.02'" in new
    assert "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS: '9'" not in new


def test_trtllm_rewrite_real_removes_forced_acceptance():
    injector = get_injector("dynamo-trt")
    injected, _ = injector.rewrite(TRTLLM_RECIPE, 2.78, _noop)
    restored, count = injector.rewrite_real(injected, _noop)
    assert count == 1
    assert "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS" not in restored
    assert restored == TRTLLM_RECIPE


def test_trtllm_rewrite_real_is_noop_without_forced_acceptance():
    injector = get_injector("dynamo-trt")
    restored, count = injector.rewrite_real(TRTLLM_RECIPE, _noop)
    assert count == 0
    assert restored == TRTLLM_RECIPE


def test_trtllm_spec_tokens_from_recipe():
    injector = get_injector("dynamo-trt")
    assert injector.spec_tokens_from_recipe(TRTLLM_RECIPE) == 3
    assert injector.spec_tokens_from_recipe("name: x\n") is None
