"""Tests for the framework-specific synthetic-acceptance injectors."""

import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthetic_injectors import get_injector  # noqa: E402

TRTLLM_RECIPE = """schema: 2
name: dynamo-agg-gb200-tp4-c20-b1-eagle3
engine: trtllm
roles:
  agg:
    nodes: 1
    env:
      HF_HUB_OFFLINE: '1'
      TRTLLM_ENABLE_PDL: '1'
    args:
      speculative_config:
        decoding_type: Eagle3
        max_draft_len: 3
        speculative_model: Inferact/MiniMax-M3-EAGLE3-GQA
frontend:
  type: dynamo
"""


def _noop(_msg):
    pass


def test_trtllm_rewrite_injects_al_minus_one_into_environment():
    injector = get_injector("dynamo-trt")
    new, count = injector.rewrite(TRTLLM_RECIPE, 2.78, _noop)
    assert count == 1
    assert yaml.safe_load(new)["roles"]["agg"]["env"] == {
        "HF_HUB_OFFLINE": "1",
        "TRTLLM_ENABLE_PDL": "1",
        "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS": "1.78",
    }


def test_trtllm_rewrite_replaces_existing_value():
    injector = get_injector("trt")
    recipe = TRTLLM_RECIPE.replace(
        "      HF_HUB_OFFLINE: '1'\n",
        "      HF_HUB_OFFLINE: '1'\n      TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS: '9'\n",
    )
    new, count = injector.rewrite(recipe, 3.02, _noop)
    assert count == 1
    assert yaml.safe_load(new)["roles"]["agg"]["env"]["TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS"] == "2.02"


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


def test_schema2_injectors_target_worker_roles_and_preserve_aliases():
    recipe = '''schema: 2
engine: sglang
roles:
  prefill:
    nodes: 1
    env: &common
      KEEP: yes
    args:
      speculative-num-steps: 3
  decode:
    nodes: 1
    env: *common
  agg:
    nodes: 1
frontend:
  env:
    KEEP_FRONTEND: yes
benchmark:
  env:
    KEEP_CLIENT: yes
'''
    for framework, variable in [("dynamo-sglang", "SGLANG_SIMULATE_ACC_LEN"),
                                ("dynamo-trt", "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS")]:
        injector = get_injector(framework)
        rewritten, count = injector.rewrite(recipe, 2.5, _noop)
        data = yaml.safe_load(rewritten)
        assert count == 3
        assert all(variable in role["env"] for role in data["roles"].values())
        assert data["roles"]["decode"]["env"]["KEEP"] is True
        assert data["frontend"] == yaml.safe_load(recipe)["frontend"]
        assert data["benchmark"] == yaml.safe_load(recipe)["benchmark"]
        real, _ = injector.rewrite_real(rewritten, _noop)
        assert variable not in real
