"""run_eval framework dispatch.

Scenario picks the default framework (agentic-coding -> swebench, fixed-seq-len
-> lm-eval); an explicit EVAL_FRAMEWORK env or --framework arg overrides it.
"""

import os
import subprocess
from pathlib import Path

BENCHMARK_LIB = Path(__file__).resolve().parents[2] / "benchmarks" / "benchmark_lib.sh"

# Stub the framework runners so dispatch is observable without a server/Docker,
# and pin EVAL_MAX_MODEL_LEN so run_eval skips context computation. CLI_FW is
# only forwarded as --framework when set (so we can test the no-arg path).
_SCRIPT = r'''
source "$BENCHMARK_LIB"
run_lm_eval()       { echo "DISPATCH=lm-eval"; }
run_swebench_eval() { echo "DISPATCH=swebench"; }
export EVAL_MAX_MODEL_LEN=16384
unset EVAL_CONCURRENT_REQUESTS
run_eval ${CLI_FW:+--framework "$CLI_FW"} --port 8888
'''


def _dispatch(*, is_agentic: str = "0", cli_fw=None, env_fw=None) -> str:
    # AgentX v1.0 added a source-time guard in benchmark_lib.sh that requires
    # KV_OFFLOADING to be set whenever the scenario is agentic (IS_AGENTIC=1 /
    # SCENARIO_TYPE=agentic-coding). KV_OFFLOADING=none satisfies it without
    # affecting framework dispatch, which only reads scenario + framework knobs.
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "IS_AGENTIC": is_agentic,
        "KV_OFFLOADING": "none",
    }
    env.pop("EVAL_FRAMEWORK", None)
    env.pop("CLI_FW", None)
    env.pop("KV_OFFLOAD_BACKEND", None)
    if cli_fw is not None:
        env["CLI_FW"] = cli_fw
    if env_fw is not None:
        env["EVAL_FRAMEWORK"] = env_fw
    res = subprocess.run(
        ["bash", "-c", _SCRIPT], env=env, text=True, capture_output=True, check=True
    )
    return res.stdout


# --- scenario default ------------------------------------------------------

def test_agentic_scenario_defaults_to_swebench():
    assert "DISPATCH=swebench" in _dispatch(is_agentic="1")


def test_fixed_seqlen_scenario_defaults_to_lm_eval():
    assert "DISPATCH=lm-eval" in _dispatch(is_agentic="0")


# --- explicit overrides win over the scenario default ----------------------

def test_explicit_framework_arg_overrides_scenario():
    # agentic, but recipe passed --framework lm-eval -> lm-eval.
    assert "DISPATCH=lm-eval" in _dispatch(is_agentic="1", cli_fw="lm-eval")


def test_env_framework_overrides_scenario():
    assert "DISPATCH=lm-eval" in _dispatch(is_agentic="1", env_fw="lm-eval")


def test_env_can_force_swebench_on_fixed_seqlen():
    assert "DISPATCH=swebench" in _dispatch(is_agentic="0", env_fw="swebench")


def test_recipe_lm_eval_arg_still_lm_eval_on_fixed_seqlen():
    # The existing fixed-seq-len recipes call `run_eval --framework lm-eval`.
    assert "DISPATCH=lm-eval" in _dispatch(is_agentic="0", cli_fw="lm-eval")
