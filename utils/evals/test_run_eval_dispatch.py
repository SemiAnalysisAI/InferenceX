"""run_eval framework dispatch: EVAL_FRAMEWORK (env) overrides the --framework arg.

This is what lets `/run-evals swebench_lite ...` run swebench even though every
recipe script hardcodes `run_eval --framework lm-eval`. With the env unset, the
CLI arg (else lm-eval) is used as before.
"""

import os
import subprocess
from pathlib import Path

BENCHMARK_LIB = Path(__file__).resolve().parents[2] / "benchmarks" / "benchmark_lib.sh"

# Stub the framework runners so dispatch is observable without a server/Docker,
# and pin EVAL_MAX_MODEL_LEN so run_eval skips context computation.
_SCRIPT = r'''
source "$BENCHMARK_LIB"
run_lm_eval()       { echo "DISPATCH=lm-eval"; }
run_swebench_eval() { echo "DISPATCH=swebench"; }
export EVAL_MAX_MODEL_LEN=16384
unset EVAL_CONCURRENT_REQUESTS
run_eval --framework "$CLI_FW" --port 8888
'''


def _dispatch(cli_fw: str, env_fw: str | None) -> str:
    env = {**os.environ, "BENCHMARK_LIB": str(BENCHMARK_LIB), "CLI_FW": cli_fw}
    env.pop("EVAL_FRAMEWORK", None)
    if env_fw is not None:
        env["EVAL_FRAMEWORK"] = env_fw
    res = subprocess.run(
        ["bash", "-c", _SCRIPT], env=env, text=True, capture_output=True, check=True
    )
    return res.stdout


def test_env_framework_overrides_cli_arg():
    # recipe passes --framework lm-eval, but EVAL_FRAMEWORK=swebench wins.
    assert "DISPATCH=swebench" in _dispatch("lm-eval", "swebench")


def test_cli_arg_used_when_env_unset():
    assert "DISPATCH=lm-eval" in _dispatch("lm-eval", None)


def test_swebench_via_cli_arg_when_env_unset():
    assert "DISPATCH=swebench" in _dispatch("swebench", None)


def test_empty_env_falls_back_to_cli_arg():
    # An empty EVAL_FRAMEWORK (how the template passes it when unset) must not
    # force anything -- the CLI arg still wins.
    assert "DISPATCH=lm-eval" in _dispatch("lm-eval", "")
