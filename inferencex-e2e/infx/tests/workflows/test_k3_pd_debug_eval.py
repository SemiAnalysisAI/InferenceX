"""Behavior of the explicitly temporary PR evaluation scope."""

import copy
import json
import subprocess
import sys

import pytest

from infx.workflows.k3_pd_debug_eval import select_debug_eval


@pytest.fixture
def plan():
    def row(decode, conc, framework):
        return {
            "prefill": {"num-worker": 1},
            "decode": {"num-worker": decode},
            "conc": [conc],
            "eval-conc": conc,
            "eval-framework": framework,
            "eval-suite": "" if framework == "lm-eval" else "vendor-suite",
            "threshold": 0.9,
            "duration": 3600,
        }

    return {
        "changelog_metadata": {
            "entries": [
                {
                    "config-keys": ["kimik3-fp4-mi355x-vllm-disagg-agentic"],
                    "pr-link": "https://github.com/SemiAnalysisAI/InferenceX/pull/3582",
                }
            ]
        },
        "multi_node": {
            "agentic": [row(d, c, None) for d, c in ((1, 1), (1, 10), (2, 24), (1, 48), (2, 48))]
        },
        "multinode_agentic_evals": [
            row(1, 48, "lm-eval"),
            row(2, 48, "lm-eval"),
            row(2, 48, "kimi-vendor"),
        ],
        "evals": [],
        "agentic_evals": [],
        "multinode_evals": [],
    }


def test_cli_preserves_benchmarks_and_complete_selected_evaluator(plan):
    completed = subprocess.run(
        [sys.executable, "-m", "infx.workflows.k3_pd_debug_eval"],
        input=json.dumps(plan),
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert result == {**plan, "multinode_agentic_evals": [plan["multinode_agentic_evals"][1]]}


def test_unrelated_pr_is_unchanged():
    unrelated = {"unrelated": "input"}
    assert select_debug_eval(unrelated) is unrelated


@pytest.mark.parametrize("change", ["missing", "duplicate", "throughput", "scope", "family"])
def test_unexpected_scope_fails_without_modifying_input(plan, change):
    if change == "missing":
        plan["multinode_agentic_evals"].pop(1)
    elif change == "duplicate":
        plan["multinode_agentic_evals"].append(plan["multinode_agentic_evals"][1])
    elif change == "throughput":
        plan["multi_node"]["agentic"].pop()
    elif change == "scope":
        plan["changelog_metadata"]["entries"][0]["config-keys"].append("unrelated")
    else:
        plan["evals"].append({"unrelated": "eval"})
    original = copy.deepcopy(plan)
    with pytest.raises(ValueError):
        select_debug_eval(plan)
    assert plan == original
