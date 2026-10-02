"""Temporary PR #3582 evaluation selection; remove before upstream qualification."""

import argparse
import json
import sys


def select_debug_eval(plan: dict) -> dict:
    """Preserve throughput and retain the generated 1P2D c48 GSM8K row only."""
    entries = plan.get("changelog_metadata", {}).get("entries", [])
    if not entries or any(
        entry.get("pr-link") != "https://github.com/SemiAnalysisAI/InferenceX/pull/3582"
        for entry in entries
    ):
        return plan
    keys = {key for entry in entries for key in entry["config-keys"]}
    if keys != {"kimik3-fp4-mi355x-vllm-disagg-agentic"}:
        raise ValueError("Temporary K3 selection requires the isolated K3 PD changelog")
    benchmarks = plan["multi_node"].get("agentic", [])
    points = [(row["decode"]["num-worker"], row["conc"]) for row in benchmarks]
    if sorted(points) != [(1, [1]), (1, [10]), (1, [48]), (2, [24]), (2, [48])]:
        raise ValueError("Temporary K3 selection requires the unchanged throughput matrix")
    selected = [
        row
        for row in plan["multinode_agentic_evals"]
        if row.get("eval-framework") == "lm-eval"
        and row.get("eval-suite", "") == ""
        and row["prefill"]["num-worker"] == 1
        and row["decode"]["num-worker"] == 2
        and row["conc"] == [48]
        and row["eval-conc"] == 48
    ]
    if len(selected) != 1:
        raise ValueError("Expected exactly one generated 1P2D c48 GSM8K evaluation")
    if any(plan.get(key) for key in ("evals", "agentic_evals", "multinode_evals")):
        raise ValueError("Unexpected evaluation family in the isolated K3 PD sweep")
    return {**plan, "multinode_agentic_evals": selected}


def main() -> None:
    """Filter the generated plan without modifying evaluator inputs or scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(select_debug_eval(json.load(sys.stdin))))


if __name__ == "__main__":
    main()
