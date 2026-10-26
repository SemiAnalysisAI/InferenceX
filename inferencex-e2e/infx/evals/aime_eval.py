"""AIME 2025+2026 adapter: answer every problem against the local endpoint and grade it.

``aime_2025_2026.jsonl`` holds the 60 problems converted from MathArena/aime_2025
(c94da77eb22bbd6439e62a323bec18493a421302) and MathArena/aime_2026
(d61f486805306e6e6b5913504c07763a99e1b112). A sample is correct when the last
``\\boxed{}`` in the final answer equals the official integer.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from aa_lcr_eval import _ask, _write

RESULT_FORMAT = "inferencex-eval-v1"
TASK_NAME = "aime_2025_2026"
DATA = Path(__file__).with_name("aime_2025_2026.jsonl")
PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def extract(response: str) -> int | None:
    """The integer in the last ``\\boxed{}``, or None."""
    start = response.rfind("\\boxed{")
    if start < 0:
        return None
    depth, i = 1, start + len("\\boxed{")
    for j in range(i, len(response)):
        depth += {"{": 1, "}": -1}.get(response[j], 0)
        if depth == 0:
            digits = re.sub(r"[\s,]|\\[a-z]+|\^\\circ|[{}$]", "", response[i:j])
            return int(digits) if digits.isdigit() else None
    return None


def _result(model: str, score: float, se: float, effective: int, error: str | None = None) -> dict[str, Any]:
    """The lm-eval style compatibility result."""
    result: dict[str, Any] = {
        "result_format": RESULT_FORMAT,
        "eval_adapter": "aime",
        "model_name": model,
        "results": {
            TASK_NAME: {"exact_match,strict-match": score, "exact_match_stderr,strict-match": se}
        },
        "configs": {
            TASK_NAME: {
                "metric_list": [{"metric": "exact_match"}],
                "filter_list": [{"name": "strict-match"}],
            }
        },
        "n-samples": {TASK_NAME: {"original": effective, "effective": effective}},
        "source": {"repository": "MathArena/aime_2025+aime_2026"},
    }
    if error is not None:
        result["integration_error"] = {"type": "aime", "message": error}
    return result


def run(args: argparse.Namespace) -> int:
    """Generate ``--repeats`` answers per problem, grade them, and write samples plus the result."""
    problems = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()]
    if len(problems) != 60:
        raise RuntimeError(f"expected 60 AIME problems, found {len(problems)}")
    jobs = [(p, r) for r in range(args.repeats) for p in problems]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_problem: dict[str, list[bool]] = defaultdict(list)
    by_repeat: dict[int, list[bool]] = defaultdict(list)
    with (
        (args.output_dir / "samples_aime.jsonl").open("w", encoding="utf-8") as samples,
        concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool,
    ):
        futures = {
            pool.submit(
                _ask, args.base_url, args.model, PROMPT.format(problem=p["problem"]), args.max_tokens
            ): (p, r)
            for p, r in jobs
        }
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            problem, repeat = futures[future]
            outcome = future.result()
            predicted = extract(outcome.get("response", ""))
            correct = predicted == problem["answer"]
            by_problem[problem["id"]].append(correct)
            by_repeat[repeat].append(correct)
            samples.write(json.dumps({
                **problem, "repeat": repeat, "predicted": predicted, "correct": correct, **outcome,
            }) + "\n")  # fmt: skip
            samples.flush()
            print(f"aime {done}/{len(jobs)} correct={sum(map(sum, by_problem.values()))}", flush=True)
    # Problems are the sampling unit; average repeats per problem first.
    means = [sum(v) / len(v) for v in by_problem.values()]
    score = sum(means) / len(means)
    se = math.sqrt(sum((m - score) ** 2 for m in means) / (len(means) - 1) / len(means))
    print("aime per-repeat:", " / ".join(f"{sum(v) / len(v):.3f}" for _, v in sorted(by_repeat.items())))
    print(f"aime accuracy={score:.4f} se={se:.4f}")
    _write(args.output_dir, _result(args.model, score, se, len(jobs)), "aime")
    return 0


def failure(args: argparse.Namespace) -> int:
    """Record an integration failure."""
    _write(args.output_dir, _result(args.model, 0.0, 0.0, 0, args.message), "aime")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse ``run`` or ``failure`` and dispatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--base-url", required=True)
    run_parser.add_argument("--concurrency", type=int, required=True)
    run_parser.add_argument("--repeats", type=int, default=int(os.environ.get("AIME_REPEATS", "8")))
    run_parser.add_argument(
        "--max-tokens", type=int, default=int(os.environ.get("AIME_MAX_TOKENS", "65536"))
    )
    failure_parser = commands.add_parser("failure")
    failure_parser.add_argument("--message", required=True)
    for sub in (run_parser, failure_parser):
        sub.add_argument("--model", required=True)
        sub.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    return run(args) if args.command == "run" else failure(args)


if __name__ == "__main__":
    sys.exit(main())
