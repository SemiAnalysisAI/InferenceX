"""Compute auditable SPEED-Bench acceptance from real vLLM counter deltas."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def read_counters(text: str) -> dict[str, float]:
    """Sum engine-labelled counters, rejecting missing or malformed measurements."""
    counters = {}
    for key in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
        name = f"vllm:spec_decode_{key}_total"
        values = [
            float(match[1])
            for match in re.finditer(
                rf"^{name}(?:\{{[^\n]*\}})?\s+(\S+)(?:\s+\S+)?$", text, re.MULTILINE
            )
        ]
        if not values or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"Missing or invalid counter: {name}")
        counters[key] = sum(values)
    return counters


def acceptance(before: str, after: str, draft_length: int) -> dict[str, float]:
    """AL includes the bonus token; AR uses the actual proposed-token count."""
    start, end = read_counters(before), read_counters(after)
    delta = {key: end[key] - start[key] for key in start}
    drafts = delta["num_drafts"]
    proposed = delta["num_draft_tokens"]
    accepted = delta["num_accepted_tokens"]
    if (
        draft_length < 1
        or drafts <= 0
        or proposed <= 0
        or accepted < 0
        or accepted > proposed
        or proposed > draft_length * drafts
    ):
        raise ValueError(f"Invalid speculative counter deltas: {delta}")
    return {**delta, "al": 1 + accepted / drafts, "ar": accepted / proposed}


def collect_cell(
    before: str, after: str, result: dict, expected_prompts: int, draft_length: int
) -> dict[str, float]:
    """Only publish a cell when all selected requests completed successfully."""
    if expected_prompts <= 0 or result.get("completed") != expected_prompts:
        raise ValueError(f"Expected {expected_prompts} completions; got {result.get('completed')}")
    errors = result.get("errors", [])
    if any(errors):
        raise ValueError("Benchmark contains request errors")
    return acceptance(before, after, draft_length)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("expected_prompts", type=int)
    parser.add_argument("draft_length", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            collect_cell(
                args.before.read_text(),
                args.after.read_text(),
                json.loads(args.result.read_text()),
                args.expected_prompts,
                args.draft_length,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
