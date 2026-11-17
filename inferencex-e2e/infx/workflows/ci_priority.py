#!/usr/bin/env python3
"""Assign deterministic CI priority scores to generated benchmark jobs.

With --facts-file it also records each job's scheduling facts, keyed by queue
token, for the priority scheduler to score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import yaml

from infx.config import repository_root

SCORE_QUANTUM = Decimal("0.001")
POLICY_KEYS = frozenset({"version", "base-score", "adjustments", "patchwork-score"})
FACTS_VERSION = 1
RUN_FACT_KEYS = ("version", "event", "patchwork", "checklist-complete")
ADJUSTMENT_KEYS = frozenset(
    {
        "additional-node",
        "event",
        "multi-node",
        "agentic",
        "eval-only",
        "checklist-complete",
        "precision",
        "spec-decoding",
        "framework-prefix",
        "model-prefix",
    }
)


@dataclass(frozen=True)
class PriorityContext:
    event_name: str = ""
    queue_namespace: str = "local"
    criteria: frozenset[str] | None = None


def _decimal(value: Any) -> Decimal:
    """Convert value to an exact Decimal via its string form."""
    return Decimal(str(value))


def load_policy(path: str | Path) -> dict[str, Any]:
    """Load the priority policy YAML at path.

    Raises ValueError on an unsupported version or unknown keys.
    """
    with Path(path).open() as policy_file:
        policy = yaml.safe_load(policy_file)
    if policy.get("version") != 1:
        raise ValueError(f"Unsupported CI priority policy version: {policy.get('version')}")
    unknown = sorted(set(policy) - POLICY_KEYS) + sorted(
        f"adjustments.{key}" for key in set(policy.get("adjustments") or {}) - ADJUSTMENT_KEYS
    )
    if unknown:
        raise ValueError(f"Unknown CI priority policy keys: {', '.join(unknown)}")
    return policy


def _first_prefix_adjustment(value: str, adjustments: dict[str, Any]) -> Decimal:
    """Return the adjustment for the first prefix matching value, or 0 if none matches.

    A prefix matches when value equals it or starts with it followed by a hyphen.
    """
    for prefix, adjustment in adjustments.items():
        if value == prefix or value.startswith(f"{prefix}-"):
            return _decimal(adjustment)
    return Decimal(0)


def supported_criteria(policy: dict[str, Any]) -> frozenset[str]:
    """Return every classifier fact understood by the configured policy."""
    adjustments = policy["adjustments"]
    return frozenset(
        {
            "multi-node",
            "agentic",
            "eval-only",
            "checklist-complete",
            "patchwork",
        }
        | set(adjustments.get("precision", {}))
        | set(adjustments.get("spec-decoding", {}))
        | set(adjustments.get("framework-prefix", {}))
        | set(adjustments.get("model-prefix", {}))
    )


def _entry_from_criteria(
    criteria: frozenset[str],
    policy: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, Any]:
    """Project entry onto only the fields named in criteria, for priority grouping.

    Raises ValueError if criteria contains names the policy does not support.
    """
    unknown = criteria - supported_criteria(policy)
    if unknown:
        raise ValueError(f"Unknown CI priority criteria: {sorted(unknown)}")
    precision = str(entry.get("precision", ""))
    spec_decoding = str(entry.get("spec-decoding", ""))
    framework = str(entry.get("framework", ""))
    model_prefix = str(entry.get("model-prefix", ""))
    framework_criteria = tuple(policy["adjustments"].get("framework-prefix", {}))
    model_criteria = tuple(policy["adjustments"].get("model-prefix", {}))
    return {
        "prefill": ({} if "multi-node" in criteria and entry.get("prefill") is not None else None),
        "scenario-type": (
            "agentic-coding"
            if "agentic" in criteria and entry.get("scenario-type") == "agentic-coding"
            else ""
        ),
        "eval-only": "eval-only" in criteria and entry.get("eval-only") is True,
        "precision": precision if precision in criteria else "",
        "spec-decoding": spec_decoding if spec_decoding in criteria else "",
        "framework": (
            framework
            if any(
                criterion in criteria
                and (framework == criterion or framework.startswith(f"{criterion}-"))
                for criterion in framework_criteria
            )
            else ""
        ),
        "model-prefix": (
            model_prefix
            if any(
                criterion in criteria
                and (model_prefix == criterion or model_prefix.startswith(f"{criterion}-"))
                for criterion in model_criteria
            )
            else ""
        ),
        "node-count": entry.get("node-count", 1),
    }


def calculate_priority(
    entry: dict[str, Any],
    policy: dict[str, Any],
    context: PriorityContext = PriorityContext(),
) -> Decimal:
    """Return a higher-is-sooner priority score for one benchmark matrix entry."""
    criteria = context.criteria
    if criteria is not None and "patchwork" in criteria:
        return _decimal(policy["patchwork-score"]).quantize(SCORE_QUANTUM, ROUND_HALF_UP)

    if criteria is not None:
        entry = _entry_from_criteria(criteria, policy, entry)
    adjustments = policy["adjustments"]
    score = _decimal(policy["base-score"])
    score += _decimal(adjustments.get("event", {}).get(context.event_name, 0))

    node_count = entry.get("node-count", 1)
    if not isinstance(node_count, int) or isinstance(node_count, bool) or node_count < 1:
        raise ValueError(f"node-count must be a positive integer, got {node_count!r}")
    score += _decimal(adjustments.get("additional-node", 0)) * (node_count - 1)

    if entry.get("prefill") is not None:
        score += _decimal(adjustments.get("multi-node", 0))
    if entry.get("scenario-type") == "agentic-coding":
        score += _decimal(adjustments.get("agentic", 0))
    if entry.get("eval-only") is True:
        score += _decimal(adjustments.get("eval-only", 0))

    score += _decimal(adjustments.get("precision", {}).get(str(entry.get("precision", "")), 0))
    score += _decimal(
        adjustments.get("spec-decoding", {}).get(str(entry.get("spec-decoding", "")), 0)
    )
    score += _first_prefix_adjustment(
        str(entry.get("framework", "")), adjustments.get("framework-prefix", {})
    )
    score += _decimal(
        adjustments.get("model-prefix", {}).get(str(entry.get("model-prefix", "")), 0)
    )

    if criteria is not None and "checklist-complete" in criteria:
        score += _decimal(adjustments.get("checklist-complete", 0))

    return score.quantize(SCORE_QUANTUM, ROUND_HALF_UP)


def format_priority(score: Decimal) -> str:
    """Format score with three decimals."""
    return f"{score:.3f}"


def queue_token(value: dict[str, Any], namespace: str, path: tuple[str, ...]) -> str:
    """Return a 32-hex-character hash of value, namespace and path for queue grouping."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    material = f"{namespace}:{'/'.join(path)}:{canonical}".encode()
    return hashlib.sha256(material).hexdigest()[:32]


def annotate_jobs(
    value: Any,
    policy: dict[str, Any],
    context: PriorityContext = PriorityContext(),
    *,
    _path: tuple[str, ...] = (),
) -> Any:
    """Copy a generated matrix payload and add scheduling metadata to every job."""
    if isinstance(value, list):
        return [
            annotate_jobs(item, policy, context, _path=(*_path, str(index)))
            for index, item in enumerate(value)
        ]
    if not isinstance(value, dict):
        return value

    annotated = {
        key: annotate_jobs(item, policy, context, _path=(*_path, key))
        for key, item in value.items()
    }
    if "runner" in value and "framework" in value:
        annotated["priority"] = format_priority(calculate_priority(value, policy, context))
        annotated["queue-token"] = queue_token(value, context.queue_namespace, _path)
    return annotated


def job_facts(
    entry: dict[str, Any],
    policy: dict[str, Any],
    criteria: frozenset[str] | None,
) -> dict[str, Any]:
    """Return the facts calculate_priority credits for entry under criteria.

    Patchwork runs score patchwork-score regardless of their facts, so like
    calculate_priority this never rejects them.
    """
    patchwork = criteria is not None and "patchwork" in criteria
    if criteria is not None and not patchwork:
        entry = _entry_from_criteria(criteria, policy, entry)
    node_count = entry.get("node-count", 1)
    if not isinstance(node_count, int) or isinstance(node_count, bool) or node_count < 1:
        if not patchwork:
            raise ValueError(f"node-count must be a positive integer, got {node_count!r}")
        node_count = 1
    return {
        "framework": str(entry.get("framework", "")),
        "model-prefix": str(entry.get("model-prefix", "")),
        "precision": str(entry.get("precision", "")),
        "spec-decoding": str(entry.get("spec-decoding", "")),
        "agentic": entry.get("scenario-type") == "agentic-coding",
        "eval-only": entry.get("eval-only") is True,
        "multi-node": entry.get("prefill") is not None,
        "node-count": node_count,
    }


def scheduling_facts(
    annotated: Any,
    policy: dict[str, Any],
    context: PriorityContext = PriorityContext(),
) -> dict[str, Any]:
    """Return the facts document for every job annotate_jobs labelled in annotated."""
    jobs: dict[str, dict[str, Any]] = {}

    def collect(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                collect(item)
        elif isinstance(node, dict):
            if "queue-token" in node:
                jobs[node["queue-token"]] = job_facts(node, policy, context.criteria)
            for item in node.values():
                collect(item)

    collect(annotated)
    criteria = context.criteria or frozenset()
    return {
        "version": FACTS_VERSION,
        "event": context.event_name,
        "patchwork": "patchwork" in criteria,
        "checklist-complete": "checklist-complete" in criteria,
        "jobs": jobs,
    }


def merge_facts(path: Path, facts: dict[str, Any]) -> None:
    """Write facts to path, merging with the jobs of an earlier invocation in the same run.

    Raises ValueError when the existing file disagrees on a run-level fact or
    records different facts for the same queue token.
    """
    if path.exists():
        existing = json.loads(path.read_text())
        mismatched = [key for key in RUN_FACT_KEYS if existing.get(key) != facts[key]]
        if mismatched:
            raise ValueError(
                f"{path} was written with different run facts: {', '.join(mismatched)}"
            )
        jobs = existing["jobs"]
        conflicting = sorted(
            token for token, job in facts["jobs"].items() if jobs.get(token, job) != job
        )
        if conflicting:
            raise ValueError(f"{path} records different facts for queue tokens: {conflicting}")
        facts = {**facts, "jobs": {**jobs, **facts["jobs"]}}
    path.write_text(json.dumps(facts, separators=(",", ":")) + "\n")


def _criteria_from_json(raw_criteria: str) -> frozenset[str] | None:
    """Parse --criteria-json into a set of names, or None when empty or null.

    Raises ValueError unless it is a JSON array of strings.
    """
    if not raw_criteria:
        return None
    value = json.loads(raw_criteria)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("--criteria-json must be a JSON array of strings")
    return frozenset(value)


def main() -> int:
    """Annotate the matrix JSON from --input or stdin with priorities, print it and return 0."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=str(repository_root() / "configs/ci-priority.yaml"))
    parser.add_argument("--event-name", default="")
    parser.add_argument("--queue-namespace", default="local")
    parser.add_argument("--criteria-json", default="")
    parser.add_argument(
        "--input",
        type=Path,
        help="Read matrix JSON from this file instead of stdin",
    )
    parser.add_argument(
        "--facts-file",
        type=Path,
        help="Merge every job's scheduling facts into this JSON file",
    )
    parser.add_argument(
        "--list-criteria",
        action="store_true",
        help="Print the policy's supported criteria as a JSON array and exit",
    )
    args = parser.parse_args()

    policy = load_policy(args.policy)
    if args.list_criteria:
        print(json.dumps(sorted(supported_criteria(policy))))
        return 0
    context = PriorityContext(
        event_name=args.event_name,
        queue_namespace=args.queue_namespace,
        criteria=_criteria_from_json(args.criteria_json),
    )
    source = args.input.read_text() if args.input else sys.stdin.read()
    payload = json.loads(source)
    annotated = annotate_jobs(payload, policy, context)
    if args.facts_file:
        merge_facts(args.facts_file, scheduling_facts(annotated, policy, context))
    json.dump(annotated, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
