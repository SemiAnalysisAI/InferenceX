#!/usr/bin/env python3
"""Assign deterministic CI priority scores to generated benchmark jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import yaml

SCORE_QUANTUM = Decimal("0.001")


@dataclass(frozen=True)
class PriorityContext:
    event_name: str = ""
    labels: frozenset[str] = frozenset()
    queue_namespace: str = "local"
    pr_number: int | None = None
    criteria: frozenset[str] | None = None


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def load_policy(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as policy_file:
        policy = yaml.safe_load(policy_file)
    if policy.get("version") != 1:
        raise ValueError(f"Unsupported CI priority policy version: {policy.get('version')}")
    return policy


def _first_prefix_adjustment(value: str, adjustments: dict[str, Any]) -> Decimal:
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
        "prefill": (
            {} if "multi-node" in criteria and entry.get("prefill") is not None else None
        ),
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
                and (
                    framework == criterion
                    or framework.startswith(f"{criterion}-")
                )
                for criterion in framework_criteria
            )
            else ""
        ),
        "model-prefix": (
            model_prefix
            if any(
                criterion in criteria
                and (
                    model_prefix == criterion
                    or model_prefix.startswith(f"{criterion}-")
                )
                for criterion in model_criteria
            )
            else ""
        ),
    }


def calculate_priority(
    entry: dict[str, Any],
    policy: dict[str, Any],
    context: PriorityContext = PriorityContext(),
) -> Decimal:
    """Return a higher-is-sooner priority score for one benchmark matrix entry."""
    patchwork = policy["labels"]["patchwork"]
    patch_labels = set(patchwork["names"])
    waiver_labels = set(patchwork.get("waived-by", []))
    criteria = context.criteria
    if (
        (
            (criteria is not None and "patchwork" in criteria)
            or context.labels & patch_labels
        )
        and not context.labels & waiver_labels
    ):
        return _decimal(patchwork["score"]).quantize(SCORE_QUANTUM, ROUND_HALF_UP)

    if criteria is not None:
        entry = _entry_from_criteria(criteria, policy, entry)
    adjustments = policy["adjustments"]
    score = _decimal(policy["base-score"])
    score += _decimal(adjustments.get("event", {}).get(context.event_name, 0))

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

    checklist = policy["labels"].get("checklist-complete", {})
    if (
        (criteria is not None and "checklist-complete" in criteria)
        or context.labels & set(checklist.get("names", []))
    ):
        score += _decimal(checklist.get("adjustment", 0))

    return score.quantize(SCORE_QUANTUM, ROUND_HALF_UP)


def format_priority(score: Decimal) -> str:
    return f"{score:.3f}"


def queue_token(value: dict[str, Any], namespace: str, path: tuple[str, ...]) -> str:
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
        if (
            context.pr_number is not None
            and policy["labels"]["skip-queue"]["name"] in context.labels
        ):
            annotated["skip-queue-pr"] = context.pr_number
    return annotated


def _labels_from_json(raw_labels: str) -> frozenset[str]:
    if not raw_labels:
        return frozenset()
    value = json.loads(raw_labels)
    if value is None:
        return frozenset()
    if not isinstance(value, list) or not all(isinstance(label, str) for label in value):
        raise ValueError("--labels-json must be a JSON array of strings")
    return frozenset(value)


def _criteria_from_json(raw_criteria: str) -> frozenset[str] | None:
    if not raw_criteria:
        return None
    value = json.loads(raw_criteria)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("--criteria-json must be a JSON array of strings")
    return frozenset(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="configs/ci-priority.yaml")
    parser.add_argument("--event-name", default="")
    parser.add_argument("--labels-json", default="[]")
    parser.add_argument("--queue-namespace", default="local")
    parser.add_argument("--pr-number", type=int)
    parser.add_argument("--criteria-json", default="")
    parser.add_argument(
        "--input",
        type=Path,
        help="Read matrix JSON from this file instead of stdin",
    )
    args = parser.parse_args()

    policy = load_policy(args.policy)
    context = PriorityContext(
        event_name=args.event_name,
        labels=_labels_from_json(args.labels_json),
        queue_namespace=args.queue_namespace,
        pr_number=args.pr_number,
        criteria=_criteria_from_json(args.criteria_json),
    )
    source = args.input.read_text() if args.input else sys.stdin.read()
    payload = json.loads(source)
    json.dump(annotate_jobs(payload, policy, context), sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
