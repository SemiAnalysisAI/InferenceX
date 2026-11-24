#!/usr/bin/env python3
"""Add queue tokens to generated benchmark jobs and record their scheduling facts.

The priority scheduler scores jobs from these facts with its own weights, so
this module only reports what each job is and never ranks it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from infx.config import MASTER_CONFIGS, repository_root

FACTS_VERSION = 1
RUN_FACT_KEYS = ("version", "event", "patchwork", "checklist-complete")
FLAG_CRITERIA = frozenset({"multi-node", "agentic", "eval-only", "checklist-complete", "patchwork"})
CATALOG_FIELDS = ("precision", "spec-decoding", "framework", "model-prefix")


@dataclass(frozen=True)
class SchedulingContext:
    event_name: str = ""
    queue_namespace: str = "local"
    criteria: frozenset[str] | None = None


def criteria_catalog(root: Path) -> dict[str, frozenset[str]]:
    """Return every value of each catalog field found anywhere in the master configs under root."""
    values: dict[str, set[str]] = {field: set() for field in CATALOG_FIELDS}

    def collect(node: Any) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                if key in values and isinstance(item, str):
                    values[key].add(item)
                else:
                    collect(item)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    for config in MASTER_CONFIGS:
        collect(yaml.safe_load((root / config).read_text()))
    return {field: frozenset(found) for field, found in values.items()}


def supported_criteria(root: Path) -> frozenset[str]:
    """Return every classifier criterion, the fixed flags plus each master-config catalog value."""
    return FLAG_CRITERIA.union(*criteria_catalog(root).values())


def _prefix_matches(value: str, criteria: frozenset[str]) -> bool:
    """Return whether value equals a criterion or starts with one followed by a hyphen."""
    return any(value == criterion or value.startswith(f"{criterion}-") for criterion in criteria)


def project(entry: dict[str, Any], criteria: frozenset[str]) -> dict[str, Any]:
    """Project entry onto only the facts its criteria credit, keeping the real node-count.

    Every criterion beyond the fixed flags is a candidate value. Precision and
    spec-decoding must equal one, framework and model-prefix may also extend one
    with a hyphen. Criteria matching nothing credit nothing.
    """
    values = criteria - FLAG_CRITERIA
    precision = str(entry.get("precision", ""))
    spec_decoding = str(entry.get("spec-decoding", ""))
    framework = str(entry.get("framework", ""))
    model_prefix = str(entry.get("model-prefix", ""))
    return {
        "prefill": ({} if "multi-node" in criteria and entry.get("prefill") is not None else None),
        "scenario-type": (
            "agentic-coding"
            if "agentic" in criteria and entry.get("scenario-type") == "agentic-coding"
            else ""
        ),
        "eval-only": "eval-only" in criteria and entry.get("eval-only") is True,
        "precision": precision if precision in values else "",
        "spec-decoding": spec_decoding if spec_decoding in values else "",
        "framework": framework if _prefix_matches(framework, values) else "",
        "model-prefix": model_prefix if _prefix_matches(model_prefix, values) else "",
        "node-count": entry.get("node-count", 1),
    }


def job_facts(entry: dict[str, Any]) -> dict[str, Any]:
    """Return the scheduling facts of one (possibly projected) matrix entry.

    Raises ValueError unless node-count is a positive integer.
    """
    node_count = entry.get("node-count", 1)
    if not isinstance(node_count, int) or isinstance(node_count, bool) or node_count < 1:
        raise ValueError(f"node-count must be a positive integer, got {node_count!r}")
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


def queue_token(value: dict[str, Any], namespace: str, path: tuple[str, ...]) -> str:
    """Return a 32-hex-character hash of value, namespace and path for queue grouping."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    material = f"{namespace}:{'/'.join(path)}:{canonical}".encode()
    return hashlib.sha256(material).hexdigest()[:32]


def annotate_jobs(
    value: Any,
    context: SchedulingContext = SchedulingContext(),
) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Copy a generated matrix payload with a queue token on every job.

    Returns the copy and each job's scheduling facts keyed by its queue token.
    Jobs are dicts holding both `runner` and `framework`.
    """
    criteria = context.criteria
    jobs: dict[str, dict[str, Any]] = {}

    def annotate(node: Any, path: tuple[str, ...]) -> Any:
        if isinstance(node, list):
            return [annotate(item, (*path, str(index))) for index, item in enumerate(node)]
        if not isinstance(node, dict):
            return node
        annotated = {key: annotate(item, (*path, key)) for key, item in node.items()}
        if "runner" in node and "framework" in node:
            token = queue_token(node, context.queue_namespace, path)
            jobs[token] = job_facts(node if criteria is None else project(node, criteria))
            annotated["queue-token"] = token
        return annotated

    return annotate(value, ()), jobs


def run_facts(context: SchedulingContext, jobs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return the facts document for one invocation."""
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
    """Print the matrix JSON from --input or stdin with queue tokens and record its facts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-name", default="")
    parser.add_argument("--queue-namespace", default="local")
    parser.add_argument("--criteria-json", default="")
    parser.add_argument(
        "--root",
        type=Path,
        help="Repository whose master configs --list-criteria reads (default: this checkout)",
    )
    parser.add_argument(
        "--facts-file",
        type=Path,
        help="Merge every job's scheduling facts into this JSON file",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Read matrix JSON from this file instead of stdin",
    )
    parser.add_argument(
        "--list-criteria",
        action="store_true",
        help="Print the supported classifier criteria as a JSON array and exit",
    )
    args = parser.parse_args()

    if args.list_criteria:
        print(json.dumps(sorted(supported_criteria(args.root or repository_root()))))
        return 0
    context = SchedulingContext(
        event_name=args.event_name,
        queue_namespace=args.queue_namespace,
        criteria=_criteria_from_json(args.criteria_json),
    )
    source = args.input.read_text() if args.input else sys.stdin.read()
    annotated, jobs = annotate_jobs(json.loads(source), context)
    if args.facts_file:
        merge_facts(args.facts_file, run_facts(context, jobs))
    json.dump(annotated, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
