#!/usr/bin/env python3
"""Verify producer/consumer sync for ISB1 preview and extension exports."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


RELEVANT_SUBTREES = (
    "extension_131k",
    "preview/long_context_500k",
    "preview/long_context_1m",
)


@dataclass
class SyncIssue:
    kind: str
    path: str


def _json_files(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {
        str(path.relative_to(root))
        for path in root.rglob("*.json")
        if path.is_file()
    }


def _compare_subtree(producer_root: Path, consumer_root: Path, subtree: str) -> list[SyncIssue]:
    issues: list[SyncIssue] = []

    producer_subtree = producer_root / subtree
    consumer_subtree = consumer_root / subtree

    producer_files = _json_files(producer_subtree)
    consumer_files = _json_files(consumer_subtree)

    if not producer_subtree.exists():
        issues.append(SyncIssue("missing_producer_subtree", subtree))
        return issues
    if not consumer_subtree.exists():
        issues.append(SyncIssue("missing_consumer_subtree", subtree))
        return issues

    for relative_path in sorted(producer_files - consumer_files):
        issues.append(SyncIssue("missing_in_consumer", f"{subtree}/{relative_path}"))

    for relative_path in sorted(consumer_files - producer_files):
        issues.append(SyncIssue("extra_in_consumer", f"{subtree}/{relative_path}"))

    for relative_path in sorted(producer_files & consumer_files):
        producer_file = producer_subtree / relative_path
        consumer_file = consumer_subtree / relative_path
        if producer_file.read_bytes() != consumer_file.read_bytes():
            issues.append(SyncIssue("content_mismatch", f"{subtree}/{relative_path}"))

    return issues


def verify_sync(producer_root: Path, consumer_root: Path) -> list[SyncIssue]:
    issues: list[SyncIssue] = []
    for subtree in RELEVANT_SUBTREES:
        issues.extend(_compare_subtree(producer_root, consumer_root, subtree))
    return issues


def _default_consumer_root() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets" / "isb1" / "exports"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that committed ISB1 consumer preview/extension exports are "
            "synced with producer exports."
        )
    )
    parser.add_argument(
        "--producer-root",
        required=True,
        type=Path,
        help="Path to ISB1 producer exports root (…/upstream/inferencex/exports)",
    )
    parser.add_argument(
        "--consumer-root",
        default=_default_consumer_root(),
        type=Path,
        help="Path to InferenceX consumer exports root (default: datasets/isb1/exports)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    issues = verify_sync(args.producer_root.resolve(), args.consumer_root.resolve())

    if not issues:
        print(
            "Producer/consumer export sync check passed for: "
            + ", ".join(RELEVANT_SUBTREES)
        )
        return 0

    print("Producer/consumer export sync check failed:", file=sys.stderr)
    for issue in issues:
        print(f"- {issue.kind}: {issue.path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
