#!/usr/bin/env python3
"""Prepare perf-changelog.yaml for a reuse-assisted PR merge."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from validate_perf_changelog import (
    ChangelogValidationError,
    PR_LINK_PLACEHOLDERS,
    compare_entries,
    parse_changelog,
    read_git_file,
    validate_raw_change,
    without_pr_link,
)


@dataclass(frozen=True)
class EntrySpan:
    """Byte offsets for one top-level changelog entry."""

    start: int
    content_end: int
    end: int


def entry_spans(raw: bytes, entries: list[dict[str, Any]]) -> list[EntrySpan]:
    """Locate top-level entry blocks without normalizing their bytes."""
    starts = [
        match.start()
        for match in re.finditer(rb"(?m)^- config-keys:[^\r\n]*\n", raw)
    ]
    if len(starts) != len(entries):
        raise ChangelogValidationError(
            "could not map parsed changelog entries to raw byte spans"
        )

    spans: list[EntrySpan] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(raw)
        segment = raw[start:end]
        lines = segment.splitlines(keepends=True)
        while lines and not lines[-1].strip(b" \t\r\n"):
            lines.pop()
        content_end = start + sum(len(line) for line in lines)
        spans.append(EntrySpan(start=start, content_end=content_end, end=end))
    return spans


def replace_pr_link(block: bytes, new_link: str) -> bytes:
    """Replace exactly one top-level pr-link line in an entry block."""
    replacement = f"  pr-link: {new_link}".encode()
    updated, count = re.subn(
        rb"(?m)^  pr-link:[^\r\n]*",
        replacement,
        block,
        count=1,
    )
    if count != 1:
        raise ChangelogValidationError(
            "could not locate exactly one pr-link line in a changelog entry"
        )
    return updated


def entry_signature(entry: dict[str, Any]) -> str:
    """Return a stable identity for an entry excluding its PR link."""
    return json.dumps(
        without_pr_link(entry),
        sort_keys=True,
        separators=(",", ":"),
    )


def canonicalize_appended_links(
    base_raw: bytes,
    head_raw: bytes,
    pr_number: int,
    repo: str,
) -> bytes:
    """Canonicalize placeholders only in entries appended by this PR."""
    base_entries = parse_changelog(base_raw, "base perf-changelog.yaml")
    head_entries = parse_changelog(head_raw, "PR perf-changelog.yaml")
    additions, corrections = compare_entries(
        base_entries,
        head_entries,
        pr_number,
    )
    validate_raw_change(base_raw, head_raw, len(additions), corrections)

    if not additions:
        return head_raw

    expected_link = f"https://github.com/{repo}/pull/{pr_number}"
    spans = entry_spans(head_raw, head_entries)
    replacements: list[tuple[EntrySpan, bytes]] = []
    for index, entry in enumerate(
        head_entries[len(base_entries):],
        start=len(base_entries),
    ):
        link = str(entry.get("pr-link") or "")
        if link == expected_link:
            continue
        if link not in PR_LINK_PLACEHOLDERS:
            raise ChangelogValidationError(
                f"appended entry {index + 1} has unexpected pr-link {link!r}"
            )
        span = spans[index]
        block = head_raw[span.start:span.content_end]
        replacements.append((span, replace_pr_link(block, expected_link)))

    result = head_raw
    for span, block in reversed(replacements):
        result = result[:span.start] + block + result[span.content_end:]

    result_entries = parse_changelog(result, "prepared perf-changelog.yaml")
    result_additions, result_corrections = compare_entries(
        base_entries,
        result_entries,
        pr_number,
    )
    validate_raw_change(
        base_raw,
        result,
        len(result_additions),
        result_corrections,
    )
    return result


def resolve_conflict_bytes(
    base_raw: bytes,
    pr_raw: bytes,
    main_raw: bytes,
    pr_number: int,
    repo: str,
) -> bytes:
    """Resolve a changelog conflict while preserving main byte-for-byte."""
    base_entries = parse_changelog(base_raw, "merge-base perf-changelog.yaml")
    pr_entries = parse_changelog(pr_raw, "PR-side perf-changelog.yaml")
    main_entries = parse_changelog(main_raw, "main-side perf-changelog.yaml")
    additions, corrections = compare_entries(
        base_entries,
        pr_entries,
        pr_number,
    )
    validate_raw_change(base_raw, pr_raw, len(additions), corrections)
    expected_link = f"https://github.com/{repo}/pull/{pr_number}"

    if additions:
        main_signatures = {entry_signature(entry) for entry in main_entries}
        pr_spans = entry_spans(pr_raw, pr_entries)
        contribution_blocks: list[bytes] = []
        for index, entry in enumerate(
            pr_entries[len(base_entries):],
            start=len(base_entries),
        ):
            if entry_signature(entry) in main_signatures:
                continue
            span = pr_spans[index]
            block = pr_raw[span.start:span.content_end]
            contribution_blocks.append(replace_pr_link(block, expected_link))

        if not contribution_blocks:
            raise ChangelogValidationError(
                "the PR has no changelog contribution remaining after merging main"
            )

        separator = b"" if main_raw.endswith(b"\n\n") else b"\n"
        result = main_raw + separator + b"\n".join(contribution_blocks)
    elif corrections:
        main_spans = entry_spans(main_raw, main_entries)
        replacements: list[tuple[EntrySpan, bytes]] = []
        for index, (base_entry, pr_entry) in enumerate(
            zip(base_entries, pr_entries)
        ):
            if base_entry == pr_entry:
                continue
            if index >= len(main_entries):
                raise ChangelogValidationError(
                    f"main is missing corrected entry {index + 1}"
                )
            main_entry = main_entries[index]
            if without_pr_link(main_entry) != without_pr_link(base_entry):
                raise ChangelogValidationError(
                    f"main changed corrected entry {index + 1}; resolve manually"
                )

            base_link = str(base_entry.get("pr-link") or "")
            desired_link = str(pr_entry.get("pr-link") or "")
            main_link = str(main_entry.get("pr-link") or "")
            if main_link == desired_link:
                continue
            if main_link != base_link:
                raise ChangelogValidationError(
                    f"main has a conflicting pr-link for entry {index + 1}"
                )

            span = main_spans[index]
            block = main_raw[span.start:span.content_end]
            replacements.append((span, replace_pr_link(block, desired_link)))

        if not replacements:
            raise ChangelogValidationError(
                "the PR's pr-link corrections are already present on main"
            )

        result = main_raw
        for span, block in reversed(replacements):
            result = result[:span.start] + block + result[span.content_end:]
    else:
        raise ChangelogValidationError(
            "the PR has no appended entry or pr-link correction to preserve"
        )

    result_entries = parse_changelog(result, "resolved perf-changelog.yaml")
    result_additions, result_corrections = compare_entries(
        main_entries,
        result_entries,
        pr_number,
    )
    validate_raw_change(
        main_raw,
        result,
        len(result_additions),
        result_corrections,
    )
    return result


def read_stage(stage: int, path: str) -> bytes:
    """Read a conflicted file from the git index."""
    result = subprocess.run(
        ["git", "show", f":{stage}:{path}"],
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ChangelogValidationError(
            f"could not read stage {stage} for {path}: {detail}"
        )
    return result.stdout


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    canonicalize = subparsers.add_parser("canonicalize")
    canonicalize.add_argument("--changelog-file", default="perf-changelog.yaml")
    canonicalize.add_argument("--base-ref", required=True)
    canonicalize.add_argument("--pr-number", required=True, type=int)
    canonicalize.add_argument(
        "--repo",
        default="SemiAnalysisAI/InferenceX",
    )

    resolve = subparsers.add_parser("resolve-conflict")
    resolve.add_argument("--changelog-file", default="perf-changelog.yaml")
    resolve.add_argument("--pr-number", required=True, type=int)
    resolve.add_argument(
        "--repo",
        default="SemiAnalysisAI/InferenceX",
    )

    args = parser.parse_args()
    path = Path(args.changelog_file)

    try:
        if args.command == "canonicalize":
            original = path.read_bytes()
            prepared = canonicalize_appended_links(
                read_git_file(args.base_ref, args.changelog_file),
                original,
                args.pr_number,
                args.repo,
            )
        else:
            original = path.read_bytes()
            prepared = resolve_conflict_bytes(
                read_stage(1, args.changelog_file),
                read_stage(2, args.changelog_file),
                read_stage(3, args.changelog_file),
                args.pr_number,
                args.repo,
            )
    except (ChangelogValidationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if prepared != original:
        path.write_bytes(prepared)
        print(f"Prepared {args.changelog_file} for PR #{args.pr_number}")
    else:
        print(f"{args.changelog_file} already prepared for PR #{args.pr_number}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
