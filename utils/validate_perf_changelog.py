#!/usr/bin/env python3
"""Validate perf-changelog.yaml before sweep reuse can skip setup."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from matrix_logic.validation import ChangelogEntry


CANONICAL_PR_LINK = re.compile(
    r"https://github\.com/SemiAnalysisAI/InferenceX/pull/\d+"
)
PR_LINK_PLACEHOLDERS = {
    "XXX",
    "https://github.com/SemiAnalysisAI/InferenceX/pull/XXX",
}


class ChangelogValidationError(ValueError):
    """Raised when the changelog or its git diff violates repository rules."""


def run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run git and return captured text output."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ChangelogValidationError(
            f"git {' '.join(args)} failed: {detail}"
        )
    return result


def read_git_file(ref: str, path: str) -> bytes:
    """Read a repository file exactly as stored at a git ref."""
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ChangelogValidationError(
            f"could not read {path} at {ref}: {detail}"
        )
    return result.stdout


def parse_changelog(raw: bytes, label: str) -> list[dict[str, Any]]:
    """Validate file-level invariants and return raw YAML entry mappings."""
    if not raw.endswith(b"\n"):
        raise ChangelogValidationError(f"{label} does not end with a newline")
    if b"\r" in raw:
        raise ChangelogValidationError(f"{label} contains CR characters")
    if b"\t" in raw:
        raise ChangelogValidationError(f"{label} contains tabs")
    if b"\0" in raw:
        raise ChangelogValidationError(f"{label} contains NUL bytes")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ChangelogValidationError(f"{label} is not UTF-8: {exc}") from exc

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ChangelogValidationError(
            f"{label} is not valid YAML: {exc}"
        ) from exc

    if not isinstance(data, list):
        raise ChangelogValidationError(f"{label} root must be a YAML list")

    top_level_entries = sum(
        line.startswith("- config-keys:") for line in text.splitlines()
    )
    if top_level_entries != len(data):
        raise ChangelogValidationError(
            f"{label} has {top_level_entries} top-level config entries "
            f"but YAML parsed {len(data)} entries"
        )

    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(data, start=1):
        if not isinstance(entry, dict):
            raise ChangelogValidationError(
                f"{label} entry {index} is not a mapping"
            )
        try:
            ChangelogEntry.model_validate(entry)
        except Exception as exc:
            raise ChangelogValidationError(
                f"{label} entry {index} fails ChangelogEntry validation: {exc}"
            ) from exc
        entries.append(entry)

    return entries


def without_pr_link(entry: dict[str, Any]) -> dict[str, Any]:
    """Return an entry copy without its pr-link field."""
    return {key: value for key, value in entry.items() if key != "pr-link"}


def validate_added_pr_link(link: str, pr_number: int | None) -> None:
    """Require a canonical link, with placeholders allowed only on PR runs."""
    if pr_number is None:
        if not CANONICAL_PR_LINK.fullmatch(link):
            raise ChangelogValidationError(
                f"new main-branch entry has invalid pr-link: {link!r}"
            )
        return

    expected = (
        f"https://github.com/SemiAnalysisAI/InferenceX/pull/{pr_number}"
    )
    if link not in PR_LINK_PLACEHOLDERS and link != expected:
        raise ChangelogValidationError(
            f"new PR entry must use {expected!r} or an XXX placeholder; "
            f"found {link!r}"
        )


def compare_entries(
    base_entries: list[dict[str, Any]],
    head_entries: list[dict[str, Any]],
    pr_number: int | None,
) -> tuple[list[dict[str, Any]], int]:
    """Validate append-only ordering and canonical pr-link-only corrections."""
    if len(head_entries) < len(base_entries):
        raise ChangelogValidationError(
            "perf-changelog.yaml entries were deleted"
        )

    corrections = 0
    for index, base_entry in enumerate(base_entries):
        head_entry = head_entries[index]
        if base_entry == head_entry:
            continue

        if without_pr_link(base_entry) != without_pr_link(head_entry):
            raise ChangelogValidationError(
                f"entry {index + 1} changed; existing entries are immutable "
                "except for pr-link-only corrections"
            )

        old_link = str(base_entry.get("pr-link") or "")
        new_link = str(head_entry.get("pr-link") or "")
        if old_link == new_link:
            raise ChangelogValidationError(
                f"entry {index + 1} was reformatted without a semantic change"
            )
        if not CANONICAL_PR_LINK.fullmatch(new_link):
            raise ChangelogValidationError(
                f"entry {index + 1} pr-link correction is not canonical: "
                f"{new_link!r}"
            )
        corrections += 1

    additions = head_entries[len(base_entries):]
    if corrections and additions:
        raise ChangelogValidationError(
            "do not mix historical pr-link corrections with new changelog entries"
        )

    for entry in additions:
        validate_added_pr_link(str(entry.get("pr-link") or ""), pr_number)

    return additions, corrections


def diff_changed_lines(base_ref: str, head_ref: str, path: str) -> list[str]:
    """Return added and deleted diff lines without file headers."""
    result = run_git(
        "diff",
        "--unified=0",
        "--no-ext-diff",
        base_ref,
        head_ref,
        "--",
        path,
    )
    changed: list[str] = []
    for line in result.stdout.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith(("+", "-")):
            changed.append(line[1:])
    return changed


def validate_metadata_diff(
    base_ref: str,
    head_ref: str,
    path: str,
) -> None:
    """Require metadata-only changes to touch pr-link lines exclusively."""
    changed = diff_changed_lines(base_ref, head_ref, path)
    unexpected = [
        line for line in changed if not line.startswith("  pr-link:")
    ]
    if unexpected:
        preview = "\n".join(repr(line) for line in unexpected[:5])
        raise ChangelogValidationError(
            "pr-link correction diff contains other changed lines:\n"
            f"{preview}"
        )


def validate_generated_config(base_ref: str, head_ref: str, path: str) -> None:
    """Run the same changelog processor used by sweep setup."""
    processor = Path(__file__).with_name("process_changelog.py")
    result = subprocess.run(
        [
            sys.executable,
            str(processor),
            "--changelog-file",
            path,
            "--base-ref",
            base_ref,
            "--head-ref",
            head_ref,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ChangelogValidationError(
            f"process_changelog.py rejected the diff:\n{detail}"
        )
    try:
        json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ChangelogValidationError(
            f"process_changelog.py returned invalid JSON: {exc}"
        ) from exc


def validate_changelog(
    base_ref: str,
    head_ref: str,
    path: str,
    pr_number: int | None,
) -> tuple[int, int]:
    """Validate the complete head file and its change from base to head."""
    diff_check = run_git(
        "diff",
        "--check",
        base_ref,
        head_ref,
        "--",
        path,
        check=False,
    )
    if diff_check.returncode != 0:
        detail = diff_check.stdout.strip() or diff_check.stderr.strip()
        raise ChangelogValidationError(
            f"git diff --check failed for {path}:\n{detail}"
        )

    base_entries = parse_changelog(
        read_git_file(base_ref, path),
        f"{path} at {base_ref}",
    )
    head_entries = parse_changelog(
        read_git_file(head_ref, path),
        f"{path} at {head_ref}",
    )
    additions, corrections = compare_entries(
        base_entries,
        head_entries,
        pr_number,
    )

    if additions:
        validate_generated_config(base_ref, head_ref, path)
    elif corrections:
        validate_metadata_diff(base_ref, head_ref, path)
    else:
        changed = diff_changed_lines(base_ref, head_ref, path)
        if changed:
            raise ChangelogValidationError(
                "changelog diff has no appended entry or pr-link correction"
            )
        raise ChangelogValidationError(f"{path} has no changes")

    return len(additions), corrections


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--head-ref", required=True)
    parser.add_argument("--changelog-file", default="perf-changelog.yaml")
    parser.add_argument("--pr-number", type=int)
    args = parser.parse_args()

    try:
        additions, corrections = validate_changelog(
            args.base_ref,
            args.head_ref,
            args.changelog_file,
            args.pr_number,
        )
    except ChangelogValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Validated {args.changelog_file}: "
        f"{additions} appended entr{'y' if additions == 1 else 'ies'}, "
        f"{corrections} pr-link correction"
        f"{'' if corrections == 1 else 's'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
