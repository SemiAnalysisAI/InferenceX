#!/usr/bin/env python3
"""Validate perf-changelog.yaml changes."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver

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


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def construct_unique_mapping(
    loader: UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    """Construct a mapping while rejecting keys that would overwrite values."""
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found unhashable key {key!r}",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    construct_unique_mapping,
)


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
        data = yaml.load(text, Loader=UniqueKeyLoader)
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


def validate_raw_change(
    base_raw: bytes,
    head_raw: bytes,
    additions: int,
    corrections: int,
) -> None:
    """Require historical bytes to remain exact outside explicit link fixes."""
    if additions:
        if not head_raw.startswith(base_raw):
            raise ChangelogValidationError(
                "appended entries changed historical perf-changelog.yaml bytes; "
                "restore the base file byte-for-byte and append at the end"
            )

        suffix = head_raw[len(base_raw):]
        expected_start = (
            b"- config-keys:"
            if base_raw.endswith(b"\n\n")
            else b"\n- config-keys:"
        )
        if not suffix.startswith(expected_start):
            raise ChangelogValidationError(
                "new changelog entries must be separated from history by one "
                "empty line and appended at the end"
            )
        entry_starts = [
            match.start()
            for match in re.finditer(
                rb"(?m)^- config-keys:[^\r\n]*\n",
                head_raw,
            )
        ]
        appended_starts = entry_starts[-additions:]
        for start in appended_starts[1:]:
            prefix = head_raw[:start]
            if (
                not prefix.endswith(b"\n\n")
                or prefix.endswith(b"\n\n\n")
            ):
                raise ChangelogValidationError(
                    "appended changelog entries must have exactly one empty "
                    "separator line"
                )
        if head_raw.endswith(b"\n\n"):
            raise ChangelogValidationError(
                "the final appended changelog entry must end with one newline"
            )
        return

    if corrections:
        base_lines = base_raw.splitlines(keepends=True)
        head_lines = head_raw.splitlines(keepends=True)
        if len(base_lines) != len(head_lines):
            raise ChangelogValidationError(
                "pr-link corrections may not add or delete lines"
            )

        changed = 0
        for line_number, (base_line, head_line) in enumerate(
            zip(base_lines, head_lines),
            start=1,
        ):
            if base_line == head_line:
                continue
            changed += 1
            if not (
                base_line.startswith(b"  pr-link:")
                and head_line.startswith(b"  pr-link:")
                and base_line.endswith(b"\n")
                and head_line.endswith(b"\n")
            ):
                raise ChangelogValidationError(
                    "historical bytes changed outside a pr-link line at "
                    f"line {line_number}"
                )

        if changed != corrections:
            raise ChangelogValidationError(
                f"expected {corrections} changed pr-link line(s), found {changed}"
            )
        return

    if base_raw != head_raw:
        raise ChangelogValidationError(
            "changelog diff has no appended entry or pr-link correction; "
            "historical whitespace and formatting are immutable"
        )


def validate_generated_config(
    base_ref: str,
    head_ref: str,
    path: str,
    *,
    all_evals: bool = False,
    evals_only: bool = False,
) -> None:
    """Run the same changelog processor used by sweep setup."""
    processor = Path(__file__).with_name("process_changelog.py")
    command = [
        sys.executable,
        str(processor),
        "--changelog-file",
        path,
        "--base-ref",
        base_ref,
        "--head-ref",
        head_ref,
    ]
    if all_evals:
        command.append("--all-evals")
    if evals_only:
        command.append("--evals-only")
    result = subprocess.run(
        command,
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


def validate_matrix_compatible_change(
    base_ref: str,
    head_ref: str,
    path: str,
    *,
    all_evals: bool = False,
    evals_only: bool = False,
) -> None:
    """Validate the final newline and the diff accepted by sweep setup."""
    head_raw = read_git_file(head_ref, path)
    if not head_raw.endswith(b"\n"):
        raise ChangelogValidationError(
            f"{path} at {head_ref} does not end with a newline"
        )

    validate_generated_config(
        base_ref,
        head_ref,
        path,
        all_evals=all_evals,
        evals_only=evals_only,
    )


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--head-ref", required=True)
    parser.add_argument("--changelog-file", default="perf-changelog.yaml")
    parser.add_argument("--all-evals", action="store_true")
    parser.add_argument("--evals-only", action="store_true")
    args = parser.parse_args()

    try:
        validate_matrix_compatible_change(
            args.base_ref,
            args.head_ref,
            args.changelog_file,
            all_evals=args.all_evals,
            evals_only=args.evals_only,
        )
    except ChangelogValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Validated {args.changelog_file}: "
        "final newline present and matrix generated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
