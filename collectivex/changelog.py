#!/usr/bin/env python3
"""collectivex/perf-changelog.yaml: one entry per PR that changes what CollectiveX measures.

Each entry names the PR and selects the cases it affects (pools, plus optional suites, backends,
modes, phases, EP sizes and precisions, matched against results/cases.json). An upload of results
names the PRs it publishes, and the database links the stored run to these entries, so a run can
be traced to the changes it measures and each change to the runs that measured it.

  changelog.py validate [--base-ref REF] [--pr-number N]   check the file (CI)
  changelog.py cases --pr N                                 list the cases N's entries select
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CHANGELOG = ROOT / "perf-changelog.yaml"
CASES = ROOT / "results" / "cases.json"
REGISTRY = ROOT / "configs" / "platform_config.json"
REPO_PATH = "collectivex/perf-changelog.yaml"

PR_LINK = re.compile(r"https://github\.com/SemiAnalysisAI/InferenceX/pull/([1-9][0-9]*)")
REQUIRED = ("date", "pr-link", "pools", "description", "description-zh")
# Optional selectors, each narrowing the pools to cases whose field is in the list.
SELECTORS = {"suites": "suite", "backends": "backend", "modes": "mode", "phases": "phase",
             "ep": "ep", "precisions": "precision"}


class ChangelogError(ValueError):
    pass


def parse(raw: bytes, label: str = REPO_PATH) -> list[dict]:
    """The file's entries, with every schema rule checked."""
    if raw and not raw.endswith(b"\n"):
        raise ChangelogError(f"{label} must end with a newline")
    if b"\t" in raw or b"\r" in raw:
        raise ChangelogError(f"{label} must not contain tabs or carriage returns")
    entries = yaml.safe_load(raw) or []
    if not isinstance(entries, list):
        raise ChangelogError(f"{label} must be a YAML list of entries")
    previous = None
    for index, entry in enumerate(entries):
        where = f"{label} entry {index + 1}"
        if not isinstance(entry, dict):
            raise ChangelogError(f"{where} is not a mapping")
        unknown = set(entry) - set(REQUIRED) - set(SELECTORS)
        missing = [key for key in REQUIRED if key not in entry]
        if unknown or missing:
            raise ChangelogError(f"{where}: unknown keys {sorted(unknown)}, missing {missing}")
        date = entry["date"]
        if isinstance(date, str):
            try:
                date = datetime.date.fromisoformat(date)
            except ValueError:
                date = None
        if not isinstance(date, datetime.date) or isinstance(date, datetime.datetime):
            raise ChangelogError(f"{where}: date must be YYYY-MM-DD")
        if previous and date < previous:
            raise ChangelogError(f"{where}: dates must not decrease (entries are appended in order)")
        previous = date
        if not isinstance(entry["pr-link"], str) or not PR_LINK.fullmatch(entry["pr-link"]):
            raise ChangelogError(f"{where}: pr-link must be https://github.com/SemiAnalysisAI/InferenceX/pull/<n>")
        for key in ("pools", *SELECTORS):
            if key in entry:
                kind = int if key == "ep" else str
                values = entry[key]
                if not isinstance(values, list) or not values or not all(
                        isinstance(value, kind) and not isinstance(value, bool) for value in values):
                    raise ChangelogError(f"{where}: {key} must be a non-empty list of {kind.__name__}")
        for key in ("description", "description-zh"):
            lines = entry[key]
            if not isinstance(lines, list) or not lines or not all(
                    isinstance(line, str) and line.strip() for line in lines):
                raise ChangelogError(f"{where}: {key} must be a non-empty list of lines")
    return entries


def pr_number(entry: dict) -> int:
    return int(PR_LINK.fullmatch(entry["pr-link"]).group(1))


def selects(entry: dict, requested: dict) -> bool:
    """Whether the entry covers one cases.json requested case."""
    if requested["sku"] not in entry["pools"]:
        return False
    case = requested["case"]
    return all(case.get(field) in entry[key] for key, field in SELECTORS.items() if key in entry)


def selected_cases(entry: dict, cases: list[dict]) -> list[str]:
    return [requested["case"]["case_id"] for requested in cases if selects(entry, requested)]


def check_coverage(entries: list[dict], cases: list[dict], pools: set[str], label: str = REPO_PATH) -> None:
    """Every entry selects at least one case, unless all of its pools have left the registry."""
    for index, entry in enumerate(entries):
        if not set(entry["pools"]) & pools:
            continue
        if not selected_cases(entry, cases):
            raise ChangelogError(f"{label} entry {index + 1} ({entry['pr-link']}) selects no case in cases.json")


def check_append_only(base: bytes, head: bytes, pr: int | None) -> list[dict]:
    """The head file is the base file plus entries appended at the end; returns the appended ones."""
    if not head.startswith(base):
        raise ChangelogError(f"{REPO_PATH}: existing entries changed; only append new entries at the end")
    base_entries, head_entries = parse(base, f"{REPO_PATH} (base)"), parse(head)
    appended = head_entries[len(base_entries):]
    # The PR that creates the file seeds it with entries for earlier PRs.
    if pr is not None and base:
        for entry in appended:
            if pr_number(entry) != pr:
                raise ChangelogError(f"appended entry links {entry['pr-link']}, not this PR (#{pr})")
    return appended


def git_show(ref: str, path: str) -> bytes:
    result = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=False)
    if result.returncode != 0:
        return b""  # the file does not exist at that ref yet
    return result.stdout


def load_cases() -> list[dict]:
    return json.loads(CASES.read_text(encoding="utf-8"))["requested_cases"]


def registry_pools() -> set[str]:
    return set(json.loads(REGISTRY.read_text(encoding="utf-8"))["platforms"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="check the changelog")
    validate.add_argument("--base-ref", help="git ref the file may only be appended to")
    validate.add_argument("--pr-number", type=int, help="the PR every appended entry must link")
    show = sub.add_parser("cases", help="list the case ids one PR's entries select")
    show.add_argument("--pr", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        head = CHANGELOG.read_bytes()
        entries = parse(head)
        if args.command == "cases":
            chosen = [entry for entry in entries if pr_number(entry) == args.pr]
            if not chosen:
                raise ChangelogError(f"no entry links #{args.pr}")
            cases = load_cases()
            print("\n".join(sorted({case for entry in chosen for case in selected_cases(entry, cases)})))
            return 0
        check_coverage(entries, load_cases(), registry_pools())
        if args.base_ref:
            appended = check_append_only(git_show(args.base_ref, REPO_PATH), head, args.pr_number)
            print(f"{REPO_PATH}: {len(entries)} entries, {len(appended)} appended")
        else:
            print(f"{REPO_PATH}: {len(entries)} entries")
    except ChangelogError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
