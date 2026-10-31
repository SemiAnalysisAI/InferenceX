#!/usr/bin/env python3
"""Validate collectivex/perf-changelog.yaml: one entry per PR that changes what CollectiveX measures.

Each entry selects the cases its PR affects (pools, plus optional suites, backends, modes, phases, EP
sizes and precisions, matched against results/cases.json); uploads link runs to these entries.

  changelog.py validate [--base-ref REF --pr-number N]
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
PATH = "collectivex/perf-changelog.yaml"
PR_LINK = re.compile(r"https://github\.com/SemiAnalysisAI/InferenceX/pull/([1-9][0-9]*)")
SELECTORS = {"suites": "suite", "backends": "backend", "modes": "mode", "phases": "phase", "ep": "ep",
             "precisions": "precision"}
REQUIRED = {"date", "pr-link", "pools", "description"}


def parse(raw: bytes) -> list[dict]:
    entries = yaml.safe_load(raw) or []
    assert isinstance(entries, list), "the file must be a list of entries"
    previous = datetime.date.min
    for entry in entries:
        assert isinstance(entry, dict) and REQUIRED <= set(entry) <= REQUIRED | set(SELECTORS), f"bad keys: {entry}"
        date = datetime.date.fromisoformat(str(entry["date"]))
        assert date >= previous, f"{entry['pr-link']}: dates must not decrease"
        previous = date
        assert PR_LINK.fullmatch(str(entry["pr-link"])), f"bad pr-link: {entry['pr-link']}"
        for key in ("pools", "description", *SELECTORS):
            values = entry.get(key, [0])
            assert isinstance(values, list) and values, f"{entry['pr-link']}: {key} must be a non-empty list"
    return entries


def selects(entry: dict, requested: dict) -> bool:
    return requested["disposition"] == "runnable" and requested["sku"] in entry["pools"] and all(
        requested["case"].get(field) in entry[key] for key, field in SELECTORS.items() if key in entry)


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, check=check)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["validate"])
    parser.add_argument("--base-ref")
    parser.add_argument("--pr-number", type=int)
    args = parser.parse_args()
    head = (ROOT / "perf-changelog.yaml").read_bytes()
    entries = parse(head)
    if args.base_ref:
        git("cat-file", "-e", f"{args.base_ref}^{{commit}}")
        exists = git("cat-file", "-e", f"{args.base_ref}:{PATH}", check=False).returncode == 0
        base = git("show", f"{args.base_ref}:{PATH}").stdout if exists else b""
        old = parse(base)
        assert head.startswith(base) and entries[:len(old)] == old, "only append entries at the end"
        cases = json.loads((ROOT / "results" / "cases.json").read_text())["requested_cases"]
        pools = set(json.loads((ROOT / "configs" / "platform_config.json").read_text())["platforms"])
        for entry in entries[len(old):]:
            link = entry["pr-link"]
            # The PR that creates the file seeds entries for earlier PRs.
            assert not base or PR_LINK.fullmatch(link).group(1) == str(args.pr_number), f"{link} is not this PR"
            assert set(entry["pools"]) <= pools, f"{link}: unknown pools"
            for key in SELECTORS:
                for value in entry.get(key, []):
                    assert any(selects({**entry, key: [value]}, c) for c in cases), f"{link}: {key} {value!r}"
            assert any(selects(entry, c) for c in cases), f"{link} selects no runnable case"
    print(f"{PATH}: {len(entries)} entries")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, ValueError, subprocess.CalledProcessError) as exc:
        sys.exit(f"ERROR: {exc}")
