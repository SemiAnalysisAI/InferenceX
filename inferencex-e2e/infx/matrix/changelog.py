"""Per-PR perf changelog files, and the checks that keep them in step with the config diff."""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from infx import yamlio
from infx.config import git_path_at_ref, git_repository_root

from .validation import ChangelogEntry

LEGACY_FILE = "perf-changelog.yaml"
DIRECTORY = "perf-changelog"
REPOSITORY = "SemiAnalysisAI/InferenceX"
FILE_NAME = re.compile(r"([1-9][0-9]*)\.yaml")
SCRIPT_PATHS = (
    "inferencex-e2e/benchmarks/",
    "inferencex-e2e/infx/launch/",
    "inferencex-e2e/runners/",
)


class ChangelogError(ValueError):
    """Raised when a PR's changelog files break the repository rules."""


def pr_link(number: int) -> str:
    """Return the canonical link of pull request ``number``."""
    return f"https://github.com/{REPOSITORY}/pull/{number}"


def git(*args: str, cwd: Path | None = None) -> str:
    """Run git and return its stdout, raising ChangelogError with its stderr on failure."""
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ChangelogError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def merge_base(base_ref: str, head_ref: str) -> str:
    """Return the commit ``head_ref`` branched from ``base_ref`` at."""
    return git("merge-base", base_ref, head_ref).strip()


def directory_at(ref: str, changelog_file: str) -> str:
    """Return the repository path of the changelog directory beside ``changelog_file``."""
    return Path(git_path_at_ref(ref, changelog_file)).with_name(DIRECTORY).as_posix()


def parse_file(text: str, number: int, label: str) -> dict[str, Any]:
    """Return the changelog entry in one per-PR file, with its pr-link derived from the name."""
    try:
        data = yamlio.load(text, yamlio.UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ChangelogError(f"{label} is not valid YAML: {error}") from error
    if not isinstance(data, dict):
        raise ChangelogError(f"{label} must hold one mapping, the changelog entry")
    if "pr-link" in data:
        raise ChangelogError(f"{label} must not set pr-link; it comes from the file name")
    entry = {**data, "pr-link": pr_link(number)}
    try:
        ChangelogEntry.model_validate(entry)
    except ValueError as error:
        raise ChangelogError(f"{label} is not a valid changelog entry: {error}") from error
    return entry


def added_files(base_ref: str, head_ref: str, changelog_file: str) -> list[tuple[int, dict]]:
    """Return the PR number and entry of each changelog file ``head_ref`` adds since the base.

    The diff starts at the merge base, so files main gained after the branch point are not
    counted. Changing, renaming or deleting an existing file raises ChangelogError.
    """
    base = merge_base(base_ref, head_ref)
    directory = directory_at(head_ref, changelog_file)
    diff = git(
        "diff", "--name-status", "--no-renames", base, head_ref, "--", f":(top){directory}/*.yaml"
    )  # fmt: skip
    added = []
    for line in diff.splitlines():
        status, path = line.split("\t", 1)
        if status != "A":
            raise ChangelogError(f"changelog files are immutable; {path} has status {status}")
        name = FILE_NAME.fullmatch(Path(path).name)
        if name is None:
            raise ChangelogError(f"{path} must be named <PR number>.yaml")
        number = int(name.group(1))
        added.append((number, parse_file(git("show", f"{head_ref}:{path}"), number, path)))
    return added


def derived_changes(base: str) -> dict[str, Any]:
    """Run ``infx.matrix.changed`` against ``base`` and return its JSON report."""
    result = subprocess.run(
        [sys.executable, "-m", "infx.matrix.changed", "--base", base, "--json"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ChangelogError(f"infx.matrix.changed failed:\n{result.stderr.strip()}")
    return json.loads(result.stdout)


def split_changes(report: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """Return the changed live keys, the removed keys, and the scenarios of the live changes."""
    live, retired, scenarios = [], [], set()
    for change in report["changed"]:
        if change["head-points"] == 0:
            retired.append(change["config-key"])
            continue
        live.append(change["config-key"])
        for point in change["added"] + change["removed"]:
            scenarios.add(point.get("scenario-type", "fixed-seq-len"))
    return live, retired, sorted(scenarios)


def draft_entry(report: dict[str, Any], description: list[str], keys: list[str]) -> dict[str, Any]:
    """Return the entry for the changes in ``report`` plus rerun ``keys``, without a pr-link."""
    live, retired, scenarios = split_changes(report)
    live += [key for key in keys if key not in live]
    if not live and not retired:
        raise ChangelogError(
            "no config changes found; pass --key for each config to rerun without a change"
        )
    if not live:
        return {"config-keys": retired, "retired": True, "description": description}
    entry: dict[str, Any] = {"config-keys": live}
    if retired:
        entry["retired-keys"] = retired
    if scenarios:
        entry["scenario-type"] = scenarios
    entry["description"] = description
    return entry


def coverage_errors(entry: dict[str, Any], report: dict[str, Any]) -> list[str]:
    """Return how ``entry`` fails to cover the config changes in ``report``."""
    live, retired, scenarios = split_changes(report)
    errors = []
    if entry.get("retired"):
        listed_retired, patterns = entry["config-keys"], []
    else:
        listed_retired, patterns = entry.get("retired-keys") or [], entry["config-keys"]
    uncovered = [key for key in live if not any(fnmatch.fnmatchcase(key, p) for p in patterns)]
    if uncovered:
        errors.append(f"config-keys must cover the changed configs {uncovered}")
    unlisted = [key for key in retired if key not in listed_retired]
    if unlisted:
        field = "config-keys" if entry.get("retired") else "retired-keys"
        errors.append(f"{field} must list the removed configs {unlisted}")
    selected = entry.get("scenario-type")
    if selected is not None and live:
        missing = [scenario for scenario in scenarios if scenario not in selected]
        if missing:
            errors.append(f"scenario-type must include the changed scenarios {missing}")
    return errors


def legacy_allowed(created_at: str, base_ref: str, changelog_file: str) -> bool:
    """Whether a PR created at ``created_at`` predates the changelog directory on the base."""
    directory = directory_at(base_ref, changelog_file)
    introduced = git(
        "log", base_ref, "--diff-filter=A", "--format=%cI", "--", f":(top){directory}/README.md"
    ).split()
    if not introduced:
        return True
    return datetime.fromisoformat(created_at) < datetime.fromisoformat(introduced[-1])


def legacy_entries(base: str, head_ref: str, changelog_file: str) -> list[dict]:
    """Return the entries ``head_ref`` appends to the legacy changelog file since ``base``."""
    from .plan import get_added_lines

    added = get_added_lines(base, head_ref, changelog_file)
    return (yamlio.load(added, yamlio.UniqueKeySafeLoader) or []) if added.strip() else []


def check(
    base_ref: str,
    head_ref: str,
    pull_request: int,
    created_at: str,
    changelog_file: str = LEGACY_FILE,
) -> list[str]:
    """Return every rule the PR's changelog breaks; an empty list means it passes.

    Prints a warning when shared scripts or launchers changed with no changelog entry.
    """
    base = merge_base(base_ref, head_ref)
    files = added_files(base, head_ref, changelog_file)
    legacy = legacy_entries(base, head_ref, changelog_file)
    command = f"python -m infx.matrix.changelog new --pr {pull_request} -m '<what changed>'"
    errors = []
    if len(files) > 1:
        errors.append(f"add one changelog file per PR, found {[n for n, _ in files]}")
    errors += [
        f"name the changelog file {pull_request}.yaml, not {number}.yaml"
        for number, _ in files
        if number != pull_request
    ]
    if files and legacy:
        errors.append(
            f"add the entry as {DIRECTORY}/{pull_request}.yaml only, not to {LEGACY_FILE}"
        )
    if legacy and not legacy_allowed(created_at, base_ref, changelog_file):
        errors.append(f"{LEGACY_FILE} is frozen; run {command}")
    if len(legacy) > 1:
        errors.append(f"merge the {len(legacy)} {LEGACY_FILE} entries into one")

    report = derived_changes(base)
    if report["changed"] and not files and not legacy:
        live, retired, _ = split_changes(report)
        errors.append(f"configs changed ({live + retired}) with no changelog entry; run {command}")
    for _, entry in files:
        errors += coverage_errors(entry, report)
    if not files and not legacy:
        changed = git("diff", "--name-only", base, head_ref).split()
        scripts = [
            path
            for path in changed
            if path.startswith(SCRIPT_PATHS) and path.endswith((".sh", ".py"))
        ]
        if scripts:
            print(
                f"::warning::shared scripts or launchers changed ({scripts}) with no changelog "
                f"entry; if this changes what a benchmark runs, run {command} --key <config>"
            )
    return errors


def detect_pr() -> int:
    """Return the number of the open PR for the current branch, from the gh CLI."""
    result = subprocess.run(
        ["gh", "pr", "view", "--json", "number", "--jq", ".number"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ChangelogError("open the PR first, or pass --pr with its number")
    return int(result.stdout)


def new(pull_request: int | None, description: list[str], keys: list[str], base_ref: str) -> Path:
    """Write the changelog file for the current branch and return its path."""
    number = pull_request or detect_pr()
    root = git_repository_root()
    path = root / directory_at("HEAD", LEGACY_FILE) / f"{number}.yaml"
    if path.exists():
        raise ChangelogError(f"{path.relative_to(root)} already exists; edit it instead")
    entry = draft_entry(derived_changes(merge_base(base_ref, "HEAD")), description, keys)
    parse_file(yaml.safe_dump(entry, sort_keys=False), number, path.name)
    path.write_text(yaml.safe_dump(entry, sort_keys=False, allow_unicode=True, width=1_000_000))
    return path


def main(argv: list[str] | None = None) -> int:
    """Run the changelog command line on ``argv`` and return the exit status."""
    parser = argparse.ArgumentParser(description="Write and check per-PR perf changelog files.")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("new", help="write perf-changelog/<PR>.yaml from the config diff")
    create.add_argument("--pr", type=int, help="PR number; read from gh when omitted")
    create.add_argument(
        "-m", "--message", action="append", required=True, help="description bullet; repeatable"
    )
    create.add_argument(
        "--key", action="append", default=[], help="config to rerun without a change; repeatable"
    )
    create.add_argument("--base-ref", default="origin/main")
    verify = commands.add_parser("check", help="check a PR's changelog against its config diff")
    verify.add_argument("--base-ref", required=True)
    verify.add_argument("--head-ref", required=True)
    verify.add_argument("--pull-request", type=int, required=True)
    verify.add_argument("--created-at", required=True, help="PR creation time, ISO 8601")
    args = parser.parse_args(argv)
    try:
        if args.command == "new":
            print(f"Wrote {new(args.pr, args.message, args.key, args.base_ref)}")
            return 0
        errors = check(args.base_ref, args.head_ref, args.pull_request, args.created_at)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    for error in errors:
        print(f"::error::{error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
