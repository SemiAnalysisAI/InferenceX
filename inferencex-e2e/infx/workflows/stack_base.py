"""Resolve the base revision a stacked append-only PR sweep plans against.

A PR can add ``append-only: true`` points to a curve that an unmerged parent PR
introduces. Its diff against ``main`` then also carries the parent's regular
changelog entry, which ``infx.matrix.plan`` rejects in an append-only sweep (and
which would rerun every parent point). When the PR is stacked on exactly one open
parent PR, this module returns the parent's head SHA as the planning base, so the
sweep runs only the points the child adds on top of the parent.

The stacked base is used only when every condition holds; otherwise the given base
is returned unchanged and the planner behaves as before:

- some appended entries link another pull request of this repository (the parent);
- every other appended entry (the child's own) is ``append-only: true``;
- all parent-linked entries name the same open, same-repository PR that targets the
  same base branch;
- the base branch is an ancestor of the parent head (the parent is up to date), and
  the parent head is an ancestor of the child head (the child contains the parent);
- the changelog lines the child adds on top of the parent head are only its own
  append-only entries.

A stacked base never changes what merges or ingests: Merge Ingest plans the child's
merge commit against ``main``, which by then contains the parent.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from infx import github, yamlio
from infx.matrix.plan import get_added_lines

PR_LINK = re.compile(r"https://github\.com/([^/\s]+/[^/\s]+)/pull/([0-9]+)", re.IGNORECASE)


class StackError(ValueError):
    """Raised when a PR looks stacked but the stack cannot be used as a planning base."""


@dataclass(frozen=True)
class StackBase:
    base_ref: str
    parent_pr: int | None = None
    reason: str = ""


def linked_pr(entry: dict[str, Any], repo: str) -> int | None:
    """Return the PR number ``entry`` links in ``repo``, or None for placeholders and others."""
    match = PR_LINK.fullmatch(str(entry.get("pr-link") or "").strip())
    if match is None or match.group(1).lower() != repo.lower():
        return None
    return int(match.group(2))


def split_entries(
    entries: list[dict[str, Any]], repo: str, pr_number: int
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    """Split appended entries into this PR's own and those linking other PRs, by PR number."""
    own: list[dict[str, Any]] = []
    others: dict[int, list[dict[str, Any]]] = {}
    for entry in entries:
        number = linked_pr(entry, repo)
        if number is None or number == pr_number:
            own.append(entry)
        else:
            others.setdefault(number, []).append(entry)
    return own, others


def is_ancestor(ancestor: str, descendant: str) -> bool:
    """Return whether ``ancestor`` is an ancestor of (or equal to) ``descendant``."""
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"git merge-base failed: {result.stderr.strip()}")
    return result.returncode == 0


def ensure_commit(sha: str, pr_number: int) -> None:
    """Fetch the parent PR head when the checkout lacks it."""
    present = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"], check=False, capture_output=True
    )
    if present.returncode == 0:
        return
    subprocess.run(
        ["git", "fetch", "--no-tags", "origin", f"+refs/pull/{pr_number}/head"],
        check=True,
        capture_output=True,
    )


def load_added(base_ref: str, head_ref: str, changelog_file: str) -> list[dict[str, Any]]:
    """Return the changelog entries added between two refs."""
    added = get_added_lines(base_ref, head_ref, changelog_file)
    if not added.strip():
        return []
    data = yamlio.load(added, yamlio.UniqueKeySafeLoader)
    if not isinstance(data, list) or not all(isinstance(entry, dict) for entry in data):
        raise StackError("changelog additions are not a list of entries")
    return data


def resolve(
    *,
    repo: str,
    pr_number: int,
    base_ref: str,
    head_ref: str,
    changelog_file: str,
    token: str,
) -> StackBase:
    """Return the planning base for PR ``pr_number``; see the module docstring."""
    try:
        added = load_added(base_ref, head_ref, changelog_file)
    except (RuntimeError, ValueError) as error:
        # Malformed diffs are reported by the changelog validator that runs next.
        return StackBase(base_ref, reason=f"changelog additions unreadable: {error}")
    own, others = split_entries(added, repo, pr_number)
    if not others:
        return StackBase(base_ref, reason="no entries link another pull request")
    if not own or not all(entry.get("append-only") is True for entry in own):
        return StackBase(base_ref, reason="this PR's own entries are not all append-only")
    if len(others) != 1:
        raise StackError(
            "a stacked append-only PR must sit on exactly one parent PR; "
            f"appended entries link {sorted(others)}"
        )
    (parent_number,) = others
    parent = github.api(repo, f"/pulls/{parent_number}", token)
    if parent.get("state") != "open":
        raise StackError(
            f"parent PR #{parent_number} is {parent.get('state')!r}; update this PR from the "
            "base branch so its diff no longer carries the parent's entries"
        )
    if str(parent["head"]["repo"]["full_name"]).lower() != repo.lower():
        raise StackError(f"parent PR #{parent_number} must come from {repo}, not a fork")
    parent_base = str(parent["base"]["ref"])
    if base_ref.removeprefix("origin/") != parent_base:
        raise StackError(
            f"parent PR #{parent_number} targets {parent_base!r}, but this PR plans against "
            f"{base_ref!r}"
        )
    parent_sha = str(parent["head"]["sha"])
    ensure_commit(parent_sha, parent_number)
    if not is_ancestor(base_ref, parent_sha):
        raise StackError(
            f"parent PR #{parent_number} ({parent_sha[:12]}) is behind {base_ref}; merge "
            f"{parent_base} into the parent, then merge the parent into this PR"
        )
    if not is_ancestor(parent_sha, head_ref):
        raise StackError(
            f"this PR does not contain parent PR #{parent_number}'s head {parent_sha[:12]}; "
            "merge the parent branch into this PR"
        )
    child_own, child_others = split_entries(
        load_added(parent_sha, head_ref, changelog_file), repo, pr_number
    )
    if child_others or not child_own:
        raise StackError(
            f"on top of parent PR #{parent_number}, this PR must append only its own entries"
        )
    if not all(entry.get("append-only") is True for entry in child_own):
        raise StackError("on top of the parent PR, every appended entry must be append-only")
    return StackBase(
        parent_sha,
        parent_pr=parent_number,
        reason=f"stacked on PR #{parent_number}",
    )


def main() -> int:
    """Print the planning base and write it to GITHUB_OUTPUT as ``base-ref``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--head-ref", required=True)
    parser.add_argument("--changelog-file", required=True)
    args = parser.parse_args()
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""

    try:
        result = resolve(
            repo=args.repo,
            pr_number=args.pr_number,
            base_ref=args.base_ref,
            head_ref=args.head_ref,
            changelog_file=args.changelog_file,
            token=token,
        )
    except StackError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1

    print(f"Planning base: {result.base_ref} ({result.reason})", file=sys.stderr)
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"base-ref={result.base_ref}\n")
            handle.write(f"parent-pr={result.parent_pr or ''}\n")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"Changelog planning base: `{result.base_ref}` ({result.reason})\n")
    print(result.base_ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
