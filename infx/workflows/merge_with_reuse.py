"""Merge a PR while reusing its completed full sweep on push to main.

Post ``/reuse-sweep-run``, merge ``origin/main`` into the PR branch (a
``perf-changelog.yaml`` conflict is resolved by keeping main's entries and
re-appending the PR's with the canonical PR URL), push a sync commit so the
reuse gate sees the authorization on the new head, then squash-merge with
``--admin``.

Usage::

    python3 -m infx.workflows.merge_with_reuse <pr-number>

Environment variables:

* ``REPO`` -- GitHub repository (default ``SemiAnalysisAI/InferenceX``)
* ``CHECK_TIMEOUT_SECONDS`` -- timeout for individual check polling (default 900)
* ``HEAD_LAG_RETRIES`` -- retries when the PR head lags after push (default 6)
* ``HEAD_LAG_DELAY`` -- seconds between head-lag retries (default 5)
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
import time
from typing import Any

from infx import github

from .prepare_perf_changelog_merge import (
    canonicalize_appended_links,
    resolve_conflict_bytes,
)
from .sweep_runs import (
    artifact_names,
    completed_pr_runs,
    has_reusable_result_artifacts,
    pr_commit_shas,
)
from .validate_perf_changelog import read_git_file

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_REPO = "SemiAnalysisAI/InferenceX"
DEFAULT_CHECK_TIMEOUT = 900
DEFAULT_HEAD_LAG_RETRIES = 6
DEFAULT_HEAD_LAG_DELAY = 5
CHANGELOG = "perf-changelog.yaml"

SWEEP_LABEL_NAMES = frozenset(
    {
        "sweep-enabled",
        "full-sweep-enabled",
        "non-canary-full-sweep-enabled",
        "full-sweep-fail-fast",
        "full-sweep-fail-fast-no-canary",
    }
)

REUSE_INCOMPATIBLE_LABELS = frozenset({"evals-only", "agentx-fast"})

# ---------------------------------------------------------------------------
# Logging helpers (match the bash colors/symbols exactly)
# ---------------------------------------------------------------------------

_CYAN_BOLD = "\033[1;36m"
_GREEN_BOLD = "\033[1;32m"
_RED_BOLD = "\033[1;31m"
_RESET = "\033[0m"


def log(msg: str) -> None:
    print(f"{_CYAN_BOLD}→{_RESET} {msg}")


def ok(msg: str) -> None:
    print(f"{_GREEN_BOLD}✓{_RESET} {msg}")


def die(msg: str) -> int:
    print(f"{_RED_BOLD}✗{_RESET} {msg}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _git(*args: str, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = ["git", *args]
    return subprocess.run(
        cmd,
        capture_output=quiet,
        text=True,
        check=check,
    )


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _worktree_clean() -> bool:
    return _git_output("status", "--porcelain") == ""


def _current_ref() -> str:
    """Return the current branch name or detached HEAD sha."""
    result = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return _git_output("rev-parse", "HEAD")


def _rev_parse(ref: str = "HEAD") -> str:
    return _git_output("rev-parse", ref)


# ---------------------------------------------------------------------------
# GitHub helpers that mirror the bash's direct gh/jq calls
# ---------------------------------------------------------------------------


def _gh_pr_view(pr: int, repo: str) -> dict[str, Any]:
    """Fetch PR metadata via gh pr view (same fields as the bash)."""
    result = subprocess.run(
        [
            "gh",
            "pr",
            "view",
            str(pr),
            "--repo",
            repo,
            "--json",
            "headRefName,isCrossRepository,state,labels",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _gh_pr_comment(pr: int, repo: str, body: str) -> None:
    subprocess.run(
        ["gh", "pr", "comment", str(pr), "--repo", repo, "--body", body],
        capture_output=True,
        text=True,
        check=True,
    )


def _gh_pr_head_oid(pr: int, repo: str) -> str:
    result = subprocess.run(
        [
            "gh",
            "pr",
            "view",
            str(pr),
            "--repo",
            repo,
            "--json",
            "headRefOid",
            "--jq",
            ".headRefOid",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _gh_pr_merge_commit(pr: int, repo: str) -> str:
    result = subprocess.run(
        [
            "gh",
            "pr",
            "view",
            str(pr),
            "--repo",
            repo,
            "--json",
            "mergeCommit",
            "--jq",
            ".mergeCommit.oid",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _gh_pr_checks_watch(pr: int, repo: str) -> None:
    subprocess.run(
        ["gh", "pr", "checks", str(pr), "--repo", repo, "--watch", "--fail-fast"],
        text=True,
        check=True,
    )


# ---------------------------------------------------------------------------
# Eligibility check (mirrors the bash's loop over workflow runs + artifacts)
# ---------------------------------------------------------------------------


def find_eligible_run(
    repo: str,
    pr: int,
    head_branch: str,
    token: str,
) -> int | None:
    """Find a successful run-sweep.yml run with reusable artifacts on a PR commit.

    Returns the run ID, or None if no eligible run exists.
    """
    pr_shas = pr_commit_shas(repo, pr, token)
    if not pr_shas:
        return None

    runs = completed_pr_runs(repo, "run-sweep.yml", head_branch, token)
    for run in runs:
        if run.get("conclusion") != "success":
            continue
        run_sha = str(run.get("head_sha") or "")
        if run_sha not in pr_shas:
            continue
        names = artifact_names(repo, int(run["id"]), token)
        if has_reusable_result_artifacts(names):
            return int(run["id"])
    return None


# ---------------------------------------------------------------------------
# Check-run polling (mirrors wait_for_check in the bash)
# ---------------------------------------------------------------------------


def wait_for_check(
    sha: str,
    check_name: str,
    repo: str,
    token: str,
    timeout: int = DEFAULT_CHECK_TIMEOUT,
) -> int:
    """Poll until a named check-run completes on a commit.

    Returns 0 on success, 1 on failure or timeout.
    """
    log(f"Waiting for {check_name} on {sha[:8]}")
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        checks = github.api(
            repo,
            f"/commits/{sha}/check-runs",
            token,
            {"per_page": "100"},
        )
        check_runs = checks.get("check_runs", [])
        matching = [cr for cr in check_runs if cr.get("name") == check_name]
        if matching:
            matching.sort(key=lambda cr: str(cr.get("started_at") or ""))
            latest = matching[-1]
            status = latest.get("status") or ""
            conclusion = latest.get("conclusion") or ""
            details = latest.get("details_url") or ""

            if status == "completed":
                if conclusion == "success":
                    detail_suffix = f" - {details}" if details else ""
                    ok(f"{check_name} passed{detail_suffix}")
                    return 0
                detail_suffix = f" - {details}" if details else ""
                return die(f"{check_name} concluded {conclusion or 'unknown'}{detail_suffix}")

        time.sleep(5)

    return die(f"Timed out after {timeout}s waiting for {check_name} on {sha}")


# ---------------------------------------------------------------------------
# Changelog conflict resolution (calls infx Python APIs, not CLI)
# ---------------------------------------------------------------------------


def _read_index_stage(stage: int, path: str) -> bytes:
    """Read a file from a given git index stage during a merge conflict."""
    result = subprocess.run(
        ["git", "show", f":{stage}:{path}"],
        capture_output=True,
        check=True,
    )
    return result.stdout


def resolve_changelog_conflict(pr: int, repo: str) -> bool:
    """Resolve a perf-changelog.yaml conflict using the Python API.

    Returns True on success, False on failure (caller should abort the merge).
    """
    from .validate_perf_changelog import ChangelogValidationError

    try:
        base_raw = _read_index_stage(1, CHANGELOG)
        pr_raw = _read_index_stage(2, CHANGELOG)
        main_raw = _read_index_stage(3, CHANGELOG)
        resolved = resolve_conflict_bytes(base_raw, pr_raw, main_raw, pr, repo)
        with open(CHANGELOG, "wb") as f:
            f.write(resolved)
        print(f"Prepared {CHANGELOG} for PR #{pr}")
        return True
    except (ChangelogValidationError, OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return False


def canonicalize_changelog(pr: int, repo: str) -> None:
    """Canonicalize appended pr-link placeholders using the Python API."""
    with open(CHANGELOG, "rb") as fh:
        original = fh.read()
    base_raw = read_git_file("origin/main", CHANGELOG)
    prepared = canonicalize_appended_links(base_raw, original, pr, repo)
    if prepared != original:
        with open(CHANGELOG, "wb") as f:
            f.write(prepared)
        print(f"Prepared {CHANGELOG} for PR #{pr}")
    else:
        print(f"{CHANGELOG} already prepared for PR #{pr}")


# ---------------------------------------------------------------------------
# Main merge flow
# ---------------------------------------------------------------------------


def merge_pr(
    pr: int,
    *,
    repo: str = DEFAULT_REPO,
    check_timeout: int = DEFAULT_CHECK_TIMEOUT,
    head_lag_retries: int = DEFAULT_HEAD_LAG_RETRIES,
    head_lag_delay: int = DEFAULT_HEAD_LAG_DELAY,
) -> int:
    """Execute the full merge-with-reuse sequence.

    Returns 0 on success, non-zero on failure.
    """
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""

    # --- Pre-flight: clean worktree ---
    if not _worktree_clean():
        return die("Working tree is not clean")

    original_branch = _current_ref()
    local_branch = ""

    def cleanup() -> None:
        with contextlib.suppress(Exception):
            _git("checkout", "--quiet", original_branch, check=False, quiet=True)
        if local_branch:
            _git("branch", "-D", local_branch, check=False, quiet=True)

    try:
        return _merge_pr_inner(
            pr,
            repo=repo,
            check_timeout=check_timeout,
            head_lag_retries=head_lag_retries,
            head_lag_delay=head_lag_delay,
            token=token,
        )
    finally:
        cleanup()


def _merge_pr_inner(
    pr: int,
    *,
    repo: str,
    check_timeout: int,
    head_lag_retries: int,
    head_lag_delay: int,
    token: str,
) -> int:
    # --- PR eligibility ---
    pr_info = _gh_pr_view(pr, repo)
    pr_state = pr_info.get("state", "")
    if pr_state != "OPEN":
        return die(f"PR #{pr} is {pr_state}, expected OPEN")

    if pr_info.get("isCrossRepository", False):
        return die(f"PR #{pr} is from a fork; the merge helper cannot update its branch")

    head_branch: str = pr_info.get("headRefName", "")
    labels = [label.get("name", "") for label in pr_info.get("labels", [])]

    sweep_labels = [name for name in labels if name in SWEEP_LABEL_NAMES]
    if len(sweep_labels) > 1:
        return die(f"PR #{pr} has multiple conflicting sweep labels")

    incompatible = [name for name in labels if name in REUSE_INCOMPATIBLE_LABELS]
    if incompatible:
        return die(
            f"PR #{pr} uses {', '.join(incompatible)}, which is not eligible for artifact reuse"
        )

    # --- Find an eligible run with reusable artifacts ---
    eligible_run = find_eligible_run(repo, pr, head_branch, token)
    if eligible_run is None:
        return die(f"PR #{pr} has no successful reusable run-sweep.yml run on a current commit")

    # --- Post reuse comment ---
    log(f"Posting /reuse-sweep-run {eligible_run} on PR #{pr}")
    _gh_pr_comment(pr, repo, f"/reuse-sweep-run {eligible_run}")
    ok("Comment posted")

    # --- Fetch and checkout PR branch ---
    local_branch = f"pr-{pr}-reuse-{os.getpid()}"
    log(f"Fetching PR branch {head_branch}")
    _git("fetch", "origin", f"pull/{pr}/head:{local_branch}", "--quiet", quiet=True)
    _git("checkout", "--quiet", local_branch, quiet=True)
    _git("fetch", "origin", "main", "--quiet", quiet=True)

    pre_merge = _rev_parse()

    # --- Merge origin/main ---
    log("Merging origin/main")
    merge_result = _git("merge", "origin/main", "--no-ff", "--no-edit", check=False)
    if merge_result.returncode != 0:
        # Check what's unresolved
        unresolved = _git_output("diff", "--name-only", "--diff-filter=U")
        if unresolved != CHANGELOG:
            _git("merge", "--abort", check=False)
            return die(
                f"Unexpected conflict(s) in: {unresolved} -- only {CHANGELOG} is auto-resolved"
            )

        log(f"Resolving {CHANGELOG} conflict")
        if not resolve_changelog_conflict(pr, repo):
            _git("merge", "--abort", check=False)
            return die(f"Could not safely resolve {CHANGELOG}")

        _git("add", CHANGELOG)
        _git("commit", "--no-edit")

    # --- Canonicalize changelog links ---
    head_after_merge = _rev_parse()
    canonicalize_changelog(pr, repo)

    if _git("diff", "--quiet", "--", CHANGELOG, check=False).returncode != 0:
        _git("add", CHANGELOG)
        if head_after_merge != pre_merge:
            _git("commit", "--amend", "--no-edit")
        else:
            _git("commit", "-m", f"fix: canonicalize PR #{pr} changelog link [skip-sweep]")

    # --- Ensure synchronize event ---
    if pre_merge == _rev_parse():
        _git(
            "commit",
            "--allow-empty",
            "-m",
            f"chore: refresh PR #{pr} for sweep reuse [skip-sweep]",
        )

    post_merge = _rev_parse()

    # --- Push ---
    log(f"Pushing prepared commit {post_merge[:8]}")
    _git("push", "origin", f"{local_branch}:{head_branch}")
    ok("Push complete; reuse authorization will be evaluated on the new head")

    # --- Verify head (with lag retry) ---
    current_head = _poll_pr_head(pr, repo, post_merge, head_lag_retries, head_lag_delay)
    if current_head != post_merge:
        return die(f"PR head changed to {current_head[:8]}; expected {post_merge[:8]}")

    # --- Wait for check-changelog ---
    rc = wait_for_check(post_merge, "check-changelog", repo, token, check_timeout)
    if rc != 0:
        return rc

    # --- Wait for all PR checks ---
    log("Waiting for all PR checks")
    _gh_pr_checks_watch(pr, repo)
    ok("All PR checks passed")

    # --- Final head verification ---
    current_head = _gh_pr_head_oid(pr, repo)
    if current_head != post_merge:
        return die(f"PR head changed to {current_head[:8]}; expected {post_merge[:8]}")

    # --- Squash merge ---
    log(f"Squash-merging PR #{pr} into main")
    subprocess.run(
        ["gh", "pr", "merge", str(pr), "--repo", repo, "--squash", "--admin"],
        capture_output=True,
        text=True,
        check=True,
    )

    merge_sha = _gh_pr_merge_commit(pr, repo)
    ok(
        f"PR #{pr} merged as {merge_sha[:8]} -- "
        f"the push-to-main run will reuse the prior successful sweep."
    )
    return 0


def _poll_pr_head(
    pr: int,
    repo: str,
    expected: str,
    retries: int,
    delay: int,
) -> str:
    """Poll the PR head OID with retries to handle post-push lag.

    After pushing, GitHub's API can briefly report the old head SHA.
    The bash script had no retry and would abort spuriously.  This
    implementation retries a configurable number of times (default 6,
    i.e. ~30 s) before giving up, which eliminates the known
    "PR head changed" race documented in KLAUD_DEBUG.md and memory.
    """
    for attempt in range(retries + 1):
        current = _gh_pr_head_oid(pr, repo)
        if current == expected:
            return current
        if attempt < retries:
            time.sleep(delay)
    return current  # type: ignore[possibly-undefined]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    if len(sys.argv) != 2 or not re.fullmatch(r"\d+", sys.argv[1]):
        print(f"Usage: {sys.argv[0]} <pr-number>", file=sys.stderr)
        return 2
    pr = int(sys.argv[1])
    repo = os.environ.get("REPO", DEFAULT_REPO)
    check_timeout = int(os.environ.get("CHECK_TIMEOUT_SECONDS", str(DEFAULT_CHECK_TIMEOUT)))
    head_lag_retries = int(os.environ.get("HEAD_LAG_RETRIES", str(DEFAULT_HEAD_LAG_RETRIES)))
    head_lag_delay = int(os.environ.get("HEAD_LAG_DELAY", str(DEFAULT_HEAD_LAG_DELAY)))

    return merge_pr(
        pr,
        repo=repo,
        check_timeout=check_timeout,
        head_lag_retries=head_lag_retries,
        head_lag_delay=head_lag_delay,
    )


if __name__ == "__main__":
    raise SystemExit(main())
