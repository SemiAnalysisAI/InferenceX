"""Merge a PR while reusing its completed full sweep on push to main.

Post ``/reuse-sweep-run``, merge ``origin/main`` into the PR branch (a
``perf-changelog.yaml`` conflict is resolved by keeping main's entries and
re-appending the PR's with the canonical PR URL), push a sync commit so the
reuse gate sees the authorization on the new head, then squash-merge with
admin bypass.

Usage::

    uv run --extra workflows python -m infx.workflows.merge_with_reuse <pr-number>

Environment variables:

* ``REPO`` -- GitHub repository (default ``SemiAnalysisAI/InferenceX``)
* ``CHECK_TIMEOUT_SECONDS`` -- timeout for individual check polling (default 900)
* ``HEAD_LAG_RETRIES`` -- retries when the PR head lags after push (default 6)
* ``HEAD_LAG_DELAY`` -- seconds between head-lag retries (default 5)
* ``GH_TOKEN`` / ``GITHUB_TOKEN`` -- GitHub personal access token
* ``GITHUB_API_URL`` -- GitHub API base URL (default ``https://api.github.com``)
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import subprocess
import sys
import time
from typing import TYPE_CHECKING

try:
    import git as gitpython

    from github import Github, GithubException
except ImportError:
    gitpython = None  # type: ignore[assignment]
    Github = None  # type: ignore[assignment, misc]

    class GithubException(Exception):  # type: ignore[no-redef]  # noqa: N818
        """Stub when PyGithub is not installed."""

        status = 0
        data: object = None
        headers: dict = {}  # type: ignore[type-arg]  # noqa: RUF012


_WORKFLOWS_AVAILABLE = gitpython is not None

if TYPE_CHECKING:
    from github.PullRequest import PullRequest
    from github.Repository import Repository

from .prepare_perf_changelog_merge import (  # noqa: E402
    canonicalize_appended_links,
    resolve_conflict_bytes,
)
from .sweep_runs import (  # noqa: E402
    artifact_names,
    completed_pr_runs,
    has_reusable_result_artifacts,
    pr_commit_shas,
)
from .validate_perf_changelog import read_git_file  # noqa: E402

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

# Suppress PyGithub/GitPython/urllib3 debug logging to avoid leaking tokens
# or request headers.
logging.getLogger("github").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("git").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

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
# Auth
# ---------------------------------------------------------------------------


def _resolve_token() -> str:
    """Resolve the GitHub token from the environment, falling back to ``gh auth token``.

    Precedence: GH_TOKEN > GITHUB_TOKEN > ``gh auth token`` (a single
    bootstrap subprocess call -- the only subprocess in this module).
    """
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
    if token:
        return token
    # Last-resort fallback: ask the gh CLI for its stored credential.
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def _make_github(token: str) -> Github:
    """Build a PyGithub ``Github`` client, respecting ``GITHUB_API_URL``."""
    base_url = os.environ.get("GITHUB_API_URL") or "https://api.github.com"
    return Github(login_or_token=token, base_url=base_url)


# ---------------------------------------------------------------------------
# Transient error handling for API polling
# ---------------------------------------------------------------------------

_TRANSIENT_HTTP_STATUSES = frozenset({403, 429, 500, 502, 503, 504})


def _is_transient_error(exc: Exception) -> bool:
    """Return True for API errors safe to retry during check polling.

    Covers server errors (5xx), rate limits (403/429), and connection-level
    failures from requests/urllib3.
    """
    if isinstance(exc, GithubException):
        return exc.status in _TRANSIENT_HTTP_STATUSES
    # Connection/timeout at any level (builtin, requests, urllib3).
    # builtins.ConnectionError covers ConnectionRefused/Reset/Aborted/BrokenPipe.
    # requests.ConnectionError and urllib3 errors inherit from OSError.
    return isinstance(exc, (ConnectionError, TimeoutError))


def _retry_delay(exc: Exception) -> float:
    """Extract Retry-After from a rate-limited response, default 10s, capped at 60s."""
    if isinstance(exc, GithubException):
        headers = getattr(exc, "headers", None) or {}
        if isinstance(headers, dict):
            raw = headers.get("Retry-After") or headers.get("retry-after")
            if raw:
                try:
                    return min(max(float(raw), 1.0), 60.0)
                except (ValueError, TypeError):
                    pass
    return 10.0


# ---------------------------------------------------------------------------
# GitPython wrapper
# ---------------------------------------------------------------------------


class GitOps:
    """Thin wrapper around GitPython for the operations this workflow needs.

    All git operations go through this class so tests can supply a
    ``gitpython.Repo`` backed by a temporary directory instead of the real
    checkout.
    """

    def __init__(self, repo: gitpython.Repo | None = None) -> None:
        self.repo = repo or gitpython.Repo(".")

    # -- queries --

    def is_clean(self) -> bool:
        return not self.repo.is_dirty(untracked_files=True)

    def current_ref(self) -> str:
        if self.repo.head.is_detached:
            return self.repo.head.commit.hexsha
        return self.repo.active_branch.name

    def rev_parse(self, ref: str = "HEAD") -> str:
        return self.repo.rev_parse(ref).hexsha

    def diff_name_only_unmerged(self) -> str:
        """Return newline-joined list of unmerged paths (like ``git diff --name-only --diff-filter=U``)."""
        return self.repo.git.diff("--name-only", "--diff-filter=U")

    def diff_quiet(self, *args: str) -> bool:
        """Return True if ``git diff --quiet`` exits 0 (no changes)."""
        try:
            self.repo.git.diff("--quiet", *args)
            return True
        except gitpython.GitCommandError:
            return False

    def show_stage(self, stage: int, path: str) -> bytes:
        """Read a file from a given git index stage during a merge conflict.

        Returns raw bytes identical to ``git show :<stage>:<path>``.
        ``repo.git.show()`` returns str by default and strips the trailing
        newline, which breaks ``parse_changelog``'s newline requirement.
        ``stdout_as_string=False`` and ``strip_newline_in_stdout=False``
        preserve the exact bytes.
        """
        return self.repo.git.show(
            f":{stage}:{path}",
            stdout_as_string=False,
            strip_newline_in_stdout=False,
        )

    # -- mutations --

    def fetch(self, *args: str) -> None:
        self.repo.git.fetch(*args)

    def checkout(self, *args: str) -> None:
        self.repo.git.checkout(*args)

    def merge(self, *args: str) -> int:
        """Run ``git merge`` and return the exit code (0 or non-zero for conflicts)."""
        try:
            self.repo.git.merge(*args)
            return 0
        except gitpython.GitCommandError as exc:
            return exc.status or 1

    def add(self, *paths: str) -> None:
        self.repo.git.add(*paths)

    def commit(self, *args: str) -> None:
        self.repo.git.commit(*args)

    def push(self, *args: str) -> None:
        self.repo.git.push(*args)

    def branch_delete(self, name: str) -> None:
        with contextlib.suppress(gitpython.GitCommandError):
            self.repo.git.branch("-D", name)

    def merge_abort(self) -> None:
        with contextlib.suppress(gitpython.GitCommandError):
            self.repo.git.merge("--abort")


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
# Check-run deduplication helpers
# ---------------------------------------------------------------------------


def _latest_check_runs(check_runs: list) -> list:
    """Dedupe check runs by name, keeping the one with the latest started_at/id.

    GitHub can return superseded re-runs for the same check name.  This mirrors
    ``gh pr checks --watch --fail-fast`` which only considers the latest run
    per check name.
    """
    by_name: dict[str, object] = {}
    for cr in check_runs:
        existing = by_name.get(cr.name)
        if existing is None:
            by_name[cr.name] = cr
            continue
        cr_key = (str(cr.started_at or ""), getattr(cr, "id", 0))
        ex_key = (
            str(getattr(existing, "started_at", "") or ""),
            getattr(existing, "id", 0),
        )
        if cr_key > ex_key:
            by_name[cr.name] = cr
    return list(by_name.values())


def _latest_statuses(statuses: list) -> list:
    """Dedupe commit statuses by context, keeping the most recent."""
    by_context: dict[str, object] = {}
    for s in statuses:
        existing = by_context.get(s.context)
        if existing is None:
            by_context[s.context] = s
            continue
        s_key = (str(getattr(s, "updated_at", "") or ""), getattr(s, "id", 0))
        ex_key = (
            str(getattr(existing, "updated_at", "") or ""),
            getattr(existing, "id", 0),
        )
        if s_key > ex_key:
            by_context[s.context] = s
    return list(by_context.values())


# Check-run conclusions that trigger fail-fast.  ``cancelled`` is deliberately
# excluded: ``gh pr checks --watch --fail-fast`` does not treat a cancelled run
# as a fast-fail trigger either -- it counts as "completed, not passing" but
# does not abort the wait.
_FAIL_FAST_CONCLUSIONS = frozenset(
    {
        "failure",
        "timed_out",
        "action_required",
        "startup_failure",
    }
)


# ---------------------------------------------------------------------------
# Check-run polling (mirrors wait_for_check in the bash)
# ---------------------------------------------------------------------------


def wait_for_checks(
    _pull: PullRequest,
    sha: str,
    gh_repo: Repository,
    timeout: int = DEFAULT_CHECK_TIMEOUT,
) -> int:
    """Poll until all check runs and commit statuses complete on *sha*.

    Fail-fast on ``failure``, ``timed_out``, ``action_required``, or
    ``startup_failure``.  ``cancelled`` is **not** a fail-fast trigger
    (matching ``gh pr checks --watch --fail-fast`` behavior).  ``stale``
    is treated as pending (the check needs re-evaluation).  ``skipped``
    and ``neutral`` count as passing.

    Check runs are deduped by name (latest ``started_at``/``id`` wins) and
    commit statuses by context to ignore superseded re-runs.

    Transient API errors (5xx, rate limits, connection failures) are retried
    with backoff until the timeout expires.

    Returns 0 when every check succeeds/is neutral/skipped and every
    commit status succeeds, 1 on failure or timeout.
    """
    log(f"Waiting for checks on {sha[:8]}")
    deadline = time.monotonic() + timeout
    commit = gh_repo.get_commit(sha)

    while time.monotonic() < deadline:
        try:
            check_runs = _latest_check_runs(list(commit.get_check_runs()))

            all_completed = True
            for cr in check_runs:
                if cr.status != "completed":
                    all_completed = False
                    continue
                if cr.conclusion == "stale":
                    # Stale means the check needs re-evaluation; treat as pending.
                    all_completed = False
                    continue
                if cr.conclusion in _FAIL_FAST_CONCLUSIONS:
                    detail = f" - {cr.details_url}" if cr.details_url else ""
                    return die(f"{cr.name} concluded {cr.conclusion}{detail}")
                # cancelled: completed but not a fail-fast trigger
                # success, neutral, skipped: passing

            # -- Commit statuses (Status API, e.g. external CI) --
            combined = commit.get_combined_status()
            statuses = _latest_statuses(list(combined.statuses))
            statuses_done = True
            for status in statuses:
                if status.state == "pending":
                    statuses_done = False
                    continue
                if status.state in ("error", "failure"):
                    detail = f" - {status.target_url}" if status.target_url else ""
                    return die(f"{status.context} concluded {status.state}{detail}")

            if all_completed and statuses_done and (check_runs or statuses):
                ok("All checks passed")
                return 0

        except Exception as exc:
            if _is_transient_error(exc):
                delay = _retry_delay(exc)
                logger.warning("Transient API error (retrying in %.0fs): %s", delay, exc)
                time.sleep(delay)
                commit = gh_repo.get_commit(sha)
                continue
            raise

        time.sleep(5)
        # Re-fetch commit object to get fresh check data
        commit = gh_repo.get_commit(sha)

    return die(f"Timed out after {timeout}s waiting for checks on {sha}")


def wait_for_check(
    sha: str,
    check_name: str,
    gh_repo: Repository,
    timeout: int = DEFAULT_CHECK_TIMEOUT,
) -> int:
    """Poll until a named check-run completes on a commit.

    Returns 0 on success, 1 on failure or timeout.
    Transient API errors are retried with backoff until the timeout expires.
    """
    log(f"Waiting for {check_name} on {sha[:8]}")
    deadline = time.monotonic() + timeout
    commit = gh_repo.get_commit(sha)

    while time.monotonic() < deadline:
        try:
            check_runs = list(commit.get_check_runs())
            matching = [cr for cr in check_runs if cr.name == check_name]
            if matching:
                matching.sort(key=lambda cr: str(cr.started_at or ""))
                latest = matching[-1]

                if latest.status == "completed":
                    detail = f" - {latest.details_url}" if latest.details_url else ""
                    if latest.conclusion == "success":
                        ok(f"{check_name} passed{detail}")
                        return 0
                    return die(f"{check_name} concluded {latest.conclusion or 'unknown'}{detail}")
        except Exception as exc:
            if _is_transient_error(exc):
                delay = _retry_delay(exc)
                logger.warning("Transient API error (retrying in %.0fs): %s", delay, exc)
                time.sleep(delay)
                commit = gh_repo.get_commit(sha)
                continue
            raise

        time.sleep(5)
        commit = gh_repo.get_commit(sha)

    return die(f"Timed out after {timeout}s waiting for {check_name} on {sha}")


# ---------------------------------------------------------------------------
# Changelog conflict resolution (calls infx Python APIs, not CLI)
# ---------------------------------------------------------------------------


def _read_index_stage(git_ops: GitOps, stage: int, path: str) -> bytes:
    """Read a file from a given git index stage during a merge conflict."""
    return git_ops.show_stage(stage, path)


def resolve_changelog_conflict(pr: int, repo: str, git_ops: GitOps) -> bool:
    """Resolve a perf-changelog.yaml conflict using the Python API.

    Returns True on success, False on failure (caller should abort the merge).
    """
    from .validate_perf_changelog import ChangelogValidationError

    try:
        base_raw = _read_index_stage(git_ops, 1, CHANGELOG)
        pr_raw = _read_index_stage(git_ops, 2, CHANGELOG)
        main_raw = _read_index_stage(git_ops, 3, CHANGELOG)
        resolved = resolve_conflict_bytes(base_raw, pr_raw, main_raw, pr, repo)
        with open(CHANGELOG, "wb") as f:
            f.write(resolved)
        print(f"Prepared {CHANGELOG} for PR #{pr}")
        return True
    except (ChangelogValidationError, OSError, gitpython.GitCommandError) as exc:
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


class _MergeState:
    """Shared mutable state between merge_pr and its cleanup closure."""

    __slots__ = ("local_branch",)

    def __init__(self) -> None:
        self.local_branch = ""


def merge_pr(
    pr: int,
    *,
    repo: str = DEFAULT_REPO,
    check_timeout: int = DEFAULT_CHECK_TIMEOUT,
    head_lag_retries: int = DEFAULT_HEAD_LAG_RETRIES,
    head_lag_delay: int = DEFAULT_HEAD_LAG_DELAY,
    _git_ops: GitOps | None = None,
    _gh: Github | None = None,
) -> int:
    """Execute the full merge-with-reuse sequence.

    Returns 0 on success, non-zero on failure.
    ``_git_ops`` and ``_gh`` are test-injection seams.
    """
    token = _resolve_token()
    gh = _gh or _make_github(token)
    git_ops = _git_ops or GitOps()

    # --- Pre-flight: clean worktree ---
    if not git_ops.is_clean():
        return die("Working tree is not clean")

    original_branch = git_ops.current_ref()
    state = _MergeState()

    def cleanup() -> None:
        # Abort any in-progress merge first so checkout doesn't fail on a
        # half-merged index.
        git_ops.merge_abort()
        with contextlib.suppress(Exception):
            git_ops.checkout("--quiet", original_branch)
        if state.local_branch:
            git_ops.branch_delete(state.local_branch)

    try:
        return _merge_pr_inner(
            pr,
            repo=repo,
            check_timeout=check_timeout,
            head_lag_retries=head_lag_retries,
            head_lag_delay=head_lag_delay,
            token=token,
            gh=gh,
            git_ops=git_ops,
            state=state,
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
    gh: Github,
    git_ops: GitOps,
    state: _MergeState,
) -> int:
    gh_repo = gh.get_repo(repo)
    pull = gh_repo.get_pull(pr)

    # --- PR eligibility ---
    pr_state = pull.state.upper()
    if pr_state != "OPEN":
        return die(f"PR #{pr} is {pr_state}, expected OPEN")

    head_repo = pull.head.repo
    if head_repo is None:
        return die(
            f"PR #{pr} head repository is unavailable (deleted fork?); "
            "the merge helper cannot update its branch"
        )
    if head_repo.full_name != pull.base.repo.full_name:
        return die(f"PR #{pr} is from a fork; the merge helper cannot update its branch")

    head_branch: str = pull.head.ref
    labels = [label.name for label in pull.labels]

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
    pull.create_issue_comment(f"/reuse-sweep-run {eligible_run}")
    ok("Comment posted")

    # --- Fetch and checkout PR branch ---
    state.local_branch = f"pr-{pr}-reuse-{os.getpid()}"
    log(f"Fetching PR branch {head_branch}")
    git_ops.fetch("origin", f"pull/{pr}/head:{state.local_branch}", "--quiet")
    git_ops.checkout("--quiet", state.local_branch)
    git_ops.fetch("origin", "main", "--quiet")

    pre_merge = git_ops.rev_parse()

    # --- Merge origin/main ---
    log("Merging origin/main")
    merge_rc = git_ops.merge("origin/main", "--no-ff", "--no-edit")
    if merge_rc != 0:
        # Check what's unresolved
        unresolved = git_ops.diff_name_only_unmerged()
        if unresolved != CHANGELOG:
            git_ops.merge_abort()
            return die(
                f"Unexpected conflict(s) in: {unresolved} -- only {CHANGELOG} is auto-resolved"
            )

        log(f"Resolving {CHANGELOG} conflict")
        if not resolve_changelog_conflict(pr, repo, git_ops):
            git_ops.merge_abort()
            return die(f"Could not safely resolve {CHANGELOG}")

        git_ops.add(CHANGELOG)
        git_ops.commit("--no-edit")

    # --- Canonicalize changelog links ---
    head_after_merge = git_ops.rev_parse()
    canonicalize_changelog(pr, repo)

    if not git_ops.diff_quiet("--", CHANGELOG):
        git_ops.add(CHANGELOG)
        if head_after_merge != pre_merge:
            git_ops.commit("--amend", "--no-edit")
        else:
            git_ops.commit("-m", f"fix: canonicalize PR #{pr} changelog link [skip-sweep]")

    # --- Ensure synchronize event ---
    if pre_merge == git_ops.rev_parse():
        git_ops.commit(
            "--allow-empty",
            "-m",
            f"chore: refresh PR #{pr} for sweep reuse [skip-sweep]",
        )

    post_merge = git_ops.rev_parse()

    # --- Push ---
    log(f"Pushing prepared commit {post_merge[:8]}")
    git_ops.push("origin", f"{state.local_branch}:{head_branch}")
    ok("Push complete; reuse authorization will be evaluated on the new head")

    # --- Verify head (with lag retry) ---
    current_head = _poll_pr_head(pull, post_merge, head_lag_retries, head_lag_delay)
    if current_head != post_merge:
        return die(f"PR head changed to {current_head[:8]}; expected {post_merge[:8]}")

    # --- Wait for check-changelog ---
    rc = wait_for_check(post_merge, "check-changelog", gh_repo, check_timeout)
    if rc != 0:
        return rc

    # --- Wait for all PR checks ---
    log("Waiting for all PR checks")
    rc = wait_for_checks(pull, post_merge, gh_repo, check_timeout)
    if rc != 0:
        return rc

    # --- Final head verification ---
    pull.update()
    current_head = pull.head.sha
    if current_head != post_merge:
        return die(f"PR head changed to {current_head[:8]}; expected {post_merge[:8]}")

    # --- Squash merge ---
    # GitHub's MergePullRequestInput has no bypass field; the "Protect main"
    # ruleset lists org admins as bypass actors, so REST PullRequest.merge
    # with an admin token is equivalent to ``gh pr merge --admin``.
    log(f"Squash-merging PR #{pr} into main")
    try:
        merge_status = pull.merge(merge_method="squash", sha=post_merge)
    except GithubException as exc:
        # Surface the API error message verbatim, but never the token.
        api_message = (
            exc.data.get("message", str(exc.status))
            if isinstance(exc.data, dict)
            else str(exc.status)
        )
        return die(f"Merge failed: {api_message}")

    if not merge_status.merged:
        return die(f"Merge failed: {merge_status.message}")

    merge_sha = merge_status.sha
    ok(
        f"PR #{pr} merged as {merge_sha[:8]} -- "
        f"the push-to-main run will reuse the prior successful sweep."
    )
    return 0


def _poll_pr_head(
    pull: PullRequest,
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
        pull.update()
        current = pull.head.sha
        if current == expected:
            return current
        if attempt < retries:
            time.sleep(delay)
    return current  # type: ignore[possibly-undefined]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_INSTALL_HINT = (
    "Missing required dependencies (PyGithub, GitPython). Install with:\n"
    "  uv run --extra workflows python -m infx.workflows.merge_with_reuse <pr>"
)


def main() -> int:
    if not _WORKFLOWS_AVAILABLE:
        print(_INSTALL_HINT, file=sys.stderr)
        return 1

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
