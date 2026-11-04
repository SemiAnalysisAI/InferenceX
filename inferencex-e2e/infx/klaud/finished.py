"""Keep finished Klaud PRs mergeable while they await review, and requeue superseded ones.

A finished PR is an open, ready-for-review Klaud PR whose session ended: only a verified
``validated`` finish marks a Klaud PR ready. Every Klaud PR appends to the tail of
perf-changelog.yaml, so each merge to main leaves the others conflicting there. Recovery merges
main into those PRs with the repository's changelog resolver and pushes a ``[skip-sweep]``
merge commit, so the validated sweep stays the PR's reusable evidence and no new sweep starts.
"""

from __future__ import annotations

import base64
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from . import github
from .lifecycle import BOT, CANDIDATE_ID, COMMITTER, owned_activity
from .models import REQUEUE_LABEL, Ownership, utc
from .reporting import translated

CHANGELOG = "inferencex-e2e/perf-changelog.yaml"
COMMITTER_EMAIL = "41898282+github-actions[bot]@users.noreply.github.com"
STALE_DAYS = 5
SYNC_LIMIT = 3
SYNC_REFS = "refs/klaud-sync/"
GIT_TIMEOUT_SECONDS = 120


def remote(repository: str) -> str:
    """The URL that recovery fetches from and pushes to for ``repository``."""
    return f"https://github.com/{repository}"


def finished(repository: str, pull: dict) -> str | None:
    """Return the candidate ID of ``pull`` if it is a finished Klaud PR of ``repository``."""
    match = re.fullmatch(rf"klaud/auto-({CANDIDATE_ID})", pull["head"]["ref"])
    if (
        match
        and not pull["draft"]
        and pull["user"]["login"] == BOT
        and (pull["head"].get("repo") or {}).get("full_name") == repository
    ):
        return match[1]
    return None


def git(*args: str, cwd: Path | None = None, env: dict | None = None) -> str:
    """Run git with ``args`` in ``cwd`` and return its stripped output."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=GIT_TIMEOUT_SECONDS,
    ).stdout.strip()


def authenticated() -> dict[str, str]:
    """The environment in which git authenticates to github.com with ``$GH_TOKEN``."""
    credential = base64.b64encode(f"x-access-token:{os.environ['GH_TOKEN']}".encode()).decode()
    return {
        **os.environ,
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": "AUTHORIZATION: basic " + credential,
    }


def changelog_only(head: str, main: str) -> bool:
    """Whether merging commit ``main`` into ``head`` conflicts on the changelog and nothing else.

    The merge runs in memory, so classifying a PR costs no worktree.
    """
    merge = subprocess.run(
        ["git", "merge-tree", "--write-tree", "--name-only", "--no-messages", head, main],
        check=False,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
    )
    if merge.returncode > 1:
        raise subprocess.CalledProcessError(
            merge.returncode, merge.args, merge.stdout, merge.stderr
        )
    return merge.stdout.splitlines()[1:] == [CHANGELOG]


def merge_main(repository: str, pull: dict, main: str) -> None:
    """Merge commit ``main`` into ``pull``'s fetched head, resolve the changelog and push it.

    Callers first check with ``changelog_only`` that the changelog is the sole conflict, which
    ``prepare_perf_changelog_merge resolve-conflict`` resolves. The push only fast-forwards.
    """
    branch = pull["head"]["ref"]
    bot = {
        **os.environ,
        "GIT_AUTHOR_NAME": COMMITTER,
        "GIT_AUTHOR_EMAIL": COMMITTER_EMAIL,
        "GIT_COMMITTER_NAME": COMMITTER,
        "GIT_COMMITTER_EMAIL": COMMITTER_EMAIL,
    }
    with tempfile.TemporaryDirectory(prefix="klaud-sync-") as temp:
        tree = Path(temp) / "tree"
        git("worktree", "add", "--quiet", "--detach", str(tree), pull["head"]["sha"])
        try:
            subprocess.run(
                ["git", "merge", "--no-ff", "--no-commit", "--quiet", main],
                check=False,
                cwd=tree,
                env=bot,
                capture_output=True,
                timeout=GIT_TIMEOUT_SECONDS,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-P",
                    "-m",
                    "infx.workflows.prepare_perf_changelog_merge",
                    "resolve-conflict",
                    "--changelog-file",
                    "perf-changelog.yaml",
                    "--pr-number",
                    str(pull["number"]),
                    "--repo",
                    repository,
                ],
                cwd=tree / "inferencex-e2e",
                capture_output=True,
                check=True,
                timeout=GIT_TIMEOUT_SECONDS,
            )
            git("add", CHANGELOG, cwd=tree)
            message = f"Merge main into {branch} [skip-sweep]"
            git("commit", "--quiet", "-m", message, cwd=tree, env=bot)
            git(
                "push",
                "--quiet",
                remote(repository),
                "HEAD:refs/heads/" + branch,
                cwd=tree,
                env=authenticated(),
            )
        finally:
            git("worktree", "remove", "--force", str(tree))


def sync_conflicts(repository: str, live: set[str], claimed: list[Ownership]) -> None:
    """Best effort: merge main into up to SYNC_LIMIT conflicting finished PRs, oldest first.

    Candidates in ``live``, in a family claim of ``claimed``, in an ownership artifact or with
    an active PR sweep keep their head: recovery still verifies an owned candidate's exact-head
    sweep and completion report, and a push would cancel the sweep. A PR whose mergeability
    GitHub has not computed, whose branch moved or that conflicts beyond the changelog is left
    as is; only attempted merges count toward the limit. Run from a full-history checkout
    of ``repository``; fetches and pushes authenticate with ``$GH_TOKEN``.
    """
    skipped = live | {candidate.id for owner in claimed for candidate in owner.candidates}
    dirty = []
    try:
        for listed in github.items(repository, "pulls?state=open&per_page=100"):
            candidate = finished(repository, listed)
            if candidate and candidate not in skipped:
                pull = github.read(repository, f"pulls/{listed['number']}")
                if pull["mergeable"] is False and finished(repository, pull) == candidate:
                    dirty.append((pull, candidate))
        if dirty:
            skipped |= owned_candidates(repository, []) | owned_activity(repository)[0]
        todo = sorted(
            (pull for pull, candidate in dirty if candidate not in skipped),
            key=lambda pull: pull["created_at"],
        )
        if todo:
            branches = ["main", *(pull["head"]["ref"] for pull in todo)]
            git(
                "fetch",
                "--quiet",
                "--no-tags",
                remote(repository),
                *(f"+refs/heads/{branch}:{SYNC_REFS}{branch}" for branch in branches),
                env=authenticated(),
            )
            main = git("rev-parse", SYNC_REFS + "main")
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"::warning::Klaud changelog sync skipped: {type(error).__name__}")
        return
    synced = 0
    for pull in todo:
        number = pull["number"]
        if synced == SYNC_LIMIT:
            print(f"Klaud: changelog sync of PR #{number} and later waits for the next pass.")
            return
        try:
            head = git("rev-parse", SYNC_REFS + pull["head"]["ref"])
            if head != pull["head"]["sha"] or not changelog_only(head, main):
                print(
                    f"Klaud: PR #{number} left as is: its branch moved or it conflicts beyond "
                    "the changelog."
                )
                continue
            synced += 1
            merge_main(repository, pull, main)
            print(f"Klaud: merged main into PR #{number} to resolve its changelog.")
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            print(f"::warning::Klaud changelog sync of PR #{number} failed: {type(error).__name__}")


def owned_candidates(repository: str, claimed: list[Ownership]) -> set[str]:
    """Return the candidate IDs of the family claims ``claimed`` and of every ownership artifact.

    Raises like the listing and download it reads, so callers act only on known ownership.
    """
    owned = {candidate.id for owner in claimed for candidate in owner.candidates}
    for artifact in github.items(
        repository, "actions/artifacts?name=klaud-ownership&per_page=100", "artifacts"
    ):
        if artifact["expired"]:
            continue
        with tempfile.TemporaryDirectory(prefix="klaud-ownership-") as temp:
            github.download_json(repository, artifact, Path(temp))
            owner = Ownership.model_validate_json((Path(temp) / "ownership.json").read_text())
        owned.update(candidate.id for candidate in owner.candidates)
    return owned


def requeue_superseded(
    repository: str,
    pulls: list[dict],
    offered: dict[str, set[str]],
    claimed: list[Ownership],
    now: datetime,
) -> list[int]:
    """Close finished PRs older than STALE_DAYS whose family the feed offers on a newer image.

    ``offered`` maps each family branch prefix to the image keys the planner would build its
    candidates from. A finished PR started from an image its family no longer offers is
    superseded, but still occupies the family. Closing it with REQUEUE_LABEL returns the family
    to the pool. PRs a session still owns are skipped. Writes use ``$KLAUD_REQUEUE_TOKEN`` and
    return the closed PR numbers.
    """
    stale = []
    for pull in pulls:
        candidate = finished(repository, pull)
        if not candidate or now - utc(pull["created_at"]) < timedelta(days=STALE_DAYS):
            continue
        prefix, image = candidate.split("-")
        if offered.get(prefix) and image not in offered[prefix]:
            stale.append((pull, candidate))
    if not stale:
        return []
    try:
        owned = owned_candidates(repository, claimed)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"::warning::Klaud requeue skipped: ownership unavailable ({type(error).__name__})")
        return []
    token = os.environ["KLAUD_REQUEUE_TOKEN"]
    comment = translated(
        f"**requeued** · Open for over {STALE_DAYS} days while the published baseline of this "
        "family moved to a newer image. Closed with `klaud-requeue` so a new candidate can "
        "refresh the family from its current image.",
        f"**requeued** · 本 PR 已开放超过 {STALE_DAYS} 天，而该配置族已发布的基线已更新为"
        "更新的镜像。已关闭并标记 `klaud-requeue`，新的候选可以基于当前镜像更新该配置族。",
    )
    closed = []
    for pull, candidate in stale:
        if candidate in owned:
            continue
        number = pull["number"]
        try:
            github.write(
                repository,
                f"issues/{number}/labels",
                "POST",
                {"labels": [REQUEUE_LABEL]},
                token=token,
            )
            github.write(
                repository, f"issues/{number}/comments", "POST", {"body": comment}, token=token
            )
            github.write(repository, f"pulls/{number}", "PATCH", {"state": "closed"}, token=token)
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            print(f"::warning::Klaud requeue of PR #{number} failed: {type(error).__name__}")
            continue
        print(f"Klaud: requeued superseded PR #{number}.")
        closed.append(number)
    return closed
