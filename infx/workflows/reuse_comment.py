#!/usr/bin/env python3
"""Acknowledge reuse requests with reactions; never post a separate comment."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from .. import github
from . import reuse


def acknowledge(repo: str, event: dict[str, Any], token: str) -> int:
    if event.get("action") not in {"created", "edited"} or not event.get("issue", {}).get("pull_request"):
        return 0
    pr_number = int(event["issue"]["number"])
    comment_id = int(event["comment"]["id"])
    comment_path = f"/issues/comments/{comment_id}"
    comment = github.api(repo, comment_path, token)
    # A queued event must not acknowledge an edited version it did not validate.
    if any(comment.get(key) != event["comment"].get(key) for key in ("body", "updated_at")):
        return 0
    github.set_comment_reaction(repo, comment_id, token, None, replace=("+1", "-1"))
    body = str(comment.get("body") or "")
    if not re.search(r"(?m)^\s*/reuse-sweep-run(?:\s|$)", body):
        return 0  # Includes edits that remove the command and its old acknowledgment.

    try:
        allowed = set(reuse.DEFAULT_ALLOWED_AUTHOR_ASSOCIATIONS)
        if comment.get("author_association") not in allowed:
            raise RuntimeError("Reuse requires an OWNER, MEMBER, or COLLABORATOR request.")
        matched, pinned_run_id = reuse.parse_reuse_command(body)
        if not matched:
            raise RuntimeError("Usage: /reuse-sweep-run [run_id]")
        selected, _ = reuse.find_reuse_request(repo, pr_number, token, "/reuse-sweep-run", allowed)
        if selected is None or selected.get("id") != comment_id:
            raise RuntimeError("A newer reuse request supersedes this comment.")
        pr = github.api(repo, f"/pulls/{pr_number}", token)
        if pr.get("state") != "open":
            raise RuntimeError(f"PR #{pr_number} is not open.")
        run = reuse.resolve_reusable_run(repo, "run-sweep.yml", pr_number, pr, token, pinned_run_id)

        latest = github.api(repo, comment_path, token)
        selected, _ = reuse.find_reuse_request(repo, pr_number, token, "/reuse-sweep-run", allowed)
        if (
            any(latest.get(key) != comment.get(key) for key in ("body", "updated_at"))
            or selected is None or selected.get("id") != comment_id
            or any(selected.get(key) != comment.get(key) for key in ("body", "updated_at"))
        ):
            github.set_comment_reaction(repo, comment_id, token, None, replace=("+1", "-1"))
            return 0
        github.set_comment_reaction(repo, comment_id, token, "+1", replace=("+1", "-1"))
        message = f"Reuse accepted / 复用请求已接受: PR #{pr_number}, run {run['id']} ({run['conclusion']})."
        code = 0
    except Exception as exc:
        latest = github.api(repo, comment_path, token)
        if any(latest.get(key) != comment.get(key) for key in ("body", "updated_at")):
            return 0
        github.set_comment_reaction(repo, comment_id, token, "-1", replace=("+1", "-1"))
        message = f"Reuse rejected / 复用请求未接受: {exc}"
        code = 1
    print(message)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    return code


def main() -> int:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GH_TOKEN or GITHUB_TOKEN is required")
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    return acknowledge(os.environ["GITHUB_REPOSITORY"], event, token)


if __name__ == "__main__":
    sys.exit(main())
