"""Resolve trusted CODEOWNER verification metadata and requester authorization."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

from infx import github
from infx.workflows.reuse import write_outputs


def _positive_id(value: Any) -> int:
    if isinstance(value, bool) or not re.fullmatch(r"[1-9][0-9]*", str(value)):
        raise ValueError("Expected a positive GitHub resource ID")
    return int(value)


def _reference(
    event_name: str, event: dict[str, Any], repo: str, token: str
) -> tuple[int, str, str, str, str]:
    if event_name == "workflow_dispatch":
        inputs = event["inputs"]
        match = re.fullmatch(
            rf"https://github\.com/{re.escape(repo)}/pull/([1-9][0-9]*)"
            r"(?:/files)?#(issuecomment-|pullrequestreview-|discussion_r)([1-9][0-9]*)",
            inputs.get("comment_url", ""),
            re.IGNORECASE,
        )
        if match is None or _positive_id(inputs["pr-number"]) != int(match[1]):
            raise ValueError("comment_url must identify a sign-off on pr-number in this repository")
        number, fragment, ref_id = int(match[1]), match[2].lower(), int(match[3])
        signoff_key = f"{fragment}{ref_id}"
        if fragment == "issuecomment-":
            kind, path = "conversation comment", f"/issues/comments/{ref_id}"
        elif fragment == "pullrequestreview-":
            kind, path = "review summary", f"/pulls/{number}/reviews/{ref_id}"
        else:
            kind, path = "inline review comment", f"/pulls/comments/{ref_id}"
        comment = github.api(repo, path, token)
        # Comment IDs are repository-wide; the URL's PR number alone is not evidence.
        if kind != "review summary":
            field, resource = (
                ("issue_url", "issues")
                if fragment == "issuecomment-"
                else ("pull_request_url", "pulls")
            )
            if comment.get(field, "").lower() != (
                f"https://api.github.com/repos/{repo}/{resource}/{number}".lower()
            ):
                raise ValueError("Sign-off comment does not belong to the requested pull request")
        author = comment["user"]["login"]
    else:
        number = _positive_id(
            event["issue"]["number"]
            if event_name == "issue_comment"
            else event["pull_request"]["number"]
        )
        comment = event["review"] if event_name == "pull_request_review" else event["comment"]
        ref_id = _positive_id(comment["id"])
        author = comment["user"]["login"]
        if event_name == "issue_comment":
            kind, path = "conversation comment", f"/issues/comments/{ref_id}"
            signoff_key = f"issuecomment-{ref_id}"
        elif event_name == "pull_request_review":
            kind, path = "review summary", f"/pulls/{number}/reviews/{ref_id}"
            signoff_key = f"pullrequestreview-{ref_id}"
        else:
            kind, path = "inline review comment", f"/pulls/comments/{ref_id}"
            signoff_key = f"discussion_r{ref_id}"
    if not isinstance(author, str) or not re.fullmatch(r"[A-Za-z0-9-]+(?:\[bot\])?", author):
        raise ValueError("Invalid sign-off author")
    return number, author, kind, signoff_key, f"gh api repos/{repo}{path} --jq .body"


def resolve(
    repo: str,
    event_name: str,
    event: dict[str, Any],
    actor: str,
    scoped_head_sha: str,
    token: str,
) -> dict[str, str]:
    if event_name not in {
        "issue_comment",
        "pull_request_review",
        "pull_request_review_comment",
        "workflow_dispatch",
    }:
        return {"proceed": "false"}
    if event_name != "workflow_dispatch":
        expected_actions = (
            {"submitted", "edited"}
            if event_name == "pull_request_review"
            else {"created", "edited"}
        )
        comment = event.get("review" if event_name == "pull_request_review" else "comment") or {}
        if event.get("action") not in expected_actions or "As a PR reviewer and CODEOWNER" not in (
            comment.get("body") or ""
        ):
            return {"proceed": "false"}
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Invalid GitHub repository")
    number, author, kind, signoff_key, fetch_cmd = _reference(event_name, event, repo, token)
    pr = github.api(repo, f"/pulls/{number}", token)
    if pr["head"]["sha"] != scoped_head_sha:
        raise RuntimeError(
            "PR head changed while determining sign-off scope; retry on the current head."
        )
    if event_name != "workflow_dispatch" and (pr["state"] != "open" or pr["draft"]):
        return {"proceed": "false"}
    permission = github.api(repo, f"/collaborators/{quote(actor, safe='')}/permission", token)
    if (
        permission.get("permission") not in {"admin", "write"}
        or permission.get("role_name") not in {"admin", "maintain", "write"}
        or actor.endswith("[bot]")
        or (event.get("sender") or {}).get("type") == "Bot"
    ):
        return {"proceed": "false"}
    return {
        "proceed": "true",
        "pr-number": str(number),
        "head-sha": pr["head"]["sha"],
        "signoff-author": author,
        "signoff-kind": kind,
        "signoff-key": signoff_key,
        "signoff-fetch-cmd": fetch_cmd,
    }


def main() -> None:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    outputs = resolve(
        os.environ["GITHUB_REPOSITORY"],
        os.environ["GITHUB_EVENT_NAME"],
        event,
        os.environ["GITHUB_ACTOR"],
        os.environ["SCOPED_HEAD_SHA"],
        os.environ["GH_TOKEN"],
    )
    write_outputs(os.environ["GITHUB_OUTPUT"], outputs)


if __name__ == "__main__":
    main()
