"""Select reuse evidence without replacing an already authorized source run."""

from __future__ import annotations

import argparse
import json
import subprocess
from typing import Any

from infx.workflows.reuse import DEFAULT_ALLOWED_AUTHOR_ASSOCIATIONS, parse_reuse_command


def gh(path: str, *, paginated: bool = False) -> Any:
    args = ["gh", "api", path]
    if paginated:
        args.extend(("--paginate", "--slurp"))
    return json.loads(subprocess.run(args, check=True, text=True, capture_output=True).stdout)


def select_source(repo: str, pr_number: int, branch: str, explicit: int | None = None) -> int:
    comments = [
        item
        for page in gh(f"repos/{repo}/issues/{pr_number}/comments?per_page=100", paginated=True)
        for item in page
    ]
    comments.sort(key=lambda value: (value["created_at"], value["id"]), reverse=True)
    authorized = None
    for comment in comments:
        if comment.get("author_association") not in DEFAULT_ALLOWED_AUTHOR_ASSOCIATIONS:
            continue
        matches, pinned = parse_reuse_command(comment.get("body", ""))
        if matches:
            authorized = pinned
            break
    if explicit is not None and authorized is not None and explicit != authorized:
        raise ValueError("explicit source conflicts with the existing maintainer authorization")
    selected = authorized if authorized is not None else explicit
    shas = {
        item["sha"]
        for page in gh(f"repos/{repo}/pulls/{pr_number}/commits?per_page=100", paginated=True)
        for item in page
    }
    if selected is None:
        from urllib.parse import quote

        runs = [
            item
            for page in gh(
                f"repos/{repo}/actions/workflows/run-sweep.yml/runs?event=pull_request&branch={quote(branch, safe='')}&status=completed&per_page=100",
                paginated=True,
            )
            for item in page["workflow_runs"]
        ]
    else:
        runs = [gh(f"repos/{repo}/actions/runs/{selected}")]
    for run in runs:
        allowed = {"success", "failure", "cancelled"} if authorized is not None else {"success"}
        valid = (
            run.get("event") == "pull_request"
            and run.get("status") == "completed"
            and run.get("conclusion") in allowed
            and run.get("head_sha") in shas
            and run.get("path", "").split("@", 1)[0] == ".github/workflows/run-sweep.yml"
        )
        if not valid:
            if selected is not None:
                raise ValueError(
                    "authorized source is no longer eligible; refusing to substitute another run"
                )
            continue
        artifacts = [
            item
            for page in gh(
                f"repos/{repo}/actions/runs/{run['id']}/artifacts?per_page=100", paginated=True
            )
            for item in page["artifacts"]
        ]
        if any(
            not item["expired"]
            and item["name"].startswith(("results_bmk", "eval_results_all", "bmk_agentic_"))
            for item in artifacts
        ):
            return int(run["id"])
        if selected is not None:
            raise ValueError(
                "authorized source artifacts are unavailable; refusing to substitute another run"
            )
    raise ValueError("no eligible successful sweep exists for this pull request")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--source-run", type=int)
    args = parser.parse_args()
    print(select_source(args.repo, args.pr, args.branch, args.source_run))


if __name__ == "__main__":
    main()
