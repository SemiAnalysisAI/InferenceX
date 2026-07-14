#!/usr/bin/env python3
"""Verify that the active skip_queue label was applied by a trusted team member."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml


class GitHubApi:
    def __init__(self, token: str, api_url: str = "https://api.github.com"):
        self.token = token
        self.api_url = api_url.rstrip("/")

    def request(self, path: str) -> Any:
        request = urllib.request.Request(
            f"{self.api_url}{path}",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"GitHub API GET {path} failed: {error.code} {detail}") from error

    def paged(self, path: str) -> list[dict[str, Any]]:
        separator = "&" if "?" in path else "?"
        values = []
        page = 1
        while True:
            batch = self.request(f"{path}{separator}per_page=100&page={page}")
            values.extend(batch)
            if len(batch) < 100:
                return values
            page += 1


def active_label_actor(events: list[dict[str, Any]], label_name: str) -> str | None:
    """Return who applied the currently active instance of a label."""
    actor = None
    for event in events:
        if event.get("label", {}).get("name") != label_name:
            continue
        if event.get("event") == "labeled":
            actor = event.get("actor", {}).get("login")
        elif event.get("event") == "unlabeled":
            actor = None
    return actor


def is_authorized(
    api: GitHubApi,
    *,
    repository: str,
    pr_number: int,
    organization: str,
    team_slug: str,
    label_name: str,
) -> tuple[bool, str | None]:
    events = api.paged(f"/repos/{repository}/issues/{pr_number}/timeline")
    actor = active_label_actor(events, label_name)
    if not actor:
        return False, None

    membership = api.request(
        f"/orgs/{organization}/teams/{team_slug}/memberships/{actor}"
    )
    return membership.get("state") == "active", actor


def load_skip_policy(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as policy_file:
        policy = yaml.safe_load(policy_file)
    return policy["labels"]["skip-queue"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--pr-number", required=True, type=int)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).parents[1] / "configs" / "ci-priority.yaml",
    )
    parser.add_argument("--organization")
    parser.add_argument("--team-slug")
    parser.add_argument("--label")
    parser.add_argument("--token-env", default="REPO_PAT")
    parser.add_argument("--api-url", default="https://api.github.com")
    args = parser.parse_args()
    skip_policy = load_skip_policy(args.policy)
    organization = args.organization or skip_policy["organization"]
    team_slug = args.team_slug or skip_policy["team-slug"]
    label_name = args.label or skip_policy["name"]

    token = os.environ.get(args.token_env)
    if not token:
        print(f"::warning::{args.token_env} is unavailable; refusing skip_queue authorization", file=sys.stderr)
        print("false")
        return 0

    try:
        authorized, actor = is_authorized(
            GitHubApi(token, args.api_url),
            repository=args.repository,
            pr_number=args.pr_number,
            organization=organization,
            team_slug=team_slug,
            label_name=label_name,
        )
    except RuntimeError as error:
        print(f"::warning::{error}; refusing skip_queue authorization", file=sys.stderr)
        print("false")
        return 0

    if authorized:
        print(f"::notice::skip_queue authorized by {actor}", file=sys.stderr)
    elif actor:
        print(
            f"::warning::skip_queue was applied by {actor}, who is not an active "
            f"{organization}/{team_slug} member",
            file=sys.stderr,
        )
    print("true" if authorized else "false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
