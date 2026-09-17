"""GitHub REST and comment-reaction primitives for internal automation."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Collection
from typing import Any
from urllib.parse import parse_qsl, urlencode


class ListingError(RuntimeError):
    """A fixed failure reason that contains no API response data."""


def api(
    repo: str,
    path: str,
    token: str | None = None,
    params: dict[str, str] | None = None,
    *,
    method: str = "GET",
    data: dict[str, Any] | None = None,
    paginate: bool = False,
) -> Any:
    if token is not None and not token.strip():
        raise RuntimeError("GitHub API requires a non-empty token")
    endpoint, _, query = path.partition("?")
    query = urlencode({**dict(parse_qsl(query, keep_blank_values=True)), **(params or {})})
    endpoint = f"repos/{repo}/{endpoint.lstrip('/')}" + (f"?{query}" if query else "")
    args = ["gh", "api", "--method", method, endpoint]
    if token is not None:
        args.extend(
            [
                "--hostname",
                "github.com",
                "--header",
                "Accept: application/vnd.github+json",
                "--header",
                "X-GitHub-Api-Version: 2022-11-28",
            ]
        )
    if paginate:
        args.extend(["--paginate", "--slurp"])
    send_data = method != "GET" or data is not None
    if send_data:
        args.extend(["--input", "-"])
    try:
        result = subprocess.run(
            args,
            input=json.dumps(data or {}) if send_data else None,
            env={**os.environ, "GH_TOKEN": token} if token is not None else None,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        if token is None:
            raise
        raise RuntimeError(f"GitHub API {path} failed: {exc.stderr.strip()}") from exc
    if method != "GET" and not result.stdout.strip() and token is None:
        return {}
    if method == "DELETE" and not result.stdout and token is not None:
        return None
    return json.loads(result.stdout)


def paginate(
    repo: str,
    path: str,
    token: str | None = None,
    item_key: str = "",
    params: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    pages = api(
        repo, path, token, {**(params or {}), "per_page": "100", "page": "1"}, paginate=True
    )
    if not isinstance(pages, list) or not pages:
        raise ListingError("Missing GitHub listing")
    rows = []
    expected = 0
    for page in pages:
        items = page.get(item_key) if isinstance(page, dict) else page
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise ListingError("GitHub listing returned an unexpected shape")
        rows.extend(items)
        if item_key:
            total = page.get("total_count") if isinstance(page, dict) else None
            if type(total) is not int or total < 0:
                raise ListingError("Invalid GitHub listing count")
            expected = max(expected, total)
    if expected > len(rows):
        raise ListingError("Incomplete GitHub listing")
    return rows


def set_comment_reaction(
    repo: str,
    comment_id: int,
    token: str,
    content: str | None,
    *,
    replace: Collection[str] = (),
) -> None:
    """Replace selected github-actions reactions while preserving human reactions.

    With no replacement set, simply add the requested reaction. GitHub makes
    repeated additions of the same reaction idempotent.
    """
    path = f"/issues/comments/{comment_id}/reactions"
    if replace:
        reactions = paginate(repo, path, token, "")
        for reaction in reactions:
            if (
                reaction.get("user", {}).get("login") == "github-actions[bot]"
                and reaction.get("content") in replace
            ):
                api(repo, f"{path}/{reaction['id']}", token, method="DELETE")
    if content is not None:
        api(repo, path, token, method="POST", data={"content": content})
