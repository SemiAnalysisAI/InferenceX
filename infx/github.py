"""GitHub REST and comment-reaction primitives for internal automation."""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Collection, Iterable
from itertools import count
from typing import Any

API_BASE = "https://api.github.com"


class ListingError(RuntimeError):
    """A fixed failure reason that contains no API response data."""


def cli_api(
    repo: str,
    path: str,
    *,
    method: str = "GET",
    data: dict[str, Any] | None = None,
    paginate: bool = False,
) -> Any:
    args = ["gh", "api", "--method", method, f"repos/{repo}/{path.lstrip('/')}"]
    if paginate:
        args.extend(["--paginate", "--slurp"])
    if method != "GET":
        args.extend(["--input", "-"])
    result = subprocess.run(
        args,
        input=json.dumps(data or {}) if method != "GET" else None,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return json.loads(result.stdout) if method == "GET" or result.stdout.strip() else {}


def _page_items(data: Any, item_key: str) -> list[dict[str, Any]]:
    items = data.get(item_key) if isinstance(data, dict) else data
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ListingError("GitHub listing returned an unexpected shape")
    return items


def _list_items(pages: Iterable[Any], item_key: str) -> list[dict[str, Any]]:
    rows = []
    expected = 0
    for page in pages:
        rows.extend(_page_items(page, item_key))
        if item_key:
            total = page.get("total_count") if isinstance(page, dict) else None
            if type(total) is not int or total < 0:
                raise ListingError("Invalid GitHub listing count")
            expected = max(expected, total)
    if expected > len(rows):
        raise ListingError("Incomplete GitHub listing")
    return rows


def cli_paginate(repo: str, path: str, item_key: str) -> list[dict[str, Any]]:
    pages = cli_api(repo, path, paginate=True)
    if not isinstance(pages, list) or not pages:
        raise ListingError("Missing GitHub listing")
    return _list_items(pages, item_key)


def api(
    repo: str,
    path: str,
    token: str,
    params: dict[str, str] | None = None,
    *,
    method: str = "GET",
    data: dict[str, Any] | None = None,
) -> Any:
    """Call the GitHub REST API and return decoded JSON."""
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    request = urllib.request.Request(  # noqa: S310
        f"{API_BASE}/repos/{repo}{path}{query}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
        method=method,
        data=json.dumps(data).encode("utf-8") if data is not None else None,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            body = response.read().decode("utf-8")
            return None if method == "DELETE" and not body else json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {path} failed: HTTP {exc.code}: {body}") from exc


def paginate(
    repo: str,
    path: str,
    token: str,
    item_key: str,
    params: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch all pages from a GitHub REST list endpoint."""

    def pages() -> Iterable[Any]:
        for page in count(1):
            page_params = {**(params or {}), "per_page": "100", "page": str(page)}
            data = api(repo, path, token, page_params)
            yield data
            if len(_page_items(data, item_key)) < 100:
                return

    return _list_items(pages(), item_key)


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
