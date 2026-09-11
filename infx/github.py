"""GitHub REST and comment-reaction primitives for internal automation."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Collection
from typing import Any

API_BASE = "https://api.github.com"


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
    request = urllib.request.Request(
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
        with urllib.request.urlopen(request, timeout=30) as response:
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
    out: list[dict[str, Any]] = []
    page = 1
    while True:
        page_params = {"per_page": "100", "page": str(page)}
        if params:
            page_params.update(params)
        data = api(repo, path, token, page_params)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get(item_key, [])
        else:
            items = []
        if not isinstance(items, list):
            raise RuntimeError(f"GitHub API {path} returned an unexpected shape")
        out.extend(items)
        if len(items) < 100:
            return out
        page += 1


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
