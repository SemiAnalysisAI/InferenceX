from __future__ import annotations

import io
import json
import subprocess

import pytest

from infx import github


def test_reaction_transport_sends_json_and_handles_empty_delete_response(monkeypatch):
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        return io.BytesIO(b'{"id": 51}' if request.method == "POST" else b"")

    monkeypatch.setattr(github.urllib.request, "urlopen", urlopen)
    api = github.api
    assert api("example/project", "/issues/comments/41/reactions", "token",
               method="POST", data={"content": "+1"}) == {"id": 51}
    assert api("example/project", "/issues/comments/41/reactions/51", "token", method="DELETE") is None
    assert requests[0].method == "POST"
    assert json.loads(requests[0].data) == {"content": "+1"}
    assert requests[1].method == "DELETE"
    assert requests[1].full_url.endswith("/issues/comments/41/reactions/51")
    with pytest.raises(json.JSONDecodeError):
        api("example/project", "/pulls/7", "token")



@pytest.mark.parametrize("content", [None, "+1", "-1"])
def test_reaction_replacement_preserves_humans_and_unmanaged_bot_reactions(monkeypatch, content):
    calls = []

    def api(repo, path, token, params=None, *, method="GET", data=None):
        calls.append((method, path, data))
        if method == "GET":
            return [
                {"id": 11, "content": "+1", "user": {"login": "maintainer"}},
                {"id": 12, "content": "heart", "user": {"login": "github-actions[bot]"}},
                {"id": 13, "content": "+1", "user": {"login": "github-actions[bot]"}},
                {"id": 14, "content": "-1", "user": {"login": "github-actions[bot]"}},
            ]
        return None

    monkeypatch.setattr(github, "api", api)
    github.set_comment_reaction("example/project", 7, "token", content, replace=("+1", "-1"))
    expected = [
        ("GET", "/issues/comments/7/reactions", None),
        ("DELETE", "/issues/comments/7/reactions/13", None),
        ("DELETE", "/issues/comments/7/reactions/14", None),
    ]
    if content is not None:
        expected.append(("POST", "/issues/comments/7/reactions", {"content": content}))
    assert calls == expected


@pytest.mark.parametrize("item_key", ["", "artifacts"])
def test_pagination_reads_following_pages_without_losing_filters(monkeypatch, item_key):
    pages = []
    first_page = [{"id": value} for value in range(100)]
    last_page = [{"id": 100}]

    def api(repo, path, token, params):
        pages.append(params)
        data = first_page if params["page"] == "1" else last_page
        return {item_key: data, "total_count": 101} if item_key else data

    monkeypatch.setattr(github, "api", api)
    result = github.paginate(
        "example/project", "/items", "token", item_key,
        {"branch": "feature", "page": "9", "per_page": "1"},
    )
    assert len(result) == 101
    assert result[0] == {"id": 0} and result[100] == {"id": 100}
    assert pages == [
        {"per_page": "100", "page": "1", "branch": "feature"},
        {"per_page": "100", "page": "2", "branch": "feature"},
    ]


@pytest.mark.parametrize("payload", [None, {}, {"jobs": None}, {"jobs": {}}, [None]])
def test_pagination_rejects_malformed_responses(monkeypatch, payload):
    monkeypatch.setattr(
        github.urllib.request, "urlopen",
        lambda request, timeout: io.BytesIO(json.dumps(payload).encode()),
    )
    with pytest.raises(RuntimeError, match="unexpected shape"):
        github.paginate("example/project", "/actions/runs/42/jobs", "token", "jobs")


@pytest.mark.parametrize("transport", ["token", "cli"])
@pytest.mark.parametrize("payload,message", [
    ({"jobs": [], "total_count": 1}, "Incomplete GitHub listing"),
    ({"jobs": []}, "Invalid GitHub listing count"),
    ({"jobs": [], "total_count": True}, "Invalid GitHub listing count"),
    ({"jobs": [], "total_count": -1}, "Invalid GitHub listing count"),
    ({"jobs": [], "total_count": "0"}, "Invalid GitHub listing count"),
    ({"jobs": [None], "total_count": 1}, "unexpected shape"),
])
def test_listing_rejects_invalid_or_incomplete_pages(monkeypatch, transport, payload, message):
    if transport == "token":
        monkeypatch.setattr(github.urllib.request, "urlopen",
                            lambda request, timeout: io.BytesIO(json.dumps(payload).encode()))
        fetch = lambda: github.paginate("example/project", "/jobs", "token", "jobs")
    else:
        monkeypatch.setattr(github.subprocess, "run", lambda args, **kwargs:
                            subprocess.CompletedProcess(args, 0, json.dumps([payload]), ""))
        fetch = lambda: github.cli_paginate("example/project", "jobs", "jobs")
    with pytest.raises(github.ListingError, match=message):
        fetch()


def test_cli_pagination_uses_all_linked_pages_even_when_first_page_is_short(monkeypatch):
    def run(args, **kwargs):
        pages = [{"artifacts": [{"id": 7}], "total_count": 2},
                 {"artifacts": [{"id": 9}], "total_count": 2}]
        output = pages if "--paginate" in args and "--slurp" in args else pages[0]
        return subprocess.CompletedProcess(args, 0, json.dumps(output), "")

    monkeypatch.setattr(github.subprocess, "run", run)
    assert github.cli_paginate("example/project", "actions/artifacts", "artifacts") == [
        {"id": 7}, {"id": 9},
    ]


@pytest.mark.parametrize("pages,key,expected", [
    ([[]], "", []),
    ([{"jobs": [], "total_count": 0}], "jobs", []),
    ([[{"id": 3}], [{"id": 5}]], "", [{"id": 3}, {"id": 5}]),
])
def test_cli_pagination_accepts_empty_and_bare_array_endpoints(monkeypatch, pages, key, expected):
    monkeypatch.setattr(github.subprocess, "run", lambda args, **kwargs:
                        subprocess.CompletedProcess(args, 0, json.dumps(pages), ""))
    assert github.cli_paginate("example/project", "items", key) == expected


@pytest.mark.parametrize("payload", [[], {}, None])
def test_cli_pagination_rejects_missing_page_envelope(monkeypatch, payload):
    monkeypatch.setattr(github.subprocess, "run", lambda args, **kwargs:
                        subprocess.CompletedProcess(args, 0, json.dumps(payload), ""))
    with pytest.raises(github.ListingError, match="Missing GitHub listing"):
        github.cli_paginate("example/project", "items", "")
