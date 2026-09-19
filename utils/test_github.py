from __future__ import annotations

import json
import os
import subprocess
from urllib.parse import parse_qs, urlsplit

import pytest

from infx import github


@pytest.mark.parametrize("token,expected", [(None, "admin"), ("workflow-token", "write")])
def test_explicit_credentials_override_inherited_auth_without_changing_it(monkeypatch, token, expected):
    monkeypatch.setenv("GH_TOKEN", "local-token")
    monkeypatch.setenv("GITHUB_TOKEN", "other-token")
    monkeypatch.setenv("GH_HOST", "enterprise.example")

    def run(args, **kwargs):
        env = kwargs["env"] if kwargs["env"] is not None else os.environ
        host = args[args.index("--hostname") + 1] if "--hostname" in args else env["GH_HOST"]
        identities = {("github.com", "workflow-token"): "write",
                      ("enterprise.example", "local-token"): "admin"}
        role = identities[(host, env["GH_TOKEN"])]
        assert "workflow-token" not in args
        return subprocess.CompletedProcess(args, 0, json.dumps({"role_name": role}), "")

    monkeypatch.setattr(github.subprocess, "run", run)
    assert github.api("example/project", "/collaborators/alice/permission", token) == {
        "role_name": expected,
    }
    assert os.environ["GH_TOKEN"] == "local-token"
    assert os.environ["GH_HOST"] == "enterprise.example"


@pytest.mark.parametrize("token", ["", " \n"])
def test_empty_explicit_token_cannot_fall_back_to_local_credentials(monkeypatch, token):
    def run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, '{"permission": "admin"}', "")

    monkeypatch.setattr(github.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="non-empty token"):
        github.api("example/project", "/collaborators/alice/permission", token)


def test_reaction_transport_sends_json_and_handles_empty_delete_response(monkeypatch):
    reactions = {}

    def run(args, **kwargs):
        endpoint = next(arg for arg in args if arg.startswith("repos/"))
        method = args[args.index("--method") + 1]
        if method == "POST":
            reactions[endpoint + "/51"] = json.loads(kwargs["input"])["content"]
            return subprocess.CompletedProcess(args, 0, '{"id": 51}', "")
        if method == "DELETE":
            del reactions[endpoint]
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(github.subprocess, "run", run)
    api = github.api
    assert api("example/project", "/issues/comments/41/reactions", "token",
               method="POST", data={"content": "+1"}) == {"id": 51}
    assert reactions == {"repos/example/project/issues/comments/41/reactions/51": "+1"}
    assert api("example/project", "/issues/comments/41/reactions/51", "token", method="DELETE") is None
    assert reactions == {}
    with pytest.raises(json.JSONDecodeError):
        api("example/project", "/pulls/7", "token")


@pytest.mark.parametrize("content", [None, "+1", "-1"])
def test_reaction_replacement_preserves_humans_and_unmanaged_bot_reactions(monkeypatch, content):
    reactions = [
        {"id": 11, "content": "+1", "user": {"login": "maintainer"}},
        {"id": 12, "content": "heart", "user": {"login": "github-actions[bot]"}},
        {"id": 13, "content": "+1", "user": {"login": "github-actions[bot]"}},
        {"id": 14, "content": "-1", "user": {"login": "github-actions[bot]"}},
    ]

    def run(args, **kwargs):
        endpoint = next(arg for arg in args if arg.startswith("repos/"))
        method = args[args.index("--method") + 1]
        if method == "GET":
            payload = [reactions] if "--slurp" in args else reactions
            return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")
        if method == "DELETE":
            reactions[:] = [r for r in reactions if r["id"] != int(endpoint.rsplit("/", 1)[1])]
        else:
            reactions.append({"id": 15, "content": json.loads(kwargs["input"])["content"],
                              "user": {"login": "github-actions[bot]"}})
        return subprocess.CompletedProcess(args, 0, "{}", "")

    monkeypatch.setattr(github.subprocess, "run", run)
    github.set_comment_reaction("example/project", 7, "token", content, replace=("+1", "-1"))
    expected = [(11, "+1"), (12, "heart")]
    if content is not None:
        expected.append((15, content))
    assert [(r["id"], r["content"]) for r in reactions] == expected


def test_pagination_reads_short_linked_pages_without_losing_filters(monkeypatch):
    def run(args, **kwargs):
        endpoint = next(arg for arg in args if arg.startswith("repos/"))
        query = parse_qs(urlsplit(endpoint).query)
        assert query == {"filter": ["all"], "branch": ["feature/space & +测试"],
                         "page": ["1"], "per_page": ["100"]}
        assert args[args.index("--method") + 1] == "GET"
        pages = [{"artifacts": [{"id": 7}], "total_count": 2},
                 {"artifacts": [{"id": 9}], "total_count": 2}]
        output = pages if "--paginate" in args and "--slurp" in args else pages[0]
        return subprocess.CompletedProcess(args, 0, json.dumps(output), "")

    monkeypatch.setattr(github.subprocess, "run", run)
    assert github.paginate("example/project", "/actions/artifacts?filter=all&per_page=1",
                           "token", "artifacts",
                           {"branch": "feature/space & +测试", "page": "9"}) == [{"id": 7}, {"id": 9}]


@pytest.mark.parametrize("payload,message", [
    (None, "unexpected shape"),
    ({}, "unexpected shape"),
    ({"jobs": None}, "unexpected shape"),
    ({"jobs": {}}, "unexpected shape"),
    ({"jobs": [None], "total_count": 1}, "unexpected shape"),
    ({"jobs": [], "total_count": 1}, "Incomplete GitHub listing"),
    ({"jobs": []}, "Invalid GitHub listing count"),
    ({"jobs": [], "total_count": True}, "Invalid GitHub listing count"),
    ({"jobs": [], "total_count": -1}, "Invalid GitHub listing count"),
    ({"jobs": [], "total_count": "0"}, "Invalid GitHub listing count"),
])
def test_listing_rejects_invalid_or_incomplete_pages(monkeypatch, payload, message):
    monkeypatch.setattr(github.subprocess, "run", lambda args, **kwargs:
                        subprocess.CompletedProcess(args, 0, json.dumps([payload]), ""))
    with pytest.raises(github.ListingError, match=message):
        github.paginate("example/project", "jobs", item_key="jobs")


def test_smaller_later_count_cannot_hide_missing_items(monkeypatch):
    pages = [{"jobs": [{"id": 7}], "total_count": 3},
             {"jobs": [{"id": 9}], "total_count": 1}]
    monkeypatch.setattr(github.subprocess, "run", lambda args, **kwargs:
                        subprocess.CompletedProcess(args, 0, json.dumps(pages), ""))
    with pytest.raises(github.ListingError, match="Incomplete GitHub listing"):
        github.paginate("example/project", "jobs", item_key="jobs")


@pytest.mark.parametrize("pages,key,expected", [
    ([[]], "", []),
    ([{"jobs": [], "total_count": 0}], "jobs", []),
    ([[{"id": 3}], [{"id": 5}]], "", [{"id": 3}, {"id": 5}]),
])
def test_pagination_accepts_empty_and_bare_array_endpoints(monkeypatch, pages, key, expected):
    monkeypatch.setattr(github.subprocess, "run", lambda args, **kwargs:
                        subprocess.CompletedProcess(args, 0, json.dumps(pages), ""))
    assert github.paginate("example/project", "items", item_key=key) == expected


@pytest.mark.parametrize("payload", [[], {}, None])
def test_pagination_rejects_missing_page_envelope(monkeypatch, payload):
    monkeypatch.setattr(github.subprocess, "run", lambda args, **kwargs:
                        subprocess.CompletedProcess(args, 0, json.dumps(payload), ""))
    with pytest.raises(github.ListingError, match="Missing GitHub listing"):
        github.paginate("example/project", "items")
