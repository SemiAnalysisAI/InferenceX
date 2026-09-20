"""Exercise resolver outputs and authorization against a controlled GitHub API."""

import json
import subprocess

import pytest

from infx import github
from infx.workflows import signoff_resolve

SHA = "a" * 40


@pytest.fixture
def service(monkeypatch):
    responses = {
        "/pulls/7": {"state": "open", "draft": False, "head": {"sha": SHA}},
        "/collaborators/writer/permission": {"permission": "write", "role_name": "write"},
        "/issues/comments/41": {
            "user": {"login": "reviewer"},
            "issue_url": "https://api.github.com/repos/example/repo/issues/7",
        },
        "/pulls/7/reviews/42": {"user": {"login": "reviewer"}},
        "/pulls/comments/43": {
            "user": {"login": "reviewer"},
            "pull_request_url": "https://api.github.com/repos/example/repo/pulls/7",
        },
    }

    def run(args, **kwargs):
        endpoint = next(arg for arg in args if arg.startswith("repos/"))
        path = endpoint.removeprefix("repos/example/repo")
        response = responses[path]
        if isinstance(response, Exception):
            raise response
        return subprocess.CompletedProcess(args, 0, json.dumps(response), "")

    monkeypatch.setattr(github.subprocess, "run", run)
    return responses


def event_for(name):
    comment = {"id": 41, "user": {"login": "reviewer"}}
    event = {
        "issue": {"number": 7},
        "pull_request": {"number": 7},
        "comment": comment,
        "review": {"id": 42, "user": {"login": "reviewer"}},
        "sender": {"type": "User"},
    }
    if name == "pull_request_review_comment":
        comment["id"] = 43
    if name == "workflow_dispatch":
        event["inputs"] = {
            "pr-number": "7",
            "comment_url": "https://github.com/example/repo/pull/7#issuecomment-41",
        }
    return event


def resolve(name="issue_comment", event=None, actor="writer"):
    return signoff_resolve.resolve(
        "example/repo", name, event if event is not None else event_for(name), actor, SHA, "token"
    )


@pytest.mark.parametrize("name,kind,path", [
    ("issue_comment", "conversation comment", "issues/comments/41"),
    ("pull_request_review", "review summary", "pulls/7/reviews/42"),
    ("pull_request_review_comment", "inline review comment", "pulls/comments/43"),
    ("workflow_dispatch", "conversation comment", "issues/comments/41"),
])
def test_event_metadata_and_fetch_commands(service, name, kind, path):
    assert resolve(name) == {
        "proceed": "true", "pr-number": "7", "head-sha": SHA,
        "signoff-author": "reviewer", "signoff-kind": kind,
        "signoff-fetch-cmd": f"gh api repos/example/repo/{path} --jq .body",
    }


@pytest.mark.parametrize("suffix,kind,path", [
    ("#pullrequestreview-42", "review summary", "pulls/7/reviews/42"),
    ("/files#discussion_r43", "inline review comment", "pulls/comments/43"),
])
def test_dispatch_supports_reviews_and_files_comments(service, suffix, kind, path):
    event = event_for("workflow_dispatch")
    event["inputs"]["comment_url"] = "https://github.com/example/repo/pull/7" + suffix
    result = resolve("workflow_dispatch", event)
    assert result["signoff-kind"] == kind
    assert result["signoff-fetch-cmd"] == f"gh api repos/example/repo/{path} --jq .body"


@pytest.mark.parametrize("url", [
    "https://github.com/other/repo/pull/7#issuecomment-41",
    "https://github.com/example/repo/pull/8#issuecomment-41",
    "https://evil.example/example/repo/pull/7#issuecomment-41",
    "https://github.com/example/repo/pull/7",
    "https://github.com/example/repo/pull/7#issuecomment-41;touch /tmp/unsafe",
])
def test_dispatch_rejects_wrong_repository_pr_or_malformed_url(service, url):
    event = event_for("workflow_dispatch")
    event["inputs"]["comment_url"] = url
    with pytest.raises(ValueError, match="comment_url"):
        resolve("workflow_dispatch", event)


@pytest.mark.parametrize("suffix,path,field", [
    ("#issuecomment-41", "/issues/comments/41", "issue_url"),
    ("#discussion_r43", "/pulls/comments/43", "pull_request_url"),
])
def test_dispatch_checks_comment_actually_belongs_to_pr(service, suffix, path, field):
    event = event_for("workflow_dispatch")
    event["inputs"]["comment_url"] = "https://github.com/example/repo/pull/7" + suffix
    service[path][field] = service[path][field][:-1] + "8"
    with pytest.raises(ValueError, match="does not belong"):
        resolve("workflow_dispatch", event)


@pytest.mark.parametrize("permission,role,expected", [
    ("admin", "admin", "true"),
    ("write", "maintain", "true"),
    ("write", "write", "true"),
    ("read", "read", "false"),
    ("triage", "triage", "false"),
    ("write", "custom-role", "false"),
    ("read", "admin", "false"),
])
def test_only_repository_writers_with_known_roles_can_request(service, permission, role, expected):
    service["/collaborators/writer/permission"] = {"permission": permission, "role_name": role}
    assert resolve()["proceed"] == expected


@pytest.mark.parametrize("actor,sender", [("writer[bot]", "User"), ("writer", "Bot")])
def test_bot_requesters_are_skipped_even_with_write_permission(service, actor, sender):
    service[f"/collaborators/{actor.replace('[', '%5B').replace(']', '%5D')}/permission"] = {
        "permission": "write", "role_name": "write",
    }
    event = event_for("issue_comment")
    event["sender"]["type"] = sender
    assert resolve(event=event, actor=actor) == {"proceed": "false"}


@pytest.mark.parametrize("state,draft", [("closed", False), ("open", True)])
def test_not_ready_pr_skips_events_but_allows_authorized_manual_verification(service, state, draft):
    service["/pulls/7"].update(state=state, draft=draft)
    assert resolve() == {"proceed": "false"}
    assert resolve("workflow_dispatch")["proceed"] == "true"


def test_head_movement_and_permission_lookup_failure_fail_closed(service):
    service["/pulls/7"]["head"]["sha"] = "b" * 40
    with pytest.raises(RuntimeError, match="PR head changed"):
        resolve()
    service["/pulls/7"]["head"]["sha"] = SHA
    service["/collaborators/writer/permission"] = subprocess.CalledProcessError(
        1, ["gh"], stderr="gh: Forbidden (HTTP 403)"
    )
    with pytest.raises(github.APIError):
        resolve()


def test_unsupported_event_cannot_proceed(service):
    assert resolve("push", {}) == {"proceed": "false"}


def test_untrusted_metadata_cannot_inject_outputs_or_commands(service):
    event = event_for("issue_comment")
    event["comment"]["user"]["login"] = "reviewer\nproceed=true"
    with pytest.raises(ValueError, match="author"):
        resolve(event=event)
    event = event_for("issue_comment")
    event["comment"]["id"] = "41; echo unsafe"
    with pytest.raises(ValueError, match="resource ID"):
        resolve(event=event)


def test_entrypoint_writes_real_actions_outputs(service, tmp_path, monkeypatch):
    event_path = tmp_path / "event.json"
    output = tmp_path / "outputs"
    event_path.write_text(json.dumps(event_for("pull_request_review")))
    for name, value in {
        "GITHUB_EVENT_PATH": str(event_path), "GITHUB_OUTPUT": str(output),
        "GITHUB_REPOSITORY": "example/repo", "GITHUB_EVENT_NAME": "pull_request_review",
        "GITHUB_ACTOR": "writer", "SCOPED_HEAD_SHA": SHA, "GH_TOKEN": "token",
    }.items():
        monkeypatch.setenv(name, value)
    signoff_resolve.main()
    assert output.read_text() == (
        f"proceed=true\npr-number=7\nhead-sha={SHA}\nsignoff-author=reviewer\n"
        "signoff-kind=review summary\n"
        "signoff-fetch-cmd=gh api repos/example/repo/pulls/7/reviews/42 --jq .body\n"
    )
