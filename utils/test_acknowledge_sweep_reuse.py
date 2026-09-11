from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from infx.workflows import reuse_comment as acknowledgment


def test_comment_entrypoint_runs_with_only_the_infx_package(tmp_path):
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "infx", tmp_path / "infx")
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({"action": "created", "issue": {"number": 7}}))
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env.update(GH_TOKEN="test-token", GITHUB_REPOSITORY="example/project",
               GITHUB_EVENT_PATH=str(event_path))
    run = subprocess.run(
        [sys.executable, "-m", "infx.workflows.reuse_comment"],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=10,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout == ""


@pytest.fixture
def request_case(monkeypatch):
    comment = {
        "id": 41, "body": "/reuse-sweep-run 123", "author_association": "MEMBER",
        "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
    }
    case = {
        "comment": comment,
        "comments": [comment],
        "pr": {"state": "open", "labels": [], "head": {"ref": "feature"}},
        "run": {"id": 123, "event": "pull_request", "status": "completed",
                "conclusion": "success", "path": ".github/workflows/run-sweep.yml",
                "head_sha": "tested-sha"},
        "commits": [{"sha": "tested-sha"}],
        "artifacts": [{"name": "results_bmk", "expired": False}],
        "reactions": [], "writes": [],
    }

    def api(repo, path, token, params=None, *, method="GET", data=None):
        assert repo == "example/project"
        assert token == "test-token"
        if case.get("fail_path") == path:
            raise RuntimeError("GitHub unavailable")
        if method == "POST":
            assert path == "/issues/comments/41/reactions"  # No separate comments.
            case["writes"].append(data["content"])
            case["reactions"].append({"id": len(case["writes"]) + 100,
                                      "content": data["content"],
                                      "user": {"login": "github-actions[bot]"}})
            return case["reactions"][-1]
        if method == "DELETE":
            assert path.startswith("/issues/comments/41/reactions/")
            reaction_id = int(path.rsplit("/", 1)[-1])
            case["reactions"][:] = [r for r in case["reactions"] if r["id"] != reaction_id]
            return None
        if path == "/issues/comments/41":
            return copy.deepcopy(case["comment"])
        if path == "/issues/comments/41/reactions":
            return list(case["reactions"])
        if path == "/issues/7/comments":
            return case["comments"]
        if path == "/pulls/7":
            return case["pr"]
        if path == "/pulls/7/commits":
            return case["commits"]
        if path == "/actions/runs/123":
            return case["run"]
        if path == "/actions/workflows/run-sweep.yml/runs":
            return {"workflow_runs": [case["run"]]}
        if path == "/actions/runs/123/artifacts":
            if callback := case.get("during_validation"):
                callback()
            return {"artifacts": case["artifacts"]}
        raise AssertionError((method, path))

    monkeypatch.setattr(acknowledgment.github, "api", api)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    return case


def event_for(case, **changes):
    return {"action": "created", "issue": {"number": 7, "pull_request": {"url": "https://api.github.com/repos/example/project/pulls/7"}},
            "comment": copy.deepcopy(case["comment"]), **changes}


def bot_status(case):
    return [r["content"] for r in case["reactions"] if r["user"]["login"] == "github-actions[bot]"]


@pytest.mark.parametrize("body", ["/reuse-sweep-run", "/reuse-sweep-run 123"])
@pytest.mark.parametrize("labels", [[], [{"name": "sweep-enabled"}]])
def test_accepts_valid_reuse_without_full_sweep_label(request_case, body, labels):
    case = request_case
    case["comment"]["body"] = body
    case["pr"]["labels"] = labels
    assert acknowledgment.acknowledge("example/project", event_for(case), "test-token") == 0
    assert case["writes"] == ["+1"]
    assert bot_status(case) == ["+1"]


@pytest.mark.parametrize("conclusion", ["failure", "cancelled"])
def test_explicit_partial_source_keeps_existing_acceptance(request_case, conclusion):
    request_case["run"]["conclusion"] = conclusion
    assert acknowledgment.acknowledge("example/project", event_for(request_case), "test-token") == 0
    assert bot_status(request_case) == ["+1"]


@pytest.mark.parametrize("problem,reason", [
    ("unauthorized", "OWNER, MEMBER, or COLLABORATOR"),
    ("syntax", "Usage:"),
    ("closed", "not open"),
    ("workflow", "expected .github/workflows/run-sweep.yml"),
    ("running", "not completed"),
    ("orphan", "not in PR #7's commit list"),
    ("empty", "no benchmark, eval, or agentic result artifact"),
    ("expired", "no benchmark, eval, or agentic result artifact"),
    ("modifier", "reuse-incompatible"),
    ("unavailable", "GitHub unavailable"),
    ("unpinned-failure", "no successful"),
])
def test_rejects_invalid_requests_without_approving(request_case, capsys, problem, reason):
    case = request_case
    if problem == "unauthorized":
        case["comment"]["author_association"] = "CONTRIBUTOR"
    elif problem == "syntax":
        case["comment"]["body"] = "/reuse-sweep-run nope"
    elif problem == "closed":
        case["pr"]["state"] = "closed"
    elif problem == "workflow":
        case["run"]["path"] = ".github/workflows/other.yml"
    elif problem == "running":
        case["run"]["status"] = "in_progress"
    elif problem == "orphan":
        case["commits"] = [{"sha": "different-sha"}]
    elif problem == "empty":
        case["artifacts"] = [{"name": "run-stats"}]
    elif problem == "expired":
        case["artifacts"][0]["expired"] = True
    elif problem == "modifier":
        case["pr"]["labels"] = [{"name": "agentx-fast"}]
    elif problem == "unavailable":
        case["fail_path"] = "/actions/runs/123"
    elif problem == "unpinned-failure":
        case["comment"]["body"] = "/reuse-sweep-run"
        case["run"]["conclusion"] = "failure"
    assert acknowledgment.acknowledge("example/project", event_for(case), "test-token") == 1
    assert case["writes"] == ["-1"]
    assert bot_status(case) == ["-1"]
    assert reason in capsys.readouterr().out


@pytest.mark.parametrize("body,status", [("withdrawn", []), ("/reuse-sweep-run nope", ["-1"])])
def test_edit_replaces_bot_status_and_preserves_human_reaction(request_case, body, status):
    case = request_case
    case["reactions"] = [
        {"id": 1, "content": "+1", "user": {"login": "github-actions[bot]"}},
        {"id": 2, "content": "+1", "user": {"login": "maintainer"}},
    ]
    case["comment"]["body"] = body
    acknowledgment.acknowledge("example/project", event_for(case, action="edited"), "test-token")
    assert bot_status(case) == status
    assert case["reactions"][0] == {"id": 2, "content": "+1", "user": {"login": "maintainer"}}


def test_stale_queued_event_does_not_change_reactions(request_case):
    event = event_for(request_case)
    request_case["comment"]["body"] = "/reuse-sweep-run 456"
    assert acknowledgment.acknowledge("example/project", event, "test-token") == 0
    assert request_case["writes"] == []


@pytest.mark.parametrize("change", ["edit", "newer-request"])
def test_changed_request_during_validation_is_not_approved(request_case, change):
    def update():
        if change == "edit":
            request_case["comment"]["body"] = "/reuse-sweep-run 456"
        else:
            request_case["comments"].append({**request_case["comment"], "id": 42,
                                             "created_at": "2026-01-02T00:00:00Z"})
    request_case["during_validation"] = update
    assert acknowledgment.acknowledge("example/project", event_for(request_case), "test-token") == 0
    assert request_case["writes"] == []
    assert bot_status(request_case) == []


def test_redelivery_leaves_one_acceptance_reaction(request_case):
    event = event_for(request_case)
    for _ in range(2):
        assert acknowledgment.acknowledge("example/project", event, "test-token") == 0
    assert bot_status(request_case) == ["+1"]


@pytest.mark.parametrize("change", ["issue", "unrelated", "inline-mention"])
def test_unrelated_activity_does_not_get_acknowledged(request_case, change):
    event = event_for(request_case)
    if change == "issue":
        event["issue"].pop("pull_request")
    else:
        request_case["comment"]["body"] = (
            "hello" if change == "unrelated" else "please use /reuse-sweep-run later"
        )
        event = event_for(request_case)
    assert acknowledgment.acknowledge("example/project", event, "test-token") == 0
    assert request_case["writes"] == []
