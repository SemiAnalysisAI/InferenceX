"""Check real policy, CLI and workflow effects against independent expectations."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from infx.workflows import authorize
from infx.workflows.github import PermissionLookupError, get_repository_permission
from test_run_sweep_gating import _Parser, _tokens

ROOT = Path(__file__).resolve().parents[2]
OPERATIONS = ("stage-results", "trusted-external-sweep")


@pytest.mark.parametrize("permission,role,tier,allowed", [
    ("admin", "admin", "MAINTAINER", True),
    ("write", "maintain", "MAINTAINER", True),
    ("maintain", "maintain", "MAINTAINER", True),
    ("write", "write", "COLLABORATOR", True),
    ("read", "triage", "PUBLIC", False),
    ("read", "read", "PUBLIC", False),
    ("none", "none", "PUBLIC", False),
    ("admin", "custom-admin", None, False),
    ("write", "custom-writer", None, False),
    ("read", "custom-reader", None, False),
    ("write", None, None, False),
    ("admin", "read", "PUBLIC", False),
    ("read", "admin", "MAINTAINER", False),
    ("", None, None, False),
])
def test_operation_access(permission, role, tier, allowed):
    for operation in OPERATIONS:
        decision = authorize(operation, permission, role)
        assert decision.allowed is allowed
        assert (decision.tier.name if decision.tier is not None else None) == tier


def test_unknown_operation_denies_even_an_admin():
    decision = authorize("misspelled-operation", "admin", "admin")
    assert not decision.allowed
    assert decision.reason == "unknown-operation"


def test_permission_reader_uses_workflow_token_and_effective_role(monkeypatch):
    run = Mock(return_value=subprocess.CompletedProcess([], 0, json.dumps({
        "permission": "write", "role_name": "maintain",
    })))
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("GH_TOKEN", "unrelated-personal-token")
    permission = get_repository_permission("example/repo", "robot[bot]", "workflow-token")
    assert (permission.role_name, permission.permission) == ("maintain", "write")
    args, kwargs = run.call_args
    assert args[0] == [
        "gh", "api", "--hostname", "github.com",
        "repos/example/repo/collaborators/robot%5Bbot%5D/permission",
    ]
    assert kwargs["env"]["GH_TOKEN"] == "workflow-token"
    assert 0 < kwargs["timeout"] <= 30


@pytest.mark.parametrize("response", [
    "not json", "null", "[]", '{}', '{"role_name":"admin"}',
    '{"permission":true,"role_name":"admin"}',
    '{"permission":null,"role_name":"admin"}',
    '{"permission":"admin"}', '{"permission":"admin","role_name":null}',
    '{"permission":"admin","role_name":true}', '{"permission":"admin","role_name":""}',
])
def test_malformed_permission_responses_never_grant_access(monkeypatch, response):
    monkeypatch.setattr(subprocess, "run", Mock(return_value=subprocess.CompletedProcess([], 0, response)))
    with pytest.raises(PermissionLookupError):
        get_repository_permission("example/repo", "requester", "workflow-token")


@pytest.mark.parametrize("error", [
    subprocess.CalledProcessError(1, "gh", stderr="private response body"),
    subprocess.TimeoutExpired("gh", 15),
    FileNotFoundError("gh"),
])
def test_lookup_errors_are_sanitized(monkeypatch, error):
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=error))
    with pytest.raises(PermissionLookupError, match="^repository-permission-lookup-failed$"):
        get_repository_permission("example/repo", "requester", "workflow-token")


@pytest.fixture
def cli_env(tmp_path):
    """Stub only gh; the real Python CLI and policy run in a child process."""
    gh = tmp_path / "gh"
    gh.write_text(
        '#!/bin/sh\n'
        'printf "%s\\n" "$*" >> "$TEST_GH_CALLS"\n'
        'printf "%s" "$TEST_PERMISSION"\n'
        'exit "${TEST_GH_STATUS:-0}"\n'
    )
    gh.chmod(0o755)
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({"action": "created", "comment": {"user": {"login": "commenter"}}}))
    return {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_TOKEN": "workflow-token",
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_EVENT_NAME": "issue_comment",
        "GITHUB_REPOSITORY": "example/repo",
        "GITHUB_ACTOR": "requester",
        "GITHUB_TRIGGERING_ACTOR": "admin-rerunner",
        "TEST_PYTHON": sys.executable,
        "TEST_GH_CALLS": str(tmp_path / "gh-calls"),
        "TEST_PERMISSION": '{"role_name":"write","permission":"write"}',
    }


def run_cli(env, operation="stage-results"):
    result = subprocess.run(
        [sys.executable, "-m", "infx.workflows", "authorize", operation],
        env=env, cwd=ROOT, capture_output=True, text=True, timeout=20,
    )
    return result.returncode, json.loads(result.stdout)


@pytest.mark.parametrize("operation,event_name,event,actor", [
    ("stage-results", "issue_comment", {"action": "created", "comment": {"user": {"login": "commenter"}}}, "commenter"),
    ("trusted-external-sweep", "pull_request_target", {"action": "labeled"}, "requester"),
])
def test_cli_uses_requester_not_rerunner_or_payload_attribution(cli_env, operation, event_name, event, actor):
    Path(cli_env["GITHUB_EVENT_PATH"]).write_text(json.dumps(event))
    cli_env.update(GITHUB_EVENT_NAME=event_name, TEST_PERMISSION='{"role_name":"maintain","permission":"write"}')
    code, result = run_cli(cli_env, operation)
    assert code == 0
    assert result["actor"] == actor
    assert result["tier"] == "MAINTAINER"
    calls = Path(cli_env["TEST_GH_CALLS"]).read_text().splitlines()
    assert calls == [f"api --hostname github.com repos/example/repo/collaborators/{actor}/permission"]


@pytest.mark.parametrize("change,event", [
    ({"GITHUB_TOKEN": ""}, None),
    ({"GITHUB_EVENT_NAME": "pull_request"}, None),
    ({"GITHUB_REPOSITORY": ""}, None),
    ({}, {}),
    ({}, {"action": "edited", "comment": {"user": {"login": "commenter"}}}),
    ({}, {"action": "created", "comment": {"user": {"login": ""}}}),
    ({}, []),
])
def test_invalid_context_cannot_query_or_authorize(cli_env, change, event):
    cli_env.update(change)
    if event is not None:
        Path(cli_env["GITHUB_EVENT_PATH"]).write_text(json.dumps(event))
    code, result = run_cli(cli_env)
    assert code == 2
    assert result["allowed"] is False
    assert not Path(cli_env["TEST_GH_CALLS"]).exists()


def test_unknown_cli_operation_denies_without_io(cli_env):
    code, result = run_cli(cli_env, "unknown")
    assert (code, result) == (1, {"allowed": False, "reason": "unknown-operation"})
    assert not Path(cli_env["TEST_GH_CALLS"]).exists()


def workflow(name, root=ROOT):
    return yaml.load((root / f".github/workflows/{name}.yml").read_text(), Loader=yaml.BaseLoader)


def scenario(operation):
    """Hand-built PR and artifact facts, independent of the production selectors."""
    pull = {
        "number": 42, "state": "open", "draft": False,
        "head": {"sha": "approved-head", "ref": "feature", "repo": {"full_name": "outside/repo"}},
        "base": {"sha": "base-head"}, "merge_commit_sha": "approved-merge",
        "labels": [{"name": "full-sweep-fail-fast"}],
    }
    payload = {
        "repository": {"full_name": "example/repo", "default_branch": "main"},
        "pull_request": copy.deepcopy(pull), "issue": {"number": 42, "pull_request": {}},
        "comment": {"body": "/stage-results", "user": {"login": "commenter"}},
        "label": {"name": "full-sweep-fail-fast"},
    }
    payload["action"] = "created" if operation == "stage-results" else "labeled"
    run = {
        "id": 101, "run_attempt": 2, "created_at": "2026-01-02T10:00:00Z",
        "head_sha": "approved-head", "status": "completed", "conclusion": "success",
        "path": ".github/workflows/run-sweep.yml", "event": "pull_request",
        "pull_requests": [{"number": 42}], "html_url": "https://example.test/runs/101",
    }
    return {
        "context": {"actor": "requester", "repo": {"owner": "example", "repo": "repo"},
                    "issue": {"number": 42}, "payload": payload},
        "permission": {"role_name": "write", "permission": "write"},
        "data": {
            "pull": pull, "commits": [{"sha": "approved-head"}], "runs": [run],
            "timeline": [{"event": "labeled", "label": {"name": "full-sweep-fail-fast"}, "created_at": "2026-01-01T00:00:00Z"}],
            "artifacts": {"101": [{"name": "changelog-metadata", "expired": False}, {"name": "results_bmk", "expired": False}]},
            "dispatchedRuns": [{"display_title": "e2e Test - External PR #42 @ approved-hea", "created_at": "2026-01-02T12:00:00Z", "html_url": "https://example.test/runs/303"}],
        },
    }


def run_workflow(operation, case, env, root=ROOT):
    job = next(iter(workflow(operation, root)["jobs"].values()))
    case = {**case, "steps": job["steps"]}
    payload = case["context"]["payload"]
    Path(env["GITHUB_EVENT_PATH"]).write_text(json.dumps(payload))
    env = {**env,
           "GITHUB_EVENT_NAME": {"stage-results": "issue_comment", "trusted-external-sweep": "pull_request_target"}[operation],
           "TEST_PERMISSION": json.dumps(case["permission"]),
           "TEST_GH_STATUS": "1" if case.get("permissionError") else "0"}
    result = subprocess.run(
        ["node", str(ROOT / "utils/changelog_gate_tests/workflow_script_runner.cjs")],
        input=json.dumps(case), env=env, cwd=ROOT, capture_output=True, text=True, timeout=30,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("role,legacy,allowed", [
    ("admin", "admin", True), ("maintain", "write", True), ("write", "write", True),
    ("triage", "read", False), ("read", "read", False), ("none", "none", False),
    ("custom-role", "admin", False), ("custom-role", "write", False), ("custom-role", "read", False),
    ("admin", "read", False), ("read", "admin", False),
])
def test_workflow_dispatch_requires_the_requested_repository_role(cli_env, operation, role, legacy, allowed):
    case = scenario(operation)
    case["permission"] = {"role_name": role, "permission": legacy}
    result = run_workflow(operation, case, cli_env)
    dispatches = [w for w in result["writes"] if w["method"] != "issues.createComment"]
    assert bool(dispatches) is allowed
    assert bool(result["failures"]) is not allowed
    if not allowed and operation == "stage-results":
        assert [w["body"] for w in result["writes"]] == [
            "@commenter `/stage-results` requires write access to this repository.",
        ]


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("permission,error", [
    ({"role_name": "admin", "permission": "admin"}, True),
    ({"role_name": "admin"}, False),
    ({"permission": "admin"}, False),
    ({"permission": "admin", "role_name": None}, False),
    ({"permission": "admin", "role_name": True}, False),
    ({"permission": "admin", "role_name": ""}, False),
])
def test_unavailable_role_never_reaches_protected_effects(cli_env, operation, permission, error):
    case = scenario(operation)
    case.update(permission=permission, permissionError=error)
    result = run_workflow(operation, case, cli_env)
    assert result["failures"]
    assert result["writes"] == []
    assert result["outputs"] == {}


def test_staging_keeps_source_outputs_and_dispatch_contract(cli_env):
    result = run_workflow("stage-results", scenario("stage-results"), cli_env)
    assert result["failures"] == []
    assert result["outputs"]["request"] == {
        "run-id": "101", "run-attempt": "2", "run-date": "2026-01-02",
        "requested-by": "commenter", "run-url": "https://example.test/runs/101",
    }
    assert result["writes"][-1] == {
        "method": "repos.createDispatchEvent", "owner": "SemiAnalysisAI", "repo": "InferenceX-app",
        "event_type": "stage-results", "client_payload": {
            "source-repository": "example/repo", "pr-number": "42", "run-id": "101",
            "run-attempt": "2", "run-date": "2026-01-02", "requested-by": "commenter", "comment-id": "501",
        },
    }


def test_invalid_staging_command_preserves_usage_reply_without_role_lookup(cli_env):
    case = scenario("stage-results")
    case["context"]["payload"]["comment"]["body"] = "/stage-results invalid"
    result = run_workflow("stage-results", case, cli_env)
    assert result["failures"] == ["Unsupported /stage-results syntax"]
    assert [w["body"] for w in result["writes"]] == ["Usage: `/stage-results` or `/stage-results <run-id>`."]
    assert not Path(cli_env["TEST_GH_CALLS"]).exists()


@pytest.mark.parametrize("field,value", [
    ("head_sha", "unrelated-head"), ("event", "push"), ("path", "other.yml"),
    ("status", "in_progress"), ("conclusion", "skipped"),
])
def test_staging_role_does_not_override_run_eligibility(cli_env, field, value):
    case = scenario("stage-results")
    case["data"]["runs"][0][field] = value
    result = run_workflow("stage-results", case, cli_env)
    assert result["failures"]
    assert result["writes"] == []


@pytest.mark.parametrize("conclusion", ["success", "failure", "cancelled"])
@pytest.mark.parametrize("artifact", ["results_bmk", "eval_results_all", "bmk_agentic_example"])
def test_staging_preserves_partial_results_for_each_supported_artifact(cli_env, conclusion, artifact):
    case = scenario("stage-results")
    case["data"]["runs"][0]["conclusion"] = conclusion
    case["data"]["artifacts"]["101"][1]["name"] = artifact
    result = run_workflow("stage-results", case, cli_env)
    assert not result["failures"]
    assert result["writes"][-1]["event_type"] == "stage-results"


@pytest.mark.parametrize("change", ["missing-metadata", "expired-results", "no-current-label", "no-historical-label"])
def test_staging_role_does_not_override_labels_or_artifacts(cli_env, change):
    case = scenario("stage-results")
    data = case["data"]
    if change == "missing-metadata":
        data["artifacts"]["101"].pop(0)
    elif change == "expired-results":
        data["artifacts"]["101"][1]["expired"] = True
    elif change == "no-current-label":
        data["pull"]["labels"] = []
    else:
        data["timeline"] = []
    result = run_workflow("stage-results", case, cli_env)
    assert result["failures"]
    assert all(w["method"] == "issues.createComment" for w in result["writes"])


@pytest.mark.parametrize("pinned,associated,expected", [(True, True, True), (True, False, False), (False, True, False)])
def test_historical_staging_association_requires_explicit_run_id(cli_env, pinned, associated, expected):
    case = scenario("stage-results")
    case["data"]["runs"][0].update(head_sha="historical-head", pull_requests=[{"number": 42 if associated else 43}])
    if pinned:
        case["context"]["payload"]["comment"]["body"] = "/stage-results 101"
    result = run_workflow("stage-results", case, cli_env)
    assert bool(result["failures"]) is not expected
    assert any(w["method"] == "repos.createDispatchEvent" for w in result["writes"]) is expected


@pytest.mark.parametrize("change", ["closed", "draft", "same-repo", "advanced-head", "conflicting-labels", "no-merge"])
def test_external_approval_does_not_override_pr_integrity(cli_env, change):
    case = scenario("trusted-external-sweep")
    pull = case["data"]["pull"]
    if change == "closed":
        pull["state"] = "closed"
    elif change == "draft":
        pull["draft"] = True
    elif change == "same-repo":
        pull["head"]["repo"]["full_name"] = "example/repo"
    elif change == "advanced-head":
        pull["head"]["sha"] = "new-head"
    elif change == "conflicting-labels":
        pull["labels"].append({"name": "sweep-enabled"})
    else:
        pull["merge_commit_sha"] = None
    result = run_workflow("trusted-external-sweep", case, cli_env)
    assert result["failures"]
    assert result["writes"] == []


def test_external_dispatch_preserves_approved_refs_and_options(cli_env):
    case = scenario("trusted-external-sweep")
    case["data"]["pull"]["labels"] += [{"name": "all-evals"}, {"name": "agentx-fast"}]
    result = run_workflow("trusted-external-sweep", case, cli_env)
    assert result["failures"] == []
    assert result["writes"][0] == {
        "method": "actions.createWorkflowDispatch", "owner": "example", "repo": "repo",
        "workflow_id": "e2e-tests.yml", "ref": "main", "inputs": {
            "test-name": "External PR #42 @ approved-hea", "ref": "approved-merge",
            "changelog-base-ref": "base-head", "changelog-head-ref": "approved-head",
            "trim-conc": "false", "all-evals": "true", "evals-only": "false", "fail-fast": "true",
            "agentx-fast": "true", "pr-labels-json": '["full-sweep-fail-fast","all-evals","agentx-fast"]',
        },
    }


@pytest.mark.parametrize("operation", OPERATIONS)
def test_authorization_loads_workflow_revision_without_persisted_credentials(operation):
    steps = next(iter(workflow(operation)["jobs"].values()))["steps"]
    checkout = next(s for s in steps if s.get("uses", "").startswith("actions/checkout@"))
    expression = re.fullmatch(r"\$\{\{\s*(.*?)\s*\}\}", checkout["with"]["ref"])[1]
    selected_ref = _Parser(_tokens(expression), {
        "github.workflow_sha": "trusted-workflow-commit",
        "github.sha": "event-commit",
        "github.event.pull_request.head.sha": "external-head",
    }).parse()
    assert selected_ref == "trusted-workflow-commit"
    assert checkout["with"]["persist-credentials"] == "false"
    assert steps.index(checkout) < next(i for i, s in enumerate(steps) if "GITHUB_TOKEN" in s.get("env", {}))
