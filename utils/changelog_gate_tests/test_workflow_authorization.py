"""Exercise the real workflow scripts with controlled GitHub responses."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
OPERATIONS = ("stage-results", "trusted-external-sweep")


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


def run_workflow(operation, case):
    job = next(iter(workflow(operation)["jobs"].values()))
    result = subprocess.run(
        ["node", str(ROOT / "utils/changelog_gate_tests/workflow_script_runner.cjs")],
        input=json.dumps({**case, "steps": job["steps"]}), cwd=ROOT,
        capture_output=True, text=True, timeout=10, check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("operation,expected_actor", [
    ("stage-results", "commenter"), ("trusted-external-sweep", "requester"),
])
def test_permission_lookup_uses_original_requester(operation, expected_actor):
    case = scenario(operation)
    case["context"]["triggering_actor"] = "admin-rerunner"
    case["context"]["payload"]["sender"] = {"login": "payload-sender"}
    result = run_workflow(operation, case)
    assert not result["failures"]
    assert result["permissionRequests"] == [{
        "owner": "example", "repo": "repo", "username": expected_actor,
    }]


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("role,legacy,allowed", [
    ("admin", "admin", True), ("maintain", "write", True),
    ("maintain", "maintain", True), ("write", "write", True),
    ("triage", "read", False), ("read", "read", False), ("none", "none", False),
    ("custom-role", "admin", False), ("custom-role", "write", False), ("custom-role", "read", False),
    ("admin", "read", False), ("read", "admin", False),
    ("admin", "unknown", False), ("WRITE", "write", False), (" write ", "write", False),
])
def test_workflow_dispatch_requires_the_requested_repository_role(operation, role, legacy, allowed):
    case = scenario(operation)
    case["permission"] = {"role_name": role, "permission": legacy}
    result = run_workflow(operation, case)
    dispatches = [w for w in result["writes"] if w["method"] != "issues.createComment"]
    assert bool(dispatches) is allowed
    assert bool(result["failures"]) is not allowed
    if not allowed:
        assert f'permission "{legacy}"' in result["failures"][0]
        assert f'role "{role}"' in result["failures"][0]
        if operation == "stage-results":
            assert len(result["writes"]) == 1
            comment = result["writes"][0]["body"]
            assert "@commenter" in comment
            assert f'permission "{legacy}"' in comment
            assert f'role "{role}"' in comment


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("permission,error", [
    ({"role_name": "admin", "permission": "admin"}, True),
    (None, False), ([], False), ("not an object", False), ({}, False),
    ({"permission": True, "role_name": "admin"}, False),
    ({"permission": None, "role_name": "admin"}, False),
    ({"permission": "", "role_name": "admin"}, False),
    ({"permission": "admin", "role_name": []}, False),
    ({"permission": "admin", "role_name": {}}, False),
    ({"role_name": "admin"}, False),
    ({"permission": "admin"}, False),
    ({"permission": "admin", "role_name": None}, False),
    ({"permission": "admin", "role_name": True}, False),
    ({"permission": "admin", "role_name": ""}, False),
])
def test_unavailable_role_never_reaches_protected_effects(operation, permission, error):
    case = scenario(operation)
    case.update(permission=permission, permissionError=error)
    result = run_workflow(operation, case)
    assert result["failures"]
    assert result["writes"] == []
    assert result["outputs"] == {}


def test_staging_keeps_source_outputs_and_dispatch_contract():
    result = run_workflow("stage-results", scenario("stage-results"))
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


def test_invalid_staging_command_preserves_usage_reply_without_role_lookup():
    case = scenario("stage-results")
    case["context"]["payload"]["comment"]["body"] = "/stage-results invalid"
    result = run_workflow("stage-results", case)
    assert result["failures"] == ["Unsupported /stage-results syntax"]
    assert [w["body"] for w in result["writes"]] == ["Usage: `/stage-results` or `/stage-results <run-id>`."]
    assert result["permissionRequests"] == []


@pytest.mark.parametrize("field,value", [
    ("head_sha", "unrelated-head"), ("event", "push"), ("path", "other.yml"),
    ("status", "in_progress"), ("conclusion", "skipped"),
])
def test_staging_role_does_not_override_run_eligibility(field, value):
    case = scenario("stage-results")
    case["data"]["runs"][0][field] = value
    result = run_workflow("stage-results", case)
    assert result["failures"]
    assert result["writes"] == []


@pytest.mark.parametrize("conclusion", ["success", "failure", "cancelled"])
@pytest.mark.parametrize("artifact", ["results_bmk", "eval_results_all", "bmk_agentic_example"])
def test_staging_preserves_partial_results_for_each_supported_artifact(conclusion, artifact):
    case = scenario("stage-results")
    case["data"]["runs"][0]["conclusion"] = conclusion
    case["data"]["artifacts"]["101"][1]["name"] = artifact
    result = run_workflow("stage-results", case)
    assert not result["failures"]
    assert result["writes"][-1]["event_type"] == "stage-results"


@pytest.mark.parametrize("change", ["missing-metadata", "expired-results", "no-current-label", "no-historical-label"])
def test_staging_role_does_not_override_labels_or_artifacts(change):
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
    result = run_workflow("stage-results", case)
    assert result["failures"]
    assert all(w["method"] == "issues.createComment" for w in result["writes"])


@pytest.mark.parametrize("pinned,associated,expected", [(True, True, True), (True, False, False), (False, True, False)])
def test_historical_staging_association_requires_explicit_run_id(pinned, associated, expected):
    case = scenario("stage-results")
    case["data"]["runs"][0].update(head_sha="historical-head", pull_requests=[{"number": 42 if associated else 43}])
    if pinned:
        case["context"]["payload"]["comment"]["body"] = "/stage-results 101"
    result = run_workflow("stage-results", case)
    assert bool(result["failures"]) is not expected
    assert any(w["method"] == "repos.createDispatchEvent" for w in result["writes"]) is expected


@pytest.mark.parametrize("change", ["closed", "draft", "same-repo", "advanced-head", "conflicting-labels", "no-merge"])
def test_external_approval_does_not_override_pr_integrity(change):
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
    result = run_workflow("trusted-external-sweep", case)
    assert result["failures"]
    assert result["writes"] == []


def test_external_dispatch_preserves_approved_refs_and_options():
    case = scenario("trusted-external-sweep")
    case["data"]["pull"]["labels"] += [{"name": "all-evals"}, {"name": "agentx-fast"}]
    result = run_workflow("trusted-external-sweep", case)
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
