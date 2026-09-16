import copy
import json

import pytest

from infx import github
from infx.workflows import stage_results


@pytest.fixture
def staging(monkeypatch, tmp_path):
    run = {
        "id": 123, "run_attempt": 2, "event": "pull_request", "status": "completed",
        "conclusion": "success", "path": ".github/workflows/run-sweep.yml",
        "head_sha": "tested", "created_at": "2026-09-01T12:00:00Z",
        "html_url": "https://github.com/example/project/actions/runs/123", "pull_requests": [],
    }
    case = {
        "event": {"issue": {"number": 7}, "comment": {
            "body": "/stage-results 123", "user": {"login": "reviewer"},
        }},
        "responses": {
            "/collaborators/reviewer/permission": {"permission": "write", "role_name": "write"},
            "/pulls/7": {"labels": [{"name": "full-sweep-enabled"}], "head": {"ref": "feature"}},
            "/pulls/7/commits": [{"sha": "tested"}, {"sha": "new-head"}],
            "/issues/7/timeline": [{"event": "labeled", "label": {"name": "full-sweep-enabled"},
                                    "created_at": "2026-09-01T11:00:00Z"}],
            "/actions/runs/123": run,
            "/actions/workflows/run-sweep.yml/runs": {"workflow_runs": [run], "total_count": 1},
            "/actions/runs/123/artifacts": {"artifacts": [
                {"name": "changelog-metadata", "expired": False},
                {"name": "results_bmk", "expired": False},
            ], "total_count": 2},
        },
        "comments": [], "reads": [], "output": tmp_path / "outputs",
    }

    def api(repo, path, token=None, params=None, *, method="GET", data=None, paginate=False):
        if method == "POST":
            assert path == "/issues/7/comments"
            case["comments"].append(data["body"])
            return {"id": 81}
        case["reads"].append(path)
        response = case["responses"][path]
        if isinstance(response, Exception):
            raise response
        return copy.deepcopy([response] if paginate else response)

    monkeypatch.setattr(github, "api", api)
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    monkeypatch.setenv("GH_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_OUTPUT", str(case["output"]))
    return case


def invoke(case, monkeypatch, tmp_path):
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(case["event"]))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    stage_results.main()


@pytest.mark.parametrize("body", [
    "/stage-results", "/stage-results 123", "\ufeff/stage-results\u00a000123\r\n",
    "/stage-results\n123",
])
def test_writes_selected_run_outputs_after_pr_head_advances(staging, monkeypatch, tmp_path, body):
    staging["event"]["comment"]["body"] = body
    invoke(staging, monkeypatch, tmp_path)
    assert staging["output"].read_text() == (
        "run-id=123\nrun-attempt=2\nrun-date=2026-09-01\nrequested-by=reviewer\n"
        "run-url=https://github.com/example/project/actions/runs/123\n"
    )
    assert staging["comments"] == []


@pytest.mark.parametrize("conclusion,result_name", [
    ("failure", "eval_results_all"), ("cancelled", "bmk_agentic_point"),
])
def test_automatic_staging_accepts_partial_results(staging, conclusion, result_name):
    staging["event"]["comment"]["body"] = "/stage-results"
    staging["responses"]["/actions/runs/123"]["conclusion"] = conclusion
    staging["responses"]["/actions/runs/123/artifacts"]["artifacts"][1]["name"] = result_name
    assert stage_results.request("example/project", staging["event"], "token")["run-id"] == "123"


@pytest.mark.parametrize("permission,role,label", [
    ("write", "maintain", "non-canary-full-sweep-enabled"),
    ("admin", "admin", "full-sweep-fail-fast"),
    ("write", "write", "full-sweep-fail-fast-no-canary"),
])
def test_standard_roles_and_full_sweep_modes_remain_eligible(staging, permission, role, label):
    staging["responses"]["/collaborators/reviewer/permission"] = {
        "permission": permission, "role_name": role,
    }
    staging["responses"]["/pulls/7"]["labels"] = [label]
    staging["responses"]["/issues/7/timeline"][0]["label"]["name"] = label
    del staging["responses"]["/actions/runs/123"]["run_attempt"]
    result = stage_results.request("example/project", staging["event"], "token")
    assert result["run-id"] == "123"
    assert result["run-attempt"] == "1"


def test_automatic_selection_skips_newer_empty_and_orphaned_runs(staging):
    staging["event"]["comment"]["body"] = "/stage-results"
    real = staging["responses"]["/actions/runs/123"]
    listing = staging["responses"]["/actions/workflows/run-sweep.yml/runs"]
    listing.update(workflow_runs=[
        {**real, "id": 126, "head_sha": "orphaned", "pull_requests": [{"number": 7}]},
        {**real, "id": 125}, {**real, "id": 124}, real,
    ], total_count=4)
    staging["responses"]["/actions/runs/125/artifacts"] = {"artifacts": [], "total_count": 0}
    staging["responses"]["/actions/runs/124/artifacts"] = {
        "artifacts": [{"name": "changelog-metadata", "expired": False}], "total_count": 1,
    }
    assert stage_results.request("example/project", staging["event"], "token")["run-id"] == "123"


@pytest.mark.parametrize("associated_pr", [7, 8])
def test_explicit_historical_pin_requires_pr_association(staging, associated_pr):
    run = staging["responses"]["/actions/runs/123"]
    run.update(head_sha="orphaned", pull_requests=[{"number": associated_pr}])
    if associated_pr == 7:
        assert stage_results.request("example/project", staging["event"], "token")["run-id"] == "123"
    else:
        with pytest.raises(RuntimeError, match="head commit is not in the PR commit list"):
            stage_results.request("example/project", staging["event"], "token")


def test_automatic_selection_rejects_all_orphaned_runs(staging):
    staging["event"]["comment"]["body"] = "/stage-results"
    staging["responses"]["/actions/runs/123"].update(
        head_sha="orphaned", pull_requests=[{"number": 7}],
    )
    with pytest.raises(RuntimeError, match="No stageable completed run-sweep.yml run"):
        stage_results.request("example/project", staging["event"], "token")


@pytest.mark.parametrize("change,reason", [
    ({"path": ".github/workflows/e2e-tests.yml"}, "not a pull-request run"),
    ({"event": "push"}, "not a pull-request run"),
    ({"status": "in_progress", "conclusion": None}, "it is in_progress/null"),
    ({"conclusion": "timed_out"}, "it is completed/timed_out"),
    ({"created_at": "not a date"}, "not created while a full-sweep label"),
])
def test_invalid_pinned_run_never_publishes_outputs(staging, monkeypatch, tmp_path, change, reason):
    staging["responses"]["/actions/runs/123"].update(change)
    with pytest.raises(RuntimeError, match=reason):
        invoke(staging, monkeypatch, tmp_path)
    assert not staging["output"].exists()
    assert staging["comments"] == []


@pytest.mark.parametrize("expired_index,reason", [
    (0, "no unexpired changelog-metadata"), (1, "no unexpired benchmark result"),
])
def test_expired_artifacts_reject_staging(staging, expired_index, reason):
    staging["responses"]["/actions/runs/123/artifacts"]["artifacts"][expired_index]["expired"] = True
    with pytest.raises(RuntimeError, match=reason):
        stage_results.request("example/project", staging["event"], "token")


@pytest.mark.parametrize("created_at,accepted", [
    ("2026-09-01T10:59:59Z", False), ("2026-09-01T11:00:00Z", True),
    ("2026-09-01T11:59:59Z", True), ("2026-09-01T12:00:00Z", False),
    ("2026-09-01T13:00:00Z", True),
])
def test_uses_label_history_at_run_creation(staging, created_at, accepted):
    staging["responses"]["/actions/runs/123"]["created_at"] = created_at
    staging["responses"]["/issues/7/timeline"].extend([
        {"event": "labeled", "label": {"name": "full-sweep-enabled"},
         "created_at": "2026-09-01T15:00:00+02:00"},
        {"event": "unlabeled", "label": {"name": "full-sweep-enabled"},
         "created_at": "2026-09-01T12:00:00Z"},
        {"event": "unlabeled", "label": {"name": "full-sweep-enabled"}, "created_at": "bad"},
        {"event": "labeled", "label": {"name": "sweep-enabled"},
         "created_at": "2026-09-01T10:00:00Z"},
    ])
    if accepted:
        assert stage_results.request("example/project", staging["event"], "token")["run-id"] == "123"
    else:
        with pytest.raises(RuntimeError, match="not created while a full-sweep label"):
            stage_results.request("example/project", staging["event"], "token")


def test_removing_one_full_sweep_label_keeps_another_active(staging):
    staging["responses"]["/issues/7/timeline"].extend([
        {"event": "labeled", "label": {"name": "full-sweep-fail-fast"},
         "created_at": "2026-09-01T11:30:00Z"},
        {"event": "unlabeled", "label": {"name": "full-sweep-enabled"},
         "created_at": "2026-09-01T12:00:00Z"},
    ])
    assert stage_results.request("example/project", staging["event"], "token")["run-id"] == "123"


@pytest.mark.parametrize("body", [
    "/stage-resultsx", "/stage-results -1", "/stage-results 123 extra",
    "/stage-results １２３", "/stage-results\x85123",
])
def test_bad_syntax_posts_usage_without_permission_or_run_lookups(staging, body):
    staging["event"]["comment"]["body"] = body
    with pytest.raises(RuntimeError, match="Unsupported /stage-results syntax"):
        stage_results.request("example/project", staging["event"], "token")
    assert staging["comments"] == ["Usage: `/stage-results` or `/stage-results <run-id>`."]
    assert staging["reads"] == []


@pytest.mark.parametrize("permission,role", [("read", "read"), ("write", "custom"), ("read", "admin")])
def test_both_permission_fields_must_be_trusted(staging, permission, role):
    staging["responses"]["/collaborators/reviewer/permission"] = {
        "permission": permission, "role_name": role,
    }
    with pytest.raises(RuntimeError, match="both must be write, maintain, or admin"):
        stage_results.request("example/project", staging["event"], "token")
    assert len(staging["comments"]) == 1
    assert "两者都必须是 write、maintain 或 admin" in staging["comments"][0]
    assert staging["reads"] == ["/collaborators/reviewer/permission"]


@pytest.mark.parametrize("access", [None, {}, {"permission": "write"},
                                      {"permission": "admin", "role_name": 1}])
def test_malformed_permission_response_fails_closed(staging, access):
    staging["responses"]["/collaborators/reviewer/permission"] = access
    with pytest.raises(RuntimeError, match="Invalid repository permission response"):
        stage_results.request("example/project", staging["event"], "token")
    assert staging["comments"] == []
    assert staging["reads"] == ["/collaborators/reviewer/permission"]


def test_current_trim_label_does_not_authorize_staging(staging):
    staging["responses"]["/pulls/7"]["labels"] = [{"name": "sweep-enabled"}]
    with pytest.raises(RuntimeError, match="PR does not have a full-sweep label"):
        stage_results.request("example/project", staging["event"], "token")
    assert "requires a completed run from a PR using one of:" in staging["comments"][0]
    assert "/issues/7/timeline" not in staging["reads"]


@pytest.mark.parametrize("path", ["/issues/7/timeline", "/pulls/7/commits", "/actions/runs/123/artifacts"])
def test_api_failures_never_publish_a_source(staging, monkeypatch, tmp_path, path):
    staging["responses"][path] = RuntimeError("GitHub unavailable")
    with pytest.raises(RuntimeError, match="GitHub unavailable"):
        invoke(staging, monkeypatch, tmp_path)
    assert not staging["output"].exists()


def test_incomplete_artifact_listing_cannot_approve_a_run(staging):
    staging["responses"]["/actions/runs/123/artifacts"]["total_count"] = 3
    with pytest.raises(github.ListingError, match="Incomplete GitHub listing"):
        stage_results.request("example/project", staging["event"], "token")
