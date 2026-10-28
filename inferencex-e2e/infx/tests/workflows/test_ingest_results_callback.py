import copy

import pytest

from infx import github
from infx.workflows import ingest_results_callback as callback


def counts(**overrides):
    rows = {c: dict.fromkeys(callback.OUTCOMES, 0) for c in callback.CATEGORIES}
    for category, values in overrides.items():
        rows[category].update(values)
    return rows


@pytest.fixture
def case(monkeypatch):
    state = {
        "event": {
            "action": "ingest-results-completed",
            "client_payload": {
                "source-repository": "SemiAnalysisAI/InferenceX-app",
                "merge-run-id": "40178629864",
                "source-run-id": "40141873883",
                "app-run-id": "40178653878",
                "app-run-attempt": "1",
                "database-target": "production",
                "ingest-succeeded": "true",
                "report": {
                    "counts": counts(benchmark={"new": 1}),
                    "detail": {"benchmarkFiles": 2, "changelogEntriesWritten": 1},
                    "unmapped": {"models": [], "hardware": ["mi999x"]},
                },
            },
        },
        "responses": {
            "/actions/runs/40178629864": {
                "path": ".github/workflows/merge-ingest.yml", "event": "push",
                "head_branch": "main", "head_sha": "abc",
            },
            "/commits/abc/pulls": [
                {"number": 4456, "merged_at": "2026-10-24T00:00:00Z", "merge_commit_sha": "abc"},
            ],
            "/issues/4456/comments": [],
        },
        "writes": [],
    }

    def api(repo, path, token=None, params=None, *, method="GET", data=None, paginate=False):
        if method != "GET":
            state["writes"].append((method, path, data["body"]))
            return {"id": 1}
        response = state["responses"][path]
        return copy.deepcopy([response] if paginate else response)

    monkeypatch.setattr(github, "api", api)
    return state


def run(state):
    callback.report("SemiAnalysisAI/InferenceX", state["event"], "t", "https://github.com")


def test_posts_counts_on_the_merged_pr(case):
    run(case)
    [(method, path, body)] = case["writes"]
    assert (method, path) == ("POST", "/issues/4456/comments")
    assert body.startswith("<!-- ingest-results-report merge-run=40178629864 -->")
    assert "| benchmark | 1 | 0 | 0 | 0 | 0 |" in body
    assert "benchmarkFiles 2, changelogEntriesWritten 1" in body
    assert "Unmapped hardware: `mi999x`" in body
    assert "<summary>中文</summary>" in body


def test_updates_existing_comment_in_place_on_failure(case):
    case["responses"]["/issues/4456/comments"] = [
        {"id": 9, "user": {"login": "someone"},
         "body": "<!-- ingest-results-report merge-run=40178629864 -->"},
        {"id": 10, "user": {"login": "github-actions[bot]"},
         "body": "<!-- ingest-results-report merge-run=40178629864 -->\nold"},
    ]
    case["event"]["client_payload"]["ingest-succeeded"] = "false"
    case["event"]["client_payload"]["report"] = None
    run(case)
    [(method, path, body)] = case["writes"]
    assert (method, path) == ("PATCH", "/issues/comments/10")
    assert "failed" in body
    assert "No ingest report was produced" in body


def test_no_merged_pr_posts_nothing(case):
    case["responses"]["/commits/abc/pulls"] = [{"number": 1, "merged_at": None}]
    run(case)
    assert case["writes"] == []


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("source-repository", "someone/fork"),
        ("merge-run-id", "1; rm"),
        ("ingest-succeeded", "yes"),
        ("database-target", "prod`uction"),
        ("report", {"counts": counts(eval={"new": -1})}),
        ("report", {"counts": counts(), "unmapped": {"models": ["[x](http://e)"]}}),
    ],
)
def test_rejects_malformed_payload(case, key, value):
    case["event"]["client_payload"][key] = value
    with pytest.raises(RuntimeError):
        run(case)
    assert case["writes"] == []


def test_rejects_runs_other_than_merge_ingest(case):
    case["responses"]["/actions/runs/40178629864"]["path"] = ".github/workflows/run-sweep.yml"
    with pytest.raises(RuntimeError):
        run(case)
