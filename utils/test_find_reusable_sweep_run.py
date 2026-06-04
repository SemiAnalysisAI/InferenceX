from __future__ import annotations

import find_reusable_sweep_run as reuse


def test_find_reuse_authorization_uses_latest_allowed_comment(monkeypatch) -> None:
    def fake_paginated_github_api(*args, **kwargs):
        return [
            {
                "created_at": "2026-05-13T00:00:00Z",
                "author_association": "MEMBER",
                "body": "/reuse-sweep-run 111",
            },
            {
                "created_at": "2026-05-13T00:01:00Z",
                "author_association": "CONTRIBUTOR",
                "body": "/reuse-sweep-run 222",
            },
            {
                "created_at": "2026-05-13T00:02:00Z",
                "author_association": "OWNER",
                "body": "approved\n/reuse-sweep-run 333",
            },
        ]

    monkeypatch.setattr(reuse, "paginated_github_api", fake_paginated_github_api)

    assert reuse.find_reuse_authorization(
        "SemiAnalysisAI/InferenceX",
        1321,
        "token",
        "/reuse-sweep-run",
        {"OWNER", "MEMBER", "COLLABORATOR"},
    ) == (True, 333)


def test_find_reuse_authorization_accepts_command_without_run_id(monkeypatch) -> None:
    def fake_paginated_github_api(*args, **kwargs):
        return [
            {
                "created_at": "2026-05-13T00:00:00Z",
                "author_association": "OWNER",
                "body": "/reuse-sweep-run",
            },
        ]

    monkeypatch.setattr(reuse, "paginated_github_api", fake_paginated_github_api)

    assert reuse.find_reuse_authorization(
        "SemiAnalysisAI/InferenceX",
        1321,
        "token",
        "/reuse-sweep-run",
        {"OWNER", "MEMBER", "COLLABORATOR"},
    ) == (True, None)


def test_find_reuse_authorization_lets_newer_no_arg_unpin_older_pin(monkeypatch) -> None:
    def fake_paginated_github_api(*args, **kwargs):
        return [
            {
                "created_at": "2026-05-13T00:00:00Z",
                "author_association": "OWNER",
                "body": "/reuse-sweep-run 111",
            },
            {
                "created_at": "2026-05-13T00:01:00Z",
                "author_association": "OWNER",
                "body": "/reuse-sweep-run",
            },
        ]

    monkeypatch.setattr(reuse, "paginated_github_api", fake_paginated_github_api)

    assert reuse.find_reuse_authorization(
        "SemiAnalysisAI/InferenceX",
        1321,
        "token",
        "/reuse-sweep-run",
        {"OWNER", "MEMBER", "COLLABORATOR"},
    ) == (True, None)


def test_find_reuse_authorization_ignores_inline_mentions(monkeypatch) -> None:
    def fake_paginated_github_api(*args, **kwargs):
        return [
            {
                "created_at": "2026-05-13T00:00:00Z",
                "author_association": "OWNER",
                "body": "please run /reuse-sweep-run 111 after merge",
            },
        ]

    monkeypatch.setattr(reuse, "paginated_github_api", fake_paginated_github_api)

    assert reuse.find_reuse_authorization(
        "SemiAnalysisAI/InferenceX",
        1321,
        "token",
        "/reuse-sweep-run",
        {"OWNER", "MEMBER", "COLLABORATOR"},
    ) == (False, None)


def test_validate_reusable_run_accepts_successful_same_pr_run(monkeypatch) -> None:
    monkeypatch.setattr(reuse, "artifact_names", lambda *args: {"results_bmk"})
    monkeypatch.setattr(reuse, "pr_commit_shas", lambda *args: {"abc123"})

    reuse.validate_reusable_run(
        "SemiAnalysisAI/InferenceX",
        "run-sweep.yml",
        1321,
        {
            "id": 25763404168,
            "event": "pull_request",
            "status": "completed",
            "conclusion": "success",
            "path": ".github/workflows/run-sweep.yml",
            "head_sha": "abc123",
            "pull_requests": [{"number": 1321}],
        },
        "token",
    )


def test_validate_reusable_run_accepts_run_for_older_pr_commit(monkeypatch) -> None:
    """Regression: pinned run survives an additional commit landing on the PR.

    GitHub recomputes ``run.pull_requests`` to empty once the PR head moves past
    the run's commit, but the run's commit is still part of the PR's history and
    should remain reusable.
    """
    monkeypatch.setattr(reuse, "artifact_names", lambda *args: {"results_bmk"})
    monkeypatch.setattr(
        reuse,
        "pr_commit_shas",
        lambda *args: {
            "e36afac48cc6165f4e1f8ea7e1977b01ef29787c",
            "5c7d7df8ce125e6c725eb37db123269380b7c97d",
        },
    )

    reuse.validate_reusable_run(
        "SemiAnalysisAI/InferenceX",
        "run-sweep.yml",
        1321,
        {
            "id": 25763404168,
            "event": "pull_request",
            "status": "completed",
            "conclusion": "success",
            "path": ".github/workflows/run-sweep.yml",
            "head_sha": "e36afac48cc6165f4e1f8ea7e1977b01ef29787c",
            "pull_requests": [],
        },
        "token",
    )


def test_validate_reusable_run_rejects_run_for_orphaned_commit(monkeypatch) -> None:
    monkeypatch.setattr(reuse, "artifact_names", lambda *args: {"results_bmk"})
    monkeypatch.setattr(reuse, "pr_commit_shas", lambda *args: {"def456"})

    try:
        reuse.validate_reusable_run(
            "SemiAnalysisAI/InferenceX",
            "run-sweep.yml",
            1321,
            {
                "id": 25763404168,
                "event": "pull_request",
                "status": "completed",
                "conclusion": "success",
                "path": ".github/workflows/run-sweep.yml",
                "head_sha": "abc123",
                "pull_requests": [],
            },
            "token",
        )
    except RuntimeError as error:
        assert "is not in PR #1321's commit list" in str(error)
    else:
        raise AssertionError("expected orphaned-commit run to be rejected")


def test_main_enables_pinned_reuse_without_extra_label(monkeypatch, tmp_path) -> None:
    comments = [
        {
            "created_at": "2026-05-13T00:00:00Z",
            "author_association": "OWNER",
            "body": "/reuse-sweep-run 25763404168",
        },
    ]
    run = {
        "id": 25763404168,
        "event": "pull_request",
        "status": "completed",
        "conclusion": "success",
        "path": ".github/workflows/run-sweep.yml",
        "pull_requests": [{"number": 1321}],
        "run_attempt": 1,
        "html_url": "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/25763404168",
        "head_sha": "abc123",
    }

    def fake_github_api(repo, path, token, params=None):
        if path == "/commits/merge-sha/pulls":
            return [{"number": 1321}]
        if path == "/pulls/1321":
            return {
                "merged_at": "2026-05-13T00:01:00Z",
                "labels": [{"name": "full-sweep-enabled"}],
                "head": {"sha": "abc123"},
            }
        if path == "/actions/runs/25763404168":
            return run
        raise AssertionError(f"unexpected GitHub API path: {path}")

    def fake_paginated_github_api(repo, path, token, item_key, params=None):
        if path == "/issues/1321/comments":
            return comments
        if path == "/pulls/1321/commits":
            return [{"sha": "abc123"}]
        if path == "/actions/runs/25763404168/artifacts":
            return [{"name": "results_bmk"}]
        raise AssertionError(f"unexpected paginated GitHub API path: {path}")

    output_path = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(reuse, "github_api", fake_github_api)
    monkeypatch.setattr(reuse, "paginated_github_api", fake_paginated_github_api)
    monkeypatch.setattr(
        reuse.sys,
        "argv",
        [
            "find_reusable_sweep_run.py",
            "--repo",
            "SemiAnalysisAI/InferenceX",
            "--commit-sha",
            "merge-sha",
            "--event-name",
            "push",
            "--ref",
            "refs/heads/main",
            "--github-output",
            str(output_path),
        ],
    )

    assert reuse.main() == 0

    outputs = dict(line.split("=", 1) for line in output_path.read_text().splitlines())
    assert outputs["reuse-enabled"] == "true"
    assert outputs["reuse-source-run-id"] == "25763404168"
    assert outputs["reuse-source-pr-number"] == "1321"
    assert outputs["reuse-source-head-sha"] == "abc123"


def test_main_resolves_no_arg_command_to_latest_head_sweep(monkeypatch, tmp_path) -> None:
    comments = [
        {
            "created_at": "2026-05-13T00:00:00Z",
            "author_association": "OWNER",
            "body": "/reuse-sweep-run",
        },
    ]
    run = {
        "id": 25763404168,
        "event": "pull_request",
        "status": "completed",
        "conclusion": "success",
        "path": ".github/workflows/run-sweep.yml",
        "pull_requests": [{"number": 1321}],
        "run_attempt": 1,
        "html_url": "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/25763404168",
        "head_sha": "abc123",
    }

    def fake_github_api(repo, path, token, params=None):
        if path == "/commits/merge-sha/pulls":
            return [{"number": 1321}]
        if path == "/pulls/1321":
            return {
                "merged_at": "2026-05-13T00:01:00Z",
                "labels": [{"name": "full-sweep-enabled"}],
                "head": {"sha": "abc123", "ref": "feature-branch"},
            }
        raise AssertionError(f"unexpected GitHub API path: {path}")

    def fake_paginated_github_api(repo, path, token, item_key, params=None):
        if path == "/issues/1321/comments":
            return comments
        if path == "/pulls/1321/commits":
            return [{"sha": "abc123"}]
        if path == "/actions/workflows/run-sweep.yml/runs":
            assert params["branch"] == "feature-branch"
            return [run]
        if path == "/actions/runs/25763404168/artifacts":
            return [{"name": "results_bmk"}]
        raise AssertionError(f"unexpected paginated GitHub API path: {path}")

    output_path = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(reuse, "github_api", fake_github_api)
    monkeypatch.setattr(reuse, "paginated_github_api", fake_paginated_github_api)
    monkeypatch.setattr(
        reuse.sys,
        "argv",
        [
            "find_reusable_sweep_run.py",
            "--repo",
            "SemiAnalysisAI/InferenceX",
            "--commit-sha",
            "merge-sha",
            "--event-name",
            "push",
            "--ref",
            "refs/heads/main",
            "--github-output",
            str(output_path),
        ],
    )

    assert reuse.main() == 0

    outputs = dict(line.split("=", 1) for line in output_path.read_text().splitlines())
    assert outputs["reuse-enabled"] == "true"
    assert outputs["reuse-source-run-id"] == "25763404168"
    assert outputs["reuse-source-pr-number"] == "1321"
    assert outputs["reuse-source-head-sha"] == "abc123"


def test_main_no_arg_picks_run_for_older_pr_commit(monkeypatch, tmp_path) -> None:
    """Regression: no-arg /reuse-sweep-run finds a run for an earlier PR commit.

    After a sweep succeeds on commit A, an additional commit B can land on the
    PR (e.g. a main merge to resolve perf-changelog.yaml). The current head is
    now B; no sweep ever ran for B, but A's run is still reusable.
    """
    comments = [
        {
            "created_at": "2026-05-13T00:00:00Z",
            "author_association": "OWNER",
            "body": "/reuse-sweep-run",
        },
    ]
    older_run = {
        "id": 25763466401,
        "event": "pull_request",
        "status": "completed",
        "conclusion": "success",
        "path": ".github/workflows/run-sweep.yml",
        "pull_requests": [],
        "run_attempt": 1,
        "html_url": "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/25763466401",
        "head_sha": "f6b43f745a7df3653d677b30372b05d3bec6f153",
    }

    def fake_github_api(repo, path, token, params=None):
        if path == "/commits/merge-sha/pulls":
            return [{"number": 1347}]
        if path == "/pulls/1347":
            return {
                "merged_at": "2026-05-13T00:01:00Z",
                "labels": [{"name": "full-sweep-enabled"}],
                "head": {
                    "sha": "f862cb8b126e2437ee793951919f40c23bba3a6f",
                    "ref": "claude/issue-1154-qwen3.5-fp8-h200-sglang-mtp",
                },
            }
        raise AssertionError(f"unexpected GitHub API path: {path}")

    def fake_paginated_github_api(repo, path, token, item_key, params=None):
        if path == "/issues/1347/comments":
            return comments
        if path == "/pulls/1347/commits":
            return [
                {"sha": "f6b43f745a7df3653d677b30372b05d3bec6f153"},
                {"sha": "f862cb8b126e2437ee793951919f40c23bba3a6f"},
            ]
        if path == "/actions/workflows/run-sweep.yml/runs":
            assert params["branch"] == "claude/issue-1154-qwen3.5-fp8-h200-sglang-mtp"
            return [older_run]
        if path == "/actions/runs/25763466401/artifacts":
            return [{"name": "results_bmk"}]
        raise AssertionError(f"unexpected paginated GitHub API path: {path}")

    output_path = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(reuse, "github_api", fake_github_api)
    monkeypatch.setattr(reuse, "paginated_github_api", fake_paginated_github_api)
    monkeypatch.setattr(
        reuse.sys,
        "argv",
        [
            "find_reusable_sweep_run.py",
            "--repo",
            "SemiAnalysisAI/InferenceX",
            "--commit-sha",
            "merge-sha",
            "--event-name",
            "push",
            "--ref",
            "refs/heads/main",
            "--github-output",
            str(output_path),
        ],
    )

    assert reuse.main() == 0

    outputs = dict(line.split("=", 1) for line in output_path.read_text().splitlines())
    assert outputs["reuse-enabled"] == "true"
    assert outputs["reuse-source-run-id"] == "25763466401"
    assert outputs["reuse-source-pr-number"] == "1347"
    assert outputs["reuse-source-head-sha"] == "f6b43f745a7df3653d677b30372b05d3bec6f153"


def test_main_disables_reuse_without_pinned_comment(monkeypatch, tmp_path) -> None:
    def fake_github_api(repo, path, token, params=None):
        if path == "/commits/merge-sha/pulls":
            return [{"number": 1321}]
        raise AssertionError(f"unexpected GitHub API path: {path}")

    def fake_paginated_github_api(repo, path, token, item_key, params=None):
        if path == "/issues/1321/comments":
            return []
        raise AssertionError(f"unexpected paginated GitHub API path: {path}")

    output_path = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(reuse, "github_api", fake_github_api)
    monkeypatch.setattr(reuse, "paginated_github_api", fake_paginated_github_api)
    monkeypatch.setattr(
        reuse.sys,
        "argv",
        [
            "find_reusable_sweep_run.py",
            "--repo",
            "SemiAnalysisAI/InferenceX",
            "--commit-sha",
            "merge-sha",
            "--event-name",
            "push",
            "--ref",
            "refs/heads/main",
            "--github-output",
            str(output_path),
        ],
    )

    assert reuse.main() == 0

    outputs = dict(line.split("=", 1) for line in output_path.read_text().splitlines())
    assert outputs["reuse-enabled"] == "false"
    assert outputs["reuse-source-pr-number"] == "1321"
    assert outputs["reuse-reason"] == "PR #1321 has no /reuse-sweep-run authorization"


def test_main_accepts_non_canary_full_sweep_label(monkeypatch, tmp_path) -> None:
    comments = [
        {
            "created_at": "2026-05-13T00:00:00Z",
            "author_association": "OWNER",
            "body": "/reuse-sweep-run 25763404168",
        },
    ]
    run = {
        "id": 25763404168,
        "event": "pull_request",
        "status": "completed",
        "conclusion": "success",
        "path": ".github/workflows/run-sweep.yml",
        "pull_requests": [{"number": 1321}],
        "run_attempt": 1,
        "html_url": "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/25763404168",
        "head_sha": "abc123",
    }

    def fake_github_api(repo, path, token, params=None):
        if path == "/commits/merge-sha/pulls":
            return [{"number": 1321}]
        if path == "/pulls/1321":
            return {
                "merged_at": "2026-05-13T00:01:00Z",
                "labels": [{"name": "non-canary-full-sweep-enabled"}],
                "head": {"sha": "abc123"},
            }
        if path == "/actions/runs/25763404168":
            return run
        raise AssertionError(f"unexpected GitHub API path: {path}")

    def fake_paginated_github_api(repo, path, token, item_key, params=None):
        if path == "/issues/1321/comments":
            return comments
        if path == "/pulls/1321/commits":
            return [{"sha": "abc123"}]
        if path == "/actions/runs/25763404168/artifacts":
            return [{"name": "results_bmk"}]
        raise AssertionError(f"unexpected paginated GitHub API path: {path}")

    output_path = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(reuse, "github_api", fake_github_api)
    monkeypatch.setattr(reuse, "paginated_github_api", fake_paginated_github_api)
    monkeypatch.setattr(
        reuse.sys,
        "argv",
        [
            "find_reusable_sweep_run.py",
            "--repo",
            "SemiAnalysisAI/InferenceX",
            "--commit-sha",
            "merge-sha",
            "--event-name",
            "push",
            "--ref",
            "refs/heads/main",
            "--github-output",
            str(output_path),
        ],
    )

    assert reuse.main() == 0

    outputs = dict(line.split("=", 1) for line in output_path.read_text().splitlines())
    assert outputs["reuse-enabled"] == "true"


def test_main_rejects_pr_with_neither_full_sweep_label(monkeypatch, tmp_path) -> None:
    comments = [
        {
            "created_at": "2026-05-13T00:00:00Z",
            "author_association": "OWNER",
            "body": "/reuse-sweep-run 25763404168",
        },
    ]

    def fake_github_api(repo, path, token, params=None):
        if path == "/commits/merge-sha/pulls":
            return [{"number": 1321}]
        if path == "/pulls/1321":
            return {
                "merged_at": "2026-05-13T00:01:00Z",
                "labels": [{"name": "sweep-enabled"}],
                "head": {"sha": "abc123"},
            }
        raise AssertionError(f"unexpected GitHub API path: {path}")

    def fake_paginated_github_api(repo, path, token, item_key, params=None):
        if path == "/issues/1321/comments":
            return comments
        raise AssertionError(f"unexpected paginated GitHub API path: {path}")

    output_path = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(reuse, "github_api", fake_github_api)
    monkeypatch.setattr(reuse, "paginated_github_api", fake_paginated_github_api)
    monkeypatch.setattr(
        reuse.sys,
        "argv",
        [
            "find_reusable_sweep_run.py",
            "--repo",
            "SemiAnalysisAI/InferenceX",
            "--commit-sha",
            "merge-sha",
            "--event-name",
            "push",
            "--ref",
            "refs/heads/main",
            "--github-output",
            str(output_path),
        ],
    )

    try:
        reuse.main()
    except RuntimeError as error:
        msg = str(error)
        assert "full-sweep-enabled" in msg
        assert "non-canary-full-sweep-enabled" in msg
    else:
        raise AssertionError(
            "expected RuntimeError when PR has neither full-sweep-enabled nor "
            "non-canary-full-sweep-enabled label"
        )
