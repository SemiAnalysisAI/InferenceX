from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from infx.workflows import reuse


@pytest.mark.parametrize("entrypoint", ["legacy", "package"])
@pytest.mark.parametrize("token_present", [False, True])
def test_reuse_entrypoints_preserve_outputs_and_errors_without_installation(
    tmp_path, entrypoint, token_present,
):
    root = Path(__file__).resolve().parents[1]
    # The package must work without utils/ and without an inherited import path.
    shutil.copytree(root / "infx", tmp_path / "infx")
    command = ([str(root / "utils/find_reusable_sweep_run.py")] if entrypoint == "legacy"
               else ["-m", "infx.workflows.reuse"])
    env = {key: value for key, value in os.environ.items()
           if key not in {"PYTHONPATH", "GH_TOKEN", "GITHUB_TOKEN", "GITHUB_OUTPUT"}}
    if token_present:
        env["GH_TOKEN"] = "test-token"
    output = tmp_path / "outputs"
    run = subprocess.run(
        [sys.executable, *command, "--repo", "example/project", "--commit-sha", "abc",
         "--event-name", "workflow_dispatch", "--ref", "refs/heads/feature",
         "--github-output", str(output)],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=10,
    )
    if not token_present:
        assert run.returncode == 1
        assert run.stderr == "error: GH_TOKEN or GITHUB_TOKEN is required\n"
        assert not output.exists()
        return
    assert run.returncode == 0, run.stderr
    expected = {
        "reuse-enabled": "false", "skip-pr-sweep": "false",
        "reuse-source-run-id": "", "reuse-source-run-attempt": "",
        "reuse-source-run-url": "", "reuse-source-pr-number": "",
        "reuse-source-head-sha": "", "reuse-reason": "not a push to main",
    }
    assert json.loads(run.stdout) == expected
    assert dict(line.split("=", 1) for line in output.read_text().splitlines()) == expected


def test_legacy_import_keeps_overrides_on_the_canonical_implementation(monkeypatch):
    import find_reusable_sweep_run as legacy

    def unexpected_artifacts(*args):
        pytest.fail("legacy override was not used")

    monkeypatch.setattr(reuse, "artifact_names", unexpected_artifacts)
    monkeypatch.setattr(legacy, "artifact_names", lambda *args: {"results_bmk"})
    run = {"id": 123, "head_sha": "tested-sha", "conclusion": "success"}
    monkeypatch.setattr(reuse.github, "paginate", lambda *args: [run])
    assert reuse.find_latest_successful_pr_run(
        "example/project", "run-sweep.yml", "feature", {"tested-sha"}, "token",
    ) == run


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

    monkeypatch.setattr(reuse.github, "paginate", fake_paginated_github_api)

    assert reuse.find_reuse_authorization(
        "SemiAnalysisAI/InferenceX",
        1321,
        "token",
        "/reuse-sweep-run",
        {"OWNER", "MEMBER", "COLLABORATOR"},
    ) == (True, 333)


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

    monkeypatch.setattr(reuse.github, "paginate", fake_paginated_github_api)

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

    monkeypatch.setattr(reuse.github, "paginate", fake_paginated_github_api)

    assert reuse.find_reuse_authorization(
        "SemiAnalysisAI/InferenceX",
        1321,
        "token",
        "/reuse-sweep-run",
        {"OWNER", "MEMBER", "COLLABORATOR"},
    ) == (False, None)


def test_find_latest_successful_pr_run_skips_newer_failed_run(monkeypatch) -> None:
    failed_run = {
        "id": 222,
        "conclusion": "failure",
        "head_sha": "abc123",
    }
    successful_run = {
        "id": 111,
        "conclusion": "success",
        "head_sha": "abc123",
    }

    def fake_paginated_github_api(repo, path, token, item_key, params=None):
        assert path == "/actions/workflows/run-sweep.yml/runs"
        return [failed_run, successful_run]

    monkeypatch.setattr(reuse.github, "paginate", fake_paginated_github_api)
    monkeypatch.setattr(reuse, "artifact_names", lambda *args: {"results_bmk"})

    assert (
        reuse.find_latest_successful_pr_run(
            "SemiAnalysisAI/InferenceX",
            "run-sweep.yml",
            "feature-branch",
            {"abc123"},
            "token",
        )
        == successful_run
    )


def test_find_latest_successful_pr_run_skips_gated_noop_run(monkeypatch) -> None:
    gated_run = {
        "id": 333,
        "conclusion": "success",
        "head_sha": "def456",
    }
    real_run = {
        "id": 222,
        "conclusion": "success",
        "head_sha": "abc123",
    }

    def fake_paginated_github_api(repo, path, token, item_key, params=None):
        assert path == "/actions/workflows/run-sweep.yml/runs"
        return [gated_run, real_run]

    artifacts_by_run = {333: set(), 222: {"results_bmk"}}
    monkeypatch.setattr(reuse.github, "paginate", fake_paginated_github_api)
    monkeypatch.setattr(
        reuse, "artifact_names", lambda repo, run_id, token: artifacts_by_run[run_id]
    )

    assert (
        reuse.find_latest_successful_pr_run(
            "SemiAnalysisAI/InferenceX",
            "run-sweep.yml",
            "feature-branch",
            {"abc123", "def456"},
            "token",
        )
        == real_run
    )


def test_find_latest_successful_pr_run_accepts_agentic_only_run(
    monkeypatch,
) -> None:
    agentic_run = {
        "id": 444,
        "conclusion": "success",
        "head_sha": "abc123",
    }

    monkeypatch.setattr(
        reuse.github,
        "paginate",
        lambda *args, **kwargs: [agentic_run],
    )
    monkeypatch.setattr(
        reuse,
        "artifact_names",
        lambda *args: {"bmk_agentic_dsv4_tp8_conc16"},
    )

    assert (
        reuse.find_latest_successful_pr_run(
            "SemiAnalysisAI/InferenceX",
            "run-sweep.yml",
            "feature-branch",
            {"abc123"},
            "token",
        )
        == agentic_run
    )


def test_artifact_names_excludes_expired_artifacts(monkeypatch) -> None:
    monkeypatch.setattr(
        reuse.github,
        "paginate",
        lambda *args, **kwargs: [
            {"name": "results_bmk", "expired": True},
            {"name": "run-stats", "expired": False},
        ],
    )

    assert reuse.artifact_names("repo", 123, "token") == {"run-stats"}


@pytest.mark.parametrize("has_artifacts", [True, False])
def test_main_checks_source_before_skipping_pr_synchronize(
    monkeypatch, tmp_path, has_artifacts
) -> None:
    comments = [
        {
            "created_at": "2026-05-13T00:00:00Z",
            "author_association": "MEMBER",
            "body": "/reuse-sweep-run",
        },
    ]

    def fake_paginated_github_api(repo, path, token, item_key, params=None):
        if path == "/issues/1321/comments":
            return comments
        if path == "/pulls/1321/commits":
            return [{"sha": "abc123"}]
        if path == "/actions/workflows/run-sweep.yml/runs":
            return [{"id": 123, "event": "pull_request", "status": "completed",
                     "conclusion": "success", "head_sha": "abc123"}]
        if path == "/actions/runs/123/artifacts":
            return [{"name": "results_bmk"}] if has_artifacts else []
        raise AssertionError(path)

    def fake_github_api(repo, path, token, params=None):
        assert path == "/pulls/1321"
        return {"labels": [], "head": {"ref": "feature-branch"}}

    monkeypatch.setattr(reuse.github, "api", fake_github_api)
    output_path = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(reuse.github, "paginate", fake_paginated_github_api)
    monkeypatch.setattr(
        reuse.sys,
        "argv",
        [
            "find_reusable_sweep_run.py",
            "--repo",
            "SemiAnalysisAI/InferenceX",
            "--commit-sha",
            "abc123",
            "--event-name",
            "pull_request",
            "--event-action",
            "synchronize",
            "--pr-number",
            "1321",
            "--ref",
            "refs/pull/1321/merge",
            "--github-output",
            str(output_path),
        ],
    )

    if not has_artifacts:
        with pytest.raises(RuntimeError, match="no successful"):
            reuse.main()
        assert not output_path.exists()
        return
    assert reuse.main() == 0

    outputs = dict(line.split("=", 1) for line in output_path.read_text().splitlines())
    assert outputs["reuse-enabled"] == "false"
    assert outputs["skip-pr-sweep"] == "true"
    assert outputs["reuse-source-pr-number"] == "1321"


def test_main_allows_pr_synchronize_without_reuse_authorization(
    monkeypatch, tmp_path
) -> None:
    def fake_paginated_github_api(repo, path, token, item_key, params=None):
        assert path == "/issues/1321/comments"
        return [
            {
                "created_at": "2026-05-13T00:00:00Z",
                "author_association": "CONTRIBUTOR",
                "body": "/reuse-sweep-run",
            },
        ]

    output_path = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(reuse.github, "paginate", fake_paginated_github_api)
    monkeypatch.setattr(
        reuse.sys,
        "argv",
        [
            "find_reusable_sweep_run.py",
            "--repo",
            "SemiAnalysisAI/InferenceX",
            "--commit-sha",
            "abc123",
            "--event-name",
            "pull_request",
            "--event-action",
            "synchronize",
            "--pr-number",
            "1321",
            "--ref",
            "refs/pull/1321/merge",
            "--github-output",
            str(output_path),
        ],
    )

    assert reuse.main() == 0

    outputs = dict(line.split("=", 1) for line in output_path.read_text().splitlines())
    assert outputs["reuse-enabled"] == "false"
    assert outputs["skip-pr-sweep"] == "false"


def test_main_does_not_check_reuse_comment_for_label_event(
    monkeypatch, tmp_path
) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("comments should not be queried for label events")

    output_path = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(reuse.github, "paginate", fail_if_called)
    monkeypatch.setattr(
        reuse.sys,
        "argv",
        [
            "find_reusable_sweep_run.py",
            "--repo",
            "SemiAnalysisAI/InferenceX",
            "--commit-sha",
            "abc123",
            "--event-name",
            "pull_request",
            "--event-action",
            "labeled",
            "--pr-number",
            "1321",
            "--ref",
            "refs/pull/1321/merge",
            "--github-output",
            str(output_path),
        ],
    )

    assert reuse.main() == 0

    outputs = dict(line.split("=", 1) for line in output_path.read_text().splitlines())
    assert outputs["skip-pr-sweep"] == "false"


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


def test_validate_reusable_run_accepts_failed_run_when_explicitly_allowed(
    monkeypatch,
) -> None:
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
            "conclusion": "failure",
            "path": ".github/workflows/run-sweep.yml",
            "head_sha": "abc123",
        },
        "token",
        allow_failed=True,
    )


def test_validate_reusable_run_rejects_failed_run_by_default(monkeypatch) -> None:
    monkeypatch.setattr(reuse, "artifact_names", lambda *args: {"results_bmk"})
    monkeypatch.setattr(reuse, "pr_commit_shas", lambda *args: {"abc123"})

    try:
        reuse.validate_reusable_run(
            "SemiAnalysisAI/InferenceX",
            "run-sweep.yml",
            1321,
            {
                "id": 25763404168,
                "event": "pull_request",
                "status": "completed",
                "conclusion": "failure",
                "path": ".github/workflows/run-sweep.yml",
                "head_sha": "abc123",
            },
            "token",
        )
    except RuntimeError as error:
        assert "expected success" in str(error)
    else:
        raise AssertionError("expected an unpinned failed run to be rejected")


def test_validate_reusable_run_accepts_cancelled_run_when_explicitly_allowed(
    monkeypatch,
) -> None:
    """A fail-fast sweep concludes ``cancelled`` once a job is cut short.

    Its completed benchmark jobs still uploaded usable artifacts, so a pinned
    ``cancelled`` run is reusable on the same terms as a pinned failure.
    """
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
            "conclusion": "cancelled",
            "path": ".github/workflows/run-sweep.yml",
            "head_sha": "abc123",
        },
        "token",
        allow_failed=True,
    )


def test_validate_reusable_run_rejects_cancelled_run_by_default(monkeypatch) -> None:
    monkeypatch.setattr(reuse, "artifact_names", lambda *args: {"results_bmk"})
    monkeypatch.setattr(reuse, "pr_commit_shas", lambda *args: {"abc123"})

    try:
        reuse.validate_reusable_run(
            "SemiAnalysisAI/InferenceX",
            "run-sweep.yml",
            1321,
            {
                "id": 25763404168,
                "event": "pull_request",
                "status": "completed",
                "conclusion": "cancelled",
                "path": ".github/workflows/run-sweep.yml",
                "head_sha": "abc123",
            },
            "token",
        )
    except RuntimeError as error:
        assert "expected success" in str(error)
    else:
        raise AssertionError("expected an unpinned cancelled run to be rejected")


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


@pytest.mark.parametrize("labels", [[], ["documentation"], ["sweep-enabled"], ["full-sweep-enabled"]])
def test_main_enables_pinned_reuse_without_sweep_label(monkeypatch, tmp_path, labels) -> None:
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
                "labels": [{"name": name} for name in labels],
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
    monkeypatch.setattr(reuse.github, "api", fake_github_api)
    monkeypatch.setattr(reuse.github, "paginate", fake_paginated_github_api)
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


def test_main_enables_explicitly_pinned_failed_run(monkeypatch, tmp_path) -> None:
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
        "conclusion": "failure",
        "path": ".github/workflows/run-sweep.yml",
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
    monkeypatch.setattr(reuse.github, "api", fake_github_api)
    monkeypatch.setattr(reuse.github, "paginate", fake_paginated_github_api)
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


@pytest.mark.parametrize("labels", [[], ["full-sweep-enabled"]])
def test_main_resolves_no_arg_command_to_latest_head_sweep(monkeypatch, tmp_path, labels) -> None:
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
                "labels": [{"name": name} for name in labels],
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
    monkeypatch.setattr(reuse.github, "api", fake_github_api)
    monkeypatch.setattr(reuse.github, "paginate", fake_paginated_github_api)
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
    monkeypatch.setattr(reuse.github, "api", fake_github_api)
    monkeypatch.setattr(reuse.github, "paginate", fake_paginated_github_api)
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
    monkeypatch.setattr(reuse.github, "api", fake_github_api)
    monkeypatch.setattr(reuse.github, "paginate", fake_paginated_github_api)
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


def test_main_accepts_all_evals_with_non_canary_full_sweep_label(
    monkeypatch,
    tmp_path,
) -> None:
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
                "labels": [
                    {"name": "non-canary-full-sweep-enabled"},
                    {"name": "all-evals"},
                ],
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
    monkeypatch.setattr(reuse.github, "api", fake_github_api)
    monkeypatch.setattr(reuse.github, "paginate", fake_paginated_github_api)
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


@pytest.mark.parametrize("incompatible_label", ["evals-only", "agentx-fast"])
def test_main_rejects_incompatible_label_for_reuse(
    monkeypatch,
    tmp_path,
    incompatible_label,
) -> None:
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
                "labels": [
                    {"name": "full-sweep-enabled"},
                    {"name": incompatible_label},
                ],
                "head": {"sha": "abc123"},
            }
        raise AssertionError(f"unexpected GitHub API path: {path}")

    def fake_paginated_github_api(repo, path, token, item_key, params=None):
        if path == "/issues/1321/comments":
            return comments
        raise AssertionError(f"unexpected paginated GitHub API path: {path}")

    output_path = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(reuse.github, "api", fake_github_api)
    monkeypatch.setattr(reuse.github, "paginate", fake_paginated_github_api)
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

    with pytest.raises(
        RuntimeError,
        match=rf"reuse-incompatible.*{incompatible_label}",
    ):
        reuse.main()
