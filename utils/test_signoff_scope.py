from __future__ import annotations

import base64
import copy
import json
from urllib.parse import quote

import pytest

from infx.workflows import signoff_scope


@pytest.fixture
def scope_case(monkeypatch):
    case = {
        "pr": {"number": 7, "head": {"sha": "head"}, "base": {"sha": "trusted-base", "ref": "main"}, "changed_files": 1},
        "branch_sha": "trusted-base",
        "target_ref": "main",
        "files": [{"filename": "configs/model.yaml"}],
        "codeowners": "* @SemiAnalysisAI/core\n/configs/ @admin @writer\n",
        "permissions": {"admin": {"permission": "admin", "role_name": "admin"},
                        "writer": {"permission": "write", "role_name": "write"}},
        "errors": [], "requests": [], "statuses": [],
    }

    def api(repo, path, token, params=None, *, method="GET", data=None):
        case["requests"].append((path, params))
        if path == case.get("fail_path"):
            raise RuntimeError("GitHub unavailable")
        if path == "/pulls/7":
            return copy.deepcopy(case["pr"])
        if path == f"/branches/{quote(case['target_ref'], safe='')}":
            return {"commit": {"sha": case["branch_sha"]}}
        if path == "/pulls/7/files":
            if callback := case.get("during_listing"):
                callback()
            page = int(params["page"]) - 1
            return case["files"][page * 100:(page + 1) * 100]
        if path == "/codeowners/errors":
            if params == {"ref": "stale-base"}:
                return {"errors": [{"kind": "Unknown owner"}]}
            if params != {"ref": "trusted-base"}:
                raise AssertionError("Ownership must come from the resolved target-branch commit")
            if callback := case.get("during_errors"):
                callback()
            return {"errors": case["errors"]}
        if path == "/contents/.github/CODEOWNERS":
            if params != {"ref": "trusted-base"}:
                raise AssertionError("Do not trust a PR's replacement CODEOWNERS")
            return {"type": "file", "encoding": "base64", "content": base64.b64encode(case["codeowners"].encode()).decode()}
        if path.startswith("/collaborators/"):
            return case["permissions"][path.split("/")[2]]
        if path == "/statuses/head" and method == "POST":
            case["statuses"].append(data)
            return data
        raise AssertionError((method, path))

    monkeypatch.setattr(signoff_scope.github, "api", api)
    return case


@pytest.mark.parametrize("rule,permission,expected", [
    ("@SemiAnalysisAI/core", None, []),
    ("@SEMIANALYSISAI/CORE", None, []),
    ("@admin", {"permission": "admin", "role_name": "admin"}, []),
    ("@admin @writer", {"permission": "admin", "role_name": "admin"}, ["@writer"]),
    ("@admin", {"permission": "admin", "role_name": "custom"}, ["@admin"]),
    ("@admin", {"permission": "admin"}, ["@admin"]),
    ("@admin", {"permission": "write", "role_name": "admin"}, ["@admin"]),
    ("@admin", {}, ["@admin"]),
    ("@another-org/core", None, ["@another-org/core"]),
    ("@SemiAnalysisAI/partners", None, ["@semianalysisai/partners"]),
    ("owner@example.com", None, ["owner@example.com"]),
])
def test_owner_exemptions_require_core_team_or_verified_repository_admin(scope_case, rule, permission, expected):
    scope_case["codeowners"] = f"* {rule}\n"
    scope_case["permissions"]["admin"] = permission
    assert signoff_scope.required_owners("example/repo", scope_case["pr"], "token") == expected


@pytest.mark.parametrize("filename,expected", [
    ("configs/model.yaml", ["@writer"]),
    ("configs/nested/model.yaml", ["@writer"]),
    ("configs/core-only.yaml", []),
    ("configs/unowned/model.yaml", []),
    ("configs-old/model.yaml", []),
    ("Configs/model.yaml", []),
    ("docs/configs/model.yaml", []),
    ("docs/space name.txt", ["@writer"]),
    ("docs/en/REVIEW.md", ["@writer"]),
])
def test_effective_ownership_respects_precedence_and_path_boundaries(scope_case, filename, expected):
    scope_case["codeowners"] += (
        "/configs/core-only.yaml @SemiAnalysisAI/core\n/configs/unowned/\n"
        "/docs/space\\ name.txt @writer\n/docs/**/REVIEW.md @writer\n"
    )
    scope_case["files"] = [{"filename": filename}]
    assert signoff_scope.required_owners("example/repo", scope_case["pr"], "token") == expected


@pytest.mark.parametrize("file", [
    {"filename": "archived/model.yaml", "previous_filename": "configs/model.yaml", "status": "renamed"},
    {"filename": "configs/model.yaml", "previous_filename": "docs/model.yaml", "status": "renamed"},
    {"filename": "configs/model.yaml", "status": "removed"},
])
def test_renaming_or_deleting_an_owned_file_still_requires_signoff(scope_case, file):
    scope_case["files"] = [file]
    assert signoff_scope.check_scope("example/repo", 7, "token")["required"] == "true"
    assert scope_case["statuses"] == []


def test_owned_file_after_first_page_is_not_missed(scope_case):
    scope_case["files"] = [{"filename": f"docs/{i}.md"} for i in range(100)] + [{"filename": "configs/model.yaml"}]
    scope_case["pr"]["changed_files"] = 101
    assert signoff_scope.check_scope("example/repo", 7, "token")["required"] == "true"
    assert scope_case["statuses"] == []


@pytest.mark.parametrize("base_ref", ["main", "release/next"])
def test_stale_pr_base_uses_current_target_ownership_pinned_to_one_commit(scope_case, base_ref):
    scope_case["pr"]["base"] = {"sha": "stale-base", "ref": base_ref}
    scope_case["target_ref"] = base_ref
    scope_case["during_errors"] = lambda: scope_case.update(branch_sha="newer-base")
    assert signoff_scope.check_scope("example/repo", 7, "token")["required"] == "true"
    assert scope_case["statuses"] == []


@pytest.mark.parametrize("preceding_files", [0, 99])
def test_type_change_entries_count_as_one_file_and_still_require_signoff(scope_case, preceding_files):
    scope_case["files"] = [
        {"filename": f"docs/{i}.md"} for i in range(preceding_files)
    ] + [
        {"filename": "configs/model.yaml", "status": "removed"},
        {"filename": "configs/model.yaml", "status": "added"},
    ]
    scope_case["pr"]["changed_files"] = preceding_files + 1
    assert signoff_scope.check_scope("example/repo", 7, "token") == {
        "required": "true", "pr-number": "7", "head-sha": "head",
    }
    assert scope_case["statuses"] == []


@pytest.mark.parametrize("changed", ["head", "base", "base-ref"])
def test_pr_changes_during_scope_resolution_do_not_publish_an_exemption(scope_case, changed):
    scope_case["files"] = [{"filename": "README.md"}]
    if changed == "base-ref":
        scope_case["during_listing"] = lambda: scope_case["pr"]["base"].update(ref="release/next")
    else:
        scope_case["during_listing"] = lambda: scope_case["pr"][changed].update(sha="new-commit")
    with pytest.raises(RuntimeError, match="PR changed"):
        signoff_scope.check_scope("example/repo", 7, "token")
    assert scope_case["statuses"][-1]["state"] == "error"


def test_matching_an_owner_more_than_once_checks_their_role_once(scope_case):
    scope_case["files"] = [{"filename": "configs/a.yaml"}, {"filename": "configs/b.yaml"}]
    scope_case["pr"]["changed_files"] = 2
    assert signoff_scope.required_owners("example/repo", scope_case["pr"], "token") == ["@writer"]
    assert sorted(path for path, _ in scope_case["requests"] if path.startswith("/collaborators/")) == [
        "/collaborators/admin/permission", "/collaborators/writer/permission",
    ]


@pytest.mark.parametrize("problem", ["files", "branch", "codeowners", "permission", "incomplete", "incomplete-duplicate", "invalid", "empty"])
def test_scope_failures_revoke_an_earlier_exemption(scope_case, problem):
    scope_case["statuses"].append({"context": "CODEOWNER sign-off", "state": "success", "description": "N/A"})
    if problem in {"files", "branch", "codeowners", "permission"}:
        scope_case["fail_path"] = {"files": "/pulls/7/files", "branch": "/branches/main", "codeowners": "/contents/.github/CODEOWNERS", "permission": "/collaborators/admin/permission"}[problem]
    elif problem in {"incomplete", "incomplete-duplicate"}:
        scope_case["pr"]["changed_files"] = 2
        if problem == "incomplete-duplicate":
            scope_case["files"] = [
                {"filename": "configs/model.yaml", "status": "removed"},
                {"filename": "configs/model.yaml", "status": "added"},
            ]
    elif problem == "invalid":
        scope_case["errors"] = [{"kind": "Unknown owner"}]
    else:
        scope_case["codeowners"] = ""
    with pytest.raises(RuntimeError):
        signoff_scope.check_scope("example/repo", 7, "token")
    assert scope_case["statuses"][-1] == {
        "context": "CODEOWNER sign-off", "state": "error",
        "description": "Could not determine sign-off scope",
    }


@pytest.mark.parametrize("event", [
    {"pull_request": {"number": 7}},
    {"issue": {"number": 7}},
    {"inputs": {"pr-number": "7", "comment_url": "https://github.com/example/repo/pull/7#pullrequestreview-9"}},
])
def test_unowned_changes_publish_not_applicable_and_disable_verifier(scope_case, monkeypatch, tmp_path, event):
    scope_case["files"] = [{"filename": "infx/github.py"}]
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event))
    output = tmp_path / "outputs"
    for key, value in {"GITHUB_EVENT_PATH": event_path, "GITHUB_OUTPUT": output,
                       "GITHUB_REPOSITORY": "example/repo", "GH_TOKEN": "token"}.items():
        monkeypatch.setenv(key, str(value))
    signoff_scope.main()
    assert output.read_text() == "required=false\npr-number=7\nhead-sha=head\n"
    assert scope_case["statuses"] == [{
        "context": "CODEOWNER sign-off", "state": "success",
        "description": "N/A",
    }]


@pytest.mark.parametrize("number,url", [
    (None, "https://github.com/example/repo/pull/7#issuecomment-9"),
    ("8", "https://github.com/example/repo/pull/7#issuecomment-9"),
    ("0", "https://github.com/example/repo/pull/0#issuecomment-9"),
    ("7", "not-a-pull-request"),
])
def test_manual_runs_cannot_update_a_pr_outside_their_concurrency_group(scope_case, monkeypatch, tmp_path, number, url):
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({"inputs": {"pr-number": number, "comment_url": url}}))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    with pytest.raises(ValueError, match="pr-number must match"):
        signoff_scope.main()
    assert scope_case["requests"] == []
