from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from validate_perf_changelog import (
    ChangelogValidationError,
    compare_entries,
    parse_changelog,
    validate_matrix_compatible_change,
    validate_raw_change,
)


def entry(
    key: str,
    link: str = "https://github.com/SemiAnalysisAI/InferenceX/pull/1",
) -> dict[str, object]:
    return {
        "config-keys": [key],
        "description": [f"Update {key}"],
        "pr-link": link,
    }


def render(entries: list[dict[str, object]]) -> bytes:
    return yaml.safe_dump(entries, sort_keys=False).encode()


def test_parse_changelog_validates_complete_file() -> None:
    parsed = parse_changelog(render([entry("config-a")]), "test changelog")

    assert parsed == [entry("config-a")]


def test_parse_changelog_rejects_missing_final_newline() -> None:
    raw = render([entry("config-a")]).rstrip(b"\n")

    with pytest.raises(ChangelogValidationError, match="end with a newline"):
        parse_changelog(raw, "test changelog")


def test_parse_changelog_rejects_malformed_nested_entry() -> None:
    raw = b"""- config-keys:
    - config-a
  description:
    - Update config-a
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/1
  - config-keys:
    - config-b
  description:
    - Update config-b
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/2
"""

    with pytest.raises(ChangelogValidationError, match="not valid YAML"):
        parse_changelog(raw, "test changelog")


def test_parse_changelog_rejects_duplicate_mapping_keys() -> None:
    raw = b"""- config-keys:
    - config-a
  description:
    - First
  description:
    - Second
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/1
"""

    with pytest.raises(ChangelogValidationError, match="duplicate key"):
        parse_changelog(raw, "test changelog")


def test_compare_entries_allows_appended_pr_entry() -> None:
    base = [entry("config-a")]
    added = entry("config-b", "XXX")

    additions, corrections = compare_entries(base, [*base, added], 42)

    assert additions == [added]
    assert corrections == 0


def test_compare_entries_rejects_wrong_pr_link_on_append() -> None:
    base = [entry("config-a")]
    added = entry(
        "config-b",
        "https://github.com/SemiAnalysisAI/InferenceX/pull/41",
    )

    with pytest.raises(ChangelogValidationError, match="new PR entry"):
        compare_entries(base, [*base, added], 42)


def test_compare_entries_requires_canonical_link_on_main() -> None:
    base = [entry("config-a")]

    with pytest.raises(ChangelogValidationError, match="main-branch entry"):
        compare_entries(base, [*base, entry("config-b", "XXX")], None)


def test_compare_entries_allows_pr_link_only_correction() -> None:
    base = [entry("config-a", "XXX")]
    head = [
        entry(
            "config-a",
            "https://github.com/SemiAnalysisAI/InferenceX/pull/42",
        )
    ]

    additions, corrections = compare_entries(base, head, 99)

    assert additions == []
    assert corrections == 1


def test_compare_entries_rejects_existing_content_change() -> None:
    base = [entry("config-a")]
    head = [entry("config-a")]
    head[0]["description"] = ["Different description"]

    with pytest.raises(ChangelogValidationError, match="entry 1 changed"):
        compare_entries(base, head, 42)


def test_compare_entries_rejects_deleted_entry() -> None:
    with pytest.raises(ChangelogValidationError, match="entries were deleted"):
        compare_entries([entry("config-a")], [], 42)


def test_compare_entries_rejects_correction_mixed_with_append() -> None:
    base = [entry("config-a", "XXX")]
    head = [
        entry(
            "config-a",
            "https://github.com/SemiAnalysisAI/InferenceX/pull/42",
        ),
        entry("config-b", "XXX"),
    ]

    with pytest.raises(ChangelogValidationError, match="do not mix"):
        compare_entries(base, head, 42)


def test_raw_append_requires_exact_historical_prefix() -> None:
    base = render([entry("config-a")])
    appended = base + b"\n" + render([entry("config-b", "XXX")])

    validate_raw_change(base, appended, additions=1, corrections=0)

    changed_history = appended.replace(b"Update config-a", b"Update config-a ")
    with pytest.raises(ChangelogValidationError, match="historical"):
        validate_raw_change(
            base,
            changed_history,
            additions=1,
            corrections=0,
        )


def test_raw_append_requires_exact_separator_and_final_newline() -> None:
    base = render([entry("config-a")])
    first = render([entry("config-b", "XXX")])
    second = render([entry("config-c", "XXX")])

    with pytest.raises(ChangelogValidationError, match="separator line"):
        validate_raw_change(
            base,
            base + b"\n" + first + b"\n\n" + second,
            additions=2,
            corrections=0,
        )

    with pytest.raises(ChangelogValidationError, match="end with one newline"):
        validate_raw_change(
            base,
            base + b"\n" + first + b"\n",
            additions=1,
            corrections=0,
        )


def test_raw_correction_rejects_whitespace_only_history_change() -> None:
    base = render([entry("config-a", "XXX")])
    corrected = base.replace(
        b"  pr-link: XXX\n",
        b"  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/42\n",
    )

    validate_raw_change(base, corrected, additions=0, corrections=1)

    changed_whitespace = corrected.replace(
        b"  - Update config-a\n",
        b"  - Update config-a  \n",
    )
    with pytest.raises(ChangelogValidationError, match="outside a pr-link"):
        validate_raw_change(
            base,
            changed_whitespace,
            additions=0,
            corrections=1,
        )


def test_matrix_compatible_check_rejects_missing_final_newline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "validate_perf_changelog.read_git_file",
        lambda *_args: b"- config-keys: []",
    )

    with pytest.raises(ChangelogValidationError, match="end with a newline"):
        validate_matrix_compatible_change("base", "head", "file")


def test_matrix_compatible_check_propagates_matrix_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "validate_perf_changelog.read_git_file",
        lambda *_args: b"- config-keys: []\n",
    )

    def reject_matrix(*_args: object, **_kwargs: object) -> None:
        raise ChangelogValidationError("matrix rejected")

    monkeypatch.setattr(
        "validate_perf_changelog.validate_generated_config",
        reject_matrix,
    )

    with pytest.raises(ChangelogValidationError, match="matrix rejected"):
        validate_matrix_compatible_change("base", "head", "file")


def test_matrix_compatible_check_forwards_eval_modifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "validate_perf_changelog.read_git_file",
        lambda *_args: b"- config-keys: []\n",
    )
    calls: list[tuple[bool, bool]] = []

    def capture_matrix(
        _base_ref: str,
        _head_ref: str,
        _path: str,
        *,
        all_evals: bool = False,
        evals_only: bool = False,
    ) -> None:
        calls.append((all_evals, evals_only))

    monkeypatch.setattr(
        "validate_perf_changelog.validate_generated_config",
        capture_matrix,
    )

    validate_matrix_compatible_change(
        "base",
        "head",
        "file",
        all_evals=True,
        evals_only=True,
    )

    assert calls == [(True, True)]


def test_matrix_compatible_check_rejects_pr_1717_conflict_resolution() -> None:
    with pytest.raises(
        ChangelogValidationError,
        match=r"Found deleted line: +pr-link: .*pull/1798",
    ):
        validate_matrix_compatible_change(
            "add33814cce15d0b71e3c98eca4bb2f7ad8aba96",
            "60bf726a7f324a01e8850d228c8f0f7a6f203dbd",
            "perf-changelog.yaml",
        )


def test_run_sweep_checks_changelog_before_reuse_and_setup() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    workflow = yaml.load(
        (repo_root / ".github/workflows/run-sweep.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    jobs = workflow["jobs"]
    pull_request_types = workflow["on"]["pull_request"]["types"]

    assert "check-changelog" in jobs
    assert "check-newline" not in jobs
    # opened/reopened are intentionally excluded so opening or reopening a PR
    # that already carries a sweep label does not start a sweep.
    assert {"synchronize", "labeled", "unlabeled", "ready_for_review"}.issubset(
        set(pull_request_types)
    )
    assert {"opened", "reopened"}.isdisjoint(set(pull_request_types))
    check_step_names = [
        step.get("name")
        for step in jobs["check-changelog"]["steps"]
    ]
    setup_step_names = [step.get("name") for step in jobs["setup"]["steps"]]
    assert "Reject conflicting sweep labels" in check_step_names
    assert "Reject conflicting sweep labels" not in setup_step_names
    conflict_script = next(
        step["run"]
        for step in jobs["check-changelog"]["steps"]
        if step.get("name") == "Reject conflicting sweep labels"
    )
    assert '"all-evals"' not in conflict_script
    assert '"evals-only"' not in conflict_script
    assert "needs" not in jobs["check-changelog"]
    assert (
        jobs["check-changelog"]["outputs"]["skip-pr-sweep"]
        == "${{ steps.sweep_policy.outputs.skip-pr-sweep }}"
    )
    assert jobs["reuse-sweep-gate"]["needs"] == "check-changelog"
    assert "classify-priority" not in jobs
    assert jobs["setup"]["needs"] == ["check-changelog", "reuse-sweep-gate"]
    classifier = next(
        step for step in jobs["setup"]["steps"] if step.get("id") == "fable"
    )
    assert classifier["if"] == "vars.PRIORITY_SCHEDULER_ENABLED == 'true'"
    assert (
        "needs.check-changelog.result == 'success'"
        in jobs["reuse-sweep-gate"]["if"]
    )
    assert "needs.check-changelog.result == 'success'" in jobs["setup"]["if"]
    assert "needs.check-changelog.result == 'skipped'" in jobs["setup"]["if"]
    check_script = "\n".join(
        step.get("run", "")
        for step in jobs["check-changelog"]["steps"]
    )
    assert "validate_perf_changelog.py" in check_script
    assert "--base-ref" in check_script
    assert "--head-ref" in check_script
    assert "--all-evals" in check_script
    assert "--evals-only" in check_script
    assert "git log -1 --format=%B" in check_script
    assert "[skip-sweep]" in check_script
    setup_script = "\n".join(
        step.get("run", "")
        for step in jobs["setup"]["steps"]
    )
    assert "--all-evals" in setup_script
    assert "--evals-only" in setup_script
    assert (
        "!contains(github.event.pull_request.labels.*.name, 'evals-only')"
        in jobs["reuse-sweep-gate"]["if"]
    )
    assert (
        "!contains(github.event.pull_request.labels.*.name, 'all-evals')"
        not in jobs["reuse-sweep-gate"]["if"]
    )
    setup_if = jobs["setup"]["if"]
    assert "needs.check-changelog.outputs.skip-pr-sweep != 'true'" in setup_if
    assert "github.event_name == 'push'" in setup_if
    assert "github.event.head_commit.message" not in setup_if


def test_merge_helper_waits_for_pr_checks_before_merge() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = (repo_root / "utils/merge_with_reuse.sh").read_text()

    push_index = script.index('git push origin "${LOCAL_BRANCH}:${HEAD_BRANCH}"')
    wait_index = script.index(
        'wait_for_check "$POST_MERGE" "check-changelog"'
    )
    checks_index = script.index(
        'gh pr checks "$PR" --repo "$REPO" --watch --fail-fast'
    )
    merge_index = script.index(
        'gh pr merge "$PR" --repo "$REPO" --squash --admin'
    )

    assert push_index < wait_index < checks_index < merge_index
    assert "CHECK_TIMEOUT_SECONDS" in script
    assert "prepare_perf_changelog_merge.py" in script
    assert "git commit --allow-empty" in script
    assert "uses all-evals, which is not eligible for artifact reuse" not in script
    assert "uses evals-only, which is not eligible for artifact reuse" in script
    assert script.count('CURRENT_HEAD="$(gh pr view') == 2
    assert "must have exactly one sweep label" in script
