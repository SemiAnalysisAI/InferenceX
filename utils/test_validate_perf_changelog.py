from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from validate_perf_changelog import (
    ChangelogValidationError,
    compare_entries,
    parse_changelog,
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


def test_run_sweep_checks_changelog_before_reuse_and_setup() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = yaml.load(
        (repo_root / ".github/workflows/run-sweep.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    jobs = workflow["jobs"]

    assert "needs" not in jobs["check-changelog"]
    assert jobs["reuse-sweep-gate"]["needs"] == "check-changelog"
    assert (
        "needs.check-changelog.result == 'success'"
        in jobs["reuse-sweep-gate"]["if"]
    )
    assert jobs["setup"]["needs"] == [
        "check-changelog",
        "reuse-sweep-gate",
    ]
    assert "needs.check-changelog.result == 'success'" in jobs["setup"]["if"]


def test_merge_helper_waits_for_changelog_check_before_merge() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "utils/merge_with_reuse.sh").read_text()

    push_index = script.index('git push origin "${LOCAL_BRANCH}:${HEAD_BRANCH}"')
    wait_index = script.index(
        'wait_for_check "$POST_MERGE" "check-changelog"'
    )
    merge_index = script.index(
        'gh pr merge "$PR" --repo "$REPO" --squash --admin'
    )

    assert push_index < wait_index < merge_index
