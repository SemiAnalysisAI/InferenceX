from __future__ import annotations

import pytest

from prepare_perf_changelog_merge import (
    canonicalize_appended_links,
    resolve_conflict_bytes,
)
from validate_perf_changelog import ChangelogValidationError, parse_changelog


def block(key: str, link: str) -> bytes:
    return f"""- config-keys:
    - {key}
  description:
    - "Update {key}"
  pr-link: {link}
""".encode()


def test_canonicalize_only_changes_appended_placeholder() -> None:
    base = block("historical", "XXX")
    head = base + b"\n" + block("new-config", "XXX")

    result = canonicalize_appended_links(
        base,
        head,
        42,
        "SemiAnalysisAI/InferenceX",
    )

    assert result.startswith(base)
    assert parse_changelog(result, "result")[0]["pr-link"] == "XXX"
    assert (
        parse_changelog(result, "result")[1]["pr-link"]
        == "https://github.com/SemiAnalysisAI/InferenceX/pull/42"
    )


def test_canonicalize_rejects_historical_whitespace_change() -> None:
    base = block("historical", "XXX")
    head = base.replace(b'    - "Update historical"\n', b'    - "Update historical"  \n')
    head += b"\n" + block("new-config", "XXX")

    with pytest.raises(ChangelogValidationError, match="historical"):
        canonicalize_appended_links(
            base,
            head,
            42,
            "SemiAnalysisAI/InferenceX",
        )


def test_conflict_resolution_preserves_main_bytes_and_appends_pr_entry() -> None:
    base = block(
        "base-config",
        "https://github.com/SemiAnalysisAI/InferenceX/pull/1",
    )
    pr = base + b"\n" + block("pr-config", "XXX")
    main = (
        base
        + b"  \n"
        + block(
            "main-config",
            "https://github.com/SemiAnalysisAI/InferenceX/pull/41",
        )
    )

    result = resolve_conflict_bytes(
        base,
        pr,
        main,
        42,
        "SemiAnalysisAI/InferenceX",
    )

    assert result.startswith(main)
    assert result[len(main):].startswith(b"\n- config-keys:")
    assert b"  \n" in result[:len(main)]
    entries = parse_changelog(result, "result")
    assert [entry["config-keys"][0] for entry in entries] == [
        "base-config",
        "main-config",
        "pr-config",
    ]
    assert (
        entries[-1]["pr-link"]
        == "https://github.com/SemiAnalysisAI/InferenceX/pull/42"
    )


def test_conflict_resolution_applies_only_requested_link_correction() -> None:
    old_link = "https://github.com/NVIDIA/InferenceMAX/pull/1722"
    new_link = "https://github.com/SemiAnalysisAI/InferenceX/pull/1722"
    base = block("base-config", old_link)
    pr = block("base-config", new_link)
    main = base + b"\n" + block(
        "main-config",
        "https://github.com/SemiAnalysisAI/InferenceX/pull/41",
    )

    result = resolve_conflict_bytes(
        base,
        pr,
        main,
        42,
        "SemiAnalysisAI/InferenceX",
    )

    assert result == main.replace(old_link.encode(), new_link.encode(), 1)


def test_conflict_resolution_rejects_duplicate_remaining_contribution() -> None:
    base = block(
        "base-config",
        "https://github.com/SemiAnalysisAI/InferenceX/pull/1",
    )
    contribution = block("same-config", "XXX")
    pr = base + b"\n" + contribution
    main = base + b"\n" + block(
        "same-config",
        "https://github.com/SemiAnalysisAI/InferenceX/pull/41",
    )

    with pytest.raises(ChangelogValidationError, match="no changelog contribution"):
        resolve_conflict_bytes(
            base,
            pr,
            main,
            42,
            "SemiAnalysisAI/InferenceX",
        )


def test_conflict_resolution_separates_multiple_contributions_by_one_blank_line() -> None:
    base = block("base-config", "https://github.com/SemiAnalysisAI/InferenceX/pull/1")
    pr = base + b"\n" + block("new-a", "XXX") + b"\n" + block("new-b", "XXX")
    main = base + b"\n" + block(
        "main-config",
        "https://github.com/SemiAnalysisAI/InferenceX/pull/41",
    )

    # resolve_conflict_bytes self-validates via validate_raw_change, which
    # requires exactly one blank line between appended entries — so a wrong
    # separator would raise here rather than return.
    result = resolve_conflict_bytes(
        base, pr, main, 42, "SemiAnalysisAI/InferenceX"
    )

    assert result.startswith(main)
    assert [e["config-keys"][0] for e in parse_changelog(result, "result")] == [
        "base-config",
        "main-config",
        "new-a",
        "new-b",
    ]
    # 4 entries -> 3 single-blank-line separators, no double blanks, one trailing newline
    assert result.count(b"- config-keys:") == 4
    assert result.count(b"\n\n- config-keys:") == 3
    assert b"\n\n\n" not in result
    assert result.endswith(b"\n") and not result.endswith(b"\n\n")
