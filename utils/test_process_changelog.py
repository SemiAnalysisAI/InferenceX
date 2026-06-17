from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import process_changelog


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(repo: Path, content: bytes, message: str) -> str:
    (repo / "perf-changelog.yaml").write_bytes(content)
    _git(repo, "add", "perf-changelog.yaml")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def changelog_repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", "Test User")
    _git(tmp_path, "config", "user.email", "test@example.com")
    return tmp_path


def test_get_added_lines_accepts_append_after_missing_newline(
    changelog_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_content = b"- config-keys:\n    - old\n  pr-link: old"
    appended = b"\n\n- config-keys:\n    - new\n  pr-link: new\n"
    base_ref = _commit(changelog_repo, base_content, "base")
    head_ref = _commit(changelog_repo, base_content + appended, "append")
    monkeypatch.chdir(changelog_repo)

    assert (
        process_changelog.get_added_lines(
            base_ref, head_ref, str(changelog_repo / "perf-changelog.yaml")
        ).encode()
        == appended
    )


def test_get_added_lines_rejects_non_whitespace_deletion(
    changelog_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_ref = _commit(changelog_repo, b"- old\n", "base")
    head_ref = _commit(changelog_repo, b"- new\n", "replace")
    monkeypatch.chdir(changelog_repo)

    with pytest.raises(ValueError, match="Deletions are not allowed"):
        process_changelog.get_added_lines(base_ref, head_ref, "perf-changelog.yaml")
