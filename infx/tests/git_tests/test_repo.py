"""Tests for infx.git.repo helpers."""

from __future__ import annotations

import subprocess
import textwrap

import pytest

from infx.git import repo as git_repo
from infx.git.repo import GitError, diff_added_lines, rev_parse, run_git, show_file, show_stage


# ---------------------------------------------------------------------------
# run_git
# ---------------------------------------------------------------------------


def test_run_git_success(tmp_path):
    """run_git returns CompletedProcess on success."""
    (tmp_path / "file.txt").write_text("hello\n")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    result = run_git("rev-parse", "HEAD", cwd=tmp_path)
    assert len(result.stdout.strip()) == 40


def test_run_git_failure(tmp_path):
    """run_git raises GitError on failure."""
    with pytest.raises(GitError, match="rev-parse"):
        run_git("rev-parse", "HEAD", cwd=tmp_path)


# ---------------------------------------------------------------------------
# show_file / show_stage
# ---------------------------------------------------------------------------


def test_show_file_reads_blob(tmp_path):
    """show_file returns exact bytes of a file at a ref."""
    content = b"line1\nline2\n"
    (tmp_path / "data.txt").write_bytes(content)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t.com"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "T"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    assert show_file("HEAD", "data.txt", cwd=tmp_path) == content


def test_show_file_missing_raises(tmp_path):
    """show_file raises GitError for a nonexistent path."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    with pytest.raises(GitError):
        show_file("HEAD", "nonexistent.txt", cwd=tmp_path)


def test_show_stage_delegates_to_show_file(monkeypatch):
    """show_stage calls show_file with the stage-colon prefix."""
    calls = []

    def fake_show(ref, path, *, cwd=None):
        calls.append((ref, path, cwd))
        return b"data"

    monkeypatch.setattr(git_repo, "show_file", fake_show)
    result = show_stage(2, "changelog.yaml", cwd="/repo")
    assert result == b"data"
    assert calls == [(":2", "changelog.yaml", "/repo")]


# ---------------------------------------------------------------------------
# rev_parse
# ---------------------------------------------------------------------------


def test_rev_parse_returns_sha(tmp_path):
    """rev_parse returns a 40-char hex SHA."""
    (tmp_path / "f").write_text("x")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t.com"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "T"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    sha = rev_parse("HEAD", cwd=tmp_path)
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


# ---------------------------------------------------------------------------
# diff_added_lines
# ---------------------------------------------------------------------------


def test_diff_added_lines_returns_additions(tmp_path):
    """diff_added_lines returns added lines between two refs."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t.com"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "T"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    (tmp_path / "log.txt").write_text("base\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "base"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    ).stdout.strip()

    (tmp_path / "log.txt").write_text("base\nnew line\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "add"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    ).stdout.strip()

    added = diff_added_lines(base, head, "log.txt", cwd=tmp_path)
    assert "new line" in added


def test_diff_added_lines_rejects_deletions(tmp_path):
    """diff_added_lines raises ValueError on non-whitespace deletions."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t.com"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "T"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    (tmp_path / "log.txt").write_text("original line\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "base"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    ).stdout.strip()

    (tmp_path / "log.txt").write_text("replacement\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "replace"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    ).stdout.strip()

    with pytest.raises(ValueError, match="Deletions are not allowed"):
        diff_added_lines(base, head, "log.txt", cwd=tmp_path)


# ---------------------------------------------------------------------------
# GitRepo (requires GitPython)
# ---------------------------------------------------------------------------


def test_gitrepo_rev_parse(tmp_path):
    """GitRepo.rev_parse returns a valid SHA."""
    git = pytest.importorskip("git")
    (tmp_path / "f").write_text("x")
    r = git.Repo.init(tmp_path)
    r.config_writer().set_value("user", "email", "t@t.com").release()
    r.config_writer().set_value("user", "name", "T").release()
    r.index.add(["f"])
    r.index.commit("init")

    from infx.git.repo import GitRepo

    repo = GitRepo(r)
    sha = repo.rev_parse("HEAD")
    assert len(sha) == 40


def test_gitrepo_is_clean(tmp_path):
    """GitRepo.is_clean correctly detects dirty worktree."""
    git = pytest.importorskip("git")
    (tmp_path / "f").write_text("x")
    r = git.Repo.init(tmp_path)
    r.config_writer().set_value("user", "email", "t@t.com").release()
    r.config_writer().set_value("user", "name", "T").release()
    r.index.add(["f"])
    r.index.commit("init")

    from infx.git.repo import GitRepo

    repo = GitRepo(r)
    assert repo.is_clean()
    (tmp_path / "dirty").write_text("y")
    assert not repo.is_clean()


# ---------------------------------------------------------------------------
# Package shadowing safety
# ---------------------------------------------------------------------------


def test_absolute_import_resolves_to_gitpython():
    """Verify that ``import git`` inside infx.git.repo resolves to GitPython."""
    git = pytest.importorskip("git")
    # If the import resolved to infx.git instead of the third-party git
    # package, it would not have a Repo attribute.
    assert hasattr(git, "Repo")
