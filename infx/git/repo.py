"""Git repository operations -- subprocess and GitPython backends.

Subprocess-based functions (``show_file``, ``rev_parse``, etc.) work without
GitPython and are safe to import in any context, including ``--no-project`` CI
runs.

The ``GitRepo`` class wraps ``git.Repo`` (GitPython) and provides the higher-level
operations used by workflow automation.  It is importable only when the
``workflows`` extra is installed.
"""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import git as _gitpython_mod


class GitError(RuntimeError):
    """A git operation failed.  The message is safe to display (no tokens)."""


# ---------------------------------------------------------------------------
# Subprocess-based helpers (no GitPython required)
# ---------------------------------------------------------------------------


def run_git(
    *args: str,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    """Run a git command and raise ``GitError`` on failure.

    Never pass secrets on the command line; use *env* for ``GIT_*`` overrides.
    """
    result = subprocess.run(
        ["git", *args],
        check=False,
        cwd=cwd,
        env=env,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise GitError(f"git {args[0] if args else ''} failed: {detail}")
    return result


def show_file(ref: str, path: str, *, cwd: str | Path | None = None) -> bytes:
    """Read a repository file exactly as stored at a git ref.

    Returns raw bytes to preserve trailing newlines and binary content.
    """
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        check=False,
        cwd=cwd,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise GitError(f"could not read {path} at {ref}: {detail}")
    return result.stdout


def show_stage(stage: int, path: str, *, cwd: str | Path | None = None) -> bytes:
    """Read a file from a given git index stage during a merge conflict.

    ``stage`` is 1 (base), 2 (ours), or 3 (theirs).
    """
    return show_file(f":{stage}", path, cwd=cwd)


def rev_parse(ref: str = "HEAD", *, cwd: str | Path | None = None) -> str:
    """Resolve a git ref to its full SHA-1 hash."""
    return run_git("rev-parse", ref, cwd=cwd).stdout.strip()


def diff_added_lines(
    base_ref: str,
    head_ref: str,
    filepath: str,
    *,
    cwd: str | Path | None = None,
    allow_deletions: bool = False,
) -> str:
    """Return lines added between two refs for a file.

    Raises ``ValueError`` on non-whitespace deletions unless
    *allow_deletions* is True.
    """
    result = subprocess.run(
        ["git", "diff", base_ref, head_ref, "--", filepath],
        check=False,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    added_lines: list[str] = []
    for line in result.stdout.split("\n"):
        if line.startswith("-") and not line.startswith("---"):
            deleted_content = line[1:]
            if deleted_content.strip() and not allow_deletions:
                raise ValueError(
                    f"Deletions are not allowed in {filepath}. "
                    f"Only additions to the changelog are permitted. "
                    f"Found deleted line: {deleted_content}"
                )
        elif line.startswith("+") and not line.startswith("+++"):
            added_lines.append(line[1:])
    return "\n".join(added_lines)


def ls_tree_batch_read(
    ref: str,
    paths: list[str],
    *,
    cwd: str | Path | None = None,
) -> dict[str, bytes]:
    """Read multiple files at a ref using ``git ls-tree`` + ``git cat-file --batch``.

    Returns a dict mapping repo-relative paths to their contents.
    Raises ``GitError`` if required blobs cannot be read.
    """
    ls_result = subprocess.run(
        ["git", "ls-tree", "-r", "-z", ref, "--", *paths],
        cwd=cwd,
        capture_output=True,
        check=True,
    )
    repo_files: dict[str, bytes] = {}
    oid_to_path: dict[bytes, str] = {}
    for entry in ls_result.stdout.split(b"\0")[:-1]:
        metadata, file_path = entry.split(b"\t", 1)
        oid = metadata.split()[2]
        decoded_path = os.fsdecode(file_path)
        oid_to_path[oid] = decoded_path
        repo_files[decoded_path] = b""

    if not oid_to_path:
        return repo_files

    cat_result = subprocess.run(
        ["git", "cat-file", "--batch"],
        input=b"\n".join(oid_to_path) + b"\n",
        cwd=cwd,
        capture_output=True,
        check=True,
    )
    blobs = io.BytesIO(cat_result.stdout)
    for file_path in oid_to_path.values():
        header = blobs.readline().split()
        if len(header) != 3 or header[1] != b"blob":
            raise GitError(f"Could not read {file_path!r} at {ref!r}: {header!r}")
        content = blobs.read(int(header[2]))
        if blobs.read(1) != b"\n":
            raise GitError(f"Incomplete Git blob for {file_path!r} at {ref!r}")
        repo_files[file_path] = content

    return repo_files


# ---------------------------------------------------------------------------
# GitPython-based class (requires ``workflows`` extra)
# ---------------------------------------------------------------------------


class GitRepo:
    """Thin wrapper around GitPython for workflow git operations.

    All git operations go through this class so tests can supply a
    ``git.Repo`` backed by a temporary directory instead of the real
    checkout.
    """

    def __init__(self, repo: _gitpython_mod.Repo | None = None) -> None:
        import git as gitpython

        self._gitpython = gitpython
        self.repo = repo or gitpython.Repo(".")

    # -- queries --

    def is_clean(self) -> bool:
        return not self.repo.is_dirty(untracked_files=True)

    def current_ref(self) -> str:
        if self.repo.head.is_detached:
            return self.repo.head.commit.hexsha
        return self.repo.active_branch.name

    def rev_parse(self, ref: str = "HEAD") -> str:
        return self.repo.rev_parse(ref).hexsha

    def diff_name_only_unmerged(self) -> str:
        """Return newline-joined list of unmerged paths."""
        return self.repo.git.diff("--name-only", "--diff-filter=U")

    def diff_quiet(self, *args: str) -> bool:
        """Return True if ``git diff --quiet`` exits 0 (no changes)."""
        try:
            self.repo.git.diff("--quiet", *args)
            return True
        except self._gitpython.GitCommandError:
            return False

    def show_file(self, ref: str, path: str) -> bytes:
        """Read a file at a ref, preserving exact bytes."""
        return self.repo.git.show(
            f"{ref}:{path}",
            stdout_as_string=False,
            strip_newline_in_stdout=False,
        )

    def show_stage(self, stage: int, path: str) -> bytes:
        """Read a file from a given git index stage during a merge conflict.

        Returns raw bytes identical to ``git show :<stage>:<path>``.
        """
        return self.show_file(f":{stage}", path)

    # -- mutations --

    def fetch(self, *args: str) -> None:
        self.repo.git.fetch(*args)

    def checkout(self, *args: str) -> None:
        self.repo.git.checkout(*args)

    def merge(self, *args: str) -> int:
        """Run ``git merge`` and return the exit code (0 or non-zero)."""
        try:
            self.repo.git.merge(*args)
            return 0
        except self._gitpython.GitCommandError as exc:
            return exc.status or 1

    def add(self, *paths: str) -> None:
        self.repo.git.add(*paths)

    def commit(self, *args: str) -> None:
        self.repo.git.commit(*args)

    def push(self, *args: str) -> None:
        self.repo.git.push(*args)

    def branch_delete(self, name: str) -> None:
        with contextlib.suppress(self._gitpython.GitCommandError):
            self.repo.git.branch("-D", name)

    def merge_abort(self) -> None:
        with contextlib.suppress(self._gitpython.GitCommandError):
            self.repo.git.merge("--abort")
