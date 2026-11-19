from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from infx.workflows import stack_base
from infx.workflows.stack_base import StackError, resolve

REPO = "SemiAnalysisAI/InferenceX"
PARENT = 10
CHILD = 11


def link(number: int) -> str:
    return f"https://github.com/{REPO}/pull/{number}"


def entry(key: str, number: int, *, append_only: bool = False) -> dict[str, object]:
    data: dict[str, object] = {
        "config-keys": [key],
        "description": [f"Update {key}"],
        "pr-link": link(number),
    }
    if append_only:
        data["append-only"] = True
    return data


class Repo:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.entries: list[dict[str, object]] = [entry("historic", 1)]
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "test")

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.root, check=True, capture_output=True, text=True
        ).stdout.strip()

    def commit(self, *added: dict[str, object]) -> str:
        self.entries = [*self.entries, *added]
        (self.root / "perf-changelog.yaml").write_text(
            yaml.safe_dump(self.entries, sort_keys=False)
        )
        self.git("add", "perf-changelog.yaml")
        self.git("commit", "-q", "-m", "change", "--allow-empty")
        return self.git("rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Repo:
    monkeypatch.chdir(tmp_path)
    return Repo(tmp_path)


def fake_parent(monkeypatch, sha: str, **overrides: object) -> None:
    parent = {
        "state": "open",
        "head": {"sha": sha, "repo": {"full_name": REPO}},
        "base": {"ref": "main"},
        **overrides,
    }

    def api(repo: str, path: str, token: str | None = None, **_: object) -> dict:
        assert path == f"/pulls/{PARENT}"
        return parent

    monkeypatch.setattr(stack_base.github, "api", api)


def run(head: str, base: str = "main") -> stack_base.StackBase:
    return resolve(
        repo=REPO,
        pr_number=CHILD,
        base_ref=base,
        head_ref=head,
        changelog_file="perf-changelog.yaml",
        token="token",
    )


def test_unstacked_pr_keeps_base(repo: Repo) -> None:
    repo.commit()
    repo.git("checkout", "-q", "-b", "child")
    head = repo.commit(entry("agg", CHILD))

    assert run(head).base_ref == "main"


def test_stacked_append_only_pr_plans_against_parent_head(repo: Repo, monkeypatch) -> None:
    repo.commit()
    repo.git("checkout", "-q", "-b", "parent")
    parent = repo.commit(entry("agg", PARENT))
    repo.git("checkout", "-q", "-b", "child")
    head = repo.commit(entry("agg", CHILD, append_only=True))
    fake_parent(monkeypatch, parent)

    result = run(head)

    assert result.base_ref == parent
    assert result.parent_pr == PARENT


def test_placeholder_child_link_counts_as_own(repo: Repo, monkeypatch) -> None:
    repo.commit()
    repo.git("checkout", "-q", "-b", "parent")
    parent = repo.commit(entry("agg", PARENT))
    repo.git("checkout", "-q", "-b", "child")
    child = entry("agg", CHILD, append_only=True)
    child["pr-link"] = f"https://github.com/{REPO}/pull/XXX"
    head = repo.commit(child)
    fake_parent(monkeypatch, parent)

    assert run(head).base_ref == parent


def test_regular_child_entry_keeps_base(repo: Repo, monkeypatch) -> None:
    repo.commit()
    repo.git("checkout", "-q", "-b", "parent")
    repo.commit(entry("agg", PARENT))
    repo.git("checkout", "-q", "-b", "child")
    head = repo.commit(entry("agg", CHILD))

    assert run(head).base_ref == "main"


def test_parent_behind_base_is_rejected(repo: Repo, monkeypatch) -> None:
    repo.commit()
    repo.git("checkout", "-q", "-b", "parent")
    parent = repo.commit(entry("agg", PARENT))
    repo.git("checkout", "-q", "main")
    repo.entries = repo.entries[:-1]
    repo.commit(entry("other", 9))
    repo.git("checkout", "-q", "-b", "child")
    repo.git("merge", "-q", "-X", "theirs", "--no-edit", "parent")
    head = repo.git("rev-parse", "HEAD")
    fake_parent(monkeypatch, parent)

    with pytest.raises(StackError, match="behind"):
        run(head)


def test_child_without_parent_head_is_rejected(repo: Repo, monkeypatch) -> None:
    repo.commit()
    repo.git("checkout", "-q", "-b", "parent")
    repo.commit(entry("agg", PARENT))
    repo.git("checkout", "-q", "-b", "child")
    head = repo.commit(entry("agg", CHILD, append_only=True))
    repo.git("checkout", "-q", "parent")
    moved = repo.commit(entry("disagg", 12))
    fake_parent(monkeypatch, moved)

    with pytest.raises(StackError, match="does not contain"):
        run(head)


def test_closed_parent_is_rejected(repo: Repo, monkeypatch) -> None:
    repo.commit()
    repo.git("checkout", "-q", "-b", "parent")
    parent = repo.commit(entry("agg", PARENT))
    repo.git("checkout", "-q", "-b", "child")
    head = repo.commit(entry("agg", CHILD, append_only=True))
    fake_parent(monkeypatch, parent, state="closed")

    with pytest.raises(StackError, match="'closed'"):
        run(head)


def test_fork_parent_is_rejected(repo: Repo, monkeypatch) -> None:
    repo.commit()
    repo.git("checkout", "-q", "-b", "parent")
    parent = repo.commit(entry("agg", PARENT))
    repo.git("checkout", "-q", "-b", "child")
    head = repo.commit(entry("agg", CHILD, append_only=True))
    fake_parent(monkeypatch, parent, head={"sha": parent, "repo": {"full_name": "fork/InferenceX"}})

    with pytest.raises(StackError, match="not a fork"):
        run(head)


def test_two_parents_are_rejected(repo: Repo, monkeypatch) -> None:
    repo.commit()
    repo.git("checkout", "-q", "-b", "child")
    head = repo.commit(
        entry("agg", PARENT), entry("disagg", 12), entry("agg", CHILD, append_only=True)
    )

    with pytest.raises(StackError, match="exactly one parent"):
        run(head)
