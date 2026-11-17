from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from infx.matrix import changelog

LINK = "https://github.com/SemiAnalysisAI/InferenceX/pull/"


def report(*changes: tuple[str, int, list[str]]) -> dict:
    """Build an ``infx.matrix.changed`` report from (key, head points, scenarios) changes."""
    return {
        "changed": [
            {
                "config-key": key,
                "head-points": head,
                "added": [{"scenario-type": scenario} for scenario in scenarios],
                "removed": [],
            }
            for key, head, scenarios in changes
        ]
    }


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A scratch repository laid out like the project, with a legacy changelog and the directory."""
    monkeypatch.chdir(tmp_path)
    project = tmp_path / "inferencex-e2e"
    (project / "perf-changelog").mkdir(parents=True)
    (project / "perf-changelog.yaml").write_text("- legacy: entry\n")
    (project / "perf-changelog/README.md").write_text("guide\n")

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")

    def commit(files: dict[str, str], message: str = "change") -> str:
        for name, text in files.items():
            (project / name).write_text(text)
        git("add", ".")
        git("commit", "-qm", message)
        return git("rev-parse", "HEAD")

    commit({}, "base")
    monkeypatch.chdir(project)
    return git, commit


def entry_text(key: str = "dsr1-fp8-b200-sglang", **extra: object) -> str:
    return yaml.safe_dump({"config-keys": [key], "description": ["Bump image"], **extra})


def test_a_file_entry_links_the_pr_named_by_its_file():
    entry = changelog.parse_file(entry_text(), 5071, "5071.yaml")

    assert entry["pr-link"] == f"{LINK}5071"
    assert entry["config-keys"] == ["dsr1-fp8-b200-sglang"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (entry_text() + f"pr-link: {LINK}1\n", "must not set pr-link"),
        ("- config-keys: [a]\n  description: [b]\n", "one mapping"),
        (yaml.safe_dump({"config-keys": ["a"], "description": []}), "not a valid changelog"),
        ("config-keys: [a]\nconfig-keys: [b]\ndescription: [c]\n", "not valid YAML"),
    ],
)
def test_a_file_entry_is_rejected_when_malformed(text, message):
    with pytest.raises(changelog.ChangelogError, match=message):
        changelog.parse_file(text, 1, "1.yaml")


def test_added_files_ignore_files_main_gained_after_the_branch_point(repo):
    git, commit = repo
    base = git("rev-parse", "HEAD")
    git("checkout", "-qb", "pr")
    commit({"perf-changelog/7.yaml": entry_text("pr-config")})
    git("checkout", "-q", "main")
    commit({"perf-changelog/6.yaml": entry_text("main-config")})

    added = changelog.added_files("main", "pr", "perf-changelog.yaml")

    assert [(number, entry["config-keys"]) for number, entry in added] == [(7, ["pr-config"])]
    assert changelog.merge_base("main", "pr") == base


def test_added_files_refuse_to_change_a_merged_file(repo):
    git, commit = repo
    base = commit({"perf-changelog/6.yaml": entry_text("old")})
    commit({"perf-changelog/6.yaml": entry_text("rewritten")})

    with pytest.raises(changelog.ChangelogError, match="immutable.*6.yaml has status M"):
        changelog.added_files(base, "HEAD", "perf-changelog.yaml")


def test_added_files_require_a_pr_number_name(repo):
    git, commit = repo
    base = git("rev-parse", "HEAD")
    commit({"perf-changelog/dsr1-b200.yaml": entry_text()})

    with pytest.raises(changelog.ChangelogError, match="named <PR number>.yaml"):
        changelog.added_files(base, "HEAD", "perf-changelog.yaml")


def test_a_draft_lists_removed_configs_beside_the_swept_ones():
    changes = report(("new-agg", 9, ["agentic-coding"]), ("old", 0, []))

    entry = changelog.draft_entry(changes, ["Split old into agg"], ["rerun-me"])

    assert entry == {
        "config-keys": ["new-agg", "rerun-me"],
        "retired-keys": ["old"],
        "scenario-type": ["agentic-coding"],
        "description": ["Split old into agg"],
    }


def test_a_draft_for_removals_only_is_a_retirement():
    entry = changelog.draft_entry(report(("old", 0, [])), ["Retire old"], [])

    assert entry == {"config-keys": ["old"], "retired": True, "description": ["Retire old"]}


def test_a_draft_without_changes_needs_rerun_keys():
    with pytest.raises(changelog.ChangelogError, match="pass --key"):
        changelog.draft_entry(report(), ["Rerun"], [])


def test_coverage_accepts_patterns_and_reports_every_gap():
    changes = report(
        ("dsr1-fp8-b200-sglang", 4, ["fixed-seq-len"]),
        ("kimik3.1-fp4-gb300-vllm", 4, ["agentic-coding"]),
        ("old", 0, []),
    )
    entry = {"config-keys": ["dsr1-*"], "scenario-type": ["fixed-seq-len"]}

    assert changelog.coverage_errors(entry, changes) == [
        "config-keys must cover the changed configs ['kimik3.1-fp4-gb300-vllm']",
        "retired-keys must list the removed configs ['old']",
        "scenario-type must include the changed scenarios ['agentic-coding']",
    ]


def test_coverage_of_a_retirement_reads_its_config_keys():
    entry = {"config-keys": ["old"], "retired": True}

    assert changelog.coverage_errors(entry, report(("old", 0, []))) == []


@pytest.fixture
def checked(repo, monkeypatch):
    """Run ``check`` for PR 7 on a branch, with ``infx.matrix.changed`` reporting ``changes``."""
    git, commit = repo

    def run(files: dict[str, str], changes: dict, created_at: str = "2999-01-01T00:00:00+00:00"):
        monkeypatch.setattr(changelog, "derived_changes", lambda base: changes)
        git("checkout", "-qB", "pr", "main")
        if files:
            commit(files)
        return changelog.check("main", "pr", 7, created_at)

    return run


def test_check_requires_an_entry_when_configs_change(checked):
    errors = checked({}, report(("dsr1-fp8-b200-sglang", 4, ["fixed-seq-len"])))

    assert len(errors) == 1
    assert "no changelog entry; run python -m infx.matrix.changelog new --pr 7" in errors[0]


def test_check_passes_a_file_that_covers_the_diff(checked):
    changes = report(("dsr1-fp8-b200-sglang", 4, ["fixed-seq-len"]))

    assert checked({"perf-changelog/7.yaml": entry_text()}, changes) == []


def test_check_rejects_a_file_named_for_another_pr_and_a_second_file(checked):
    changes = report(("dsr1-fp8-b200-sglang", 4, ["fixed-seq-len"]))
    files = {"perf-changelog/7.yaml": entry_text(), "perf-changelog/8.yaml": entry_text()}

    errors = checked(files, changes)

    assert errors == [
        "add one changelog file per PR, found [7, 8]",
        "name the changelog file 7.yaml, not 8.yaml",
    ]


def test_check_freezes_the_legacy_file_for_prs_opened_after_the_directory(checked):
    legacy = "- legacy: entry\n" + yaml.safe_dump(
        [{"config-keys": ["dsr1-fp8-b200-sglang"], "description": ["x"], "pr-link": f"{LINK}7"}]
    )
    changes = report(("dsr1-fp8-b200-sglang", 4, ["fixed-seq-len"]))

    late = checked({"perf-changelog.yaml": legacy}, changes, "2999-01-01T00:00:00+00:00")
    early = checked({"perf-changelog.yaml": legacy}, changes, "2000-01-01T00:00:00+00:00")

    assert late == [
        "perf-changelog.yaml is frozen; run python -m infx.matrix.changelog new --pr 7 "
        "-m '<what changed>'"
    ]
    assert early == []


def test_check_warns_about_script_changes_without_an_entry(checked, capsys):
    Path("benchmarks").mkdir()

    assert checked({"benchmarks/srt_fixed_sequence.sh": "echo run\n"}, report()) == []
    assert "inferencex-e2e/benchmarks/srt_fixed_sequence.sh" in capsys.readouterr().out
