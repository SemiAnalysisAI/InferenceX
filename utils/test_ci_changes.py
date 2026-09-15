import os
import subprocess
from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope="module")
def ci_job():
    path = Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"
    return yaml.safe_load(path.read_text())["jobs"]["check"]


@pytest.fixture
def repository(tmp_path):
    def git(*args):
        return subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *args],
            cwd=tmp_path, check=True, capture_output=True, text=True,
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "CI test")
    git("config", "user.email", "ci@example.invalid")
    (tmp_path / "source.py").write_text("value = 1\n")
    (tmp_path / "README.md").write_text("Example\n")
    git("add", ".")
    git("commit", "-qm", "Initial files")
    return tmp_path, git, git("rev-parse", "HEAD")


def run_filter(ci_job, directory, base, head):
    step = next(step for step in ci_job["steps"] if step.get("id") == "changes")
    results = {}
    for entry in ci_job["strategy"]["matrix"]["include"]:
        output = directory / (entry["check"] + ".output")
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]],
            cwd=directory, capture_output=True, text=True,
            env={**os.environ, "BASE_SHA": base, "GITHUB_SHA": head,
                 "CHECK_PATHS": entry["paths"], "GITHUB_OUTPUT": str(output)},
        )
        results[entry["check"]] = (
            result.returncode, output.read_text() if output.exists() else "",
        )
    return results


@pytest.mark.parametrize("path,lint,tests", [
    (None, False, False),
    ("tool.py", True, True),
    ("infx/nested/tool.py", True, True),
    ("infx/ruff.toml", True, False),
    (".github/workflows/ci.yml", True, True),
    (".github/workflows/sweep.yml", False, True),
    (".github/scripts/comment.cjs", False, True),
    (".github/mcp-ci.json", False, True),
    (".claude/requirements-mcp.txt", False, True),
    ("configs/model.yaml", False, True),
    ("runners/launch.sh", False, True),
    ("benchmarks/fixed_seq_len/run.sh", False, True),
    ("utils/results/fixture.json", False, True),
    ("experimental/CollectiveX/fixture.csv", False, True),
    ("README.md", False, False),
    ("infx/README.md", False, False),
    (".github/workflows/README.md", False, False),
    (".claude/commands/review.md", False, False),
    ("unrelated.txt", False, False),
])
def test_changed_paths_select_work(ci_job, repository, path, lint, tests):
    directory, git, base = repository
    if path:
        target = directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("changed\n")
        git("add", ".")
        git("commit", "-qm", "Change input")
    assert run_filter(ci_job, directory, base, git("rev-parse", "HEAD")) == {
        "Lint": (0, "needed=true\n" if lint else "needed=false\n"),
        "Tests": (0, "needed=true\n" if tests else "needed=false\n"),
    }


@pytest.mark.parametrize("destination", [None, "archived.txt"])
def test_removing_python_still_runs_checks(ci_job, repository, destination):
    directory, git, base = repository
    if destination:
        git("mv", "source.py", destination)
    else:
        git("rm", "source.py")
    git("commit", "-qm", "Remove Python path")
    assert run_filter(ci_job, directory, base, git("rev-parse", "HEAD")) == {
        "Lint": (0, "needed=true\n"), "Tests": (0, "needed=true\n"),
    }


@pytest.mark.parametrize("base", ["", "0" * 40])
def test_manual_or_initial_push_runs_both(ci_job, repository, base):
    directory, git, _ = repository
    assert run_filter(ci_job, directory, base, git("rev-parse", "HEAD")) == {
        "Lint": (0, "needed=true\n"), "Tests": (0, "needed=true\n"),
    }


def test_missing_history_fails_instead_of_skipping(ci_job, repository):
    directory, git, _ = repository
    for code, output in run_filter(ci_job, directory, "f" * 40, git("rev-parse", "HEAD")).values():
        assert code != 0
        assert output == ""
