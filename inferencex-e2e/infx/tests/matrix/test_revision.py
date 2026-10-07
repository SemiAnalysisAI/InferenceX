"""Tests for running a revision's own matrix tooling in isolation."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from infx.matrix import revision

ROOT = Path(__file__).resolve().parents[3]


def git_in(root):
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip()
    return git


@pytest.fixture
def generation_repo(tmp_path, monkeypatch):
    """An isolated history containing the real generator and controlled inputs."""
    shutil.copytree(ROOT / "infx", tmp_path / "infx", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/amd-master.yaml").write_text("{}\n")
    (tmp_path / "configs/runners.yaml").write_text(
        "labels: {fixture: [node-a], 'cluster:fixture': [node-a]}\n"
        "clusters:\n  fixture: {gpus-per-node: 8, arch: x86_64, scheduler: slurm,\n"
        "    slurm: {partition: batch, exclusive: true}}\n"
    )
    (tmp_path / "infx/data.bin").write_bytes(b"\x00\nblob\xff\n")
    (tmp_path / "infx/data 中文\t\r\n.bin").write_bytes(b"named asset")
    (tmp_path / "infx/data-link").symlink_to("data.bin")
    (tmp_path / ".gitattributes").write_text("infx/data.bin export-ignore\n")
    recipe = tmp_path / "benchmarks/single_node/srt-slurm-recipes/fixture/recipe.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("{}\n")

    git = git_in(tmp_path)
    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    for name, conc in (("older", 2), ("newer", 6)):
        master = {"fixture": {
            "image": "example/image:stable", "model": name, "model-prefix": "dsr1",
            "precision": "fp8", "framework": "sglang", "runner": "fixture",
            "multinode": False, "srt-recipe-dir": "fixture",
            "scenarios": {"fixed-seq-len": [{
                "isl": 1024, "osl": 1024,
                "search-space": [{"tp": 1, "conc-list": [conc], "srt-recipe": "recipe.yaml"}],
            }]},
        }}
        (tmp_path / "configs/nvidia-master.yaml").write_text(yaml.safe_dump(master))
        git("add", ".")
        git("commit", "-qm", name)
        git("tag", name)

    (tmp_path / "configs/nvidia-master.yaml").write_text("invalid working tree\n")
    for path in (tmp_path / "infx").rglob("*.py"):
        path.write_text('raise RuntimeError("working tree source was used")\n')
    monkeypatch.chdir(tmp_path)
    return tmp_path, git


@pytest.mark.parametrize("ref,expected", [
    ("older", ("older", 2)), ("newer", ("newer", 6)), ("moving", ("older", 2)),
])
@pytest.mark.parametrize("safe_path", [False, True])
def test_historical_generation_uses_committed_source_and_inputs(generation_repo, ref, expected, monkeypatch, safe_path, capsys):
    root, git = generation_repo
    monkeypatch.setenv("PYTHONPATH", str(root))
    if safe_path:
        monkeypatch.setenv("PYTHONSAFEPATH", "1")
    else:
        monkeypatch.delenv("PYTHONSAFEPATH", raising=False)
    if ref == "moving":
        git("update-ref", "refs/heads/moving", "older")
        run = subprocess.run

        def advance_after_listing(command, **kwargs):
            result = run(command, **kwargs)
            if command[:2] == ["git", "ls-tree"]:
                run(["git", "update-ref", "refs/heads/moving", "newer"], cwd=root, check=True)
            return result

        monkeypatch.setattr(subprocess, "run", advance_after_listing)
    with revision.snapshot(ref) as inputs:
        rows = inputs.generate(["fixture"], ["--no-evals"])
        assert [(row["model"], row["conc"]) for row in rows] == [expected]
        assert capsys.readouterr().err == ""
        snapshot = inputs.root
        assert (snapshot / "infx/data.bin").read_bytes() == b"\x00\nblob\xff\n"
        assert (snapshot / "infx/data 中文\t\r\n.bin").read_bytes() == b"named asset"
        assert (snapshot / "infx/data-link").read_bytes() == b"data.bin"
        assert not (snapshot / "infx/data-link").is_symlink()
    assert not snapshot.exists()


def test_historical_generation_rejects_missing_inputs(generation_repo):
    _, git = generation_repo
    git("rm", "-f", "configs/runners.yaml")
    git("commit", "-qm", "missing runner inventory")

    with pytest.raises(ValueError, match="missing generation inputs.*configs/runners.yaml"):
        with revision.snapshot("HEAD"):
            pytest.fail("an incomplete snapshot must not be used for generation")


@pytest.mark.parametrize("safe_path", [False, True])
def test_historical_generation_supports_legacy_script_layout(generation_repo, monkeypatch, safe_path):
    root, git = generation_repo
    if safe_path:
        monkeypatch.setenv("PYTHONSAFEPATH", "1")
    else:
        monkeypatch.delenv("PYTHONSAFEPATH", raising=False)
    git("rm", "-rf", "--ignore-unmatch", "infx")
    script = root / "utils/matrix_logic/generate_sweep_configs.py"
    schema = script.with_name("validation.py")
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("import json\nfrom validation import revision\nprint(json.dumps([{'model': revision, 'conc': 2}]))\n")
    schema.write_text('revision = "legacy snapshot"\n')
    git("add", str(script), str(schema))
    git("commit", "-qm", "legacy generator")
    schema.write_text('raise RuntimeError("wrong revision")\n')

    with revision.snapshot("HEAD") as inputs:
        assert inputs.generate(["fixture"], ["--no-evals"]) == [{"model": "legacy snapshot", "conc": 2}]


@pytest.mark.parametrize("ref,expected", [("HEAD", ("newer", 6)), ("older", ("older", 2))])
def test_historical_generation_from_nested_project(generation_repo, monkeypatch, ref, expected):
    root, git = generation_repo
    git("restore", ".")
    (root / "inferencex-e2e").mkdir()
    git("mv", "infx", "configs", "benchmarks", "inferencex-e2e/")
    (root / "infx/matrix").mkdir(parents=True)
    (root / "infx/matrix/generate.py").write_text(
        'raise RuntimeError("root-level decoy source was used")\n'
    )
    (root / "configs").mkdir()
    (root / "configs/nvidia-master.yaml").write_text("root-level decoy config\n")
    git("add", ".")
    git("commit", "-qm", "relocate project")
    monkeypatch.chdir(root / "inferencex-e2e")
    with revision.snapshot(ref) as inputs:
        rows = inputs.generate(["fixture"], ["--no-evals"])
    assert [(row["model"], row["conc"]) for row in rows] == [expected]


LEGACY_GENERATOR = """\
import argparse, json
import yaml
from validation import runner_nodes

parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("--config-keys", nargs="+")
parser.add_argument("--config-files", nargs="+")
parser.add_argument("--runner-config", default=".github/configs/runners.yaml")
parser.add_argument("--no-evals", action="store_true")
args = parser.parse_args()
master = {}
for path in args.config_files:
    master.update(yaml.safe_load(open(path)))
runners = yaml.safe_load(open(args.runner_config))
print(json.dumps([
    {"runner": node, "conc": conc}
    for key in args.config_keys
    for node in runner_nodes(runners, master[key]["runner"])
    for conc in master[key]["conc"]
]))
"""


def test_snapshot_runs_pre_root_move_revisions_with_their_own_config_directory(tmp_path, monkeypatch):
    configs = tmp_path / ".github/configs"
    configs.mkdir(parents=True)
    (configs / "amd-master.yaml").write_text("{}\n")
    (configs / "nvidia-master.yaml").write_text(
        yaml.safe_dump({"fixture": {"runner": "fixture", "conc": [4, 8]}})
    )
    (configs / "runners.yaml").write_text("fixture: [node-a]\n")
    script = tmp_path / revision.GENERATOR.legacy_script
    script.parent.mkdir(parents=True)
    script.write_text(LEGACY_GENERATOR)
    script.with_name("validation.py").write_text(
        "def runner_nodes(runners, label):\n    return runners[label]\n"
    )
    git = git_in(tmp_path)
    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    git("add", ".")
    git("commit", "-qm", "pre-root-move layout")
    monkeypatch.chdir(tmp_path)

    with revision.snapshot("HEAD") as producer:
        assert producer.generate(["fixture"], ["--no-evals"]) == [
            {"runner": "node-a", "conc": 4}, {"runner": "node-a", "conc": 8},
        ]


PLANNER = """\
import json, os, sys
from {origin} import NAME
print(json.dumps({{"planner": NAME, "args": sys.argv[1:], "cwd": os.getcwd(),
                  "root": os.environ["INFERENCEX_REPOSITORY_ROOT"], "env": dict(os.environ)}}))
print("planner diagnostics", file=sys.stderr)
sys.exit(int(sys.argv[sys.argv.index("--exit") + 1]) if "--exit" in sys.argv else 0)
"""


def planner_checkout(root, layout):
    if layout == "module":
        (root / "infx/matrix").mkdir(parents=True)
        (root / "infx/__init__.py").write_text("")
        (root / "infx/matrix/__init__.py").write_text("")
        (root / "infx/matrix/origin.py").write_text('NAME = "checkout package"\n')
        (root / revision.PLANNER.module_path).write_text(PLANNER.format(origin="infx.matrix.origin"))
    else:
        script = root / revision.PLANNER.legacy_script
        script.parent.mkdir(parents=True)
        script.with_name("constants.py").write_text('NAME = "checkout script"\n')
        script.write_text(PLANNER.format(origin="constants"))
    return root


def run_cli(cwd, *args, **environment):
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-P", "-m", "infx.matrix.revision", *args],
        cwd=cwd, env={**env, **environment}, capture_output=True, text=True, timeout=30,
    )


@pytest.mark.parametrize("layout,name", [("module", "checkout package"), ("legacy", "checkout script")])
@pytest.mark.parametrize("safe_path", [False, True])
def test_cli_runs_the_checkouts_own_planner_and_forwards_its_output_and_status(tmp_path, layout, name, safe_path):
    checkout = planner_checkout(tmp_path / "checkout", layout)
    arguments = ["--changelog-file", "perf-changelog.yaml", "--base-ref", "base", "--head-ref", "head", "--exit", "3"]

    result = run_cli(tmp_path, "plan", str(checkout), *arguments,
                     **({"PYTHONSAFEPATH": "1"} if safe_path else {}))

    assert result.returncode == 3, result.stderr
    output = json.loads(result.stdout)
    del output["env"]
    assert output == {
        "planner": name, "args": arguments,
        "cwd": str(checkout.resolve()), "root": str(checkout.resolve()),
    }
    assert result.stderr == "planner diagnostics\n"


def test_cli_runs_the_tool_without_the_callers_credentials(tmp_path):
    checkout = planner_checkout(tmp_path / "checkout", "module")
    caller = {
        "GH_TOKEN": "ghp_fixture", "GITHUB_TOKEN": "ghs_fixture", "AGENT_PAT": "pat_fixture",
        "KLAUD_DASHBOARD_API_KEY": "key_fixture", "VIRTUAL_ENV": "/caller/venv",
        "LANG": "en_US.UTF-8", "TMPDIR": str(tmp_path),
    }  # fmt: skip

    result = run_cli(tmp_path, "plan", str(checkout), **caller)

    assert result.returncode == 0, result.stderr
    env = json.loads(result.stdout)["env"]
    assert not {"GH_TOKEN", "GITHUB_TOKEN", "AGENT_PAT", "KLAUD_DASHBOARD_API_KEY", "VIRTUAL_ENV"} & env.keys()
    assert (env["PATH"], env["LANG"], env["TMPDIR"]) == (os.environ["PATH"], "en_US.UTF-8", str(tmp_path))
    assert env["PYTHONPATH"] == str(checkout.resolve())


def test_cli_rejects_a_checkout_without_the_tool(tmp_path):
    result = run_cli(tmp_path, "generate", str(tmp_path))

    assert result.returncode == 2
    assert "has neither infx/matrix/generate.py nor utils/matrix_logic/generate_sweep_configs.py" in result.stderr
    assert result.stdout == ""
