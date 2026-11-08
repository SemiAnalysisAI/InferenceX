"""``python -m infx.matrix.changed`` lists the config keys whose points differ from a base."""

import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import yaml

from infx.matrix import changed
from infx.matrix.revision import Revision
from infx.tests.srt_recipes import single_node_fragment, write_shared_blocks

ROOT = Path(__file__).resolve().parents[3]
RECIPES = "benchmarks/single_node/srt-slurm-recipes/fixture"


def entry(recipe: str, concurrencies: list[int]) -> dict:
    return {
        "image": "example/image:1", "model": "org/model", "model-prefix": "dsr1",
        "precision": "fp8", "framework": "sglang", "runner": "cluster:fixture",
        "multinode": False, "srt-recipe-dir": "fixture",
        "scenarios": {"fixed-seq-len": [{"isl": 1024, "osl": 128, "search-space": [
            {"tp": 4, "conc-list": concurrencies, "srt-recipe": recipe},
        ]}]},
    }  # fmt: skip


def write(root: Path, master: dict, edited: dict) -> None:
    (root / "configs/nvidia-master.yaml").write_text(yaml.safe_dump(master))
    (root / RECIPES / "edited.yaml").write_text(yaml.safe_dump(edited))


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout


BASE_MASTER = {
    "edited": entry("edited.yaml", [2, 4]), "grown": entry("stable.yaml", [2]),
    "same": entry("stable.yaml", [8]), "gone": entry("stable.yaml", [2]),
}  # fmt: skip
GROWN_MASTER = {
    "edited": entry("edited.yaml", [2, 4]), "grown": entry("stable.yaml", [2, 4]),
    "same": entry("stable.yaml", [8]),
}  # fmt: skip


def project(root: Path, master: dict = BASE_MASTER, **recipes: dict) -> None:
    """A project holding the real generator, ``master`` and extra ``recipes``, tagged ``base``."""
    shutil.copytree(ROOT / "infx", root / "infx", ignore=shutil.ignore_patterns("__pycache__"))
    (root / RECIPES).mkdir(parents=True)
    write_shared_blocks(root)
    (root / ".gitignore").write_text("__pycache__/\n")
    (root / "configs/amd-master.yaml").write_text("{}\n")
    (root / "configs/runners.yaml").write_text(yaml.safe_dump({
        "labels": {"cluster:fixture": ["node-a"]},
        "clusters": {"fixture": {"gpus-per-node": 8, "arch": "x86_64", "scheduler": "slurm",
                                 "slurm": {"partition": "batch", "exclusive": True}}},
    }))  # fmt: skip
    (root / RECIPES / "stable.yaml").write_text(yaml.safe_dump(single_node_fragment(4)))
    for name, recipe in recipes.items():
        (root / RECIPES / f"{name}.yaml").write_text(yaml.safe_dump(recipe))
    write(root, master, single_node_fragment(4))
    git(root, "init", "-q")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.com")
    git(root, "add", ".")
    git(root, "commit", "-qm", "base")
    git(root, "tag", "base")


def changes(report: dict) -> list[tuple]:
    return [
        (change["config-key"], change["base-points"], change["head-points"],
         [point["conc"] for point in change["added"]], [point["conc"] for point in change["removed"]])
        for change in report["changed"]
    ]  # fmt: skip


def test_keys_with_changed_recipes_or_points_are_listed(tmp_path, monkeypatch, capsys):
    project(tmp_path)
    write(tmp_path, GROWN_MASTER, single_node_fragment(4, **{"max-running-requests": 16}))
    monkeypatch.chdir(tmp_path)

    changed.main(["--base", "base", "--json"])
    report = json.loads(capsys.readouterr().out)

    assert (report["config-keys"], report["errors"]) == (4, [])
    assert changes(report) == [
        ("edited", 2, 2, [2, 4], [2, 4]), ("grown", 1, 2, [4], []), ("gone", 1, 0, [], [2]),
    ]  # fmt: skip
    assert git(tmp_path, "status", "--porcelain").splitlines() == [
        f" M {RECIPES}/edited.yaml", " M configs/nvidia-master.yaml",
    ]

    changed.main(["--base", "base"])
    assert capsys.readouterr().out.splitlines()[-1] == "3 of 4 config keys changed since base."


def run_from(root: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    """``python -m infx.matrix.changed`` run in ``root`` with ``root``'s own code."""
    return subprocess.run(
        [sys.executable, "-m", "infx.matrix.changed", *args],
        cwd=root, env={**os.environ, "INFX_REUSE_UNCHANGED": "", **env}, capture_output=True,
        text=True, check=False, timeout=120,
    )  # fmt: skip


def test_a_base_with_the_same_code_and_recipes_still_reports_every_difference(tmp_path):
    unbindable = {"unbindable": entry("unbindable.yaml", [2])}
    bound = {**single_node_fragment(4), "model": {"path": "hf:org/other"}}
    project(tmp_path, {**BASE_MASTER, **unbindable}, unbindable=bound)
    grown = [("grown", 1, 2, [4], []), ("gone", 1, 0, [], [2])]

    write(tmp_path, {**GROWN_MASTER, **unbindable}, single_node_fragment(4))
    reports = [
        json.loads(run_from(tmp_path, "--base", "base", "--json", **env).stdout)
        for env in ({}, {"INFX_REUSE_UNCHANGED": "0"})
    ]
    for report in reports:
        assert changes(report) == grown
        base_error, head_error = report["errors"]
        assert (base_error["config-key"], base_error["revision"]) == ("unbindable", "base")
        assert (head_error["config-key"], head_error["revision"]) == ("unbindable", "head")
        assert str(tmp_path) in head_error["error"] and str(tmp_path) not in base_error["error"]

    (tmp_path / "configs/nvidia-master.yaml").write_text(
        yaml.safe_dump({"every*": entry("stable.yaml", [2])})
    )
    invalid = run_from(tmp_path, "--base", "base")
    assert invalid.returncode == 1
    assert "Wildcard '*' is not allowed" in invalid.stderr

    write(tmp_path, {**GROWN_MASTER, **unbindable}, single_node_fragment(4, **{"max-running-requests": 16}))
    git(tmp_path, "add", RECIPES)
    git(tmp_path, "commit", "-qm", "edit")
    edited = run_from(tmp_path, "--base", "base", "--json")
    assert changes(json.loads(edited.stdout)) == [("edited", 2, 2, [2, 4], [2, 4]), *grown]

    write(tmp_path, {**BASE_MASTER, "late": entry("late.yaml", [2])}, single_node_fragment(4))
    git(tmp_path, "commit", "-qam", "late")
    (tmp_path / RECIPES / "late.yaml").write_text(yaml.safe_dump(single_node_fragment(4)))
    untracked = run_from(tmp_path, "--base", "HEAD")
    assert untracked.returncode == 1
    assert "HEAD could not generate its matrix" in untracked.stderr


def test_only_keys_whose_entry_changed_run_the_bases_generator(tmp_path, monkeypatch, capsys):
    project(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(changed, "__file__", str(tmp_path / "infx/matrix/changed.py"))
    generated = []
    generate_each = Revision.generate_each

    def recording(self, requests, workers):
        generated.extend(request[0][0] for request in requests)
        return generate_each(self, requests, workers)

    monkeypatch.setattr(Revision, "generate_each", recording)
    # The head loads only after the inputs are compared, so a missing wait or join shows.
    compared = threading.Event()
    load, same_inputs = changed.load_config_files, changed.same_inputs

    def load_after_comparison(*args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            compared.wait(timeout=5)
        return load(*args, **kwargs)

    def comparison(*args):
        try:
            return same_inputs(*args)
        finally:
            compared.set()

    monkeypatch.setattr(changed, "load_config_files", load_after_comparison)
    monkeypatch.setattr(changed, "same_inputs", comparison)

    def report(reuse: str = "") -> tuple[list, list]:
        monkeypatch.setenv("INFX_REUSE_UNCHANGED", reuse)
        compared.set() if reuse else compared.clear()
        generated.clear()
        changed.main(["--base", "base", "--json"])
        return changes(json.loads(capsys.readouterr().out)), sorted(generated)

    assert report() == ([], [])
    (tmp_path / "configs/nvidia-master.yaml").write_text(yaml.safe_dump(GROWN_MASTER))
    grown = [("grown", 1, 2, [4], []), ("gone", 1, 0, [], [2])]
    assert report() == (grown, ["gone", "grown"])
    assert report("0") == (grown, ["edited", "gone", "grown", "same"])


def test_a_bases_points_are_reused_only_when_git_vouches_for_every_other_input(
    tmp_path, monkeypatch, capfd
):
    project(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(changed, "__file__", str(tmp_path / "infx/matrix/changed.py"))
    recipe = tmp_path / RECIPES / "stable.yaml"

    def same(ref: str) -> bool:
        return changed.same_inputs(ref, Revision(tmp_path), changed.MASTER_CONFIGS, changed.RUNNER_CONFIG)

    (tmp_path / "configs/nvidia-master.yaml").write_text(yaml.safe_dump(GROWN_MASTER))
    (tmp_path / "infx/__pycache__").mkdir()
    (tmp_path / "infx/__pycache__/ignored.pyc").write_text("")
    assert same("base")
    elsewhere = tmp_path.with_name(f"{tmp_path.name}-runners.yaml")
    shutil.copyfile(tmp_path / changed.RUNNER_CONFIG, elsewhere)
    assert not changed.same_inputs("base", Revision(tmp_path), changed.MASTER_CONFIGS, str(elsewhere))

    git(tmp_path, "branch", "base")
    capfd.readouterr()
    assert same("base")
    assert capfd.readouterr().err == ""
    git(tmp_path, "branch", "-D", "base")

    (tmp_path / RECIPES / "untracked.yaml").write_text("{}\n")
    assert not same("base")
    (tmp_path / RECIPES / "untracked.yaml").unlink()

    recipe.write_text(recipe.read_text() + "# edited\n")
    assert not same("base")
    for hide, show in (("--skip-worktree", "--no-skip-worktree"),
                       ("--assume-unchanged", "--no-assume-unchanged")):  # fmt: skip
        git(tmp_path, "update-index", hide, f"{RECIPES}/stable.yaml")
        assert not same("base")
        git(tmp_path, "update-index", show, f"{RECIPES}/stable.yaml")

    git(tmp_path, "commit", "-qm", "edit", RECIPES)
    assert not same("base")
    assert same("HEAD")

    clone = tmp_path.with_name(f"{tmp_path.name}-clone")
    git(tmp_path, "clone", "-q", ".", str(clone))
    monkeypatch.chdir(clone)
    configs = [str(tmp_path / path) for path in changed.MASTER_CONFIGS]
    runners = str(tmp_path / changed.RUNNER_CONFIG)
    assert not changed.same_inputs("HEAD", Revision(clone), configs, runners)
    monkeypatch.chdir(tmp_path)

    (tmp_path / RECIPES / "link.yaml").symlink_to("stable.yaml")
    git(tmp_path, "add", RECIPES)
    git(tmp_path, "commit", "-qm", "link")
    assert not same("HEAD")


def test_rows_without_fingerprints_are_compared_individually(tmp_path):
    def matrix(*tps: int) -> changed.Matrix:
        rows = [
            {"exp-name": f"key_tp{tp}", "tp": tp, "ep": 1, "conc": [4], "srt-recipe": "gone.yaml"}
            for tp in tps
        ]
        return {"key": changed._points(rows, tmp_path)}

    base, head = matrix(4, 8), matrix(4, 2)

    assert [len(errors) for _, errors in (base["key"], head["key"])] == [2, 2]
    assert [
        (change["base-points"], change["head-points"],
         [point["tp"] for point in change["added"]], [point["tp"] for point in change["removed"]],
         [point["recipe-fingerprint"] for point in change["added"] + change["removed"]])
        for change in changed.compare(base, head)
    ] == [(2, 2, [2], [8], [None, None])]  # fmt: skip


def test_points_sharing_a_name_and_concurrency_keep_generation_order(tmp_path):
    tps = [8, 4, 2, 1, 16, 32, 64, 128]
    rows = [{"exp-name": "dsr1_8k1k", "tp": tp, "ep": 1, "conc": [32]} for tp in tps]
    head = {"key": changed._points(rows, tmp_path)}

    [change] = changed.compare({}, head)

    assert [point["tp"] for point in change["added"]] == tps


def test_a_head_config_error_wins_over_a_base_that_cannot_generate(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/nvidia-master.yaml").write_text("key: [unclosed\n")
    (tmp_path / "configs/amd-master.yaml").write_text("{}\n")
    (tmp_path / "configs/runners.yaml").write_text("labels: {}\n")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(yaml.YAMLError, match="while parsing a flow sequence"):
        changed.main(["--base", "missing-ref"])
