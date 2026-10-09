"""``python -m infx.matrix.changed`` lists the config keys whose points differ from a base."""

import json
import shutil
import subprocess
from pathlib import Path

import yaml

from infx.matrix import changed
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


def test_keys_with_changed_recipes_or_points_are_listed(tmp_path, monkeypatch, capsys):
    # The base revision generates with its own committed generator.
    shutil.copytree(ROOT / "infx", tmp_path / "infx", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / RECIPES).mkdir(parents=True)
    write_shared_blocks(tmp_path)
    (tmp_path / "configs/amd-master.yaml").write_text("{}\n")
    (tmp_path / "configs/runners.yaml").write_text(yaml.safe_dump({
        "labels": {"cluster:fixture": ["node-a"]},
        "clusters": {"fixture": {"gpus-per-node": 8, "arch": "x86_64", "scheduler": "slurm",
                                 "slurm": {"partition": "batch", "exclusive": True}}},
    }))  # fmt: skip
    (tmp_path / RECIPES / "stable.yaml").write_text(yaml.safe_dump(single_node_fragment(4)))
    write(tmp_path, {
        "edited": entry("edited.yaml", [2, 4]), "grown": entry("stable.yaml", [2]),
        "same": entry("stable.yaml", [8]), "gone": entry("stable.yaml", [2]),
    }, single_node_fragment(4))  # fmt: skip
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "base")
    git(tmp_path, "tag", "base")
    # The head is the working tree: an edited fragment, a longer conc-list and a removed key.
    write(tmp_path, {
        "edited": entry("edited.yaml", [2, 4]), "grown": entry("stable.yaml", [2, 4]),
        "same": entry("stable.yaml", [8]),
    }, single_node_fragment(4, **{"max-running-requests": 16}))  # fmt: skip
    monkeypatch.chdir(tmp_path)

    changed.main(["--base", "base", "--json"])
    report = json.loads(capsys.readouterr().out)

    assert (report["config-keys"], report["errors"]) == (4, [])
    assert [
        (change["config-key"], change["base-points"], change["head-points"],
         [point["conc"] for point in change["added"]], [point["conc"] for point in change["removed"]])
        for change in report["changed"]
    ] == [("edited", 2, 2, [2, 4], [2, 4]), ("grown", 1, 2, [4], []), ("gone", 1, 0, [], [2])]  # fmt: skip
    # Generating the base leaves the checkout as it was.
    assert git(tmp_path, "status", "--porcelain").splitlines() == [
        f" M {RECIPES}/edited.yaml", " M configs/nvidia-master.yaml",
    ]

    changed.main(["--base", "base"])
    assert capsys.readouterr().out.splitlines()[-1] == "3 of 4 config keys changed since base."
