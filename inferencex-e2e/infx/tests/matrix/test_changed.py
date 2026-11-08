"""``python -m infx.matrix.changed`` lists the config keys whose points differ from a base."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from infx.matrix import changed
from infx.tests.srt_recipes import (
    fixed_seq_entry,
    single_node_fragment,
    video_entry,
    video_fragment,
    write_shared_blocks,
    write_video_specs,
)

ROOT = Path(__file__).resolve().parents[3]
RECIPES = "benchmarks/single_node/srt-slurm-recipes/fixture"


def write(project: Path, master: dict, edited: dict, video: dict) -> None:
    (project / "configs/nvidia-master.yaml").write_text(yaml.safe_dump(master))
    (project / RECIPES / "edited.yaml").write_text(yaml.safe_dump(edited))
    (project / RECIPES / "video.yaml").write_text(yaml.safe_dump(video))


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout


def test_keys_with_changed_recipes_or_points_are_listed(tmp_path, monkeypatch, capsys):
    project = tmp_path / "inferencex-e2e"
    shutil.copytree(ROOT / "infx", project / "infx", ignore=shutil.ignore_patterns("__pycache__"))
    (project / RECIPES).mkdir(parents=True)
    write_shared_blocks(project)
    write_video_specs(tmp_path)
    (project / "configs/amd-master.yaml").write_text("{}\n")
    (project / "configs/runners.yaml").write_text(yaml.safe_dump({
        "labels": {"cluster:fixture": ["node-a"]},
        "clusters": {"fixture": {"gpus-per-node": 8, "arch": "x86_64", "scheduler": "slurm",
                                 "slurm": {"partition": "batch", "exclusive": True}}},
    }))  # fmt: skip
    (project / RECIPES / "stable.yaml").write_text(yaml.safe_dump(single_node_fragment(4)))
    unbound = video_fragment()
    del unbound["services"][0]["env"]["VIDEO_RUNTIME_REVISION"]
    (project / RECIPES / "unbound.yaml").write_text(yaml.safe_dump(unbound))
    write(project, {
        "edited": fixed_seq_entry("edited.yaml", [2, 4]),
        "grown": fixed_seq_entry("stable.yaml", [2]),
        "same": fixed_seq_entry("stable.yaml", [8]),
        "video": video_entry(),
        "unbound": video_entry("unbound.yaml"),
        "gone": fixed_seq_entry("stable.yaml", [2]),
    }, single_node_fragment(4), video_fragment())  # fmt: skip
    (tmp_path / ".gitignore").write_text("__pycache__/\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "base")
    git(tmp_path, "tag", "base")
    write(project, {
        "edited": fixed_seq_entry("edited.yaml", [2, 4]),
        "grown": fixed_seq_entry("stable.yaml", [2, 4]),
        "same": fixed_seq_entry("stable.yaml", [8]),
        "video": video_entry(),
        "unbound": video_entry("unbound.yaml"),
    }, single_node_fragment(4, **{"max-running-requests": 16}), video_fragment(**{"max-running-requests": "7"}))  # fmt: skip
    (project / RECIPES / "unbound.yaml").write_text(yaml.safe_dump(video_fragment()))
    monkeypatch.chdir(project)

    changed.main(["--base", "base", "--json"])
    report = json.loads(capsys.readouterr().out)

    assert report["config-keys"] == 6
    [error] = report["errors"]
    assert (error["config-key"], error["revision"]) == ("unbound", "base")
    assert error["error"].endswith("video service requires a pinned VIDEO_RUNTIME_REVISION")
    assert [
        (change["config-key"], change["base-points"], change["head-points"],
         [point["conc"] for point in change["added"]], [point["conc"] for point in change["removed"]])
        for change in report["changed"]
    ] == [
        ("edited", 2, 2, [2, 4], [2, 4]), ("grown", 1, 2, [4], []), ("unbound", 0, 1, [1], []),
        ("video", 1, 1, [1], [1]), ("gone", 1, 0, [], [2]),
    ]  # fmt: skip
    assert git(tmp_path, "status", "--porcelain").splitlines() == [
        f" M inferencex-e2e/{RECIPES}/edited.yaml",
        f" M inferencex-e2e/{RECIPES}/unbound.yaml",
        f" M inferencex-e2e/{RECIPES}/video.yaml",
        " M inferencex-e2e/configs/nvidia-master.yaml",
    ]

    changed.main(["--base", "base"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[-2] == "5 of 6 config keys changed since base."
    assert lines[-1].startswith("warning: 1 rows of unbound have no fingerprint at base: ")


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
