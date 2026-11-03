"""Agentic GSM8K evals cover every engine environment a deployment's recipes serve with."""

import pytest
import yaml

from infx.matrix.generate import generate_config_matrix
from infx.matrix.validation import load_config_files, load_runner_file
from infx.tests.srt_recipes import single_node_fragment, write_shared_blocks

SINGLE = "benchmarks/single_node/srt-slurm-recipes/fixture"
MULTI = "benchmarks/multi_node/srt-slurm-recipes/fixture"
ROLES = {role: {"num-worker": 1, "tp": 8, "ep": 1, "dp-attn": False} for role in ("prefill", "decode")}


def config(multinode: bool, *spaces: dict) -> dict:
    return {
        "image": "example/image:1", "model": "org/model", "model-prefix": "dsr1",
        "precision": "fp8", "framework": "sglang", "runner": "cluster:fixture",
        "multinode": multinode, "srt-recipe-dir": "fixture",
        "scenarios": {"agentic-coding": [{"search-space": list(spaces)}]},
        **({"disagg": True, "kv-p2p-transfer": "nixl"} if multinode else {}),
    }  # fmt: skip


def point(conc: int, **env: str) -> dict:
    return {"roles": {"agg": {"env": env}}, "benchmark": {"env": {"CONC": str(conc)}}}


MASTER = {
    "single": config(False, {"tp": 4, "kv-offloading": "none", "conc-list": [1, 2, 4, 8],
                             "srt-recipe": "agentic.yaml"}),
    "multi": config(
        True,
        {**ROLES, "conc-list": [4], "srt-recipe": "agentic.yaml:override_mono"},
        {**ROLES, "conc-list": [8], "srt-recipe": "agentic.yaml:override_narrow"},
        {**ROLES, "conc-list": [16, 32], "srt-recipe": "agentic.yaml:override_wide"},
    ),
}  # fmt: skip
RECIPES = {
    f"{SINGLE}/agentic.yaml": {
        "base": {**single_node_fragment(4), "environment": {"PYTHONHASHSEED": "0"}},
        "override_c1": point(1, MONO="1"),
        "override_c2": point(2, MONO="1"),
        "override_c4": point(4),
        "override_c8": point(8),
    },
    f"{MULTI}/agentic.yaml": {
        "base": {"schema": 2, "engine": "sglang", "roles": {
            "prefill": {"nodes": 1}, "decode": {"nodes": 1},
        }},
        "override_mono": {"roles": {"decode": {"env": {"MONO": "1"}}}},
        "override_narrow": {"roles": {"decode": {"args": {"max-running-requests": 8}}}},
        "override_wide": {"roles": {"decode": {"args": {"max-running-requests": 32}}}},
    },
}  # fmt: skip
RUNNERS = {"labels": {"cluster:fixture": ["node-a"]}, "clusters": {"fixture": {
    "gpus-per-node": 8, "available-cpu-dram-mib": 1024000, "arch": "x86_64", "scheduler": "slurm",
    "slurm": {"partition": "batch", "exclusive": True, "srt-slurm": {
        "network-interface": "eth0", "mounts": {"/data/models": "/models"},
        "power-exporter-port": 9400, "fabric": {"rdma-device-ports": ["mlx5_0:1"]},
    }},
}}}  # fmt: skip


@pytest.fixture
def project(tmp_path):
    files = {"configs/master.yaml": MASTER, "configs/runners.yaml": RUNNERS, **RECIPES}
    for path, data in files.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(yaml.safe_dump(data))
    write_shared_blocks(tmp_path)
    return tmp_path


def gsm8k_evals(project, key: str) -> list[tuple[str, int]]:
    master = load_config_files([str(project / "configs/master.yaml")])
    runners = load_runner_file(str(project / "configs/runners.yaml"))
    rows = generate_config_matrix([key], master, runners, eval_mode="subset", root=project)
    return sorted(
        (row["srt-recipe"].rpartition("/")[2], row.get("eval-conc", row["conc"]))
        for row in rows
        if row["eval-framework"] == "lm-eval"
    )


def test_single_node_evaluates_each_engine_environment_at_its_highest_conc(project):
    assert gsm8k_evals(project, "single") == [("agentic.yaml", 2), ("agentic.yaml", 8)]


def test_multi_node_evaluates_an_engine_environment_no_eligible_topology_covers(project):
    # override_narrow shares override_wide's environment, which wide's conc-32 eval covers.
    assert gsm8k_evals(project, "multi") == [
        ("agentic.yaml:override_mono", 4), ("agentic.yaml:override_wide", 32),
    ]


def test_selection_without_the_project_root_cannot_read_agentic_recipes(project):
    master = load_config_files([str(project / "configs/master.yaml")])
    runners = load_runner_file(str(project / "configs/runners.yaml"))

    with pytest.raises(ValueError, match="needs the master config's project root"):
        generate_config_matrix(["single"], master, runners, eval_mode="subset")
