"""Recipe fingerprints change with the recipe a point runs, not with concurrency or clusters."""

from pathlib import Path

import pytest
import yaml

from infx.matrix.fingerprint import recipe_fingerprint
from infx.matrix.generate import generate_config_matrix
from infx.matrix.validation import load_config_files, load_runner_file
from infx.tests.srt_recipes import single_node_fragment, write_shared_blocks

SINGLE = "benchmarks/single_node/srt-slurm-recipes/fixture"
MULTI = "benchmarks/multi_node/srt-slurm-recipes/fixture"
ROLES = {role: {"num-worker": 1, "tp": 8, "ep": 1, "dp-attn": False} for role in ("prefill", "decode")}


def config(multinode: bool, scenarios: dict) -> dict:
    return {
        "image": "example/image:1", "model": "org/model", "model-prefix": "dsr1",
        "precision": "fp8", "framework": "sglang", "runner": "cluster:fixture",
        "multinode": multinode, "srt-recipe-dir": "fixture", "scenarios": scenarios,
        **({"disagg": True, "kv-p2p-transfer": "nixl"} if multinode else {}),
    }  # fmt: skip


def fixed(*spaces: dict) -> dict:
    return {"fixed-seq-len": [{"isl": 1024, "osl": 128, "search-space": list(spaces)}]}


MASTER = {
    "single": config(False, fixed({"tp": 4, "conc-list": [2, 4], "srt-recipe": "bundle.yaml"})),
    "other": config(False, fixed({"tp": 4, "conc-list": [2, 4], "srt-recipe": "plain.yaml"})),
    "multi": config(True, fixed(
        {**ROLES, "conc-list": [2, 4], "srt-recipe": "disagg.yaml:override_a"},
        {**ROLES, "conc-list": [8], "srt-recipe": "disagg.yaml:override_b"},
    )),
    "agentx": config(True, {"agentic-coding": [{"search-space": [{
        **ROLES, "conc-list": [2], "srt-recipe": "agentx.yaml:override_bench",
        "eval-srt-recipe": "agentx.yaml:override_eval",
    }]}]}),
}  # fmt: skip
RECIPES = {
    # Each concurrency pairs with its own tuning.
    f"{SINGLE}/bundle.yaml": {"base": single_node_fragment(4), "zip_override_conc": {
        "roles": {"agg": {"args": {"max-running-requests": [2, 4]}}},
        "benchmark": {"env": {"CONC": ["2", "4"]}},
    }},
    f"{SINGLE}/plain.yaml": single_node_fragment(4),
    # Telemetry binds the point's concurrencies into the recipe.
    f"{MULTI}/disagg.yaml": {
        "base": {"schema": 2, "engine": "sglang", "telemetry": {"enabled": True}, "roles": {
            "prefill": {"nodes": 1, "args": {"mem-fraction-static": 0.8}}, "decode": {"nodes": 1},
        }},
        "override_a": {"roles": {"decode": {"args": {"max-running-requests": 64}}}},
        "override_b": {"roles": {"decode": {"args": {"max-running-requests": 128}}}},
    },
    f"{MULTI}/agentx.yaml": {
        "base": {
            "schema": 2, "engine": "sglang",
            "model": {"path": "alias", "container": "example/image:1", "precision": "fp8"},
            "roles": {"prefill": {"nodes": 1}, "decode": {"nodes": 1}},
            "benchmark": {"type": "custom", "command": "bash srt_agentic.sh", "concurrencies": [2]},
        },
        "override_bench": {"roles": {"decode": {"args": {"max-running-requests": 2}}}},
        "override_eval": {"roles": {"decode": {"args": {"max-running-requests": 4}}}},
    },
}  # fmt: skip
RUNNERS = {"labels": {"cluster:fixture": ["node-a"]}, "clusters": {"fixture": {
    "gpus-per-node": 8, "available-cpu-dram-mib": 1024000, "arch": "x86_64", "scheduler": "slurm",
    "slurm": {"partition": "batch", "exclusive": True, "srt-slurm": {
        "network-interface": "eth0", "mounts": {"/data/models": "/models"},
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


def fingerprints(project: Path) -> dict[tuple, str]:
    """Each point's fingerprint, by config key, recipe and concurrency."""
    master = load_config_files([str(project / "configs/master.yaml")])
    runners = load_runner_file(str(project / "configs/runners.yaml"))
    points = {}
    for key in master:
        for row in generate_config_matrix([key], master, runners, eval_mode="none", root=project):
            recipe = row["srt-recipe"].rpartition("/")[2]
            for conc in row["conc"] if isinstance(row["conc"], list) else [row["conc"]]:
                points[key, recipe, conc] = recipe_fingerprint(row, project)
    return points


def edit(project: Path, path: str, change) -> None:
    data = yaml.safe_load((project / path).read_text())
    change(data)
    (project / path).write_text(yaml.safe_dump(data))


OTHER_POINTS = {("other", "plain.yaml", 2), ("other", "plain.yaml", 4)}
SINGLE_POINTS = {("single", "bundle.yaml", 2), ("single", "bundle.yaml", 4), *OTHER_POINTS}
MULTI_POINTS = {
    ("multi", "disagg.yaml:override_a", 2), ("multi", "disagg.yaml:override_a", 4),
    ("multi", "disagg.yaml:override_b", 8),
}  # fmt: skip


@pytest.mark.parametrize(("path", "change", "changed"), [
    pytest.param(
        f"{SINGLE}/bundle.yaml",
        lambda recipe: recipe["zip_override_conc"]["roles"]["agg"]["args"].update(
            {"max-running-requests": [2, 6]}
        ),
        {("single", "bundle.yaml", 4)}, id="single-node-variant",
    ),
    pytest.param(
        f"{MULTI}/disagg.yaml",
        lambda recipe: recipe["override_a"]["roles"]["decode"]["args"].update(
            {"max-running-requests": 65}
        ),
        {("multi", "disagg.yaml:override_a", 2), ("multi", "disagg.yaml:override_a", 4)},
        id="multi-node-variant",
    ),
    pytest.param(
        "configs/srt-recipes/fixed-sequence-single.yaml",
        lambda block: block["benchmark"].update(env={"EXTRA": "1"}),
        SINGLE_POINTS, id="single-node-shared-block",
    ),
    pytest.param(
        "configs/srt-recipes/fixed-sequence-multi.yaml",
        lambda block: block["benchmark"].update(env={"EXTRA": "1"}),
        MULTI_POINTS, id="multi-node-shared-block",
    ),
    pytest.param(
        "configs/master.yaml", lambda master: master["other"].update(image="example/image:2"),
        OTHER_POINTS, id="master-image",
    ),
    pytest.param(
        f"{MULTI}/agentx.yaml",
        lambda recipe: recipe["override_bench"]["roles"]["decode"]["args"].update(
            {"max-running-requests": 3}
        ),
        {("agentx", "agentx.yaml:override_bench", 2)}, id="agentx-variant",
    ),
    # Eval-only runs produce no benchmark results.
    pytest.param(
        f"{MULTI}/agentx.yaml",
        lambda recipe: recipe["override_eval"]["roles"]["decode"]["args"].update(
            {"max-running-requests": 5}
        ),
        set(), id="agentx-eval-variant",
    ),
    pytest.param(
        "configs/runners.yaml",
        lambda runners: runners["clusters"]["fixture"]["slurm"]["srt-slurm"].update(
            mounts={"/scratch/models": "/models"}
        ),
        set(), id="cluster-mount",
    ),
])  # fmt: skip
def test_an_edit_changes_exactly_the_points_whose_recipe_it_changes(project, path, change, changed):
    before = fingerprints(project)
    edit(project, path, change)
    after = fingerprints(project)

    assert after.keys() == before.keys()
    assert {point for point in before if after[point] != before[point]} == changed


def test_concurrency_lists_leave_the_remaining_points_alone(project):
    before = fingerprints(project)

    def concurrencies(master):
        master["single"]["scenarios"]["fixed-seq-len"][0]["search-space"][0]["conc-list"] = [4]
        master["multi"]["scenarios"]["fixed-seq-len"][0]["search-space"][0]["conc-list"] += [16]

    edit(project, "configs/master.yaml", concurrencies)
    after = fingerprints(project)

    assert before.keys() - after.keys() == {("single", "bundle.yaml", 2)}
    assert after.keys() - before.keys() == {("multi", "disagg.yaml:override_a", 16)}
    assert all(after[point] == before[point] for point in after.keys() & before.keys())
    # One recipe serves every concurrency of a row, or of a fragment without CONC variants.
    assert after["multi", "disagg.yaml:override_a", 16] == before["multi", "disagg.yaml:override_a", 2]
    assert after["other", "plain.yaml", 2] == after["other", "plain.yaml", 4]


def test_a_point_no_variant_serves_cannot_be_fingerprinted(project):
    edit(project, f"{SINGLE}/bundle.yaml", lambda recipe: recipe["base"]["roles"]["agg"].update(gpus=8))
    master = load_config_files([str(project / "configs/master.yaml")])
    runners = load_runner_file(str(project / "configs/runners.yaml"))
    [row, _] = generate_config_matrix(["single"], master, runners, eval_mode="none", root=project)

    with pytest.raises(ValueError, match=r"fixture/bundle\.yaml .*Single-node SRT gpus"):
        recipe_fingerprint(row, project)


def test_rows_without_an_srt_recipe_hash_the_row_alone():
    row = {"image": "img", "model": "m", "conc": 4, "exp-name": "x", "recipe-fingerprint": "old"}
    # sha256 of {"image":"img","model":"m"}: published results of older revisions match it.
    assert recipe_fingerprint(row, Path("/nonexistent")) == (
        "adfebb88b80b867b258d2e9f972eb51f9f8c8bb2ae2bef10ab80ca8226913878"
    )
