"""Recipe preflight over controlled matrices, recipes, and runner inventories."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from infx.clusters import load_inventory
from infx.srt_slurm.preflight import check_matrix

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "utils/srt-slurm/src"))

SINGLE = "benchmarks/single_node/srt-slurm-recipes"
MULTI = "benchmarks/multi_node/srt-slurm-recipes"
INVENTORY = {
    "labels": {"cluster:c": ["c_0"], "pool": ["c_0"]},
    "clusters": {
        "c": {
            "gpus-per-node": 8, "arch": "x86_64", "scheduler": "slurm", "models": {"entries": {}},
            "slurm": {
                "partition": "batch", "exclusive": True,
                "srt-slurm": {"network-interface": "", "container-aliases": ["dynamo-sglang"]},
            },
        },
    },
}  # fmt: skip


def single_node_recipe(container: str, ep: int) -> dict:
    return {
        "engine": "sglang",
        "resources": {"gpus_per_node": 8},
        "model": {"path": "hf:test/model", "container": container, "precision": "fp8"},
        "roles": {"agg": {"nodes": 1, "workers": 1, "gpus": 8, "args": {
            "tensor-parallel-size": 8, "expert-parallel-size": ep,
        }}},
        "benchmark": {"type": "custom", "command": "bash /bench/srt_agentic.sh", "env": {"MODEL": "test/model"}},
    }  # fmt: skip


def single_node_point(recipe: str, image: str, ep: int = 1) -> dict:
    return {
        "exp-name": f"ep{ep}", "runner": "cluster:c", "srt-recipe": recipe, "image": image,
        "model": "test/model", "framework": "sglang", "precision": "fp8", "tp": 8, "ep": ep,
        "pp": 1, "dcp-size": 1, "pcp-size": 1, "dp-attn": False, "conc": 4, "spec-decoding": "none",
        "scenario-type": "agentic-coding", "kv-offloading": "none", "total-cpu-dram-gb": 0,
    }  # fmt: skip


def multi_node_point(*settings: str, image: str = "img:1", runner: str = "cluster:c",
                     framework: str = "dynamo-sglang") -> dict:  # fmt: skip
    return {
        "exp-name": "disagg", "runner": runner, "image": image, "framework": framework,
        "prefill": {"additional-settings": list(settings)}, "decode": {"additional-settings": []},
    }  # fmt: skip


def write_yaml(root: Path, relative: str, data: dict) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))
    return relative


def multi_node_recipe(root: Path, container: str, identity: str | None = None) -> str:
    recipe = {"base": {"name": "m", "model": {"path": "hf:t/m", "container": "img:1", "precision": "fp8"}},
              "override_run": {"model": {"container": container}}}  # fmt: skip
    if identity is not None:
        recipe["base"]["identity"] = {"container": {"image": identity}}
    write_yaml(root, f"{MULTI}/m/recipe.yaml", recipe)
    return "CONFIG_FILE=recipes/m/recipe.yaml:override_run"


def tilert_recipe(root: Path, decode: str, prefill: str) -> str:
    write_yaml(root, f"{MULTI}/t/recipe.yaml", {
        "name": "t", "model": {"path": "hf:t/m", "container": decode, "precision": "fp8"},
        "roles": {"prefill": {"container": prefill}, "decode": {"container": decode}},
        "frontend": {"container_image": decode}, "benchmark": {"container_image": prefill},
    })  # fmt: skip
    return "CONFIG_FILE=recipes/t/recipe.yaml"


def run_cli(root: Path, runner_config: Path, matrix: object) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": f"{ROOT}{os.pathsep}{ROOT / 'utils/srt-slurm/src'}"}
    command = [sys.executable, "-m", "infx.srt_slurm.preflight", "--root", str(root),
               "--runner-config", str(runner_config)]  # fmt: skip
    return subprocess.run(
        command, input=json.dumps(matrix), capture_output=True, text=True, env=env, check=False
    )


def test_cli_echoes_a_consistent_matrix_and_rejects_a_one_sided_image_update(tmp_path):
    recipe = write_yaml(
        tmp_path, f"{SINGLE}/a.yaml", {"base": single_node_recipe("img:2", 1), "override_c4": {}}
    )
    inventory = tmp_path / "runners.yaml"
    inventory.write_text(yaml.safe_dump(INVENTORY))

    good = {"single_node": {"agentic": [single_node_point(recipe, "img:2")]}}
    passed = run_cli(tmp_path, inventory, good)
    assert (passed.returncode, passed.stdout) == (0, json.dumps(good))

    throughput = single_node_point(recipe, "img:3")
    failed = run_cli(
        tmp_path, inventory, [throughput, {**throughput, "run-eval": True, "eval-only": True}]
    )
    assert (failed.returncode, failed.stdout) == (1, "")
    report = failed.stderr.splitlines()
    assert report[0] == "srt-slurm recipe preflight found 1 problem(s) affecting 2 point(s):"
    assert report[1].startswith(f"  srt-recipe={SINGLE}/a.yaml: ")
    assert "Single-node SRT image: recipe/matrix 'img:2' != 'img:3'" in report[1]
    assert report[2:] == ["    - ep1 on cluster:c", "    - ep1 on cluster:c | eval-only"]


def test_cli_reads_the_runner_inventory_only_for_multi_node_points(tmp_path):
    single = write_yaml(tmp_path, f"{SINGLE}/a.yaml", single_node_recipe("img:1", 1))
    inventory = tmp_path / "runners.yaml"
    inventory.write_text(yaml.safe_dump({**INVENTORY, "retired-section": {}}))

    assert run_cli(tmp_path, inventory, [single_node_point(single, "img:1")]).returncode == 0
    multi = multi_node_point(multi_node_recipe(tmp_path, "img:1"))
    failed = run_cli(tmp_path, inventory, [single_node_point(single, "img:1"), multi])
    assert failed.returncode == 1
    assert failed.stderr.startswith(
        f"srt-slurm recipe preflight cannot read {inventory} with this tooling's runner schema: "
    )


def test_shared_recipe_rejects_variant_images_swapped_between_its_master_keys(tmp_path):
    def recipe_with(base_image: str, override_image: str) -> str:
        return write_yaml(tmp_path, f"{SINGLE}/shared.yaml", {
            "base": single_node_recipe(base_image, 8),
            "override_ep8": {},
            "override_ep1": {"model": {"container": override_image},
                             "roles": {"agg": {"args": {"expert-parallel-size": 1}}}},
        })  # fmt: skip

    matrix = [single_node_point(f"{SINGLE}/shared.yaml", "img:a", ep=8),
              single_node_point(f"{SINGLE}/shared.yaml", "img:b", ep=1)]  # fmt: skip
    inventory = load_inventory(INVENTORY)
    recipe_with("img:a", "img:b")
    assert check_matrix(matrix, tmp_path, inventory) == {}
    recipe_with("img:b", "img:a")
    problems = check_matrix(matrix, tmp_path, inventory)
    assert list(problems.values()) == [["ep8 on cluster:c"], ["ep1 on cluster:c"]]


def test_eval_recipe_is_checked_alongside_the_throughput_recipe(tmp_path):
    throughput = multi_node_recipe(tmp_path, "img:1")
    write_yaml(tmp_path, f"{MULTI}/m/eval.yaml",
               {"name": "e", "model": {"path": "hf:t/m", "container": "img:0", "precision": "fp8"}})  # fmt: skip
    point = multi_node_point(throughput, "EVAL_CONFIG_FILE=recipes/m/eval.yaml")
    assert check_matrix([point], tmp_path, load_inventory(INVENTORY)) == {
        "EVAL_CONFIG_FILE=recipes/m/eval.yaml: model.container 'img:0' does not resolve to img:1 "
        "(container aliases: dynamo-sglang)": ["disagg on cluster:c"]
    }


@pytest.mark.parametrize(
    ("eval_only", "settings", "launches"),
    [
        (False, [], False),
        (False, ["EVAL_CONFIG_FILE=recipes/m/eval.yaml"], False),
        (True, ["EVAL_CONFIG_FILE=recipes/m/eval.yaml"], True),
        (True, [], False),
        (True, ["CONFIG_FILE=recipes/m/eval.yaml"], True),
    ],
    ids=["throughput-none", "throughput-eval-only-recipe", "eval-only-eval-recipe", "eval-only-none",
         "eval-only-config-recipe"],
)  # fmt: skip
def test_multi_node_point_needs_the_recipe_its_launcher_selects(
    tmp_path, eval_only, settings, launches
):
    write_yaml(tmp_path, f"{MULTI}/m/eval.yaml",
               {"name": "e", "model": {"path": "hf:t/m", "container": "img:1", "precision": "fp8"}})  # fmt: skip
    point = {**multi_node_point(*settings), "eval-only": eval_only}
    label = "disagg on cluster:c" + (" | eval-only" if eval_only else "")
    missing = (
        "CONFIG_FILE is not set; only an eval-only point may launch its EVAL_CONFIG_FILE instead"
    )
    expected = {} if launches else {missing: [label]}
    assert check_matrix([point], tmp_path, load_inventory(INVENTORY)) == expected


def test_override_recipe_without_overrides_selects_nothing_unless_base_is_named(tmp_path):
    write_yaml(tmp_path, f"{MULTI}/b.yaml",
               {"base": {"name": "b", "model": {"path": "hf:t/m", "container": "img:1", "precision": "fp8"}}})  # fmt: skip
    inventory = load_inventory(INVENTORY)
    assert list(
        check_matrix([multi_node_point("CONFIG_FILE=recipes/b.yaml")], tmp_path, inventory)
    ) == ["CONFIG_FILE=recipes/b.yaml: selects no variant, so srtctl would submit nothing"]
    assert (
        check_matrix([multi_node_point("CONFIG_FILE=recipes/b.yaml:base")], tmp_path, inventory)
        == {}
    )


@pytest.mark.parametrize(
    ("image", "container", "resolves"),
    [
        ("img:1", "dynamo-sglang", True),
        ("img:1", "dynamo-sglan", False),
        ("nvcr.io/nvidia/x:1", "nvcr.io#nvidia/x:1", True),
        ("img:1", "img:2", False),
    ],
)
def test_multi_node_container_must_be_a_cluster_alias_or_the_point_image(
    tmp_path, image, container, resolves
):
    point = multi_node_point(multi_node_recipe(tmp_path, container), image=image)
    assert (check_matrix([point], tmp_path, load_inventory(INVENTORY)) == {}) is resolves


@pytest.mark.parametrize(
    ("image", "identity", "matches"),
    [
        ("nvcr.io#nvidia/x:1", "nvcr.io/nvidia/x:1", True),
        ("nvcr.io#nvidia/x:1", "nvcr.io/nvidia/x:2", False),
    ],
)
def test_identity_image_accepts_either_registry_spelling_only(tmp_path, image, identity, matches):
    point = multi_node_point(multi_node_recipe(tmp_path, "dynamo-sglang", identity), image=image)
    assert (check_matrix([point], tmp_path, load_inventory(INVENTORY)) == {}) is matches


@pytest.mark.parametrize("runner", ["cluster:c", "pool", "c"])
def test_aliases_follow_cluster_labels_runner_pools_and_bare_cluster_ids(tmp_path, runner):
    point = multi_node_point(multi_node_recipe(tmp_path, "dynamo-sglang"), runner=runner)
    assert check_matrix([point], tmp_path, load_inventory(INVENTORY)) == {}


@pytest.mark.parametrize(
    ("image", "prefill_image", "stale"),
    [
        ("dec:1", "pre:1", []),
        (
            "dec:2",
            "pre:1",
            ["frontend.container_image", "model.container", "roles.decode.container"],
        ),
        ("dec:1", "pre:2", ["benchmark.container_image", "roles.prefill.container"]),
    ],
)
def test_tilert_roles_follow_the_decode_image_and_prefill_image(
    tmp_path, image, prefill_image, stale
):
    point = multi_node_point(tilert_recipe(tmp_path, "dec:1", "pre:1"), f"PREFILL_IMAGE={prefill_image}",
                             image=image, framework="tilert")  # fmt: skip
    problems = check_matrix([point], tmp_path, load_inventory(INVENTORY))
    assert sorted(problem.split(": ", 1)[1].split(" ", 1)[0] for problem in problems) == stale


def test_tilert_point_without_prefill_image_is_reported(tmp_path):
    point = multi_node_point(
        tilert_recipe(tmp_path, "dec:1", "pre:1"), image="dec:1", framework="tilert"
    )
    assert list(check_matrix([point], tmp_path, load_inventory(INVENTORY))) == [
        "TileRT needs a PREFILL_IMAGE setting for its prefill role"
    ]


@pytest.mark.parametrize(
    ("block", "image", "client", "stale"),
    [
        ("benchmark", "img:2", "img:2", False),
        ("benchmark", "img:2", "img:1", True),
        ("benchmark", "img:2", "nginx", False),
        ("benchmark", "nvcr.io/nvidia/x:1", "nvcr.io#nvidia/x:1", False),
        ("benchmark", "nvcr.io/nvidia/x:1", "nvcr.io/nvidia/x@sha256:0", True),
        ("frontend", "img:2", "img@sha256:0", False),
    ],
)
def test_benchmark_client_must_not_name_another_tag_of_the_point_image(
    tmp_path, block, image, client, stale
):
    single = single_node_recipe(image, 1)
    single.setdefault(block, {})["container_image"] = client
    recipe = write_yaml(tmp_path, f"{SINGLE}/a.yaml", single)
    write_yaml(tmp_path, f"{MULTI}/m.yaml", {
        "name": "m", "model": {"path": "hf:t/m", "container": image, "precision": "fp8"},
        block: {"container_image": client},
    })  # fmt: skip
    matrix = [
        single_node_point(recipe, image),
        multi_node_point("CONFIG_FILE=recipes/m.yaml", image=image),
    ]
    problems = check_matrix(matrix, tmp_path, load_inventory(INVENTORY))
    assert list(problems.values()) == (
        [["ep1 on cluster:c"], ["disagg on cluster:c"]] if stale else []
    )


@pytest.mark.parametrize(
    "point",
    [
        lambda root: single_node_point(str(root / "elsewhere/r.yaml"), "img:1"),
        lambda root: single_node_point(f"{SINGLE}/../../../elsewhere/r.yaml", "img:1"),
        lambda root: multi_node_point(f"CONFIG_FILE=recipes/{root}/elsewhere/r.yaml"),
        lambda root: multi_node_point("CONFIG_FILE=recipes/../../../elsewhere/r.yaml"),
        lambda root: multi_node_point("CONFIG_FILE=elsewhere/r.yaml"),
    ],
    ids=["single-absolute", "single-parent", "multi-absolute", "multi-parent", "multi-unprefixed"],
)
def test_recipe_paths_must_stay_inside_their_recipe_trees(tmp_path, point):
    write_yaml(tmp_path, "elsewhere/r.yaml", single_node_recipe("img:1", 1))
    [problem] = check_matrix([point(tmp_path)], tmp_path, load_inventory(INVENTORY))
    assert problem.endswith((f"not inside {SINGLE}", f"not a recipes/ path inside {MULTI}"))


def test_variant_without_a_model_container_is_reported(tmp_path):
    write_yaml(
        tmp_path, f"{MULTI}/n.yaml", {"name": "n", "model": {"path": "hf:t/m", "precision": "fp8"}}
    )
    point = multi_node_point("CONFIG_FILE=recipes/n.yaml")
    assert list(check_matrix([point], tmp_path, load_inventory(INVENTORY))) == [
        "CONFIG_FILE=recipes/n.yaml: model.container is not set"
    ]


def test_missing_recipe_is_reported_instead_of_raised(tmp_path):
    point = multi_node_point("CONFIG_FILE=recipes/absent.yaml")
    [(problem, points)] = check_matrix([point], tmp_path, load_inventory(INVENTORY)).items()
    assert problem.startswith("CONFIG_FILE=recipes/absent.yaml: ")
    assert "No such file" in problem
    assert points == ["disagg on cluster:c"]
