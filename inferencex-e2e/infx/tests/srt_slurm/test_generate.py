"""Recipe generation executes matrix expansion, native selection, and upstream validation."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from infx.cli import main


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


@pytest.fixture
def project(tmp_path):
    (tmp_path / "utils").mkdir()
    (tmp_path / "utils/srt-slurm").symlink_to(Path(__file__).resolve().parents[3] / "utils/srt-slurm", target_is_directory=True)
    common = {
        "model": {"path": "hf:${MODEL}", "container": "${IMAGE}", "precision": "${PRECISION}"},
        "benchmark": {"type": "custom", "command": "bash /infmax-workspace/benchmarks/single_node/srt_fixed_sequence.sh", "env": {
            "MODEL": "${MODEL}", "ISL": "${ISL}", "OSL": "${OSL}",
        }},
    }
    dump(tmp_path / "configs/srt-recipes/fixed-sequence-single.yaml", common)
    multi_common = deepcopy(common)
    multi_common["benchmark"].update({"command": "bash /infmax-workspace/benchmarks/multi_node/srt_fixed_sequence.sh", "concurrencies": "${CONCURRENCIES}"})
    multi_common["benchmark"]["env"]["CONC_LIST"] = "${CONC_LIST}"
    dump(tmp_path / "configs/srt-recipes/fixed-sequence-multi.yaml", multi_common)
    recipe = {
        "schema": 2, "name": "controlled-test",
        "resources": {"gpu_type": "h200", "gpus_per_node": 8},
        "engine": "sglang", "frontend": {"type": "sglang", "enable_multiple_frontends": False},
        "roles": {"agg": {"nodes": 1, "workers": 1, "gpus": 2, "args": {
            "tensor-parallel-size": 2, "max-running-requests": 8,
        }, "env": {"SGLANG_SIMULATE_ACC_LEN": "3"}}},
    }
    raw = {"base": recipe, "zip_override_conc": {
        "roles": {"agg": {"args": {"cuda-graph-max-bs": [16, 32]}}},
        "benchmark": {"env": {"CONC": ["2", "4"]}},
    }}
    dump(tmp_path / "recipe.yaml", raw)
    master = {"model-test": {
        "image": "registry/server:new", "model": "org/new", "model-prefix": "fixture",
        "runner": "h200", "precision": "fp8", "framework": "sglang", "multinode": False,
        "scenarios": {"fixed-seq-len": [{"isl": 1024, "osl": 128, "search-space": [{
            "tp": 2, "conc-list": [2, 4], "srt-recipe": "recipe.yaml",
        }]}]},
    }}
    dump(tmp_path / "master.yaml", master)
    runners = {
        "labels": {"h200": ["h200-test_00"], "cluster:test": ["h200-test_00"]},
        "clusters": {"test": {"gpus-per-node": 8, "arch": "x86_64", "scheduler": "slurm",
            "slurm": {"partition": "test", "exclusive": True}}},
    }
    dump(tmp_path / "runners.yaml", runners)
    return tmp_path, master, raw


def argv(root, *extra):
    return ["generate", "--project-root", str(root), "--config-file", str(root / "master.yaml"),
        "--runner-config", str(root / "runners.yaml"), "--config-key", "model-test",
        "--output-dir", str(root / "output"), *extra]


def outputs(root):
    manifest = json.loads((root / "output/manifest.json").read_text())
    return manifest, [yaml.safe_load((root / "output" / item["file"]).read_text()) for item in manifest["recipes"]]


def test_cli_generates_bound_native_recipes_with_coupled_server_tuning(project):
    root, _, _ = project
    original = (root / "recipe.yaml").read_bytes()
    assert main(argv(root)) == 0
    manifest, recipes = outputs(root)
    assert len(recipes) == 2
    assert [(recipe["benchmark"]["env"]["CONC"], recipe["roles"]["agg"]["args"]["cuda-graph-max-bs"]) for recipe in recipes] == [("2", 16), ("4", 32)]
    assert recipes[0]["model"] == {"path": "hf:org/new", "container": "registry/server:new", "precision": "fp8"}
    assert recipes[0]["roles"]["agg"]["args"]["served-model-name"] == "org/new"
    assert "SGLANG_SIMULATE_ACC_LEN" not in recipes[0]["roles"]["agg"]["env"]
    env = recipes[0]["benchmark"]["env"]
    assert (env["ISL"], env["OSL"], env["RUN_EVAL"], env["EVAL_ONLY"]) == ("1024", "128", "false", "false")
    assert env["RESULT_FILENAME"] == Path(manifest["recipes"][0]["file"]).stem
    assert manifest["recipes"][1]["source"].endswith(":zip_override_conc[1]")
    assert manifest["recipes"][1]["matrix"]["conc"] == 4
    assert (root / "recipe.yaml").read_bytes() == original


@pytest.mark.parametrize("selector,expected_tuning", [(":override_selected", [19]), ("", [99, 19])])
def test_multi_fixed_sequence_uses_selected_native_variant_and_list(project, selector, expected_tuning):
    root, master, raw = project
    config = master["model-test"]
    config["multinode"] = True
    config["scenarios"]["fixed-seq-len"][0]["search-space"] = [{
        "worker": {"num-worker": 1, "tp": 2, "ep": 1, "dp-attn": False,
            "additional-settings": [f"CONFIG_FILE=recipes/test.yaml{selector}"]},
        "conc-list": [2, 4], "num-nodes": 1,
    }]
    recipe = deepcopy(raw["base"])
    recipe["model"] = {"path": "hf:org/old", "container": "old:image", "precision": "fp8"}
    recipe["benchmark"] = {"type": "sa-bench", "isl": 8, "osl": 4, "concurrencies": [8]}
    dump(root / "benchmarks/multi_node/srt-slurm-recipes/test.yaml", {"base": recipe,
        "override_selected": {"roles": {"agg": {"args": {"max-running-requests": 19}}}},
        "override_other": {"roles": {"agg": {"args": {"max-running-requests": 99}}}},
    })
    dump(root / "master.yaml", master)
    assert main(argv(root)) == 0
    manifest, recipes = outputs(root)
    assert [recipe["roles"]["agg"]["args"]["max-running-requests"] for recipe in recipes] == expected_tuning
    for recipe in recipes:
        assert recipe["benchmark"] == {"type": "sa-bench", "isl": 1024, "osl": 128, "concurrencies": [2, 4], "env": {}}
    assert len({record["file"] for record in manifest["recipes"]}) == len(expected_tuning)
    assert [record["variant"] for record in manifest["recipes"]] == (["override_selected"] if selector else ["override_other", "override_selected"])
    assert manifest["recipes"][0]["matrix"]["node-count"] == 1


def test_upstream_schema_failure_writes_no_partial_outputs(project, capsys):
    root, _, raw = project
    raw["base"]["not_a_native_srt_field"] = "invalid"
    dump(root / "recipe.yaml", raw)
    with pytest.raises(SystemExit) as error:
        main(argv(root))
    assert error.value.code == 2
    assert "Invalid generated SRT recipe" in capsys.readouterr().err
    assert not (root / "output").exists()


@pytest.mark.parametrize("mode,message", [("legacy", "legacy"), ("tilert", "TileRT"), ("unknown", "not found")])
def test_unsupported_or_missing_selection_is_an_actionable_error(project, mode, message, capsys):
    root, master, _ = project
    if mode == "legacy":
        del master["model-test"]["scenarios"]["fixed-seq-len"][0]["search-space"][0]["srt-recipe"]
    elif mode == "tilert":
        master["model-test"]["framework"] = "tilert"
    else:
        master["other"] = master.pop("model-test")
    dump(root / "master.yaml", master)
    with pytest.raises(SystemExit) as error:
        main(argv(root))
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not (root / "output").exists()


def test_native_fragment_is_composed_automatically_without_rewriting_sources(project):
    root, _, _ = project
    source = root / "configs/srt-recipes/fixed-sequence-single.yaml"
    common_before = source.read_bytes()
    fragment = root / "recipe.yaml"
    fragment_before = fragment.read_bytes()
    assert main(argv(root)) == 0
    manifest, recipes = outputs(root)
    assert [recipe["model"]["path"] for recipe in recipes] == ["hf:org/new", "hf:org/new"]
    assert [recipe["roles"]["agg"]["args"]["cuda-graph-max-bs"] for recipe in recipes] == [16, 32]
    assert [record["variant"] for record in manifest["recipes"]] == ["zip_override_conc[0]", "zip_override_conc[1]"]
    assert fragment.read_bytes() == fragment_before
    assert source.read_bytes() == common_before


def test_same_native_fragment_generates_distinct_master_workloads(project):
    root, master, _ = project
    original = (root / "recipe.yaml").read_bytes()
    other = deepcopy(master["model-test"])
    other.update({"model": "org/different", "image": "registry/server:other"})
    other["scenarios"]["fixed-seq-len"][0].update({"isl": 2048, "osl": 256})
    master["other"] = other
    dump(root / "master.yaml", master)
    assert main(argv(root, "--config-key", "other")) == 0
    manifest, recipes = outputs(root)
    assert len(recipes) == 4
    assert {record["config-key"] for record in manifest["recipes"]} == {"model-test", "other"}
    assert [(recipe["model"]["path"], recipe["model"]["container"], recipe["benchmark"]["env"]["ISL"], recipe["benchmark"]["env"]["OSL"]) for recipe in recipes] == [
        ("hf:org/new", "registry/server:new", "1024", "128"),
        ("hf:org/new", "registry/server:new", "1024", "128"),
        ("hf:org/different", "registry/server:other", "2048", "256"),
        ("hf:org/different", "registry/server:other", "2048", "256"),
    ]
    assert (root / "recipe.yaml").read_bytes() == original


def test_nonempty_output_is_preserved(project, capsys):
    root, _, _ = project
    (root / "output").mkdir()
    (root / "output/existing.txt").write_text("keep me")
    with pytest.raises(SystemExit):
        main(argv(root))
    assert "must be empty" in capsys.readouterr().err
    assert (root / "output/existing.txt").read_text() == "keep me"


def test_explicit_project_requires_its_own_pinned_srt_checkout(project, capsys):
    root, _, _ = project
    (root / "utils/srt-slurm").unlink()
    with pytest.raises(SystemExit) as error:
        main(argv(root))
    assert error.value.code == 2
    assert "Initialize the pinned SRT dependency" in capsys.readouterr().err
    assert not (root / "output").exists()
