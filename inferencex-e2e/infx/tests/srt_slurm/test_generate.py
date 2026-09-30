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
    recipe = {
        "schema": 2, "name": "controlled-test",
        "model": {"path": "hf:org/old", "container": "old:image", "precision": "fp8"},
        "resources": {"gpu_type": "h200", "gpus_per_node": 8},
        "engine": "sglang", "frontend": {"type": "sglang", "enable_multiple_frontends": False},
        "roles": {"agg": {"nodes": 1, "workers": 1, "gpus": 2, "args": {
            "tensor-parallel-size": 2, "served-model-name": "org/old", "max-running-requests": 8,
        }, "env": {"SGLANG_SIMULATE_ACC_LEN": "3"}}},
        "benchmark": {"type": "custom", "command": "bash /infmax-workspace/benchmarks/single_node/srt_fixed_sequence.sh", "env": {
            "MODEL": "org/old", "ISL": "8", "OSL": "4", "RANDOM_RANGE_RATIO": "0.8",
            "USE_CHAT_TEMPLATE": "false",
        }},
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


def register(root, raw):
    common = deepcopy(raw)
    common["base"]["model"].update({"path": "hf:${MODEL}", "container": "${IMAGE}"})
    common["base"]["benchmark"]["env"].update({"MODEL": "${MODEL}", "ISL": "${ISL}", "OSL": "${OSL}"})
    dump(root / "configs/srt-recipes/common.yaml", common)
    dump(root / "configs/srt-recipes/tuning.yaml", {})
    dump(root / "configs/srt-recipes/sources.yaml", {"recipe.yaml": {"common": "common.yaml", "tuning": "tuning.yaml"}})


def test_refresh_exports_produces_native_bundle_from_registered_sources(project):
    root, _, raw = project
    register(root, raw)
    assert main(argv(root, "--refresh-exports")) == 0
    exported = yaml.safe_load((root / "recipe.yaml").read_text())
    assert exported["base"]["model"]["path"] == "hf:org/new"
    assert exported["base"]["model"]["container"] == "registry/server:new"
    assert exported["zip_override_conc"]["roles"]["agg"]["args"]["cuda-graph-max-bs"] == [16, 32]
    manifest, _ = outputs(root)
    assert manifest["refreshed-exports"] == [str(root / "recipe.yaml")]


def test_conflicting_export_sources_leave_every_file_untouched(project, capsys):
    root, master, raw = project
    register(root, raw)
    original = (root / "recipe.yaml").read_bytes()
    other = deepcopy(master["model-test"])
    other["model"] = "org/different"
    master["other"] = other
    dump(root / "master.yaml", master)
    with pytest.raises(SystemExit):
        main(argv(root, "--config-key", "other", "--refresh-exports"))
    assert "Conflicting selected workloads" in capsys.readouterr().err
    assert (root / "recipe.yaml").read_bytes() == original
    assert not (root / "output").exists()


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
