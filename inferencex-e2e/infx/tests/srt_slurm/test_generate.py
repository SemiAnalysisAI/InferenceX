"""``infx generate``: bound fixed-sequence recipes of master-config points, validated by srtctl."""

import json
from pathlib import Path

import pytest
import yaml

from infx.cli import main

ROOT = Path(__file__).resolve().parents[3]
FRAGMENT = {
    "base": {
        "schema": 2, "name": "fixture", "engine": "sglang",
        "resources": {"gpu_type": "h200", "gpus_per_node": 8},
        "frontend": {"type": "sglang", "enable_multiple_frontends": False},
        "roles": {"agg": {"nodes": 1, "workers": 1, "gpus": 4, "args": {
            "tensor-parallel-size": 4, "speculative-algorithm": "EAGLE", "speculative-num-steps": 2,
        }, "env": {"SGLANG_SIMULATE_ACC_LEN": "2.5"}}},
    },
    "zip_override_conc": {"benchmark": {"env": {"CONC": ["2", "4"]}}},
}  # fmt: skip
MASTER = {"fixture-sglang": {
    "image": "org/image:1", "model": "org/model", "model-prefix": "dsr1", "precision": "fp8",
    "framework": "sglang", "runner": "fixture", "multinode": False, "srt-recipe-dir": "fixture",
    "scenarios": {"fixed-seq-len": [{"isl": 1024, "osl": 128, "search-space": [
        {"tp": 4, "conc-start": 2, "conc-end": 4, "spec-decoding": "mtp", "srt-recipe": "recipe.yaml"},
    ]}]},
}}  # fmt: skip


@pytest.fixture
def project(tmp_path):
    (tmp_path / "configs/srt-recipes").mkdir(parents=True)
    (tmp_path / "configs/srt-recipes/fixed-sequence-single.yaml").write_text(
        yaml.safe_dump({"benchmark": {"type": "custom", "command": "bash client.sh"}})
    )
    (tmp_path / "configs/master.yaml").write_text(yaml.safe_dump(MASTER))
    (tmp_path / "configs/runners.yaml").write_text(yaml.safe_dump({
        "labels": {"fixture": ["fixture_0"], "cluster:fixture": ["fixture_0"]},
        "clusters": {"fixture": {"gpus-per-node": 8, "arch": "x86_64", "scheduler": "slurm",
                                 "slurm": {"partition": "p", "exclusive": True}}},
    }))  # fmt: skip
    recipe = tmp_path / "benchmarks/single_node/srt-slurm-recipes/fixture/recipe.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(yaml.safe_dump(FRAGMENT))
    (tmp_path / "utils/srt-slurm").mkdir(parents=True)
    (tmp_path / "utils/srt-slurm/src").symlink_to(ROOT / "utils/srt-slurm/src")
    return tmp_path


def generate(project: Path, output: Path) -> int:
    return main([
        "generate", "--config-key", "fixture-*", "--output-dir", str(output),
        "--config-file", str(project / "configs/master.yaml"),
        "--runner-config", str(project / "configs/runners.yaml"),
    ])  # fmt: skip


def test_each_point_gets_its_bound_variant_and_a_manifest_entry(project, tmp_path):
    output = tmp_path / "out"
    assert generate(project, output) == 0

    manifest = json.loads((output / "manifest.json").read_text())
    assert [(r["config-key"], r["variant"], r["matrix"]["conc"]) for r in manifest["recipes"]] == [
        ("fixture-sglang", "zip_override_conc[0]", 2),
        ("fixture-sglang", "zip_override_conc[1]", 4),
    ]
    recipe = yaml.safe_load((output / manifest["recipes"][1]["file"]).read_text())
    assert recipe["model"] == {"path": "hf:org/model", "container": "org/image:1", "precision": "fp8"}
    assert recipe["benchmark"] == {"type": "custom", "command": "bash client.sh", "env": {
        "CONC": "4", "ISL": "1024", "OSL": "128", "MODEL": "org/model",
        "RANDOM_RANGE_RATIO": "0.8", "USE_CHAT_TEMPLATE": "true",
    }}  # fmt: skip
    # Fixed-sequence runs verify real drafts; simulated acceptance is unset.
    assert recipe["roles"]["agg"]["env"] == {}


def test_output_must_be_new_or_empty(project, tmp_path, capsys):
    output = tmp_path / "out"
    output.mkdir()
    (output / "stale.yaml").write_text("{}\n")
    with pytest.raises(SystemExit) as exit_info:
        generate(project, output)
    assert exit_info.value.code == 2
    assert "Output directory must be empty" in capsys.readouterr().err


def test_a_recipe_srtctl_rejects_fails_without_writing(project, tmp_path, capsys):
    recipe = project / "benchmarks/single_node/srt-slurm-recipes/fixture/recipe.yaml"
    recipe.write_text(yaml.safe_dump({**FRAGMENT, "base": {**FRAGMENT["base"], "no_such_field": 1}}))
    output = tmp_path / "out"
    with pytest.raises(SystemExit):
        generate(project, output)
    assert "srtctl rejects the bound recipe" in capsys.readouterr().err
    assert not output.exists()
