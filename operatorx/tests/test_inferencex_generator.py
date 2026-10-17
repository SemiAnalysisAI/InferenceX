"""Check InferenceX generator subprocesses against selected checkout layouts."""

import os
import shutil
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from operatorx.scripts.inferencex_testlist.matrix import _run_inferencex_generator


@pytest.mark.parametrize("layout", ["legacy", "flat", "nested"])
def test_generator_uses_requested_checkout_with_safe_path(tmp_path, monkeypatch, layout):
    root = tmp_path / "selected checkout"
    root.mkdir()
    foreign = tmp_path / "other checkout"
    (foreign / "infx").mkdir(parents=True)
    (foreign / "infx/__init__.py").write_text("raise RuntimeError('wrong checkout')\n")
    monkeypatch.setenv("PYTHONPATH", str(foreign))
    monkeypatch.setenv("PYTHONSAFEPATH", "1")
    monkeypatch.chdir(tmp_path)

    if layout == "legacy":
        # The old generator is an external executable. Its sibling dependency
        # distinguishes the selected revision from inherited import paths.
        script = root / "utils/matrix_logic/generate_sweep_configs.py"
        script.parent.mkdir(parents=True)
        script.write_text(
            "import json, sys\nfrom validation import model\n"
            "assert sys.argv[1:] == ['full-sweep', '--config-files', 'configs/master.yaml', '--no-evals']\n"
            "print(json.dumps([{'model': model, 'conc': 2}]))\n"
        )
        script.with_name("validation.py").write_text("model = 'legacy revision'\n")
        expected = "legacy revision"
    else:
        source = Path(__file__).resolve().parents[2] / "inferencex-e2e"
        shutil.copytree(source / "infx", root / "infx",
                        ignore=shutil.ignore_patterns("__pycache__", "tests"))
        (root / "configs").mkdir()
        (root / "configs/runners.yaml").write_text(
            'labels: {fixture: [node-a], "cluster:fixture": [node-a]}\n'
            "clusters: {fixture: {gpus-per-node: 8, arch: x86_64, scheduler: slurm,"
            " slurm: {partition: p, exclusive: false}}}\n"
        )
        (root / "configs/master.yaml").write_text(yaml.safe_dump({"fixture": {
            "image": "example/image:stable", "model": "selected revision",
            "model-prefix": "dsr1", "precision": "fp8", "framework": "sglang",
            "runner": "fixture", "multinode": False, "srt-recipe-dir": "fixture",
            "scenarios": {"fixed-seq-len": [{
                "isl": 1024, "osl": 1024,
                "search-space": [{"tp": 1, "conc-list": [2], "srt-recipe": "recipe.yaml"}],
            }]},
        }}))
        recipe = root / "benchmarks/single_node/srt-slurm-recipes/fixture/recipe.yaml"
        recipe.parent.mkdir(parents=True)
        recipe.write_text("{}\n")
        expected = "selected revision"

    if layout == "nested":
        nested = root / "inferencex-e2e"
        nested.mkdir()
        for name in ("infx", "configs", "benchmarks"):
            (root / name).rename(nested / name)

    rows = _run_inferencex_generator(os.path.relpath(root), ["configs/master.yaml"], ["--no-evals"])
    assert [(row["model"], row["conc"]) for row in rows] == [(expected, 2)]
