import json
import subprocess
import sys


def test_installed_tools_use_callers_repository(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "runners.yaml").write_text("labels:\n  cluster:fixture-gpu: [self-hosted]\n")
    recipes = tmp_path / "benchmarks/multi_node/srt-slurm-recipes"
    recipes.mkdir(parents=True)
    (recipes / "fixture.yaml").write_text(
        "schema: 2\nroles:\n  prefill: {nodes: 2}\n  decode: {nodes: 4}\n"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", """
import json
from infx.matrix.generate import recipe_node_count
from infx.workflows.calc_success_rate import load_hardware_labels
print(json.dumps({
    "nodes": recipe_node_count({"additional-settings": ["CONFIG_FILE=recipes/fixture.yaml"]}, {}),
    "hardware": load_hardware_labels(),
}))
"""],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )

    assert json.loads(result.stdout) == {"nodes": 6, "hardware": ["fixture-gpu"]}
