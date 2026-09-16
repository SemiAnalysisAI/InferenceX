import json
import os
import subprocess
from functools import cache
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def installed_python(tmp_path_factory):
    @cache
    def install(extra):
        environment = tmp_path_factory.mktemp(f"infx-{extra or 'core'}")
        subprocess.run(
            [
                "uv",
                "sync",
                "--project",
                str(Path(__file__).resolve().parents[1]),
                "--locked",
                "--no-default-groups",
                "--no-editable",
                *(["--extra", extra] if extra else []),
            ],
            env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(environment)},
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        return str(environment / "bin/python")

    return install


@pytest.fixture
def run_installed(installed_python, tmp_path):
    def run(*args, extra="", input=None):
        return subprocess.run(
            [installed_python(extra), "-I", *args],
            cwd=tmp_path,
            input=input,
            capture_output=True,
            text=True,
            timeout=30,
        )

    return run


def test_installed_tools_use_callers_repository(tmp_path, run_installed):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "runners.yaml").write_text("labels:\n  cluster:fixture-gpu: [self-hosted]\n")
    recipes = tmp_path / "benchmarks/multi_node/srt-slurm-recipes"
    recipes.mkdir(parents=True)
    (recipes / "fixture.yaml").write_text(
        "schema: 2\nroles:\n  prefill: {nodes: 2}\n  decode: {nodes: 4}\n"
    )

    result = run_installed(
        "-c",
        """
import json
from infx.matrix.generate import recipe_node_count
from infx.workflows.calc_success_rate import load_hardware_labels
print(json.dumps({
    "nodes": recipe_node_count({"additional-settings": ["CONFIG_FILE=recipes/fixture.yaml"]}, {}),
    "hardware": load_hardware_labels(),
}))
""",
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"nodes": 6, "hardware": ["fixture-gpu"]}


def test_installed_matrix_validation_rejects_zero_concurrency(run_installed):
    matrix = [
        {
            "image": "fixture:latest",
            "model": "fixture/model",
            "model-prefix": "fixture",
            "precision": "fp8",
            "framework": "vllm",
            "spec-decoding": "none",
            "runner": "gpu",
            "isl": 128,
            "osl": 32,
            "tp": 1,
            "ep": 1,
            "dp-attn": False,
            "conc": 0,
            "max-model-len": 160,
            "exp-name": "fixture",
            "disagg": False,
            "run-eval": False,
        }
    ]

    result = run_installed("-m", "infx.workflows.benchmark_schema", input=json.dumps(matrix))

    assert result.returncode == 2
    assert "matrix[0]: 1 validation error" in result.stderr
    assert "conc\n  Input should be greater than 0" in result.stderr
    assert result.stdout == ""


def test_installed_hardware_matching_works_without_repository_files(run_installed):
    result = run_installed(
        "-c",
        """
from infx.workflows.calc_success_rate import build_hardware_match_patterns, extract_hardware_from_name
patterns = build_hardware_match_patterns(["gpu.a"])
print(extract_hardware_from_name("benchmark cluster:gpu.a tp8", patterns))
print(extract_hardware_from_name("benchmark gpuXa tp8", patterns))
""",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "gpu.a\nNone\n"


@pytest.mark.parametrize("score,exit_code,verdict", [(0.9, 0, "PASS"), (0.7, 1, "FAIL")])
def test_installed_eval_validation_loads_resources(
    tmp_path, run_installed, score, exit_code, verdict
):
    (tmp_path / "meta_env.json").write_text("{}")
    (tmp_path / "results.json").write_text(
        json.dumps(
            {
                "results": {"packaging-fixture": {"exact_match,strict": score}},
                "n-samples": {"packaging-fixture": {"effective": 4}},
            }
        )
    )

    result = run_installed("-m", "infx.evals.validate_scores", "--min-score", "0.85")

    assert result.returncode == exit_code, result.stderr
    assert "Loaded thresholds from " in result.stdout
    assert f"{verdict}: packaging-fixture exact_match,strict = " in result.stdout + result.stderr


def test_installed_bfcl_attribution_includes_license(tmp_path, run_installed):
    result = run_installed(
        "-c",
        """
from pathlib import Path
from infx.evals.bfcl_adapter import _write_upstream_attribution
_write_upstream_attribution(Path("bfcl"))
""",
    )

    assert result.returncode == 0, result.stderr
    attribution = json.loads((tmp_path / "bfcl/BFCL_ATTRIBUTION.json").read_text())
    upstream = attribution["upstream"]
    assert upstream["license"] == "Apache-2.0"
    license_text = (tmp_path / "bfcl" / upstream["license_file"]).read_text()
    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text


def test_installed_eval_collection_writes_rows_and_summary(tmp_path, run_installed):
    artifact = tmp_path / "evals" / "job"
    artifact.mkdir(parents=True)
    (artifact / "meta_env.json").write_text(
        json.dumps(
            {
                "infmax_model_prefix": "fixture",
                "hw": "gpu",
                "conc": "4",
            }
        )
    )
    (artifact / "results.json").write_text(
        json.dumps(
            {
                "result_format": "inferencex-eval-v1",
                "results": {"fixture-task": {"acc": 0.75}},
                "n-samples": {"fixture-task": {"effective": 4}},
            }
        )
    )

    result = run_installed(
        "-m", "infx.results.collect_eval_results", "evals", "fixture", extra="results"
    )

    assert result.returncode == 0, result.stderr
    rows = json.loads((tmp_path / "agg_eval_fixture.json").read_text())
    assert [(r["task"], r["score"], r["conc"], r["hw"], r["n_eff"]) for r in rows] == [
        ("fixture-task", 0.75, 4, "GPU", 4),
    ]
    assert "75.00%" in result.stdout
