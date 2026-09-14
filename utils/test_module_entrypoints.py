"""Exercise file and module entrypoints across checkout boundaries."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["module", "legacy"])
def invoke(request, tmp_path):
    def run(module, legacy, *args, **environment):
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        if request.param == "module":
            command = ["-P", "-m", module]
            env["PYTHONPATH"] = str(ROOT)
        else:
            command = [str(ROOT / legacy)]
        return subprocess.run(
            [sys.executable, *command, *args], cwd=tmp_path,
            env={**env, **environment}, capture_output=True, text=True, timeout=10,
        )
    return run


@pytest.mark.parametrize("payload,expected", [
    (None, []),
    ('{"value": 3}', [{"value": 3}]),
    ('[{"value": 3}, {"value": 4}]', [[{"value": 3}, {"value": 4}]]),
])
def test_collector_preserves_nested_json_and_empty_inputs(invoke, tmp_path, payload, expected):
    inputs = tmp_path / "inputs" / "nested"
    inputs.mkdir(parents=True)
    if payload is not None:
        (inputs / "result.json").write_text(payload)
    (inputs / "ignored.txt").write_text("not JSON")

    result = invoke("infx.results.collect_results", "utils/collect_results.py", "inputs", "test")

    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ""
    assert json.loads((tmp_path / "agg_test.json").read_text()) == expected


@pytest.mark.parametrize("published", [None, b'[{"previous": true}]\n'])
def test_collector_does_not_publish_partial_output_on_invalid_json(invoke, tmp_path, published):
    inputs = tmp_path / "inputs"
    (inputs / "nested").mkdir(parents=True)
    (inputs / "valid.json").write_text('{"value": 3}')
    (inputs / "nested/invalid.json").write_text("{")
    output = tmp_path / "agg_test.json"
    if published is not None:
        output.write_bytes(published)

    result = invoke("infx.results.collect_results", "utils/collect_results.py", "inputs", "test")

    assert result.returncode != 0
    assert "JSONDecodeError" in result.stderr
    if published is None:
        assert not output.exists()
    else:
        assert output.read_bytes() == published


def test_filename_entrypoint_retains_environment_and_point_arguments(invoke):
    result = invoke("infx.results.result_filename", "utils/result_filename.py",
                    RESULT_FILENAME_BASE="model_tp8", RECIPE_FINGERPRINT="abc")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "model_tp8_recipe-abc\n"

    point = invoke("infx.results.result_filename", "utils/result_filename.py",
                   "--point", "model_tp8", "config", "4", "8", "1024", "512")
    assert point.returncode == 0, point.stderr
    assert point.stdout == "model_tp8_config_conc4_gpus_8_ctx_1024_gen_512.json\n"

    invalid = invoke("infx.results.result_filename", "utils/result_filename.py",
                     "--point", "model_tp8", "config", "four", "8", "", "")
    assert invalid.returncode != 0
    assert "Expected numeric point identity" in invalid.stderr


def test_importing_command_modules_does_not_run_them(tmp_path):
    result = subprocess.run(
        [sys.executable, "-P", "-c",
         "import infx.results.collect_results; import infx.evals._kimi_verifier_archive"],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == result.stderr == ""
    assert list(tmp_path.iterdir()) == []


def test_legacy_benchmark_entrypoint_supports_spawned_workers(tmp_path):
    """Exercise the shipped wrapper's execution context with a minimal worker."""
    wrapper = tmp_path / "utils/bench_serving/benchmark_serving.py"
    wrapper.parent.mkdir(parents=True)
    shutil.copy(ROOT / "utils/bench_serving/benchmark_serving.py", wrapper)
    package = tmp_path / "infx/bench_serving"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").touch()
    (package / "__init__.py").touch()
    (package / "benchmark_serving.py").write_text('''from multiprocessing import get_context
def work(value):
    return value * value
if __name__ == "__main__":
    with get_context("spawn").Pool(1) as pool:
        print(pool.map(work, [2, 3]))
''')
    result = subprocess.run(
        [sys.executable, str(wrapper)], cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "[4, 9]\n"
