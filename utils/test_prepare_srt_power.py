"""Run with the candidate srt-slurm source on PYTHONPATH; never dispatches a job."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("overrides", "enabled"),
    [({}, True), ({"REQUIRE_POWER": "0"}, False), ({"SCENARIO_TYPE": "agentic-coding"}, False), ({"OSL": "8192"}, False), ({"IS_AGENTIC": "1"}, False),
     ({"EVAL_ONLY": "true"}, False), ({"ISL": "1024"}, False)],
)
def test_only_normal_8k1k_selects_power_runtime(overrides, enabled):
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; powerx_fixed_8k1k', "test", str(ROOT / "runners/powerx_8k1k.sh")],
        env={**os.environ, "REQUIRE_POWER": "1", "ISL": "8192", "OSL": "1024", "IS_AGENTIC": "0", "EVAL_ONLY": "false", **overrides},
    )
    assert (result.returncode == 0) is enabled


def test_resolves_override_before_injecting_required_power(tmp_path):
    pytest.importorskip("srtctl", reason="requires the pinned runtime checkout on PYTHONPATH")
    from runners.prepare_srt_power import prepare_recipe

    base = {
        "name": "fixture", "model": {"path": "/model", "container": "fixture.sqsh", "precision": "fp8"},
        "resources": {"gpu_type": "h100", "gpus_per_node": 8, "agg_nodes": 1, "agg_workers": 1},
        "benchmark": {"type": "sa-bench", "isl": 8192, "osl": 1024, "concurrencies": [1]},
    }
    raw = {"base": base, "override_selected": {"benchmark": {"concurrencies": [99], "tokenizer_mode": "deepseek_v4"}, "telemetry": {"enabled": False}},
           "override_other": {"benchmark": {"osl": 8192}}}
    source = tmp_path / "recipe.yaml"
    original = yaml.safe_dump(raw)
    source.write_text(original)
    target = prepare_recipe(f"{source}:override_selected", [4, 8])
    effective = yaml.safe_load(target.read_text())
    assert source.read_text() == original
    assert effective["benchmark"]["concurrencies"] == [4, 8]
    assert effective["benchmark"]["custom_tokenizer"] == "sa_bench_tokenizers.vllm_deepseek_v4.VLLMDeepseekV4Tokenizer"
    assert "tokenizer_mode" not in effective["benchmark"]
    assert effective["telemetry"]["enabled"] is True
    assert effective["telemetry"]["required"] is True
    assert effective["resources"]["agg_workers"] == 1
    with pytest.raises(ValueError, match="exactly one"):
        prepare_recipe(str(source), [4])
    with pytest.raises(ValueError, match="8192/1024"):
        prepare_recipe(f"{source}:override_other", [4])
    with pytest.raises(ValueError, match="positive unique integers"):
        prepare_recipe(f"{source}:override_selected", [4, 4])


@pytest.mark.parametrize("job_exit", [0, 7, 143])
def test_h200_fixed_sequence_prepares_selected_recipe_and_stages_results(tmp_path, job_exit):
    srtctl = pytest.importorskip("srtctl", reason="requires the pinned runtime checkout on PYTHONPATH")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = runtime / "recipe.yaml"
    original = yaml.safe_dump({
        "base": {
            "name": "fixture", "model": {"path": "/model", "container": "fixture.sqsh", "precision": "fp8"},
            "resources": {"gpu_type": "h200", "gpus_per_node": 8, "agg_nodes": 1, "agg_workers": 1},
            "benchmark": {"type": "sa-bench", "isl": 8192, "osl": 1024, "concurrencies": [1]},
        },
        "override_selected": {"benchmark": {"concurrencies": [99]}},
    })
    source.write_text(original)
    logs = runtime / "outputs/123/logs"
    raw = logs / "sa-bench_isl_8192_osl_1024/results_concurrency_4_gpus_8.json"
    raw.parent.mkdir(parents=True)
    raw_result = '{"completed": 4}'
    raw.write_text(raw_result)
    (logs / "power").mkdir()
    (logs / "power/manifest.json").write_text('{"retained": true}')
    (workspace / "exporter-image.sha256").write_text("fixture-exporter")
    (workspace / "power-producer-sha.txt").write_text("fixture-producer")

    launcher = (ROOT / "runners/launch_h200-dgxc-slurm.sh").read_text()
    block = launcher[launcher.index('    if [[ -f "$LOCAL_CONFIG_FILE" ]]'):]
    block = block[:block.index('\nelse\n')]
    shell = '''set -eo pipefail
source "$ROOT/runners/slurm_utils.sh"
source "$ROOT/runners/powerx_8k1k.sh"
python() {
    if [[ "$1" == -m ]]; then echo 'Unexpected AgentX adapter' >&2; return 98; fi
    "$PYTHON" "$ROOT/runners/$(basename "$1")" "${@:2}"
}
srtctl() {
    cp "$3" "$GITHUB_WORKSPACE/submitted.yaml"
    echo 'Job 123'
}
grep() {
    if [[ "$1" == -oP ]]; then cat >/dev/null; echo 123; else command grep "$@"; fi
}
stream_slurm_job_log() { if [[ "$JOB_EXIT" == 143 ]]; then kill -TERM $$; fi; return "$JOB_EXIT"; }
scancel() { :; }
'''
    if sys.platform == "darwin":
        shell += 'sed() { if [[ "$1" == -i ]]; then shift; command sed -i "" "$@"; else command sed "$@"; fi; }\n'
    result = subprocess.run(
        ["bash", "-c", shell + block], cwd=runtime, capture_output=True, text=True,
        env={**os.environ, "ROOT": str(ROOT), "PYTHON": sys.executable,
             "PYTHONPATH": str(Path(srtctl.__file__).resolve().parent.parent),
             "GITHUB_WORKSPACE": str(workspace), "LOCAL_CONFIG_FILE": "missing.yaml",
             "CONFIG_FILE": "recipe.yaml:override_selected", "CONFIG_PATH": "recipe.yaml",
             "USES_DCGM_POWER": "1", "IS_AGENTIC": "0", "REQUIRE_POWER": "1", "ISL": "8192", "OSL": "1024",
             "EVAL_ONLY": "false", "RUN_EVAL": "false", "CONC_LIST": "4 8",
             "RUNNER_NAME": "fixture-runner", "MODEL_PREFIX": "fixture", "PRECISION": "fp8",
             "RESULT_FILENAME": "run", "JOB_EXIT": str(job_exit)},
        timeout=15,
    )
    assert result.returncode == job_exit, result.stderr
    submitted = yaml.safe_load((workspace / "submitted.yaml").read_text())
    assert submitted["name"] == "fixture-runner"
    assert submitted["health_check"] == {"max_attempts": 720, "interval_seconds": 10}
    assert submitted["benchmark"]["concurrencies"] == [4, 8]
    assert submitted["telemetry"]["required"] is True
    assert source.read_text() == original
    if job_exit != 143:
        assert (workspace / "run_sa-bench_isl_8192_osl_1024_conc4_gpus_8.json").read_text() == raw_result
        assert not (runtime / "outputs").exists()
    assert (workspace / "LOGS/power/manifest.json").read_text() == '{"retained": true}'
    assert (workspace / "LOGS/power/power-producer-sha.txt").read_text() == "fixture-producer"
