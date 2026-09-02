import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_UTILS = REPO_ROOT / "runners" / "slurm_utils.sh"


def run_bash(command: str, *args: Path | str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", command, "bash", *(str(arg) for arg in args)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_copy_agentic_results_stages_only_matching_points(tmp_path: Path) -> None:
    source = tmp_path / "source"
    workspace = tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()
    (source / "run_conc1.json").write_text('{"conc": 1}\n')
    (source / "run_conc16.json").write_text('{"conc": 16}\n')
    (source / "other_conc1.json").write_text('{"conc": 1}\n')

    result = run_bash(
        'source "$1"; copy_agentic_results "$2" "$3" run',
        SLURM_UTILS,
        source,
        workspace,
    )

    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in workspace.iterdir()) == [
        "run_conc1.json",
        "run_conc16.json",
    ]


def test_copy_agentic_results_fails_when_aggregate_is_missing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    workspace = tmp_path / "workspace"
    source.mkdir()
    workspace.mkdir()

    result = run_bash(
        'source "$1"; copy_agentic_results "$2" "$3" run',
        SLURM_UTILS,
        source,
        workspace,
    )

    assert result.returncode != 0
    assert "no run_conc*.json results found" in result.stderr


def test_llmd_agentic_adapter_uses_envoy_and_all_engine_metrics(tmp_path: Path) -> None:
    """Execute the real adapter with a recording client, without any GPUs."""
    client = tmp_path / "benchmarks/multi_node/agentic_srt.sh"
    client.parent.mkdir(parents=True)
    client.write_text("python3 -c 'import json, os; print(json.dumps(dict(os.environ)))'\n")
    env = dict(os.environ, INFMAX_CONTAINER_WORKSPACE=str(tmp_path),
               MODEL_NAME="deepseek-ai/DeepSeek-V4-Pro-0813",
               ENVOY_PORT="8080", VLLM_PORT="8200", BENCHMARK_LOGS_DIR=str(tmp_path / "logs"),
               BENCH_MAX_CONCURRENCY="64", ALL_IPS="10.0.0.1,10.0.0.2",
               PREFILL_NODES="1", DECODE_NODES="1", PREFILL_WORKERS="1", DECODE_WORKERS="1",
               SPEC_DECODING="mtp", KV_OFFLOADING="dram", KV_OFFLOAD_BACKEND="mooncake")
    result = subprocess.run(["bash", str(REPO_ROOT / "benchmarks/multi_node/llm-d/agentic.sh")],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    recorded = json.loads(result.stdout)
    assert recorded["MODEL"] == env["MODEL_NAME"]
    assert recorded["AIPERF_SERVER_URL"] == "http://localhost:8080"
    assert recorded["PORT"] == "8200"
    assert recorded["AIPERF_SERVER_METRICS_URLS"] == "http://10.0.0.1:8200/metrics,http://10.0.0.2:8200/metrics"
    assert recorded["AGENTIC_OUTPUT_DIR"] == env["BENCHMARK_LOGS_DIR"]
    assert recorded["CONC"] == "64"
    assert recorded["SPEC_DECODING"] == "mtp"
    assert recorded["KV_OFFLOADING"] == "dram"
