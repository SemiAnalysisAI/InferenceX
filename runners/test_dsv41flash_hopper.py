import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_LIB = REPO_ROOT / "benchmarks" / "benchmark_lib.sh"

LAUNCH_HARNESS = '''
    salloc() { :; }
    squeue() { echo 123; }
    srun() {
        python3 -c 'import json,os,sys; open(sys.argv[1], "a").write(json.dumps({"args": sys.argv[2:], "result_dir": os.environ.get("RESULT_DIR", "")})+"\\n")' "$SRUN_LOG" "$@"
    }
    scancel() { :; }
    export IS_MULTINODE=false IS_AGENTIC=1 SCENARIO_SUBDIR=agentic/
    export IMAGE=vllm/vllm-openai:deepseekv41-flash-0909
    export HF_HUB_CACHE=/mnt/hf_hub_cache/ RESULT_DIR=/workspace/results
    export GITHUB_WORKSPACE="$1" SRUN_LOG="$2"
    cd "$GITHUB_WORKSPACE"
'''


def launch(sku: str, log: Path, **env: str) -> dict:
    exports = " ".join(f"export {key}={value};" for key, value in env.items())
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"{LAUNCH_HARNESS}\n{exports}\nsource runners/launch_{sku}-dgxc-slurm.sh",
            "bash",
            str(REPO_ROOT),
            str(log),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    # The last srun call is the benchmark launch; earlier ones import the image.
    return json.loads(log.read_text().splitlines()[-1])


@pytest.mark.parametrize("sku", ["h100", "h200"])
def test_hopper_flash_runs_the_vllm_script_from_an_ix_mount(sku: str, tmp_path: Path) -> None:
    serve = launch(
        sku,
        tmp_path / "launch.jsonl",
        MODEL_PREFIX="dsv41flash",
        MODEL="deepseek-ai/DeepSeek-V4.1-Flash",
        PRECISION="fp4",
        FRAMEWORK="vllm",
        SPEC_DECODING="mtp",
        TP="8",
        RUNNER_NAME=f"{sku}-test",
        EXP_NAME="dsv41flash_tp8_conc1",
    )
    script = serve["args"][-1]
    assert script == f"benchmarks/single_node/agentic/dsv41flash_fp4_{sku}_vllm_mtp.sh"
    assert (REPO_ROOT / script).is_file()

    mounts = next(arg for arg in serve["args"] if arg.startswith("--container-mounts="))
    assert f"{REPO_ROOT}:/ix/," in mounts
    assert ":/mnt/hf_hub_cache/," in mounts
    # AgentX must not write runtime directories under /workspace.
    assert "/workspace" not in mounts
    assert serve["result_dir"] == "/ix/results"
    assert "--container-workdir=/ix/" in serve["args"]


def test_h100_still_resolves_scripts_without_a_framework_tag(tmp_path: Path) -> None:
    """The pre-framework h100 recipes keep working after the suffix change."""
    serve = launch(
        "h100",
        tmp_path / "launch.jsonl",
        MODEL_PREFIX="qwen3.5",
        MODEL="Qwen/Qwen3.5-397B-A17B-FP8",
        PRECISION="fp8",
        FRAMEWORK="sglang",
        SPEC_DECODING="mtp",
        TP="8",
        RUNNER_NAME="h100-test",
        EXP_NAME="qwen3.5_tp8_conc1",
    )
    script = serve["args"][-1]
    assert script == "benchmarks/single_node/agentic/qwen3.5_fp8_h100_mtp.sh"
    assert (REPO_ROOT / script).is_file()

    mounts = next(arg for arg in serve["args"] if arg.startswith("--container-mounts="))
    assert f"{REPO_ROOT}:/workspace/," in mounts
    assert serve["result_dir"] == "/workspace/results"


def resolve_loader(model_prefix: str) -> str:
    """The public-dataset loader `resolve_trace_source` picks for a prefix."""
    result = subprocess.run(
        [
            "bash",
            "-c",
            # Stub the CLI bootstrap *after* sourcing, so the real one does
            # not overwrite the stub, and no dataset is actually downloaded.
            'source "$BENCHMARK_LIB"; ensure_hf_cli() { AIPERF_HF_CLI=true; }; '
            "resolve_trace_source",
        ],
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": str(REPO_ROOT),
            "BENCHMARK_LIB": str(BENCHMARK_LIB),
            "MODEL_PREFIX": model_prefix,
        },
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.split("public-dataset: ")[1].split(" ")[0]


def test_flash_replays_the_uncapped_1m_trace_corpus() -> None:
    """DeepSeek-V4.1-Flash serves 1M context, so it must not get the 256k corpus.

    This lives with the recipe rather than in the benchmark_lib tests because
    the recipe never names a corpus: it inherits one from `resolve_trace_source`
    only because the `dsv4*` case arm also matches the `dsv41flash` prefix.
    Narrowing that arm would silently downgrade this recipe's traces.
    """
    assert resolve_loader("dsv41flash") == "semianalysis_cc_traces_weka_062126"


def test_short_context_families_still_get_the_capped_corpus() -> None:
    """The uncapped default is context-driven, not a blanket default."""
    assert resolve_loader("qwen3.8next") == "semianalysis_cc_traces_weka_062126_256k"
