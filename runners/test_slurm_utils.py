import json
import os
import runpy
import subprocess
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_UTILS = REPO_ROOT / "runners" / "slurm_utils.sh"
PATCH_SRT_EVAL = REPO_ROOT / "runners" / "patch_srt_eval_dispatch.py"
PATCH_SRT_DP_RANKS = REPO_ROOT / "runners" / "patch_srt_vllm_dp_ranks.py"
PATCH_TRTLLM_CHAT_STORE = REPO_ROOT / "runners" / "patch_trtllm_chat_store.py"
PATCH_VLLM_SIMPLE_KV = REPO_ROOT / "runners" / "patch_vllm_simple_kv_offload.py"
INJECT_ACCEPTANCE = REPO_ROOT / "runners" / "inject_synthetic_acceptance.py"


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


def test_patch_srt_eval_dispatch_forwards_selection_and_is_idempotent(
    tmp_path: Path,
) -> None:
    do_sweep = tmp_path / "src/srtctl/cli/do_sweep.py"
    eval_script = tmp_path / "src/srtctl/benchmarks/scripts/lm-eval/bench.sh"
    do_sweep.parent.mkdir(parents=True)
    eval_script.parent.mkdir(parents=True)
    do_sweep.write_text(
        "def forwarded(environment):\n"
        "    forwarded = {}\n"
        "    if environment:\n"
        "        for var in [\n"
        '            "RUN_EVAL",\n'
        '            "EVAL_ONLY",\n'
        '            "IS_MULTINODE",\n'
        "        ]:\n"
        "            if var in environment:\n"
        "                forwarded[var] = environment[var]\n"
        "    return forwarded\n"
    )
    eval_script.write_text(
        'run_eval --framework lm-eval --port "$PORT" || eval_rc=$?\n'
        "cp -v results*.json /logs/eval_results/ 2>/dev/null || true\n"
        "cp -v sample*.jsonl /logs/eval_results/ 2>/dev/null || true\n"
    )

    first = subprocess.run(
        ["python3", str(PATCH_SRT_EVAL), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    patched_sweep = do_sweep.read_text()
    patched_eval = eval_script.read_text()
    second = subprocess.run(
        ["python3", str(PATCH_SRT_EVAL), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert do_sweep.read_text() == patched_sweep
    assert eval_script.read_text() == patched_eval
    settings = {name: f"value-{name}" for name in (
        "EVAL_FRAMEWORK", "EVAL_SUITE", "EVAL_CONC", "EVAL_LIMIT",
        "SWEBENCH_GEN_MODE", "SWEBENCH_USE_MODAL", "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET", "IS_AGENTIC", "SCENARIO_TYPE",
    )}
    forwarded = runpy.run_path(str(do_sweep))["forwarded"]
    assert forwarded({**settings, "UNRELATED": "do not forward"}) == settings

    execution = run_bash(
        'PORT=12345; run_eval() { printf "eval:%s\\n" "$*"; }; '
        'stage_eval_artifacts() { printf "stage:%s\\n" "$1"; }; source "$1"',
        eval_script,
    )
    assert execution.returncode == 0, execution.stderr
    assert execution.stdout.splitlines() == ["eval:--port 12345", "stage:/logs/eval_results"]


def test_patch_srt_vllm_dp_ranks_is_idempotent_and_preserves_surrounding_code(
    tmp_path: Path,
) -> None:
    symbols = runpy.run_path(str(PATCH_SRT_DP_RANKS))
    backend = tmp_path / "src/srtctl/backends/vllm.py"
    backend.parent.mkdir(parents=True)
    original = f"prefix\n{symbols['OLD_BLOCK']}suffix\n"
    backend.write_text(original)

    first = subprocess.run(
        ["python3", str(PATCH_SRT_DP_RANKS), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    patched = backend.read_text()
    second = subprocess.run(
        ["python3", str(PATCH_SRT_DP_RANKS), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert patched != original
    assert patched.startswith("prefix\n") and patched.endswith("suffix\n")
    assert backend.read_text() == patched


def test_patch_srt_vllm_dp_ranks_rejects_unknown_source(tmp_path: Path) -> None:
    backend = tmp_path / "src/srtctl/backends/vllm.py"
    backend.parent.mkdir(parents=True)
    backend.write_text("unsupported backend\n")

    result = subprocess.run(
        ["python3", str(PATCH_SRT_DP_RANKS), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert backend.read_text() == "unsupported backend\n"


def test_patch_trtllm_chat_store_accepts_false_and_is_idempotent(
    tmp_path: Path,
) -> None:
    protocol = tmp_path / "openai_protocol.py"
    protocol.write_text(
        "from typing import Literal, Optional\n\n"
        "class ChatCompletionRequest(OpenAIBaseModel):\n"
        "    messages: list\n"
        "    stream: Optional[bool] = False\n"
        "    user: Optional[str] = None\n\n"
        "class ResponsesRequest(OpenAIBaseModel):\n"
        "    store: Optional[bool] = True\n"
    )

    first = subprocess.run(
        ["python3", str(PATCH_TRTLLM_CHAT_STORE), str(protocol)],
        check=False,
        capture_output=True,
        text=True,
    )
    patched = protocol.read_text()
    second = subprocess.run(
        ["python3", str(PATCH_TRTLLM_CHAT_STORE), str(protocol)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert protocol.read_text() == patched
    models = runpy.run_path(str(protocol), init_globals={"OpenAIBaseModel": BaseModel})
    chat = models["ChatCompletionRequest"]
    assert chat(messages=[], store=False).store is False
    with pytest.raises(ValidationError):
        chat(messages=[], store=True)
    assert models["ResponsesRequest"](store=True).store is True


def test_patch_trtllm_chat_store_rejects_unknown_source(tmp_path: Path) -> None:
    protocol = tmp_path / "openai_protocol.py"
    protocol.write_text("unsupported protocol\n")

    result = subprocess.run(
        ["python3", str(PATCH_TRTLLM_CHAT_STORE), str(protocol)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert protocol.read_text() == "unsupported protocol\n"

def test_patch_vllm_simple_kv_offload_is_idempotent_and_preserves_surrounding_code(
    tmp_path: Path,
) -> None:
    symbols = runpy.run_path(str(PATCH_VLLM_SIMPLE_KV))
    worker = tmp_path / "worker.py"
    original = f"prefix\n{symbols['OLD_SETUP']}{symbols['OLD_LOOP']}suffix\n"
    worker.write_text(original)

    first = subprocess.run(
        ["python3", str(PATCH_VLLM_SIMPLE_KV), str(worker)],
        check=False,
        capture_output=True,
        text=True,
    )
    patched = worker.read_text()
    second = subprocess.run(
        ["python3", str(PATCH_VLLM_SIMPLE_KV), str(worker)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert patched != original
    assert patched.startswith("prefix\n") and patched.endswith("suffix\n")
    assert worker.read_text() == patched


def test_patch_vllm_simple_kv_offload_rejects_unknown_source(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text("unsupported worker\n")

    result = subprocess.run(
        ["python3", str(PATCH_VLLM_SIMPLE_KV), str(worker)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert worker.read_text() == "unsupported worker\n"


def test_patch_srt_eval_dispatch_preflights_before_writing(tmp_path: Path) -> None:
    do_sweep = tmp_path / "src/srtctl/cli/do_sweep.py"
    eval_script = tmp_path / "src/srtctl/benchmarks/scripts/lm-eval/bench.sh"
    do_sweep.parent.mkdir(parents=True)
    eval_script.parent.mkdir(parents=True)
    original_do_sweep = '            "EVAL_ONLY",\n            "IS_MULTINODE",\n'
    original_eval_script = "unsupported eval hook\n"
    do_sweep.write_text(original_do_sweep)
    eval_script.write_text(original_eval_script)

    result = subprocess.run(
        ["python3", str(PATCH_SRT_EVAL), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert do_sweep.read_text() == original_do_sweep
    assert eval_script.read_text() == original_eval_script


def test_patch_srt_eval_dispatch_rejects_mixed_patch_state(tmp_path: Path) -> None:
    do_sweep = tmp_path / "src/srtctl/cli/do_sweep.py"
    eval_script = tmp_path / "src/srtctl/benchmarks/scripts/lm-eval/bench.sh"
    do_sweep.parent.mkdir(parents=True)
    eval_script.parent.mkdir(parents=True)
    original_do_sweep = (
        '            "EVAL_ONLY",\n'
        '            "IS_MULTINODE",\n'
        '            "EVAL_ONLY",\n'
        '            "EVAL_FRAMEWORK",\n'
        '            "EVAL_SUITE",\n'
        '            "IS_MULTINODE",\n'
    )
    original_eval_script = (
        'run_eval --framework lm-eval --port "$PORT" || eval_rc=$?\n'
        "cp -v results*.json /logs/eval_results/ 2>/dev/null || true\n"
        "cp -v sample*.jsonl /logs/eval_results/ 2>/dev/null || true\n"
    )
    do_sweep.write_text(original_do_sweep)
    eval_script.write_text(original_eval_script)

    result = subprocess.run(
        ["python3", str(PATCH_SRT_EVAL), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "invalid patch state" in result.stderr
    assert do_sweep.read_text() == original_do_sweep
    assert eval_script.read_text() == original_eval_script


def test_eval_only_restores_real_vllm_acceptance(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "speculative-config: "
        """'{\"method\":\"dspark\",\"num_speculative_tokens\":2,"""
        """\"rejection_sample_method\":\"synthetic\","""
        """\"synthetic_acceptance_length\":2.51}'\n"""
    )
    env = {
        **os.environ,
        "EVAL_ONLY": "true",
        "SYNTHETIC_ACCEPTANCE": "true",
    }

    result = subprocess.run(
        ["python3", str(INJECT_ACCEPTANCE), str(recipe), "vllm"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    speculative_config = json.loads(yaml.safe_load(recipe.read_text())["speculative-config"])
    assert speculative_config["rejection_sample_method"] == "block"
    assert "synthetic_acceptance_length" not in speculative_config


def test_eval_only_removes_sglang_simulated_acceptance(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "backend:\n"
        "  sglang_config:\n"
        "    decode_environment:\n"
        '      SGLANG_SIMULATE_ACC_LEN: "2.99"\n'
        '      SGLANG_SIMULATE_ACC_METHOD: "match-expected"\n'
        '      SGLANG_SIMULATE_ACC_TOKEN_MODE: "real-draft-token"\n'
        "      KEEP_ME: unchanged\n"
    )

    result = subprocess.run(
        ["python3", str(INJECT_ACCEPTANCE), str(recipe), "dynamo-sglang"],
        env={**os.environ, "EVAL_ONLY": "true"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    environment = yaml.safe_load(recipe.read_text())["backend"]["sglang_config"]["decode_environment"]
    assert environment == {"KEEP_ME": "unchanged"}


def test_sglang_throughput_rejects_existing_simulated_acceptance(
    tmp_path: Path,
) -> None:
    recipe = tmp_path / "recipe.yaml"
    original = (
        "backend:\n"
        "  aggregated_environment:\n"
        '    SGLANG_SIMULATE_ACC_LEN: "2.99"\n'
        '    SGLANG_SIMULATE_ACC_METHOD: "match-expected"\n'
        '    SGLANG_SIMULATE_ACC_TOKEN_MODE: "real-draft-token"\n'
        "    KEEP_ME: unchanged\n"
    )
    recipe.write_text(original)

    result = subprocess.run(
        ["python3", str(INJECT_ACCEPTANCE), str(recipe), "dynamo-sglang"],
        env={
            **os.environ,
            "SYNTHETIC_ACCEPTANCE": "true",
            "SYNTHETIC_ACCEPTANCE_LENGTH": "3.39",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "already contains SGLANG_SIMULATE_ACC_" in result.stderr
    assert recipe.read_text() == original


def test_eval_only_acceptance_rewrite_allows_non_speculative_recipe(
    tmp_path: Path,
) -> None:
    recipe = tmp_path / "recipe.yaml"
    original = "backend:\n  type: vllm\n"
    recipe.write_text(original)

    result = subprocess.run(
        ["python3", str(INJECT_ACCEPTANCE), str(recipe), "dynamo-vllm"],
        env={**os.environ, "EVAL_ONLY": "true"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert recipe.read_text() == original
