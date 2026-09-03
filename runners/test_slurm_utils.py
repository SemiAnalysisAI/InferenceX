import json
import os
import runpy
import subprocess
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
SRT_ADAPTER = REPO_ROOT / "utils" / "srt_slurm.py"
SLURM_UTILS = REPO_ROOT / "runners" / "slurm_utils.sh"
PATCH_SRT_EVAL = REPO_ROOT / "runners" / "patch_srt_eval_dispatch.py"
PATCH_SRT_DP_RANKS = REPO_ROOT / "runners" / "patch_srt_vllm_dp_ranks.py"
PATCH_TRTLLM_CHAT_STORE = REPO_ROOT / "runners" / "patch_trtllm_chat_store.py"
PATCH_VLLM_SIMPLE_KV = REPO_ROOT / "runners" / "patch_vllm_simple_kv_offload.py"
INJECT_ACCEPTANCE = REPO_ROOT / "runners" / "inject_synthetic_acceptance.py"


def test_srt_adapter_preserves_serving_recipe_and_uses_native_image_fallback(tmp_path: Path) -> None:
    prepare = runpy.run_path(str(SRT_ADAPTER))["prepare_recipe"]
    backend = {"type": "atom", "atom_config": {"prefill": {"kv_cache_dtype": "fp8", "max-num-seqs": 256}}}
    recipe = {
        "model": {"container": "model-image"},
        "backend": backend,
        "benchmark": {"type": "custom", "command": "run-the-original-benchmark"},
    }
    profile = {"default_mounts": {"/host/rdma": "/container/rdma"}}
    paths = {
        "workspace": tmp_path / "workspace",
        "results_root": tmp_path / "results",
        "aiperf_cache": tmp_path / "aiperf",
        "image_cache": tmp_path / "images",
    }
    env = {"IMAGE": "vendor/engine:pinned", "CONC_LIST": "4 8", "UNRELATED_SECRET": "do-not-forward"}
    prepared, cluster = prepare(recipe, profile, env, **paths)

    assert prepared["backend"] == backend
    assert prepared["benchmark"]["command"] == "run-the-original-benchmark"
    assert prepared["benchmark"]["env"] == {"CONC_LIST": "4 8"}
    assert "env" not in recipe["benchmark"]
    assert profile == {"default_mounts": {"/host/rdma": "/container/rdma"}}
    assert cluster["containers"] == {"model-image": "vendor/engine:pinned"}
    assert cluster["default_mounts"][str(paths["workspace"])] == "/infmax-workspace"
    paths["image_cache"].mkdir()
    cached_image = paths["image_cache"] / "vendor_engine_pinned.sqsh"
    cached_image.write_bytes(b"cached-image")
    _, cached_cluster = prepare(recipe, profile, env, **paths)
    assert cached_cluster["containers"] == {"model-image": str(cached_image)}


def test_srt_adapter_preserves_legacy_sglang_admission_and_eval_contract(tmp_path: Path) -> None:
    prepare = runpy.run_path(str(SRT_ADAPTER))["prepare_recipe"]
    recipe = {
        "model": {"container": "image"},
        "backend": {
            "type": "sglang",
            "sglang_config": {
                "prefill": {"tp-size": 8, "ep-size": 8, "context-length": 9472},
                "decode": {"tp-size": 8, "ep-dispatch-algorithm": "experimental"},
            },
            "decode_environment": {
                "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "17",
                "MORI_MAX_DISPATCH_TOKENS_DECODE": "1",
                "SGLANG_SIMULATE_ACC_LEN": "3",
            },
        },
        "benchmark": {"type": "custom", "command": "original-throughput-command"},
    }
    env = {
        "IMAGE": "vendor/engine:pinned",
        "MODEL": "vendor/model",
        "CONC_LIST": "16 64",
        "PREFILL_DP_ATTN": "true",
        "PREFILL_EP": "8",
        "DECODE_TP": "8",
        "DECODE_MTP_SIZE": "3",
    }
    paths = {
        "workspace": tmp_path,
        "results_root": tmp_path / "results",
        "aiperf_cache": tmp_path / "cache",
        "image_cache": tmp_path / "images",
    }
    throughput, _ = prepare(recipe, {}, env, **paths)
    backend = throughput["backend"]
    assert backend["sglang_config"]["prefill"]["max-running-requests"] == 64
    assert backend["sglang_config"]["decode"]["max-running-requests"] == 64
    assert backend["decode_environment"]["SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] == "17"
    assert backend["decode_environment"]["MORI_MAX_DISPATCH_TOKENS_DECODE"] == "32"
    assert backend["decode_environment"]["SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD"] == "16"
    assert backend["decode_environment"]["SGLANG_SIMULATE_ACC_LEN"] == "3"

    evaluation, _ = prepare(recipe, {}, {**env, "EVAL_ONLY": "true", "EVAL_CONC": "64"}, **paths)
    assert "SGLANG_SIMULATE_ACC_LEN" not in evaluation["backend"]["decode_environment"]
    assert "ep-dispatch-algorithm" not in evaluation["backend"]["sglang_config"]["decode"]
    assert evaluation["benchmark"]["env"]["MODEL_NAME"] == "vendor/model"
    assert evaluation["benchmark"]["env"]["EVAL_MAX_MODEL_LEN"] == "9472"
    assert "original-throughput-command" not in evaluation["benchmark"]["command"]
    assert 'run_eval --port "${SRT_FRONTEND_PORT}"' in evaluation["benchmark"]["command"]


def test_srt_result_collection_is_job_scoped_and_preserves_artifact_names(tmp_path: Path) -> None:
    collect = runpy.run_path(str(SRT_ADAPTER))["collect_results"]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    results_root = tmp_path / "results"
    for job_id in ("42", "99"):
        fixed = results_root / job_id / "fixed-seq"
        fixed.mkdir(parents=True)
        (fixed / "benchmark-c8.json").write_text(json.dumps({"job": job_id}))
    env = {
        "RESULT_FILENAME": "benchmark",
        "PREFILL_NUM_WORKERS": "1",
        "PREFILL_TP": "8",
        "DECODE_NUM_WORKERS": "1",
        "DECODE_TP": "8",
        "DISAGG": "true",
    }
    submission = {"slurm_job_id": "42", "output_dir": str(tmp_path / "outputs" / "42")}
    collect(submission, env, workspace=workspace, results_root=results_root)
    artifact = workspace / "benchmark_srt-42_conc8_gpus_16_ctx_8_gen_8.json"
    assert json.loads(artifact.read_text()) == {"job": "42"}
    assert not list(workspace.glob("*srt-99*"))

    with pytest.raises(ValueError, match="No eval metadata"):
        collect(submission, {**env, "EVAL_ONLY": "true"}, workspace=workspace, results_root=results_root)


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
