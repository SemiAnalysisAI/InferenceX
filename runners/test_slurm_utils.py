import json
import os
import runpy
import subprocess
from pathlib import Path

import yaml

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


def test_patch_srt_eval_dispatch_forwards_selection_and_is_idempotent(
    tmp_path: Path,
) -> None:
    do_sweep = tmp_path / "src/srtctl/cli/do_sweep.py"
    eval_script = tmp_path / "src/srtctl/benchmarks/scripts/lm-eval/bench.sh"
    do_sweep.parent.mkdir(parents=True)
    eval_script.parent.mkdir(parents=True)
    do_sweep.write_text(
        "        for var in [\n"
        '            "RUN_EVAL",\n'
        '            "EVAL_ONLY",\n'
        '            "IS_MULTINODE",\n'
        "        ]:\n"
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
    second = subprocess.run(
        ["python3", str(PATCH_SRT_EVAL), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert do_sweep.read_text().count('"EVAL_FRAMEWORK"') == 1
    assert do_sweep.read_text().count('"EVAL_SUITE"') == 1
    assert do_sweep.read_text().count('"EVAL_CONC"') == 1
    assert do_sweep.read_text().count('"EVAL_LIMIT"') == 1
    assert do_sweep.read_text().count('"SWEBENCH_GEN_MODE"') == 1
    assert do_sweep.read_text().count('"SWEBENCH_USE_MODAL"') == 1
    assert do_sweep.read_text().count('"MODAL_TOKEN_ID"') == 1
    assert do_sweep.read_text().count('"MODAL_TOKEN_SECRET"') == 1
    assert do_sweep.read_text().count('"IS_AGENTIC"') == 1
    assert do_sweep.read_text().count('"SCENARIO_TYPE"') == 1
    assert 'run_eval --port "$PORT"' in eval_script.read_text()
    assert "--framework lm-eval" not in eval_script.read_text()
    assert 'stage_eval_artifacts /logs/eval_results "$PWD" || true' in eval_script.read_text()
    assert "cp -v" not in eval_script.read_text()
    assert "already patched" in second.stdout


def test_patch_srt_vllm_dp_ranks_groups_tensor_parallel_devices(
    tmp_path: Path,
) -> None:
    symbols = runpy.run_path(str(PATCH_SRT_DP_RANKS))
    backend = tmp_path / "src/srtctl/backends/vllm.py"
    backend.parent.mkdir(parents=True)
    backend.write_text(f"prefix\n{symbols['OLD_BLOCK']}suffix\n")

    first = subprocess.run(
        ["python3", str(PATCH_SRT_DP_RANKS), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        ["python3", str(PATCH_SRT_DP_RANKS), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    patched = backend.read_text()
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert symbols["OLD_BLOCK"] not in patched
    assert patched.count(symbols["NEW_BLOCK"]) == 1
    assert "gpus_per_dp_rank = tp_size * pp_size" in patched
    assert "gpu_indices=rank_gpus" in patched
    assert "already patched" in second.stdout.lower()


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
    second = subprocess.run(
        ["python3", str(PATCH_TRTLLM_CHAT_STORE), str(protocol)],
        check=False,
        capture_output=True,
        text=True,
    )

    patched = protocol.read_text()
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert patched.count("store: Optional[Literal[False]] = False") == 1
    assert patched.count("store: Optional[bool] = True") == 1
    assert "already patched" in second.stdout.lower()


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

def test_patch_vllm_simple_kv_offload_splits_heterogeneous_layers(
    tmp_path: Path,
) -> None:
    symbols = runpy.run_path(str(PATCH_VLLM_SIMPLE_KV))
    worker = tmp_path / "worker.py"
    worker.write_text(
        f"prefix\n{symbols['OLD_SETUP']}{symbols['OLD_LOOP']}suffix\n"
    )

    first = subprocess.run(
        ["python3", str(PATCH_VLLM_SIMPLE_KV), str(worker)],
        check=False,
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        ["python3", str(PATCH_VLLM_SIMPLE_KV), str(worker)],
        check=False,
        capture_output=True,
        text=True,
    )

    patched = worker.read_text()
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert symbols["OLD_SETUP"] not in patched
    assert symbols["OLD_LOOP"] not in patched
    assert patched.count(symbols["NEW_SETUP"]) == 1
    assert patched.count(symbols["NEW_LOOP"]) == 1
    assert "split_storage_by_layer" in patched
    assert "tensor.storage_offset() * tensor.element_size()" in patched
    assert "already patched" in second.stdout.lower()


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


def test_minimax_vllm_launchers_patch_simple_kv_offload() -> None:
    launchers = (
        REPO_ROOT / "benchmarks/single_node/agentic/minimaxm3_fp4_b200_mtp.sh",
        REPO_ROOT / "benchmarks/single_node/agentic/minimaxm3_fp4_b300_mtp.sh",
    )

    for launcher in launchers:
        assert "patch_vllm_simple_kv_offload.py" in launcher.read_text(), launcher


def test_minimax_trt_launchers_patch_chat_store_request() -> None:
    launchers = (
        REPO_ROOT / "benchmarks/single_node/agentic/minimaxm3_fp4_b200_trt_mtp.sh",
        REPO_ROOT / "benchmarks/single_node/agentic/minimaxm3_fp4_b300_trt_mtp.sh",
    )

    for launcher in launchers:
        assert "patch_trtllm_chat_store.py" in launcher.read_text(), launcher

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
    rewritten = recipe.read_text()
    assert '"rejection_sample_method":"block"' in rewritten
    assert "synthetic_acceptance_length" not in rewritten



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
    rewritten = recipe.read_text()
    assert "SGLANG_SIMULATE_ACC_" not in rewritten
    assert "KEEP_ME: unchanged" in rewritten


def test_sglang_throughput_replaces_existing_simulated_acceptance(
    tmp_path: Path,
) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "backend:\n"
        "  aggregated_environment:\n"
        '    SGLANG_SIMULATE_ACC_LEN: "2.99"\n'
        '    SGLANG_SIMULATE_ACC_METHOD: "match-expected"\n'
        '    SGLANG_SIMULATE_ACC_TOKEN_MODE: "real-draft-token"\n'
        "    KEEP_ME: unchanged\n"
    )

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

    assert result.returncode == 0, result.stderr
    rewritten = recipe.read_text()
    assert rewritten.count("SGLANG_SIMULATE_ACC_LEN") == 1
    assert 'SGLANG_SIMULATE_ACC_LEN: "3.39"' in rewritten
    assert "KEEP_ME: unchanged" in rewritten

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


def test_nvidia_srt_launchers_prepare_kimi_eval_dispatch() -> None:
    launchers = (
        REPO_ROOT / "runners/launch_h100-dgxc-slurm.sh",
        REPO_ROOT / "runners/launch_h200-dgxc-slurm.sh",
        REPO_ROOT / "runners/launch_b200-nscale-slurm.sh",
        REPO_ROOT / "runners/launch_b300-nv.sh",
        REPO_ROOT / "runners/launch_gb200-nv.sh",
        REPO_ROOT / "runners/launch_gb300-nv.sh",
    )

    for launcher in launchers:
        content = launcher.read_text()
        assert "patch_srt_eval_dispatch.py" in content
        patch_command = content.index("patch_srt_eval_dispatch.py")
        assert "|| exit 1" in content[patch_command : patch_command + 200]
        assert 'EVAL_FRAMEWORK:-lm-eval}" != "lm-eval"' in content
        assert "inject_synthetic_acceptance" in content


def test_gb200_kimi_compilation_config_preserves_all_settings() -> None:
    recipes = {
        "agg-gb200-tep16-balanced-agentic.yaml": 96,
        "agg-gb200-tp16-latency-agentic.yaml": 24,
    }
    recipe_dir = (
        REPO_ROOT / "benchmarks/multi_node/srt-slurm-recipes/vllm/kimi-k3/agentic"
    )

    for filename, largest_capture in recipes.items():
        recipe = yaml.safe_load((recipe_dir / filename).read_text())
        raw_config = recipe["backend"]["vllm_config"]["aggregated"][
            "compilation-config"
        ]
        compilation_config = json.loads(raw_config)

        assert compilation_config["cudagraph_capture_sizes"][-1] == largest_capture
        assert compilation_config["pass_config"]["fuse_allreduce_rms"] is False


def test_gb200_kimi_recipes_configure_tool_parser() -> None:
    recipe_dir = (
        REPO_ROOT / "benchmarks/multi_node/srt-slurm-recipes/vllm/kimi-k3/agentic"
    )
    recipe_paths = sorted(recipe_dir.glob("agg-gb200-*-agentic.yaml"))
    frontend_counts = {"dynamo": 0, "vllm": 0}

    assert len(recipe_paths) == 11
    for recipe_path in recipe_paths:
        recipe = yaml.safe_load(recipe_path.read_text())
        frontend = recipe["frontend"]
        frontend_type = frontend["type"]
        config = recipe["backend"]["vllm_config"]["aggregated"]
        assert frontend_type in frontend_counts, recipe_path
        frontend_counts[frontend_type] += 1
        if frontend_type == "dynamo":
            args = frontend["args"]
            assert args["dyn-chat-processor"] == "vllm", recipe_path
            assert args["tool-call-parser"] == "kimi_k3", recipe_path
            assert args["reasoning-parser"] == "kimi_k3", recipe_path
            assert args["enable-auto-tool-choice"] is True, recipe_path
            assert config["dyn-tool-call-parser"] == "kimi_k3", recipe_path
            assert config["dyn-reasoning-parser"] == "kimi_k3", recipe_path
        else:
            assert config["enable-auto-tool-choice"] is True, recipe_path
            assert config["tool-call-parser"] == "kimi_k3", recipe_path
            assert config["reasoning-parser"] == "kimi_k3", recipe_path

    assert frontend_counts == {"dynamo": 6, "vllm": 5}


def test_gb200_dynamo_minimax_recipes_configure_frontend_tool_parser() -> None:
    recipe_dir = (
        REPO_ROOT
        / "benchmarks/multi_node/srt-slurm-recipes/vllm/minimax-m3"
        / "gb200-fp4/agentic"
    )
    recipe_paths = sorted(recipe_dir.glob("*.yaml"))

    assert len(recipe_paths) == 6
    for recipe_path in recipe_paths:
        recipe = yaml.safe_load(recipe_path.read_text())
        args = recipe["frontend"]["args"]
        assert args["dyn-chat-processor"] == "vllm", recipe_path
        assert args["tool-call-parser"] == "minimax_m3", recipe_path
        assert args["reasoning-parser"] == "minimax_m3", recipe_path
        assert args["enable-auto-tool-choice"] is True, recipe_path


def test_mi355_minimax_launcher_configures_reasoning_parser() -> None:
    launcher = (
        REPO_ROOT
        / "benchmarks/single_node/agentic/minimaxm3_fp4_mi355x_mtp.sh"
    ).read_text()

    assert "--tool-call-parser minimax_m3" in launcher
    assert "--reasoning-parser minimax_m3" in launcher
    assert "--enable-auto-tool-choice" in launcher


def test_dynamo_sglang_agentic_recipes_parse_tools_at_frontend() -> None:
    recipe_roots = (
        (
            REPO_ROOT
            / "benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4/agentic",
            ("deepseekv4", "deepseek-v4"),
        ),
        (
            REPO_ROOT
            / "benchmarks/multi_node/srt-slurm-recipes/sglang/qwen3.5/gb300-fp4/agentic",
            ("qwen3_coder", "qwen3"),
        ),
    )
    checked = 0

    for recipe_root, (tool_parser, reasoning_parser) in recipe_roots:
        for recipe_path in recipe_root.glob("*.yaml"):
            recipe = yaml.safe_load(recipe_path.read_text())
            frontend = recipe["frontend"]
            if frontend["type"] != "dynamo":
                continue
            args = frontend["args"]
            assert args["dyn-chat-processor"] == "sglang", recipe_path
            assert args["tool-call-parser"] == tool_parser, recipe_path
            assert args["reasoning-parser"] == reasoning_parser, recipe_path
            checked += 1

    assert checked == 15


def test_swebench_container_paths_forward_modal_credentials() -> None:
    paths = (
        REPO_ROOT / "benchmarks/multi_node/llm-d/submit.sh",
        REPO_ROOT / "benchmarks/multi_node/llm-d/job.slurm",
        REPO_ROOT / "runners/launch_h100-cr.sh",
        REPO_ROOT / "runners/launch_mi325x-tw.sh",
    )

    for path in paths:
        content = path.read_text()
        assert "SWEBENCH_USE_MODAL" in content, path
        assert "MODAL_TOKEN_ID" in content, path
        assert "MODAL_TOKEN_SECRET" in content, path
        assert "IS_AGENTIC" in content, path
        assert "SCENARIO_TYPE" in content, path
        if path.name == "job.slurm" and "llm-d" in path.parts:
            assert "-e MODAL_TOKEN_ID \\" in content
            assert "-e MODAL_TOKEN_SECRET \\" in content
            assert "-e MODAL_TOKEN_ID=" not in content
            assert "-e MODAL_TOKEN_SECRET=" not in content
