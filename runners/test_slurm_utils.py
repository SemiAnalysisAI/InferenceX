import json
import os
import re
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


def test_copy_fixed_sequence_results_bounds_long_utf8_config_names(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    for suffix in ('a', 'b'):
        directory = source / ('isl8192_osl1024_' + '长' * 50 + suffix)
        directory.mkdir(parents=True)
        (directory / 'results_concurrency_16_gpus_8_ctx_4_gen_4.json').write_text('{"completed": 16}')
    result = run_bash(
        'set -eo pipefail; source "$1"; copy_fixed_sequence_results "$2" "$3" "$4"',
        SLURM_UTILS, source, workspace, 'r' * 120,
    )
    assert result.returncode == 0, result.stderr
    outputs = list(workspace.glob('*.json'))
    assert len(outputs) == 2
    for path in outputs:
        assert path.name.endswith('_conc16_gpus_8_ctx_4_gen_4.json')
        assert len(('power_validation_' + path.name + '.tmp').encode()) <= 255
        assert json.loads(path.read_text()) == {'completed': 16}


def test_copy_fixed_sequence_results_preserves_names_contents_and_discovery(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for relative, content in {
        "isl128_osl256/nested/results_concurrency_16_gpus_8_ctx_128_gen_256.json": b'{"completed": 16}\n',
        "isl32_osl64/results_concurrency_2_gpus_16.json": b'{"completed": 2}\n',
        "isl32_osl64/server.json": b"ignored",
        "unrelated/results_concurrency_4_gpus_8.json": b"ignored",
        "results_concurrency_1_gpus_8.json": b"ignored",
    }.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    result = run_bash(
        'set -eo pipefail; source "$1"; copy_fixed_sequence_results "$2" "$3" run',
        SLURM_UTILS,
        source,
        workspace,
    )

    assert result.returncode == 0, result.stderr
    expected = {
        "run_isl128_osl256_conc16_gpus_8_ctx_128_gen_256.json": b'{"completed": 16}\n',
        "run_isl32_osl64_conc2_gpus_16.json": b'{"completed": 2}\n',
    }
    assert {path.name: path.read_bytes() for path in workspace.iterdir()} == expected
    for name in expected:
        assert f"Copied result file to: {workspace / name}\n" in result.stdout
    assert result.stdout.endswith("All result files processed\n")


def test_copy_fixed_sequence_results_allows_empty_discovery(tmp_path: Path) -> None:
    result = run_bash(
        'set -e; source "$1"; copy_fixed_sequence_results "$2" "$2" run',
        SLURM_UTILS,
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        f"Warning: No result subdirectories found in {tmp_path}\n"
        "All result files processed\n"
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("errexit", [False, True])
@pytest.mark.parametrize("failure", ["discovery", "copy"])
def test_copy_fixed_sequence_results_preserves_caller_error_mode(
    tmp_path: Path, errexit: bool, failure: str,
) -> None:
    source = tmp_path / "source"
    if failure == "copy":
        result_dir = source / "isl1_osl1"
        result_dir.mkdir(parents=True)
        (result_dir / "results_concurrency_1_gpus_8.json").write_text("{}\n")

    result = run_bash(
        f'set {"-e" if errexit else "+e"}; source "$1"; '
        'copy_fixed_sequence_results "$2" "$3" run; echo continued',
        SLURM_UTILS,
        source,
        tmp_path / "missing-workspace",
    )

    assert result.returncode == (1 if errexit else 0)
    assert ("continued\n" in result.stdout) is not errexit
    assert ("All result files processed\n" in result.stdout) is not errexit
    if failure == "copy":
        assert result.stderr
        assert ("Copied result file to:" in result.stdout) is not errexit
    else:
        assert not result.stderr
        assert ("No result subdirectories found" in result.stdout) is not errexit


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


def test_sglang_throughput_injects_nested_override_environment(
    tmp_path: Path,
) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "base:\n"
        "  backend:\n"
        "    aggregated_environment:\n"
        "      KEEP_ME: unchanged\n"
        "    sglang_config:\n"
        "      aggregated:\n"
        "        speculative-num-steps: 5\n"
        "        speculative-num-draft-tokens: 6\n"
    )

    result = subprocess.run(
        ["python3", str(INJECT_ACCEPTANCE), str(recipe), "dynamo-sglang"],
        env={
            **os.environ,
            "SYNTHETIC_ACCEPTANCE": "true",
            "SYNTHETIC_ACCEPTANCE_LENGTH": "4.41",
            "MODEL_PREFIX": "qwen3.5",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    environment = yaml.safe_load(recipe.read_text())["base"]["backend"][
        "aggregated_environment"
    ]
    assert environment == {
        "KEEP_ME": "unchanged",
        "SGLANG_SIMULATE_ACC_LEN": "4.41",
        "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
        "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
    }


def test_qwen35_explicit_acceptance_must_match_recipe_golden_al(
    tmp_path: Path,
) -> None:
    recipe = tmp_path / "recipe.yaml"
    original = (
        "base:\n"
        "  backend:\n"
        "    aggregated_environment:\n"
        "      KEEP_ME: unchanged\n"
        "    sglang_config:\n"
        "      aggregated:\n"
        "        speculative-num-steps: 5\n"
        "        speculative-num-draft-tokens: 6\n"
    )
    recipe.write_text(original)

    result = subprocess.run(
        ["python3", str(INJECT_ACCEPTANCE), str(recipe), "dynamo-sglang"],
        env={
            **os.environ,
            "SYNTHETIC_ACCEPTANCE": "true",
            "SYNTHETIC_ACCEPTANCE_LENGTH": "4.80",
            "MODEL_PREFIX": "qwen3.5",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "does not match golden AL=4.41" in result.stderr
    assert "num_spec_tokens=5" in result.stderr
    assert recipe.read_text() == original


def test_qwen35_auto_lookup_uses_committed_golden_distribution(
    tmp_path: Path,
) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "base:\n"
        "  backend:\n"
        "    aggregated_environment:\n"
        "      KEEP_ME: unchanged\n"
        "    sglang_config:\n"
        "      aggregated:\n"
        "        speculative-num-steps: 6\n"
        "        speculative-num-draft-tokens: 7\n"
    )

    result = subprocess.run(
        ["python3", str(INJECT_ACCEPTANCE), str(recipe), "dynamo-sglang"],
        env={
            **os.environ,
            "SYNTHETIC_ACCEPTANCE": "true",
            "MODEL_PREFIX": "qwen3.5",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "golden_al_distribution/qwen3.5_mtp.yaml" in result.stdout
    assert 'SGLANG_SIMULATE_ACC_LEN: "4.8"' in recipe.read_text()


def test_qwen35_acceptance_rejects_wrong_draft_window(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.yaml"
    original = (
        "base:\n"
        "  backend:\n"
        "    aggregated_environment:\n"
        "      KEEP_ME: unchanged\n"
        "    sglang_config:\n"
        "      aggregated:\n"
        "        speculative-num-steps: 6\n"
        "        speculative-num-draft-tokens: 6\n"
    )
    recipe.write_text(original)

    result = subprocess.run(
        ["python3", str(INJECT_ACCEPTANCE), str(recipe), "dynamo-sglang"],
        env={
            **os.environ,
            "SYNTHETIC_ACCEPTANCE": "true",
            "SYNTHETIC_ACCEPTANCE_LENGTH": "4.80",
            "MODEL_PREFIX": "qwen3.5",
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "must equal speculative-num-steps + 1 (7)" in result.stderr
    assert recipe.read_text() == original


def test_b300_qwen35_pareto_points_match_thinking_on_golden_al() -> None:
    master = yaml.safe_load((REPO_ROOT / "configs/nvidia-master.yaml").read_text())
    points = master["qwen3.5-fp4-b300-dynamo-sglang-agentic-pareto"][
        "scenarios"
    ]["agentic-coding"][0]["search-space"]
    golden = yaml.safe_load(
        (REPO_ROOT / "golden_al_distribution/qwen3.5_mtp.yaml").read_text()
    )["qwen3.5-397b-a17b-nvfp4"]["thinking_on"]
    recipe_root = REPO_ROOT / "benchmarks/multi_node/srt-slurm-recipes"
    expected = {
        64: (3, 3.39),
        44: (7, 5.04),
        36: (7, 5.04),
        40: (5, 4.41),
        4: (6, 4.80),
    }

    actual = {}
    for point in points:
        settings = dict(
            item.split("=", 1)
            for item in point["worker"]["additional-settings"]
        )
        recipe_rel = settings["CONFIG_FILE"].split(":", 1)[0]
        recipe_rel = recipe_rel.removeprefix("recipes/")
        recipe_text = (recipe_root / recipe_rel).read_text()
        steps = {
            int(value)
            for value in re.findall(r"speculative-num-steps:\s*(\d+)", recipe_text)
        }
        drafts = {
            int(value)
            for value in re.findall(
                r"speculative-num-draft-tokens:\s*(\d+)", recipe_text
            )
        }
        assert len(steps) == 1
        assert len(drafts) == 1
        step = next(iter(steps))
        assert next(iter(drafts)) == step + 1
        al = float(settings["SYNTHETIC_ACCEPTANCE_LENGTH"])
        assert al == float(golden[step])
        actual[point["conc-list"][0]] = (step, al)

    assert actual == expected


def test_b300_dsxe_acceptance_driver_is_unconditional_and_fail_closed() -> None:
    content = (REPO_ROOT / "runners/launch_b300-dsxe.sh").read_text()
    command = 'python3 "$GITHUB_WORKSPACE/runners/inject_synthetic_acceptance.py"'
    command_index = content.index(command)

    assert content.rfind("\nfi", 0, command_index) > content.rfind(
        "\nif ", 0, command_index
    )
    assert "|| exit 1" in content[command_index : command_index + 180]


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
