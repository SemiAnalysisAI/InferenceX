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
    assert "All result files processed\n" not in result.stdout
    if failure == "copy":
        assert result.stderr
        assert "Copied result file to:" not in result.stdout
    else:
        assert not result.stderr
        assert "No result subdirectories found" not in result.stdout


def test_result_copy_uses_source_location_after_chdir(tmp_path: Path) -> None:
    source = tmp_path / 'results' / 'isl8192_osl1024'
    source.mkdir(parents=True)
    (source / 'results_concurrency_4_gpus_8.json').write_text('{"completed": 40}')
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    result = run_bash('cd "$1"; source runners/slurm_utils.sh; cd "$2"; '
                      'copy_fixed_sequence_results "$3" "$4" run',
                      REPO_ROOT, tmp_path, source.parent, workspace)
    assert result.returncode == 0, result.stderr
    assert [json.loads(p.read_text()) for p in workspace.glob('*.json')] == [{'completed': 40}]


def test_filename_failure_in_conditional_is_not_reported_as_copied(tmp_path: Path) -> None:
    source = tmp_path / 'isl8192_osl1024'
    source.mkdir()
    (source / 'results_concurrency_4_gpus_8.json').write_text('{}')
    result = run_bash('source "$1"; python3() { return 1; }; '
                      'if copy_fixed_sequence_results "$2" "$2" run; then echo unexpected-success; fi',
                      SLURM_UTILS, tmp_path)
    assert 'unexpected-success' not in result.stdout
    assert 'Copied result file' not in result.stdout
    assert 'All result files processed' not in result.stdout


def test_h100_srt_mapping_uses_requested_image(tmp_path: Path) -> None:
    result = run_bash('source "$1"; resolve_h100_srt_container "$2" dynamo-sglang; '
                      'printf "%s\\n%s\\n" "$CONTAINER_KEY" "$SQUASH_FILE"',
                      SLURM_UTILS, 'example/engine:v-next')
    assert result.returncode == 0
    assert result.stdout.splitlines() == ['example/engine:v-next', '/mnt/nfs/lustre/containers/example_engine_v-next.sqsh']


def test_staged_asset_check_rejects_missing_model(tmp_path: Path) -> None:
    result = run_bash('source "$1"; unsquashfs() { return 0; }; check_staged_srt_assets "$2" image.sqsh',
                      SLURM_UTILS, tmp_path)
    assert result.returncode == 1
    (tmp_path / 'config.json').write_text('{}')
    result = run_bash('source "$1"; unsquashfs() { return 1; }; check_staged_srt_assets "$2" image.sqsh',
                      SLURM_UTILS, tmp_path)
    assert result.returncode == 1


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
        "schema: 2\nengine: sglang\nroles:\n"
        "  decode:\n"
        "    env:\n"
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
    environment = yaml.safe_load(recipe.read_text())["roles"]["decode"]["env"]
    assert environment == {"KEEP_ME": "unchanged"}


def test_sglang_throughput_rejects_existing_simulated_acceptance(
    tmp_path: Path,
) -> None:
    recipe = tmp_path / "recipe.yaml"
    original = (
        "schema: 2\nengine: sglang\nroles:\n"
        "  agg:\n"
        "    env:\n"
        '      SGLANG_SIMULATE_ACC_LEN: "2.99"\n'
        '      SGLANG_SIMULATE_ACC_METHOD: "match-expected"\n'
        '      SGLANG_SIMULATE_ACC_TOKEN_MODE: "real-draft-token"\n'
        "      KEEP_ME: unchanged\n"
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
    original = "schema: 2\nengine: vllm\nroles:\n  agg:\n    nodes: 1\n"
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


@pytest.mark.parametrize(
    ("model", "prefix", "mount", "cache", "framework", "spec"),
    [
        ("deepseek-ai/DeepSeek-V4.1-Flash", "dsv41flash", "/ix", "/it-share/hf-hub-cache/", "vllm", "mtp"),
        ("deepseek-ai/DeepSeek-V4-Pro", "dsv4", "/workspace", "/it-share/hf-hub-cache/", "vllm", "mtp"),
        ("amd/Qwen3.8-Flash-Next-Quark-MXFP4", "qwen3.8next", "/ix", "/var/lib/hf-hub-cache/", "sglang", "mtp"),
    ],
)
def test_mi355x_agentic_model_mount_and_routing(
    tmp_path: Path, model: str, prefix: str, mount: str, cache: str,
    framework: str, spec: str,
) -> None:
    capture = tmp_path / "launch.txt"
    env = {
        **os.environ,
        "IS_MULTINODE": "false", "MODEL": model,
        "EXP_NAME": f"{prefix}_tp4_conc1", "FRAMEWORK": framework,
        "PRECISION": "fp4", "SPEC_DECODING": spec,
        "SCENARIO_SUBDIR": "agentic/", "TP": "4", "GPU_COUNT": "4",
        "RUNNER_NAME": "mi355x-amds_01", "IMAGE": "test/image:mock",
        "GITHUB_WORKSPACE": str(REPO_ROOT), "HF_HUB_CACHE": "/mnt/hf_hub_cache/",
        "RESULT_DIR": "/workspace/results", "CAPTURE": str(capture),
    }
    result = subprocess.run(
        ["bash", "-c", '''
        salloc() { :; }
        squeue() { echo 123; }
        scancel() { :; }
        srun() {
            case "$*" in
                *--container-image=*)
                    printf '%s\n' "$@" > "$CAPTURE"
                    printf 'RESULT_DIR=%s\n' "$RESULT_DIR" >> "$CAPTURE"
                    bash -c 'source benchmarks/benchmark_lib.sh;
                        printf "%s\n" "$INFMAX_CONTAINER_WORKSPACE" "$AGENTIC_DIR" "$AIPERF_DIR"' > "$CAPTURE.paths"
                    ;;
            esac
        }
        source runners/launch_mi355x-amds.sh
        '''], cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    args = capture.read_text().splitlines()
    assert f"--container-workdir={mount}/" in args
    assert f"RESULT_DIR={mount}/results" in args
    assert Path(f"{capture}.paths").read_text().splitlines() == [
        mount, f"{mount}/utils/agentic-benchmark", f"{mount}/utils/aiperf",
    ]
    assert (
        f"--container-mounts={REPO_ROOT}:{mount}/,{cache}:/mnt/hf_hub_cache/,"
        "/it-share/aiperf-cache/:/aiperf_mmap_cache"
    ) in args
    suffix = "_mtp" if spec == "mtp" else ""
    script = f"benchmarks/single_node/agentic/{prefix}_fp4_mi355x_{framework}{suffix}.sh"
    assert args[-2] == script


@pytest.mark.parametrize(
    ("model", "framework", "image", "cache_valid", "expected_uri"),
    [
        ("amd/Qwen3.8-Flash-Next-Quark-MXFP4", "sglang",
         "lmsysorg/sglang-rocm:fixture@sha256:abc123", False,
         "docker://registry-1.docker.io#lmsysorg/sglang-rocm:sha256:abc123"),
        ("amd/Qwen3.8-Flash-Next-Quark-MXFP4", "sglang",
         "lmsysorg/sglang-rocm:fixture@sha256:abc123", True, None),
        ("unrelated/model", "sglang",
         "lmsysorg/sglang-rocm:fixture@sha256:abc123", False,
         "docker://lmsysorg/sglang-rocm:fixture@sha256:abc123"),
        ("amd/Qwen3.8-Flash-Next-Quark-MXFP4", "vllm",
         "lmsysorg/sglang-rocm:fixture@sha256:abc123", False,
         "docker://lmsysorg/sglang-rocm:fixture@sha256:abc123"),
        ("amd/Qwen3.8-Flash-Next-Quark-MXFP4", "sglang",
         "lmsysorg/sglang-rocm:fixture", False,
         "docker://lmsysorg/sglang-rocm:fixture"),
        ("amd/Qwen3.8-Flash-Next-Quark-MXFP4", "sglang",
         "other/image:fixture@sha256:abc123", False,
         "docker://other/image:fixture@sha256:abc123"),
    ],
)
def test_mi355x_import_executes_scoped_digest_uri(
    tmp_path: Path, model: str, framework: str, image: str,
    cache_valid: bool, expected_uri: str | None,
) -> None:
    capture = tmp_path / "enroot.txt"
    env = {
        **os.environ,
        "IS_MULTINODE": "false", "MODEL": model, "FRAMEWORK": framework,
        "EXP_NAME": "qwen3.8next_tp8_conc1", "PRECISION": "fp4",
        "SPEC_DECODING": "mtp", "SCENARIO_SUBDIR": "agentic/", "TP": "8",
        "RUNNER_NAME": "mi355x-amds_01", "IMAGE": image,
        "GITHUB_WORKSPACE": str(REPO_ROOT), "HF_HUB_CACHE": "/mnt/hf_hub_cache/",
        "RESULT_DIR": "/workspace/results", "CAPTURE": str(capture),
        "CACHE_ROOT": str(tmp_path), "CACHE_STATUS": "0" if cache_valid else "1",
    }
    result = subprocess.run(
        ["bash", "-c", r'''
        salloc() { :; }
        squeue() { echo 123; }
        scancel() { :; }
        docker() { :; }
        unsquashfs() { return "$CACHE_STATUS"; }
        enroot() { printf '%s\n' "$@" > "$CAPTURE"; }
        export -f docker unsquashfs enroot
        srun() {
            shift
            if [[ "$1" == bash ]]; then
                # Simulate the remote filesystem in scratch, then execute the
                # launcher's actual import shell, including cache selection.
                local remote_command="$3"
                remote_command="${remote_command//\/var\/lib\/squash/$CACHE_ROOT}"
                bash -c "$remote_command"
            else
                printf '%s\n' "$@" > "$CAPTURE.launch"
            fi
        }
        source runners/launch_mi355x-amds.sh
        '''], cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    if expected_uri is None:
        assert not capture.exists()
        assert "skipping import" in result.stdout
    else:
        args = capture.read_text().splitlines()
        assert args[:2] == ["import", "-o"]
        assert args[3:] == [expected_uri]
        if image == "lmsysorg/sglang-rocm:fixture@sha256:abc123":
            assert Path(args[2]).name == "lmsysorg_sglang-rocm_fixture_sha256_abc123.sqsh"
    assert any(
        arg.startswith("--container-image=/var/lib/squash/")
        for arg in Path(f"{capture}.launch").read_text().splitlines()
    )
