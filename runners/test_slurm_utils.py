import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import BaseModel, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_UTILS = REPO_ROOT / "runners" / "slurm_utils.sh"
PATCH_SRT_EVAL = REPO_ROOT / "runners" / "patch_srt_eval_dispatch.py"
PATCH_SRT_DP_RANKS = REPO_ROOT / "runners" / "patch_srt_vllm_dp_ranks.py"
PATCH_TRTLLM_CHAT_STORE = REPO_ROOT / "runners" / "patch_trtllm_chat_store.py"
PATCH_VLLM_SIMPLE_KV = REPO_ROOT / "runners" / "patch_vllm_simple_kv_offload.py"
PATCH_AITER_QUIET = REPO_ROOT / "runners" / "patch_aiter_quiet_auto_backend.py"
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


@pytest.fixture
def aiter_backend_source() -> str:
    # Independent, reduced upstream source: never assembled from patch constants.
    return '''# Surrounding source must survive unchanged.
import itertools

import torch

_GLUON_SUPPORTED_ARCHS = ("gfx1250",)


def _is_gluon_available():
    return get_arch() in _GLUON_SUPPORTED_ARCHS


def moe_gemm_a16w4(backend=None):
    if backend in (None, "gluon"):
        if _is_gluon_available():
            backend = "gluon"
        else:
            _LOGGER.warning("GLUON backend not available. Using TRITON backend!!!")
            backend = "triton"

    backend = backend.lower()
    assert backend in ("triton", "gluon"), "Unknown backend"
    return backend


UNCHANGED = "surrounding source"
'''


def test_patch_aiter_quiet_is_idempotent_with_provenance(
    tmp_path: Path, aiter_backend_source: str,
) -> None:
    backend = tmp_path / "backend.py"
    backend.write_text(aiter_backend_source)
    before = hashlib.sha256(backend.read_bytes()).hexdigest()
    first = subprocess.run(
        [sys.executable, str(PATCH_AITER_QUIET), str(backend)],
        capture_output=True, text=True, check=False,
    )
    patched = backend.read_bytes()
    modified_at = backend.stat().st_mtime_ns
    after = hashlib.sha256(patched).hexdigest()
    second = subprocess.run(
        [sys.executable, str(PATCH_AITER_QUIET), str(backend)],
        capture_output=True, text=True, check=False,
    )

    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    assert before != after
    assert f"Patch applied: AITER quiet auto backend at {backend}" in first.stdout
    assert f"SHA256 before={before} after={after}" in first.stdout
    assert f"Already patched: AITER quiet auto backend at {backend}" in second.stdout
    assert f"SHA256 before={after} after={after}" in second.stdout
    assert backend.read_bytes() == patched
    assert backend.stat().st_mtime_ns == modified_at
    assert patched.startswith(b"# Surrounding source must survive unchanged.\n")
    assert patched.endswith(b'UNCHANGED = "surrounding source"\n')


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1100", "gfx1250"])
def test_patch_aiter_quiet_preserves_dispatch_and_warns_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aiter_backend_source: str, arch: str,
) -> None:
    backend = tmp_path / "backend.py"
    backend.write_text(aiter_backend_source)
    runpy.run_path(str(PATCH_AITER_QUIET))["patch_backend"](backend)
    warnings: list[str] = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    symbols = runpy.run_path(
        str(backend),
        init_globals={"get_arch": lambda: arch, "_LOGGER": SimpleNamespace(warning=warnings.append)},
    )
    dispatch = symbols["moe_gemm_a16w4"]
    expected_auto = "gluon" if arch == "gfx1250" else "triton"
    for _ in range(3):
        assert dispatch() == expected_auto
        assert dispatch("triton") == "triton"
    assert warnings == []
    for _ in range(3):
        assert dispatch("gluon") == expected_auto
        assert dispatch() == expected_auto
    assert len(warnings) == (0 if arch == "gfx1250" else 1)
    if warnings:
        assert "explicitly requested" in warnings[0]
    with pytest.raises(AssertionError, match="Unknown backend"):
        dispatch("invalid")
    # Exercise the real cached helper independently of dispatch.
    symbols["_warn_gluon_fallback_once"]()
    symbols["_warn_gluon_fallback_once"]()
    assert len(warnings) == 1


@pytest.mark.parametrize(
    "failure", ["partial-import", "partial-helper", "unknown", "duplicate", "syntax"],
)
def test_patch_aiter_quiet_preflights_before_writing(
    tmp_path: Path, aiter_backend_source: str, failure: str,
) -> None:
    if failure == "partial-import":
        source = aiter_backend_source.replace(
            "import itertools\n", "import itertools\nfrom functools import lru_cache\n",
        )
    elif failure == "partial-helper":
        source = aiter_backend_source + "\ndef _warn_gluon_fallback_once():\n    pass\n"
    elif failure == "unknown":
        source = aiter_backend_source.replace("backend.lower()", "str(backend).lower()")
    elif failure == "duplicate":
        source = aiter_backend_source + aiter_backend_source
    else:
        source = aiter_backend_source + "\ndef invalid(:\n"
    backend = tmp_path / "backend.py"
    backend.write_bytes(source.encode())
    modified_at = backend.stat().st_mtime_ns
    result = subprocess.run(
        [sys.executable, str(PATCH_AITER_QUIET), str(backend)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert "failed to patch AITER quiet auto backend" in result.stderr
    assert "Patch applied" not in result.stdout
    assert backend.read_bytes() == source.encode()
    assert backend.stat().st_mtime_ns == modified_at


@pytest.mark.parametrize("supported", [True, False])
def test_mi355x_recipe_patches_installed_aiter_before_download(
    tmp_path: Path, aiter_backend_source: str, supported: bool,
) -> None:
    package = tmp_path / "aiter"
    backend = package / "ops/triton/moe/moe_op_gemm_a16w4.py"
    backend.parent.mkdir(parents=True)
    # Discovery must not import AITER (and initialize GPU dependencies).
    (package / "__init__.py").write_text('raise AssertionError("AITER imported")\n')
    original = aiter_backend_source if supported else "# unknown backend\n"
    backend.write_text(original)
    recipe = REPO_ROOT / "benchmarks/single_node/agentic/dsv41flash_fp4_mi355x_vllm_mtp.sh"
    result = subprocess.run(
        ["bash", "-c", 'hf() { echo hf-called; exit 0; }; export -f hf; bash "$1"',
         "bash", str(recipe)],
        cwd=tmp_path,
        env={
            **os.environ, "PYTHONPATH": str(tmp_path), "MODEL": "test/model",
            "MODEL_PATH": "", "TP": "4", "CONC": "1", "KV_OFFLOADING": "none",
            "KV_OFFLOAD_BACKEND": "", "TOTAL_CPU_DRAM_GB": "100",
            "RESULT_DIR": str(tmp_path / "results"), "DURATION": "1",
            "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache"),
        },
        capture_output=True, text=True, check=False,
    )
    if supported:
        assert result.returncode == 0, result.stderr
        assert backend.read_text() != original
        assert result.stdout.index("Patch applied") < result.stdout.index("hf-called")
        assert str(backend) in result.stdout
    else:
        assert result.returncode == 1
        assert "hf-called" not in result.stdout
        assert backend.read_text() == original


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


@pytest.mark.parametrize(
    ("model", "prefix", "mount", "cache"),
    [
        ("deepseek-ai/DeepSeek-V4.1-Flash", "dsv41flash", "/ix", "/it-share/hf-hub-cache/"),
        ("deepseek-ai/DeepSeek-V4-Pro", "dsv4", "/workspace", "/it-share/hf-hub-cache/"),
    ],
)
def test_mi355x_agentic_model_mount_and_routing(
    tmp_path: Path, model: str, prefix: str, mount: str, cache: str,
) -> None:
    capture = tmp_path / "launch.txt"
    env = {
        **os.environ,
        "IS_MULTINODE": "false", "MODEL": model,
        "EXP_NAME": f"{prefix}_tp4_conc1", "FRAMEWORK": "vllm",
        "PRECISION": "fp4", "SPEC_DECODING": "mtp",
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
    script = f"benchmarks/single_node/agentic/{prefix}_fp4_mi355x_vllm_mtp.sh"
    assert args[-2] == script
    assert (REPO_ROOT / script).is_file()
