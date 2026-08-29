from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_LIB = REPO_ROOT / "benchmarks" / "benchmark_lib.sh"
MULTINODE_AGENTIC_SCRIPT = REPO_ROOT / "benchmarks/multi_node/agentic_srt.sh"
SINGLE_NODE_WORKFLOW = REPO_ROOT / ".github/workflows/benchmark-tmpl.yml"
MULTINODE_WORKFLOW = REPO_ROOT / ".github/workflows/benchmark-multinode-tmpl.yml"
E2E_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "e2e-tests.yml"
QWEN_SGLANG_MTP_LAUNCHERS = (
    REPO_ROOT / "benchmarks" / "single_node" / "agentic" / "qwen3.5_fp8_h100_mtp.sh",
    REPO_ROOT / "benchmarks" / "single_node" / "agentic" / "qwen3.5_fp8_h200_mtp.sh",
)

_SCRIPT = r"""
source "$BENCHMARK_LIB"
run_lm_eval()       { echo "DISPATCH=lm-eval"; }
run_swebench_eval() { echo "DISPATCH=swebench"; }
run_kimi_vendor_eval() { echo "DISPATCH=kimi-vendor"; }
run_minimax_vendor_eval() { echo "DISPATCH=minimax-vendor"; }
run_bfcl_eval() { echo "DISPATCH=bfcl"; }
append_lm_eval_summary() { echo "STAGED=summary"; }
export EVAL_MAX_MODEL_LEN=16384
export EVAL_CONCURRENT_REQUESTS=""
run_eval ${CLI_FW:+--framework "$CLI_FW"} --port 8888
"""


def _dispatch(
    *,
    is_agentic: str = "0",
    eval_only: str = "false",
    cli_fw=None,
    env_fw=None,
) -> str:
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "IS_AGENTIC": is_agentic,
        "EVAL_ONLY": eval_only,
        "KV_OFFLOADING": "none",
    }
    env.pop("EVAL_FRAMEWORK", None)
    env.pop("CLI_FW", None)
    env.pop("KV_OFFLOAD_BACKEND", None)
    env.pop("EVAL_SUITE", None)
    if cli_fw is not None:
        env["CLI_FW"] = cli_fw
    if env_fw is not None:
        env["EVAL_FRAMEWORK"] = env_fw
    res = subprocess.run(
        ["bash", "-c", _SCRIPT], env=env, text=True, capture_output=True, check=True
    )
    return res.stdout


def test_agentic_scenario_defaults_to_gsm8k_lm_eval():
    assert "DISPATCH=lm-eval" in _dispatch(is_agentic="1")


def test_fixed_seqlen_scenario_defaults_to_lm_eval():
    assert "DISPATCH=lm-eval" in _dispatch(is_agentic="0")


def test_agentic_eval_only_stages_summary():
    output = _dispatch(is_agentic="1", eval_only="true")
    assert "DISPATCH=lm-eval" in output
    assert "STAGED=summary" in output


def test_fixed_seqlen_eval_only_leaves_staging_to_recipe():
    assert "STAGED=summary" not in _dispatch(is_agentic="0", eval_only="true")


def test_fixed_seqlen_provider_leaves_staging_to_recipe() -> None:
    output = _dispatch(
        is_agentic="0",
        eval_only="true",
        env_fw="minimax-vendor",
    )
    assert "DISPATCH=minimax-vendor" in output
    assert "STAGED=summary" not in output


def test_explicit_framework_arg_overrides_scenario():
    assert "DISPATCH=lm-eval" in _dispatch(is_agentic="1", cli_fw="lm-eval")


def test_env_framework_overrides_scenario():
    assert "DISPATCH=lm-eval" in _dispatch(is_agentic="1", env_fw="lm-eval")


def test_environment_framework_overrides_legacy_recipe_argument() -> None:
    assert "DISPATCH=kimi-vendor" in _dispatch(
        is_agentic="1",
        cli_fw="bfcl",
        env_fw="kimi-vendor",
    )


def test_env_can_force_swebench_on_fixed_seqlen():
    assert "DISPATCH=swebench" in _dispatch(is_agentic="0", env_fw="swebench")


def test_env_can_force_kimi_vendor_on_agentic_eval() -> None:
    assert "DISPATCH=kimi-vendor" in _dispatch(
        is_agentic="1",
        eval_only="true",
        env_fw="kimi-vendor",
    )


def test_kimi_vendor_skips_unused_model_context_loading() -> None:
    script = r"""
source "$BENCHMARK_LIB"
unset EVAL_MAX_MODEL_LEN
compute_eval_context_length() { echo "UNEXPECTED_CONTEXT_LOAD"; return 99; }
run_kimi_vendor_eval() { echo "DISPATCH=kimi-vendor"; }
export EVAL_FRAMEWORK=kimi-vendor
export EVAL_CONCURRENT_REQUESTS=""
export EVAL_ONLY=false
export IS_AGENTIC=0
run_eval --port 8888
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "BENCHMARK_LIB": str(BENCHMARK_LIB)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "DISPATCH=kimi-vendor" in result.stdout
    assert "UNEXPECTED_CONTEXT_LOAD" not in result.stdout


def test_kimi_failure_preserves_rc_without_eval_only() -> None:
    script = r"""
set -u
source "$BENCHMARK_LIB"
run_kimi_vendor_eval() { return 7; }
export EVAL_FRAMEWORK=kimi-vendor
export EVAL_CONCURRENT_REQUESTS=""
export EVAL_MAX_MODEL_LEN=16384
export IS_AGENTIC=0
unset EVAL_ONLY
run_eval --port 8888
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "BENCHMARK_LIB": str(BENCHMARK_LIB)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 7
    assert "unbound variable" not in result.stderr


def test_recipe_lm_eval_arg_still_lm_eval_on_fixed_seqlen():
    assert "DISPATCH=lm-eval" in _dispatch(is_agentic="0", cli_fw="lm-eval")


def _run_invalid_call(call: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "KV_OFFLOADING": "none",
    }
    return subprocess.run(
        ["bash", "-c", f'source "$BENCHMARK_LIB"; {call}'],
        env=env,
        text=True,
        capture_output=True,
    )


def test_run_eval_rejects_missing_framework_value():
    result = _run_invalid_call("run_eval --framework")
    assert result.returncode == 2
    assert "--framework requires a value" in result.stderr


def test_run_eval_rejects_unsafe_suite_name() -> None:
    result = _run_invalid_call(
        "EVAL_SUITE='kimi\"suite' run_eval --framework kimi-vendor"
    )

    assert result.returncode == 2
    assert "EVAL_SUITE may contain only" in result.stderr


def test_run_eval_rejects_suite_override_for_lm_eval() -> None:
    result = _run_invalid_call("EVAL_SUITE=gpqa_diamond run_eval --framework lm-eval")

    assert result.returncode == 2
    assert "only supported with kimi-vendor, minimax-vendor, or bfcl" in result.stderr


def test_run_eval_scopes_runner_selected_suite_to_one_call() -> None:
    script = r"""
source "$BENCHMARK_LIB"
run_kimi_vendor_eval() {
    export EVAL_SUITE=kimi_tool_call_schema
    echo "DISPATCH=kimi-vendor SUITE=$EVAL_SUITE"
}
run_lm_eval() {
    echo "DISPATCH=lm-eval SUITE=${EVAL_SUITE:-unset} COMPLETED=${EVAL_COMPLETED_SUITE:-unset}"
}
append_lm_eval_summary() {
    echo "METADATA=${EVAL_COMPLETED_SUITE:-gsm8k}"
}
export EVAL_MAX_MODEL_LEN=16384
export EVAL_CONCURRENT_REQUESTS=""
export EVAL_ONLY=false
export IS_AGENTIC=0
unset EVAL_SUITE
export EVAL_FRAMEWORK=kimi-vendor
run_eval --port 8888
printf 'KIMI_COMPLETED=%s\n' "${EVAL_COMPLETED_SUITE:-unset}"
append_lm_eval_summary
export EVAL_FRAMEWORK=lm-eval
run_eval --port 8888
printf 'LM_COMPLETED=%s\n' "${EVAL_COMPLETED_SUITE:-unset}"
append_lm_eval_summary
printf 'FINAL_SUITE=%s\n' "${EVAL_SUITE-unset}"
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "BENCHMARK_LIB": str(BENCHMARK_LIB)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "DISPATCH=kimi-vendor SUITE=kimi_tool_call_schema" in result.stdout
    assert "KIMI_COMPLETED=kimi_tool_call_schema" in result.stdout
    assert "METADATA=kimi_tool_call_schema" in result.stdout
    assert "DISPATCH=lm-eval SUITE=unset COMPLETED=unset" in result.stdout
    assert "LM_COMPLETED=unset" in result.stdout
    assert "METADATA=gsm8k" in result.stdout
    assert "FINAL_SUITE=unset" in result.stdout


def test_kimi_default_suite_reaches_eval_only_metadata() -> None:
    script = r"""
source "$BENCHMARK_LIB"
run_kimi_vendor_eval() { echo "DISPATCH=$EVAL_SUITE"; }
append_lm_eval_summary() { echo "METADATA=$EVAL_COMPLETED_SUITE"; }
export EVAL_FRAMEWORK=kimi-vendor
export EVAL_ONLY=true
export IS_AGENTIC=1
export EVAL_CONCURRENT_REQUESTS=""
unset EVAL_SUITE
run_eval --port 8888
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "BENCHMARK_LIB": str(BENCHMARK_LIB)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "DISPATCH=kimi_tool_call_schema" in result.stdout
    assert "METADATA=kimi_tool_call_schema" in result.stdout


def test_agentic_eval_propagates_artifact_staging_failure() -> None:
    script = r"""
source "$BENCHMARK_LIB"
run_kimi_vendor_eval() { :; }
append_lm_eval_summary() { return 73; }
export EVAL_FRAMEWORK=kimi-vendor
export EVAL_ONLY=true
export IS_AGENTIC=1
export EVAL_CONCURRENT_REQUESTS=""
unset EVAL_SUITE
run_eval --port 8888
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "BENCHMARK_LIB": str(BENCHMARK_LIB)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 73
    assert "eval artifact staging failed with exit code 73" in result.stderr


def test_kimi_full_suite_dispatches_to_schema_runner() -> None:
    script = r"""
source "$BENCHMARK_LIB"
_run_kimi_tool_call_schema_eval() {
    printf 'DISPATCH=%s ARGS=<%s>\n' "$EVAL_SUITE" "$*"
}
EVAL_SUITE=kimi_tool_call_schema_full run_kimi_vendor_eval --port 9999
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "BENCHMARK_LIB": str(BENCHMARK_LIB)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "DISPATCH=kimi_tool_call_schema_full ARGS=<--port 9999>" in result.stdout


def test_minimax_full_suite_dispatches_to_full_runner() -> None:
    script = r"""
source "$BENCHMARK_LIB"
_run_minimax_m3_full_eval() {
    printf 'DISPATCH=%s ARGS=<%s>\n' "$EVAL_SUITE" "$*"
}
EVAL_SUITE=minimax_m3_full run_minimax_vendor_eval --port 9999
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "BENCHMARK_LIB": str(BENCHMARK_LIB)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "DISPATCH=minimax_m3_full ARGS=<--port 9999>" in result.stdout


def test_kimi_vendor_rejects_batched_concurrency() -> None:
    result = _run_invalid_call(
        "EVAL_MAX_MODEL_LEN=16384 "
        "EVAL_CONCURRENT_REQUESTS='1 4' "
        "run_eval --framework kimi-vendor"
    )
    assert result.returncode == 1
    assert "batched eval concurrency is only supported for lm-eval" in result.stderr


def test_kimi_vendor_rejects_unsupported_suite() -> None:
    result = _run_invalid_call("EVAL_SUITE=gsm8k run_kimi_vendor_eval")
    assert result.returncode == 2
    assert "unsupported Kimi Vendor Verifier suite 'gsm8k'" in result.stderr


def _run_minimax_dispatch(*, suite: str | None = None, concurrency: str = "") -> str:
    script = r"""
source "$BENCHMARK_LIB"
unset EVAL_MAX_MODEL_LEN
compute_eval_context_length() { echo "UNEXPECTED_CONTEXT_LOAD"; return 99; }
MINIMAX_DISPATCH_COUNT=0
run_minimax_vendor_eval() {
    MINIMAX_DISPATCH_COUNT=$((MINIMAX_DISPATCH_COUNT + 1))
    printf 'DISPATCH=minimax-vendor SUITE=%s ARGS=<%s>\n' "$EVAL_SUITE" "$*"
}
export EVAL_CONCURRENT_REQUESTS="$TEST_EVAL_CONCURRENCY"
export EVAL_ONLY=false
export IS_AGENTIC=0
run_eval --framework minimax-vendor --port 9999
printf 'DISPATCH_COUNT=%s\n' "$MINIMAX_DISPATCH_COUNT"
printf 'COMPLETED_SUITE=%s\n' "$EVAL_COMPLETED_SUITE"
"""
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "MODEL": "served-model",
        "MODEL_PREFIX": "minimaxm3",
        "TEST_EVAL_CONCURRENCY": concurrency,
    }
    for key in ("EVAL_FRAMEWORK", "EVAL_SUITE", "EVAL_COMPLETED_SUITE"):
        env.pop(key, None)
    if suite is not None:
        env["EVAL_SUITE"] = suite
    result = subprocess.run(
        ["bash", "-c", script],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "UNEXPECTED_CONTEXT_LOAD" not in result.stdout
    return result.stdout


def test_minimax_vendor_defaults_suite_dispatches_once_and_records_completion() -> None:
    output = _run_minimax_dispatch()

    assert "DISPATCH=minimax-vendor SUITE=minimax_m3_smoke" in output
    assert "ARGS=<--port 9999>" in output
    assert "DISPATCH_COUNT=1" in output
    assert "COMPLETED_SUITE=minimax_m3_smoke" in output


def test_minimax_vendor_accepts_explicit_supported_suite() -> None:
    output = _run_minimax_dispatch(suite="minimax_m3_smoke")

    assert "DISPATCH=minimax-vendor SUITE=minimax_m3_smoke" in output
    assert "DISPATCH_COUNT=1" in output
    assert "COMPLETED_SUITE=minimax_m3_smoke" in output


def test_minimax_vendor_rejects_unsupported_suite() -> None:
    result = _run_invalid_call(
        "MODEL_PREFIX=minimaxm3 EVAL_SUITE=gsm8k run_eval --framework minimax-vendor"
    )

    assert result.returncode == 2
    assert "unsupported MiniMax Provider Verifier suite 'gsm8k'" in result.stderr


def test_run_eval_rejects_unknown_framework() -> None:
    result = _run_invalid_call(
        "EVAL_MAX_MODEL_LEN=16384 run_eval --framework not-a-framework"
    )

    assert result.returncode == 1
    assert "Unknown framework 'not-a-framework'" in result.stdout


def test_minimax_vendor_rejects_concurrency_sweep_for_sequential_smoke() -> None:
    result = _run_invalid_call(
        "MODEL_PREFIX=minimaxm3 "
        "EVAL_CONCURRENT_REQUESTS='1 4' "
        "run_eval --framework minimax-vendor"
    )

    assert result.returncode == 1
    assert "batched eval concurrency is only supported for lm-eval" in result.stderr


def test_minimax_vendor_ignores_single_launcher_concurrency_value() -> None:
    output = _run_minimax_dispatch(concurrency="128")

    assert "DISPATCH=minimax-vendor SUITE=minimax_m3_smoke" in output
    assert "DISPATCH_COUNT=1" in output


def test_minimax_vendor_accepts_non_m3_model() -> None:
    script = r"""
source "$BENCHMARK_LIB"
_run_minimax_m3_smoke_eval() { echo "DISPATCH=$EVAL_SUITE"; }
unset EVAL_SUITE EVAL_RESULT_DIR
MODEL=moonshotai/Kimi-K3 MODEL_PREFIX=kimik3 run_minimax_vendor_eval
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "BENCHMARK_LIB": str(BENCHMARK_LIB)},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "DISPATCH=minimax_m3_smoke" in result.stdout


def test_minimax_vendor_accepts_case_insensitive_m3_model_name() -> None:
    script = r"""
source "$BENCHMARK_LIB"
_run_minimax_m3_smoke_eval() { echo "DISPATCH=$EVAL_SUITE"; }
unset MODEL_PREFIX EVAL_SUITE EVAL_RESULT_DIR
MODEL_NAME=vendor/MINIMAX-M3-custom run_minimax_vendor_eval
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "BENCHMARK_LIB": str(BENCHMARK_LIB)},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "DISPATCH=minimax_m3_smoke" in result.stdout


def test_minimax_vendor_setup_failure_uses_integration_error_and_stages(
    tmp_path: Path,
) -> None:
    script = r"""
source "$BENCHMARK_LIB"
unset EVAL_SUITE EVAL_RESULT_DIR EVAL_COMPLETED_SUITE
unset VENDOR_VERIFIER_PYTHON VENDOR_VERIFIER_PYTHON_CLEANUP_DIR
_prepare_vendor_verifier_python() {
    if compgen -G "$RESULTS_DIR/results_minimax_vendor_*.json" >/dev/null \
        || [ -e "$RESULTS_DIR/minimax_vendor_report.json" ]; then
        echo "STALE_MINIMAX_ARTIFACT"
        return 99
    fi
    mkdir -p "$PYTHON_DIR"
    cat >"$PYTHON_DIR/python3" <<'PY'
#!/bin/bash
printf 'ADAPTER_ARG=<%s>\n' "$@"
PY
    chmod +x "$PYTHON_DIR/python3"
    export VENDOR_VERIFIER_PYTHON="$PYTHON_DIR/python3"
    export VENDOR_VERIFIER_PYTHON_CLEANUP_DIR="$PYTHON_DIR"
}
_prepare_minimax_m3_full_runtime() { return 12; }
append_lm_eval_summary() {
    printf 'STAGED=<%s>\n' "$EVAL_RESULT_DIR"
    printf 'STAGED_CONC=<%s>\n' "$CONC"
}
export MODEL_PREFIX=minimaxm3
export MODEL=test-model
export EVAL_CONCURRENT_REQUESTS=7
export EVAL_ONLY=false
export IS_AGENTIC=0
run_eval --framework minimax-vendor --results-dir "$RESULTS_DIR"
eval_rc=$?
printf 'EVAL_RC=%s\n' "$eval_rc"
"""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "results_minimax_vendor_stale.json").write_text("{}")
    (results_dir / "minimax_vendor_report.json").write_text("{}")
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "BENCHMARK_LIB": str(BENCHMARK_LIB),
            "RESULTS_DIR": str(results_dir),
            "PYTHON_DIR": str(tmp_path / "python"),
        },
        text=True,
        capture_output=True,
        check=True,
    )
    output = result.stdout + result.stderr

    assert "EVAL_RC=12" in output
    assert "STALE_MINIMAX_ARTIFACT" not in output
    assert not (tmp_path / "python").exists()
    assert (
        f"ADAPTER_ARG=<{REPO_ROOT / 'utils/evals/minimax_provider_eval.py'}>" in output
    )
    assert "ADAPTER_ARG=<test-model>" in output
    assert f"ADAPTER_ARG=<{results_dir}>" in output
    assert "ADAPTER_ARG=<failure>" in output
    assert "ADAPTER_ARG=<--message>" in output
    assert (
        "ADAPTER_ARG=<MiniMax Provider Verifier pinned runtime preparation "
        "failed with exit code 12>"
    ) in output
    assert f"STAGED=<{results_dir}>" in output
    assert output.count("STAGED=<") == 1
    assert "STAGED_CONC=<7>" in output


def test_minimax_full_dependency_install_matches_pinned_upstream_requirements(
    tmp_path: Path,
) -> None:
    script = r"""
source "$BENCHMARK_LIB"
selected_python() { printf 'PYTHON_ARG=<%s>\n' "$@"; }
VENDOR_VERIFIER_PYTHON=selected_python
_install_minimax_m3_full_deps "$RUNTIME_DIR"
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "BENCHMARK_LIB": str(BENCHMARK_LIB),
            "RUNTIME_DIR": str(tmp_path / "runtime"),
        },
        text=True,
        capture_output=True,
        check=True,
    )

    for requirement in (
        "jsonschema==4.25.1",
        "loguru==0.7.3",
        "megfile==4.2.5",
        "numpy==2.3.4",
        "openai==2.7.1",
        "tqdm==4.67.1",
    ):
        assert f"PYTHON_ARG=<{requirement}>" in result.stdout
    assert "--break-system-packages" not in result.stdout


def test_minimax_runtime_prepares_source_with_full_adapter(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    calls_path = tmp_path / "calls"
    script = r"""
source "$BENCHMARK_LIB"
mktemp() {
    mkdir -p "$RUNTIME_DIR"
    printf '%s\n' "$RUNTIME_DIR"
}
selected_python() {
    printf 'PYTHON_ARG=<%s>\n' "$@" >> "$CALLS_PATH"
    mkdir -p "$RUNTIME_DIR/source"
}
_install_minimax_m3_full_deps() { mkdir -p "$1"; }
VENDOR_VERIFIER_PYTHON=selected_python
prepared_runtime=$(_prepare_minimax_m3_full_runtime)
printf 'RUNTIME=<%s>\n' "$prepared_runtime"
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "BENCHMARK_LIB": str(BENCHMARK_LIB),
            "RUNTIME_DIR": str(runtime_dir),
            "CALLS_PATH": str(calls_path),
        },
        text=True,
        capture_output=True,
        check=True,
    )
    calls = calls_path.read_text()

    assert f"PYTHON_ARG=<{REPO_ROOT / 'utils/evals/minimax_m3_full_eval.py'}>" in calls
    assert "PYTHON_ARG=<prepare-source>" in calls
    assert f"PYTHON_ARG=<{runtime_dir / 'source'}>" in calls
    assert "minimax_provider_eval.py" not in calls
    assert f"RUNTIME=<{runtime_dir}>" in result.stdout


def test_minimax_vendor_runner_uses_fixed_adapter_contract(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    runtime_dir = tmp_path / "runtime"
    python_dir = tmp_path / "python"
    script = r"""
source "$BENCHMARK_LIB"
selected_python() {
    printf 'PYTHONPATH=<%s>\n' "$PYTHONPATH" >&2
    printf 'PYTHON_ARG=<%s>\n' "$@" >&2
}
_prepare_vendor_verifier_python() {
    mkdir "$PYTHON_DIR"
    VENDOR_VERIFIER_PYTHON=selected_python
    VENDOR_VERIFIER_PYTHON_CLEANUP_DIR="$PYTHON_DIR"
    export VENDOR_VERIFIER_PYTHON VENDOR_VERIFIER_PYTHON_CLEANUP_DIR
}
_prepare_minimax_m3_full_runtime() {
    mkdir -p "$RUNTIME_DIR/source" "$RUNTIME_DIR/deps"
    printf '%s\n' "$RUNTIME_DIR"
}
mktemp() { echo "UNEXPECTED_DEFAULT_RESULTS_DIR" >&2; return 99; }
run_minimax_vendor_eval --port 9999 --results-dir "$RESULTS_DIR"
printf 'EVAL_SUITE=%s\n' "$EVAL_SUITE"
printf 'EVAL_RESULT_DIR=%s\n' "$EVAL_RESULT_DIR"
"""
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "RESULTS_DIR": str(results_dir),
        "RUNTIME_DIR": str(runtime_dir),
        "PYTHON_DIR": str(python_dir),
        "MODEL": "test-model",
        "MODEL_PREFIX": "minimaxm3",
        "OPENAI_API_KEY": "must-not-be-forwarded",
    }
    for key in ("EVAL_SUITE", "EVAL_RESULT_DIR", "MODEL_NAME"):
        env.pop(key, None)
    result = subprocess.run(
        ["bash", "-c", script],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    output = result.stdout + result.stderr
    adapter = REPO_ROOT / "utils/evals/minimax_provider_eval.py"
    fixture = REPO_ROOT / "utils/evals/minimax_m3_smoke.json"

    for value in (
        adapter,
        "run",
        "selected_python",
        runtime_dir / "source",
        runtime_dir / "deps",
        "http://127.0.0.1:9999/v1",
        "test-model",
        results_dir,
        fixture,
    ):
        assert f"PYTHON_ARG=<{value}>" in output
    for option in (
        "--python",
        "--source-dir",
        "--dependency-dir",
        "--base-url",
        "--model",
        "--output-dir",
        "--fixture",
    ):
        assert f"PYTHON_ARG=<{option}>" in output
    assert "must-not-be-forwarded" not in output
    assert "UNEXPECTED_DEFAULT_RESULTS_DIR" not in output
    assert "EVAL_SUITE=minimax_m3_smoke" in output
    assert f"EVAL_RESULT_DIR={results_dir}" in output
    assert not runtime_dir.exists()
    assert not python_dir.exists()


def test_kimi_vendor_setup_failure_writes_compatibility_result(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "results_kimi_vendor_stale.json").write_text("{}")
    (results_dir / "kimi_vendor_report.json").write_text("{}")
    python_dir = tmp_path / "python"
    script = r"""
source "$BENCHMARK_LIB"
_prepare_vendor_verifier_python() {
    if compgen -G "$RESULTS_DIR/results_kimi_vendor_*.json" >/dev/null \
        || [ -e "$RESULTS_DIR/kimi_vendor_report.json" ]; then
        echo "STALE_KIMI_ARTIFACT"
        return 99
    fi
    mkdir "$PYTHON_DIR"
    cat >"$PYTHON_DIR/python3" <<'PY'
#!/bin/bash
exec /usr/bin/env python3 "$@"
PY
    chmod +x "$PYTHON_DIR/python3"
    VENDOR_VERIFIER_PYTHON="$PYTHON_DIR/python3"
    VENDOR_VERIFIER_PYTHON_CLEANUP_DIR="$PYTHON_DIR"
    export VENDOR_VERIFIER_PYTHON VENDOR_VERIFIER_PYTHON_CLEANUP_DIR
}
_prepare_kimi_vendor_runtime() { return 12; }
run_kimi_vendor_eval --results-dir "$RESULTS_DIR"
printf 'SETUP_RC=%s\n' "$?"
"""
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "RESULTS_DIR": str(results_dir),
        "PYTHON_DIR": str(python_dir),
        "MODEL": "test-model",
        "IS_MULTINODE": "false",
        "KV_OFFLOADING": "none",
    }
    for key in (
        "EVAL_SUITE",
        "EVAL_RESULT_DIR",
        "MODEL_NAME",
    ):
        env.pop(key, None)

    result = subprocess.run(
        ["bash", "-c", script],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    message = "Kimi Vendor Verifier dependency installation failed with exit code 12"
    score_files = list(results_dir.glob("results*.json"))

    assert "SETUP_RC=12" in result.stdout
    assert "STALE_KIMI_ARTIFACT" not in result.stdout + result.stderr
    assert message in result.stderr
    assert "failed to write Kimi verifier failure artifact" not in result.stderr
    assert len(score_files) == 1
    score_result = json.loads(score_files[0].read_text())
    assert (
        score_result["results"]["kimi_tool_call_schema"]["exact_match,strict-match"]
        == 0.0
    )
    assert score_result["integration_error"]["message"] == message
    native_result = json.loads((results_dir / "kimi_vendor_report.json").read_text())
    assert native_result["completed"] is False
    assert native_result["integration_error"]["message"] == message
    assert not python_dir.exists()


def test_preclear_failure_cannot_stage_stale_provider_result(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    stale_result = results_dir / "results_kimi_vendor_stale.json"
    stale_result.write_text('{"stale": true}\n')
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    script = r"""
source "$BENCHMARK_LIB"
rm() { return 73; }
append_lm_eval_summary() {
    if [ -n "${EVAL_RESULT_DIR:-}" ]; then
        echo "UNEXPECTED_STAGING"
    fi
    return 1
}
unset EVAL_FRAMEWORK EVAL_SUITE EVAL_RESULT_DIR EVAL_COMPLETED_SUITE
export MODEL=test-model
cd "$WORK_DIR"
run_eval --framework kimi-vendor --results-dir "$RESULTS_DIR"
printf 'EVAL_RC=%s\n' "$?"
printf 'EVAL_RESULT_DIR=<%s>\n' "${EVAL_RESULT_DIR:-}"
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "BENCHMARK_LIB": str(BENCHMARK_LIB),
            "RESULTS_DIR": str(results_dir),
            "WORK_DIR": str(work_dir),
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert "EVAL_RC=73" in result.stdout
    assert "EVAL_RESULT_DIR=<>" in result.stdout
    assert "UNEXPECTED_STAGING" not in result.stdout
    assert "failed to remove stale eval artifact" in result.stderr
    assert stale_result.is_file()
    assert list(work_dir.iterdir()) == []


_KIMI_VERIFIER_REQUIRED_FILES = {
    "pyproject.toml",
    "tests/conftest.py",
    "tests/__init__.py",
    "tests/tool_call_json_schema/conftest.py",
    "tests/tool_call_json_schema/__init__.py",
    "tests/tool_call_json_schema/test_tool_call_json_schema.py",
    "tests/tool_call_json_schema/validator.py",
    *{
        f"testdata/walle_validator_cases/validator_cases/{case}/valid.jsonl"
        for case in (
            "TestAdditionalProperties",
            "TestAnyOf",
            "TestBasicTypes",
            "TestDefs",
            "TestDescription",
            "TestEnforcerCases",
            "TestID",
            "TestKeywordsValidation",
            "TestNestedDefsDepth",
            "TestNumberFormat",
            "TestRangeConstraints",
            "TestRefInProperties",
            "TestReferences",
            "TestRequired",
            "TestSingleTypeInArray",
            "TestTypeLocation",
        )
    },
}


def _kimi_verifier_archive(
    *,
    missing: str | None = None,
    unsafe_member: tarfile.TarInfo | None = None,
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for relative_path in sorted(_KIMI_VERIFIER_REQUIRED_FILES - {missing}):
            payload = relative_path.encode()
            member = tarfile.TarInfo(f"verifier-pinned/{relative_path}")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        extra = b"must not be extracted"
        member = tarfile.TarInfo("verifier-pinned/README.md")
        member.size = len(extra)
        archive.addfile(member, io.BytesIO(extra))
        if unsafe_member is not None:
            archive.addfile(
                unsafe_member,
                io.BytesIO(b"unsafe") if unsafe_member.isfile() else None,
            )
    return output.getvalue()


@contextmanager
def _serve_archive(payload: bytes, *, transient_failures: int = 0):
    request_paths = []
    request_count = 0

    class ArchiveHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal request_count
            request_paths.append(self.path)
            request_count += 1
            if request_count <= transient_failures:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), ArchiveHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield (
            f"http://127.0.0.1:{server.server_port}/owner/verifier.git",
            request_paths,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _prepare_local_kimi_verifier(
    tmp_path: Path,
    payload: bytes,
    verifier_ref: str = "1" * 40,
    transient_failures: int = 0,
    archive_sha256: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, list[str]]:
    checkout = tmp_path / "checkout"
    script = r"""
source "$BENCHMARK_LIB"
git() { echo "git must not be invoked" >&2; return 127; }
mktemp() { mkdir "$CHECKOUT"; printf '%s\n' "$CHECKOUT"; }
_prepare_kimi_vendor_verifier "$REPO_URL" "$VERIFIER_REF" "$ARCHIVE_SHA256"
"""
    with _serve_archive(
        payload,
        transient_failures=transient_failures,
    ) as (repo_url, request_paths):
        result = subprocess.run(
            ["bash", "-c", script],
            env={
                **os.environ,
                "BENCHMARK_LIB": str(BENCHMARK_LIB),
                "CHECKOUT": str(checkout),
                "REPO_URL": repo_url,
                "VERIFIER_REF": verifier_ref,
                "ARCHIVE_SHA256": archive_sha256 or hashlib.sha256(payload).hexdigest(),
            },
            text=True,
            capture_output=True,
        )
    return result, checkout, request_paths


def test_kimi_vendor_verifier_fetches_expected_subset_without_git(
    tmp_path: Path,
) -> None:
    result, checkout, request_paths = _prepare_local_kimi_verifier(
        tmp_path,
        _kimi_verifier_archive(),
    )
    verifier_ref = "1" * 40

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(checkout)
    assert request_paths == [
        f"/owner/verifier/archive/{verifier_ref}.tar.gz",
    ]
    assert {
        path.relative_to(checkout).as_posix()
        for path in checkout.rglob("*")
        if path.is_file()
    } == _KIMI_VERIFIER_REQUIRED_FILES
    assert "git must not be invoked" not in result.stderr


def test_kimi_vendor_verifier_retries_transient_archive_failure(
    tmp_path: Path,
) -> None:
    result, checkout, request_paths = _prepare_local_kimi_verifier(
        tmp_path,
        _kimi_verifier_archive(),
        transient_failures=1,
    )
    verifier_ref = "1" * 40

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(checkout)
    assert request_paths == [
        f"/owner/verifier/archive/{verifier_ref}.tar.gz",
        f"/owner/verifier/archive/{verifier_ref}.tar.gz",
    ]
    assert "archive download attempt 1/3 failed" in result.stderr


def test_kimi_vendor_verifier_rejects_archive_hash_mismatch(
    tmp_path: Path,
) -> None:
    result, checkout, _ = _prepare_local_kimi_verifier(
        tmp_path,
        _kimi_verifier_archive(),
        archive_sha256="0" * 64,
    )

    assert result.returncode == 1
    assert "archive SHA256 mismatch" in result.stderr
    assert not checkout.exists()


def test_kimi_vendor_verifier_removes_partial_checkout_when_member_missing(
    tmp_path: Path,
) -> None:
    missing = "tests/tool_call_json_schema/validator.py"
    result, checkout, _ = _prepare_local_kimi_verifier(
        tmp_path,
        _kimi_verifier_archive(missing=missing),
    )

    assert result.returncode == 1
    assert missing in result.stderr
    assert not checkout.exists()


def test_kimi_vendor_verifier_rejects_unsafe_archive_members(tmp_path: Path) -> None:
    unsafe = tarfile.TarInfo("verifier-pinned/../../escaped")
    unsafe.size = len(b"unsafe")
    result, checkout, _ = _prepare_local_kimi_verifier(
        tmp_path,
        _kimi_verifier_archive(unsafe_member=unsafe),
    )

    assert result.returncode == 1
    assert "unsafe archive member path" in result.stderr
    assert not checkout.exists()
    assert not (tmp_path / "escaped").exists()


def test_kimi_vendor_uses_system_python_fast_path() -> None:
    script = r"""
source "$BENCHMARK_LIB"
python3() {
    printf 'SYSTEM_PYTHON_ARG=<%s>\n' "$@"
    [ "$1" = "-c" ]
}
mktemp() { echo "UNEXPECTED_MKTEMP"; return 99; }
VENDOR_VERIFIER_PYTHON=/previous/python
VENDOR_VERIFIER_PYTHON_CLEANUP_DIR=/previous/runtime
_prepare_vendor_verifier_python "Kimi Vendor Verifier" "kimi-vendor-python"
printf 'SELECTED_PYTHON=<%s>\n' "$VENDOR_VERIFIER_PYTHON"
printf 'PYTHON_CLEANUP=<%s>\n' "$VENDOR_VERIFIER_PYTHON_CLEANUP_DIR"
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "BENCHMARK_LIB": str(BENCHMARK_LIB)},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "SYSTEM_PYTHON_ARG=<-c>" in result.stdout
    assert "SELECTED_PYTHON=<python3>" in result.stdout
    assert "PYTHON_CLEANUP=<>" in result.stdout
    assert "UNEXPECTED_MKTEMP" not in result.stdout


def test_kimi_vendor_bootstraps_pinned_python_and_cleans_it(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "bootstrap.log"
    fake_uv = tmp_path / "fake-uv"
    fake_uv.write_text(
        r"""#!/usr/bin/env bash
printf 'UV_CACHE_DIR=<%s>\n' "$UV_CACHE_DIR" >> "$KIMI_LOG"
printf 'UV_PYTHON_INSTALL_DIR=<%s>\n' "$UV_PYTHON_INSTALL_DIR" >> "$KIMI_LOG"
printf 'UV_ARG=<%s>\n' "$@" >> "$KIMI_LOG"
venv_dir="${!#}"
mkdir -p "$venv_dir/bin"
cat > "$venv_dir/bin/python" <<'PYTHON'
#!/usr/bin/env bash
printf 'SELECTED_PYTHON_ARG=<%s>\n' "$@" >> "$KIMI_LOG"
PYTHON
chmod +x "$venv_dir/bin/python"
"""
    )
    fake_uv.chmod(0o755)
    script = r"""
source "$BENCHMARK_LIB"
python3() {
    if [ "$1" = "-c" ]; then
        printf 'VERSION_CHECK\n' >> "$KIMI_LOG"
        return 1
    fi
    printf 'SYSTEM_PYTHON_ARG=<%s>\n' "$@" >> "$KIMI_LOG"
    local prefix=""
    while [[ $# -gt 0 ]]; do
        if [ "$1" = "--prefix" ]; then
            prefix="$2"
            break
        fi
        shift
    done
    mkdir -p "$prefix/bin"
    cp "$FAKE_UV" "$prefix/bin/uv"
    chmod +x "$prefix/bin/uv"
}
_prepare_vendor_verifier_python "Kimi Vendor Verifier" "kimi-vendor-python"
cleanup_dir="$VENDOR_VERIFIER_PYTHON_CLEANUP_DIR"
printf 'SELECTED_PYTHON=<%s>\n' "$VENDOR_VERIFIER_PYTHON"
printf 'PYTHON_CLEANUP=<%s>\n' "$cleanup_dir"
runtime_dir="$TEST_ROOT/runtime"
mkdir "$runtime_dir"
_install_kimi_vendor_eval_deps "$runtime_dir"
_cleanup_vendor_eval "$runtime_dir" "$cleanup_dir"
[ ! -e "$runtime_dir" ] && [ ! -e "$cleanup_dir" ] && printf 'CLEANED\n'
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "BENCHMARK_LIB": str(BENCHMARK_LIB),
            "FAKE_UV": str(fake_uv),
            "KIMI_LOG": str(log_path),
            "TEST_ROOT": str(tmp_path),
        },
        text=True,
        capture_output=True,
        check=True,
    )
    log = log_path.read_text()

    assert "VERSION_CHECK" in log
    assert "SYSTEM_PYTHON_ARG=<--prefix>" in log
    assert "SYSTEM_PYTHON_ARG=<--break-system-packages>" in log
    assert "SYSTEM_PYTHON_ARG=<uv==0.11.33>" in log
    assert "UV_ARG=<venv>" in log
    assert "UV_ARG=<--python>" in log
    assert "UV_ARG=<3.12>" in log
    assert "UV_ARG=<--seed>" in log
    assert "UV_CACHE_DIR=</tmp/kimi-vendor-python-" in log
    assert "UV_PYTHON_INSTALL_DIR=</tmp/kimi-vendor-python-" in log
    assert "SELECTED_PYTHON_ARG=<--target>" in log
    assert "SELECTED_PYTHON_ARG=<pytest-rerunfailures" not in log
    assert "SELECTED_PYTHON=</tmp/kimi-vendor-python-" in result.stdout
    assert "PYTHON_CLEANUP=</tmp/kimi-vendor-python-" in result.stdout
    assert "CLEANED" in result.stdout


def test_kimi_vendor_dependency_install_is_isolated(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    script = r"""
source "$BENCHMARK_LIB"
selected_python() { printf 'PYTHON_ARG=<%s>\n' "$@"; }
VENDOR_VERIFIER_PYTHON=selected_python
_install_kimi_vendor_eval_deps "$RUNTIME_DIR"
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "BENCHMARK_LIB": str(BENCHMARK_LIB),
            "RUNTIME_DIR": str(runtime_dir),
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert "PYTHON_ARG=<--target>" in result.stdout
    assert f"PYTHON_ARG=<{runtime_dir}>" in result.stdout
    assert "PYTHON_ARG=<pytest-rerunfailures" not in result.stdout
    assert "pytest-xdist" not in result.stdout
    assert "--break-system-packages" not in result.stdout


def test_kimi_vendor_surfaces_failure_artifact_error(tmp_path: Path) -> None:
    script = r"""
source "$BENCHMARK_LIB"
_prepare_vendor_verifier_python() { return 12; }
_write_kimi_vendor_integration_error() { return 23; }
run_kimi_vendor_eval --results-dir "$RESULTS_DIR"
printf 'EVAL_RC=%s\n' "$?"
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "BENCHMARK_LIB": str(BENCHMARK_LIB),
            "RESULTS_DIR": str(tmp_path / "results"),
            "MODEL": "test-model",
            "IS_MULTINODE": "false",
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert "EVAL_RC=12" in result.stdout
    assert "failed to write Kimi verifier failure artifact" in result.stderr


def test_kimi_vendor_multinode_runner_uses_fixed_upstream_contract(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    verifier_dir = tmp_path / "verifier"
    runtime_dir = tmp_path / "runtime"
    python_dir = tmp_path / "python"
    verifier_dir.mkdir()
    script = r"""
source "$BENCHMARK_LIB"
_prepare_vendor_verifier_python() {
    mkdir "$PYTHON_DIR"
    VENDOR_VERIFIER_PYTHON=selected_python
    VENDOR_VERIFIER_PYTHON_CLEANUP_DIR="$PYTHON_DIR"
    export VENDOR_VERIFIER_PYTHON VENDOR_VERIFIER_PYTHON_CLEANUP_DIR
}
_prepare_kimi_vendor_runtime() {
    mkdir "$RUNTIME_DIR"
    _install_kimi_vendor_eval_deps "$RUNTIME_DIR" >&2
    printf '%s\n' "$RUNTIME_DIR"
}
_prepare_kimi_vendor_verifier() {
    printf 'CHECKOUT=%s@%s\n' "$1" "$2" >&2
    printf 'CHECKOUT_SHA=%s\n' "$3" >&2
    "$VENDOR_VERIFIER_PYTHON" - "$1" "$2" "$3" "$VERIFIER_DIR" <<'PY' >&2
archive extraction
PY
    printf '%s\n' "$VERIFIER_DIR"
}
selected_python() {
    printf 'PYTHONPATH=<%s>\n' "$PYTHONPATH" >&2
    printf 'PYTHON_ARG=<%s>\n' "$@" >&2
}
python3() { echo "SYSTEM_PYTHON_UNEXPECTED" >&2; return 99; }
run_kimi_vendor_eval --port 9999 --results-dir "$RESULTS_DIR"
printf 'EVAL_SUITE=%s\n' "$EVAL_SUITE"
printf 'EVAL_RESULT_DIR=%s\n' "$EVAL_RESULT_DIR"
"""
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "RESULTS_DIR": str(results_dir),
        "VERIFIER_DIR": str(verifier_dir),
        "MODEL": "test-model",
        "MODEL_PREFIX": "dsv4",
        "RUNTIME_DIR": str(runtime_dir),
        "PYTHON_DIR": str(python_dir),
        "OPENAI_API_KEY": "must-not-be-forwarded",
        "KV_OFFLOADING": "none",
        "IS_MULTINODE": "true",
    }
    for key in (
        "EVAL_SUITE",
        "EVAL_RESULT_DIR",
        "MODEL_NAME",
    ):
        env.pop(key, None)

    result = subprocess.run(
        ["bash", "-c", script],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    output = result.stdout + result.stderr
    adapter = BENCHMARK_LIB.parents[1] / "utils/evals/kimi_vendor_eval.py"

    assert f"PYTHONPATH=<{tmp_path / 'runtime'}" in output
    assert (
        "CHECKOUT=https://github.com/MoonshotAI/Kimi-Vendor-Verifier.git"
        "@b9ed3a6665bdff2c943246f7d2903cd003d6ddd6"
    ) in output
    assert (
        "CHECKOUT_SHA=ab933117c894a785978f8aee0f052e5a9096b3029e7962354b1c07ea430588c3"
        in output
    )
    assert "PYTHON_ARG=<->" in output
    assert "PYTHON_ARG=<b9ed3a6665bdff2c943246f7d2903cd003d6ddd6>" in output
    for value in (
        adapter,
        verifier_dir,
        "http://127.0.0.1:9999/v1",
        "EMPTY",
        "test-model",
        results_dir,
    ):
        assert f"PYTHON_ARG=<{value}>" in output
    assert "PYTHON_ARG=<--model-prefix>" in output
    assert "PYTHON_ARG=<dsv4>" in output
    assert "PYTHON_ARG=<--task-name>" in output
    assert "PYTHON_ARG=<kimi_tool_call_schema>" in output
    assert "PYTHON_ARG=<--timeout-seconds>" in output
    assert "PYTHON_ARG=<900>" in output
    assert "must-not-be-forwarded" not in output
    assert "SYSTEM_PYTHON_UNEXPECTED" not in output
    assert "EVAL_SUITE=kimi_tool_call_schema" in output
    assert f"EVAL_RESULT_DIR={results_dir}" in output
    assert not (tmp_path / "runtime").exists()
    assert not verifier_dir.exists()
    assert not python_dir.exists()


def test_kimi_full_runner_installs_xdist_sets_timeout_and_cleans_runtimes(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    verifier_dir = tmp_path / "verifier"
    runtime_dir = tmp_path / "runtime"
    python_dir = tmp_path / "python"
    script = r"""
source "$BENCHMARK_LIB"
_prepare_vendor_verifier_python() {
    mkdir "$PYTHON_DIR"
    VENDOR_VERIFIER_PYTHON=selected_python
    VENDOR_VERIFIER_PYTHON_CLEANUP_DIR="$PYTHON_DIR"
    export VENDOR_VERIFIER_PYTHON VENDOR_VERIFIER_PYTHON_CLEANUP_DIR
}
_prepare_kimi_vendor_runtime() {
    printf 'RUNTIME_SUITE=<%s>\n' "$1" >&2
    mkdir "$RUNTIME_DIR"
    _install_kimi_vendor_eval_deps "$RUNTIME_DIR" "$1" >&2
    printf '%s\n' "$RUNTIME_DIR"
}
_prepare_kimi_vendor_verifier() {
    mkdir "$VERIFIER_DIR"
    printf '%s\n' "$VERIFIER_DIR"
}
selected_python() {
    printf 'PYTHONPATH=<%s>\n' "$PYTHONPATH" >&2
    printf 'PYTHON_ARG=<%s>\n' "$@" >&2
}
run_kimi_vendor_eval --port 9999 --results-dir "$RESULTS_DIR"
printf 'EVAL_SUITE=%s\n' "$EVAL_SUITE"
printf 'EVAL_RESULT_DIR=%s\n' "$EVAL_RESULT_DIR"
"""
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "EVAL_SUITE": "kimi_tool_call_schema_full",
        "RESULTS_DIR": str(results_dir),
        "VERIFIER_DIR": str(verifier_dir),
        "MODEL": "test-model",
        "RUNTIME_DIR": str(runtime_dir),
        "PYTHON_DIR": str(python_dir),
    }
    for key in ("EVAL_RESULT_DIR", "MODEL_NAME"):
        env.pop(key, None)

    result = subprocess.run(
        ["bash", "-c", script],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    output = result.stdout + result.stderr

    assert "RUNTIME_SUITE=<kimi_tool_call_schema_full>" in output
    assert "PYTHON_ARG=<pytest-xdist==3.8.0>" in output
    assert "PYTHON_ARG=<--task-name>" in output
    assert "PYTHON_ARG=<kimi_tool_call_schema_full>" in output
    assert "PYTHON_ARG=<--timeout-seconds>" in output
    assert "PYTHON_ARG=<7200>" in output
    assert "EVAL_SUITE=kimi_tool_call_schema_full" in output
    assert f"EVAL_RESULT_DIR={results_dir}" in output
    assert not runtime_dir.exists()
    assert not verifier_dir.exists()
    assert not python_dir.exists()


def test_run_lm_eval_rejects_missing_option_value():
    result = _run_invalid_call("run_lm_eval --port")
    assert result.returncode == 2
    assert "--port requires a value" in result.stderr


def test_lm_patch_copy_resolves_outside_repo(tmp_path):
    script = r"""
source "$BENCHMARK_LIB"
cd "$OTHER_CWD"
_patch_lm_eval
patch_dir=${PYTHONPATH%%:*}
cmp "$(_eval_patches_dir)/lm_eval_sitecustomize.py" "$patch_dir/sitecustomize.py"
"""
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "OTHER_CWD": str(tmp_path),
        "KV_OFFLOADING": "none",
    }
    subprocess.run(["bash", "-c", script], env=env, check=True)


_EVAL_LIMIT_SCRIPT = r"""
set -e
SHIM_DIR=$(mktemp -d)
cat > "$SHIM_DIR/python3" <<'PY'
#!/usr/bin/env bash
echo "PYTHON_ARGS: $*"
exit 0
PY
chmod +x "$SHIM_DIR/python3"

source "$BENCHMARK_LIB"

export EVAL_MAX_MODEL_LEN=16384
export MODEL_NAME=test-model
export OPENAI_API_KEY=EMPTY
export INFERENCEX_LM_EVAL_RUNTIME_READY=true

_install_lm_eval_deps() { :; }
_patch_lm_eval() { :; }

PATH="$SHIM_DIR:$PATH" run_lm_eval --port 9999 2>&1
"""


def _run_lm_eval_cmdline(*, eval_limit=None) -> str:
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "KV_OFFLOADING": "none",
    }
    env.pop("EVAL_LIMIT", None)
    if eval_limit is not None:
        env["EVAL_LIMIT"] = str(eval_limit)
    res = subprocess.run(
        ["bash", "-c", _EVAL_LIMIT_SCRIPT],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return res.stdout + res.stderr


def test_eval_limit_appended_when_set():
    out = _run_lm_eval_cmdline(eval_limit=10)
    assert "--limit 10" in out, f"Expected '--limit 10' in output:\n{out}"


def test_eval_limit_absent_when_unset():
    out = _run_lm_eval_cmdline(eval_limit=None)
    assert "--limit" not in out, f"Expected no '--limit' in output:\n{out}"


def test_lm_eval_defaults_to_gsm8k():
    out = _run_lm_eval_cmdline()
    assert "utils/evals/gsm8k.yaml" in out


def _summary_metadata(tmp_path: Path, **overrides: str) -> dict:
    work_dir = tmp_path / "work"
    results_dir = tmp_path / "results"
    work_dir.mkdir(parents=True)
    results_dir.mkdir()
    script = r"""
source "$BENCHMARK_LIB"
cd "$WORK_DIR"
append_lm_eval_summary >/dev/null
"""
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "WORK_DIR": str(work_dir),
        "EVAL_RESULT_DIR": str(results_dir),
        "MODEL": "test-model",
        "CONC": "7",
        "KV_OFFLOADING": "none",
    }
    for key in ("EVAL_COMPLETED_SUITE", "EVAL_SUITE", "EVAL_TASKS_DIR"):
        env.pop(key, None)
    env.update(overrides)
    subprocess.run(["bash", "-c", script], env=env, check=True)
    return json.loads((work_dir / "meta_env.json").read_text())


def test_summary_stages_bfcl_upstream_archive_before_cleanup(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    results_dir = tmp_path / "results"
    work_dir.mkdir()
    results_dir.mkdir()
    archive = results_dir / "bfcl_upstream_artifacts.tar.gz"
    archive.write_bytes(b"bfcl-archive")
    script = r"""
source "$BENCHMARK_LIB"
cd "$WORK_DIR"
append_lm_eval_summary >/dev/null
"""
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "WORK_DIR": str(work_dir),
        "EVAL_RESULT_DIR": str(results_dir),
        "MODEL": "test-model",
        "CONC": "7",
        "KV_OFFLOADING": "none",
    }

    subprocess.run(["bash", "-c", script], env=env, check=True)

    assert (work_dir / archive.name).read_bytes() == b"bfcl-archive"
    assert not results_dir.exists()


def test_stage_eval_artifacts_copies_eval_outputs_only(tmp_path: Path) -> None:
    source_one = tmp_path / "source-one"
    source_two = tmp_path / "source-two"
    destination = tmp_path / "destination"
    source_one.mkdir()
    source_two.mkdir()
    expected = {
        "meta_env.json",
        "results_bfcl.json",
        "kimi_vendor_report.json",
        "kimi_vendor_results.jsonl",
        "bfcl_report.json",
        "bfcl_upstream_artifacts.tar.gz",
        "sample_eval.jsonl",
        "agent_preds.json",
        "predictions.jsonl",
        "swebench_report_eval.json",
        "trace.traj.json",
    }
    for filename in expected:
        source = source_one if filename.endswith(".json") else source_two
        (source / filename).write_text(filename)
    (source_one / "unrelated.log").write_text("skip")
    script = r"""
source "$BENCHMARK_LIB"
stage_eval_artifacts "$DESTINATION" "$SOURCE_ONE" "$SOURCE_TWO"
"""
    subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "BENCHMARK_LIB": str(BENCHMARK_LIB),
            "DESTINATION": str(destination),
            "SOURCE_ONE": str(source_one),
            "SOURCE_TWO": str(source_two),
            "KV_OFFLOADING": "none",
        },
        check=True,
    )

    assert {path.name for path in destination.iterdir()} == expected


def test_stage_eval_artifacts_propagates_copy_failure(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "bfcl_report.json").write_text("{}")
    script = r"""
source "$BENCHMARK_LIB"
cp() { return 73; }
stage_eval_artifacts "$DESTINATION" "$SOURCE"
"""

    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "BENCHMARK_LIB": str(BENCHMARK_LIB),
            "DESTINATION": str(tmp_path / "destination"),
            "SOURCE": str(source),
            "KV_OFFLOADING": "none",
        },
        check=False,
    )

    assert result.returncode == 73


def test_stage_eval_artifacts_fails_when_no_artifacts_exist(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    script = r"""
source "$BENCHMARK_LIB"
stage_eval_artifacts "$DESTINATION" "$SOURCE"
"""

    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "BENCHMARK_LIB": str(BENCHMARK_LIB),
            "DESTINATION": str(tmp_path / "destination"),
            "SOURCE": str(source),
            "KV_OFFLOADING": "none",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "no eval artifacts found to stage" in result.stderr


def test_summary_propagates_artifact_staging_failure(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    results_dir = tmp_path / "results"
    work_dir.mkdir()
    results_dir.mkdir()
    (results_dir / "results_eval.json").write_text("{}")
    script = r"""
source "$BENCHMARK_LIB"
cp() { return 73; }
cd "$WORK_DIR"
append_lm_eval_summary
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "BENCHMARK_LIB": str(BENCHMARK_LIB),
            "WORK_DIR": str(work_dir),
            "EVAL_RESULT_DIR": str(results_dir),
            "MODEL": "test-model",
            "CONC": "7",
            "KV_OFFLOADING": "none",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 73


def test_summary_metadata_preserves_lm_eval_gsm8k_defaults(tmp_path: Path) -> None:
    meta = _summary_metadata(tmp_path)

    assert meta["eval_suite"] == "gsm8k"
    assert meta["conc"] == 7


def test_summary_metadata_preserves_single_node_expert_parallelism(
    tmp_path: Path,
) -> None:
    meta = _summary_metadata(tmp_path, TP="8", EP_SIZE="8")

    assert meta["ep"] == 8
    assert meta["prefill_ep"] == 8
    assert meta["decode_ep"] == 8


def test_run_lm_eval_exports_cli_task_path(tmp_path: Path) -> None:
    script = r"""
source "$BENCHMARK_LIB"
python3() { :; }
export EVAL_MAX_MODEL_LEN=16384
export INFERENCEX_LM_EVAL_RUNTIME_READY=true
run_lm_eval --task custom.yaml --results-dir "$RESULTS_DIR"
printf 'EVAL_TASKS_DIR=%s\n' "$EVAL_TASKS_DIR"
"""
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "RESULTS_DIR": str(tmp_path / "results"),
        "MODEL_NAME": "test-model",
        "OPENAI_API_KEY": "EMPTY",
        "KV_OFFLOADING": "none",
    }
    env.pop("EVAL_TASKS_DIR", None)
    result = subprocess.run(
        ["bash", "-c", script],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "EVAL_TASKS_DIR=custom.yaml" in result.stdout


def test_summary_metadata_prefers_explicit_suite_then_task_basename(
    tmp_path: Path,
) -> None:
    from_task = _summary_metadata(
        tmp_path / "task",
        EVAL_TASKS_DIR="/tmp/custom_reasoning.yaml",
    )
    explicit = _summary_metadata(
        tmp_path / "explicit",
        EVAL_SUITE="kimi_tool_call_schema",
        EVAL_TASKS_DIR="/tmp/ignored.yaml",
    )

    assert from_task["eval_suite"] == "custom_reasoning"
    assert explicit["eval_suite"] == "kimi_tool_call_schema"


def test_summary_metadata_prefers_completed_eval_identity(tmp_path: Path) -> None:
    meta = _summary_metadata(
        tmp_path,
        EVAL_COMPLETED_SUITE="kimi_tool_call_schema",
        EVAL_SUITE="stale_input_selector",
        EVAL_TASKS_DIR="/tmp/ignored.yaml",
    )

    assert meta["eval_suite"] == "kimi_tool_call_schema"


def test_env_is_true_is_case_insensitive_and_unset_safe() -> None:
    script = r"""
set -u
source "$BENCHMARK_LIB"
for value in TrUe yEs oN 1 false 0; do
    if _env_is_true "$value"; then
        echo true
    else
        echo false
    fi
done
for empty_call in with-argument without-argument; do
    if [ "$empty_call" = "with-argument" ]; then
        _env_is_true ""
    else
        _env_is_true
    fi
    if [ "$?" -eq 0 ]; then
        echo true
    else
        echo false
    fi
done
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "BENCHMARK_LIB": str(BENCHMARK_LIB)},
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.splitlines() == [
        "true",
        "true",
        "true",
        "true",
        "false",
        "false",
        "false",
        "false",
    ]


_MODAL_CREDS_SCRIPT = r"""
source "$BENCHMARK_LIB"
_ensure_modal_credentials
echo "HOME_AFTER=$HOME"
if [ -f "$HOME/.modal.toml" ]; then
    echo "TOML_EXISTS=true"
    PERMS=$(stat -c '%a' "$HOME/.modal.toml" 2>/dev/null || stat -f '%A' "$HOME/.modal.toml" 2>/dev/null)
    echo "TOML_PERMS=$PERMS"
fi
"""


def _run_modal_creds(
    tmp_path: Path, *, home: str, token_id="tok-id", token_secret="tok-secret"
) -> str:
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "KV_OFFLOADING": "none",
        "SWEBENCH_USE_MODAL": "true",
        "MODAL_TOKEN_ID": token_id,
        "MODAL_TOKEN_SECRET": token_secret,
        "HOME": home,
    }
    res = subprocess.run(
        ["bash", "-c", _MODAL_CREDS_SCRIPT],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return res.stdout + res.stderr


def test_modal_creds_no_remap_when_home_writable(tmp_path):
    home = str(tmp_path / "writable_home")
    Path(home).mkdir()
    out = _run_modal_creds(tmp_path, home=home)
    assert f"HOME_AFTER={home}" in out, f"HOME should not be remapped:\n{out}"
    assert "TOML_EXISTS=true" in out
    toml_path = Path(home) / ".modal.toml"
    assert toml_path.exists()
    mode = oct(stat.S_IMODE(toml_path.stat().st_mode))
    assert mode == "0o600", f"Expected 0o600 got {mode}"


def test_modal_creds_remaps_home_when_not_writable_parent(tmp_path):
    readonly_parent = tmp_path / "readonly_parent"
    readonly_parent.mkdir(mode=0o555)
    nested_home = str(readonly_parent / "nested_home")
    try:
        out = _run_modal_creds(tmp_path, home=nested_home)
        assert "HOME_AFTER=/tmp/inferencex-modal-home" in out, (
            f"Expected HOME remap:\n{out}"
        )
        assert "remapped" in out.lower() or "HOME remapped" in out
        assert "TOML_EXISTS=true" in out
        toml_path = Path("/tmp/inferencex-modal-home/.modal.toml")
        assert toml_path.exists()
        mode = oct(stat.S_IMODE(toml_path.stat().st_mode))
        assert mode == "0o600", f"Expected 0o600 got {mode}"
    finally:
        readonly_parent.chmod(0o755)


def test_modal_creds_remaps_home_when_not_writable(tmp_path):
    readonly_home = tmp_path / "readonly_home"
    readonly_home.mkdir(mode=0o555)
    try:
        out = _run_modal_creds(tmp_path, home=str(readonly_home))
        assert "HOME_AFTER=/tmp/inferencex-modal-home" in out, (
            f"Expected HOME remap:\n{out}"
        )
        assert "TOML_EXISTS=true" in out
    finally:
        readonly_home.chmod(0o755)


def test_modal_creds_no_remap_when_disabled(tmp_path):
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "KV_OFFLOADING": "none",
        "SWEBENCH_USE_MODAL": "false",
        "MODAL_TOKEN_ID": "tok",
        "MODAL_TOKEN_SECRET": "sec",
        "HOME": str(tmp_path),
    }
    res = subprocess.run(
        ["bash", "-c", _MODAL_CREDS_SCRIPT],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    out = res.stdout + res.stderr
    assert "remapped" not in out.lower()
    assert "TOML_EXISTS" not in out


_INCLUDE_PATH_SCRIPT = r"""
set -e
SHIM_DIR=$(mktemp -d)
cat > "$SHIM_DIR/python3" <<'PY'
#!/usr/bin/env bash
echo "PYTHON_ARGS: $*"
exit 0
PY
chmod +x "$SHIM_DIR/python3"

source "$BENCHMARK_LIB"

export EVAL_MAX_MODEL_LEN=16384
export MODEL_NAME=test-model
export OPENAI_API_KEY=EMPTY
export INFERENCEX_LM_EVAL_RUNTIME_READY=true

_install_lm_eval_deps() { :; }
_patch_lm_eval() { :; }

PATH="$SHIM_DIR:$PATH" run_lm_eval --port 9999 2>&1
"""


def _run_lm_eval_with_include_path(
    *,
    eval_include_path: str | None = None,
    eval_tasks_dir: str | None = None,
) -> str:
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "KV_OFFLOADING": "none",
    }
    env.pop("EVAL_INCLUDE_PATH", None)
    env.pop("EVAL_TASKS_DIR", None)
    if eval_include_path is not None:
        env["EVAL_INCLUDE_PATH"] = eval_include_path
    if eval_tasks_dir is not None:
        env["EVAL_TASKS_DIR"] = eval_tasks_dir
    res = subprocess.run(
        ["bash", "-c", _INCLUDE_PATH_SCRIPT],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return res.stdout + res.stderr


def test_include_path_injected_when_eval_include_path_set():
    out = _run_lm_eval_with_include_path(
        eval_include_path="utils/evals",
        eval_tasks_dir="swebench_lite",
    )
    assert "--include_path utils/evals" in out, (
        f"Expected '--include_path utils/evals' in output:\n{out}"
    )
    assert "--tasks swebench_lite" in out, (
        f"Expected '--tasks swebench_lite' in output:\n{out}"
    )
    assert ".yaml" not in out.split("--tasks")[1].split()[0], (
        f"--tasks must not contain a .yaml path when include_path is set:\n{out}"
    )


def test_include_path_absent_when_eval_include_path_unset():
    out = _run_lm_eval_with_include_path()
    assert "--include_path" not in out, (
        f"Expected no '--include_path' in output:\n{out}"
    )
    assert "--tasks utils/evals/gsm8k.yaml" in out, (
        f"Expected '--tasks utils/evals/gsm8k.yaml' in output:\n{out}"
    )


def test_swebench_single_shot_registers_task_yaml():
    script = r"""
source "$BENCHMARK_LIB"
run_lm_eval() {
    echo "TASK=$EVAL_TASKS_DIR"
    echo "INCLUDE=$EVAL_INCLUDE_PATH"
    return 9
}
export SWEBENCH_GEN_MODE=single-shot
export EVAL_TASKS_DIR="$TASK_YAML"
export MODEL=test-model
run_swebench_eval
"""
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "TASK_YAML": str(BENCHMARK_LIB.parents[1] / "utils/evals/swebench_lite.yaml"),
        "KV_OFFLOADING": "none",
    }
    result = subprocess.run(
        ["bash", "-c", script],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 9
    assert "TASK=swebench_lite" in result.stdout
    assert f"INCLUDE={BENCHMARK_LIB.parents[1] / 'utils/evals'}" in result.stdout


def test_modal_credentials_sanitizes_whitespace_contaminated_tokens(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    script = r"""
source "$BENCHMARK_LIB" 2>/dev/null
export SWEBENCH_USE_MODAL=true
export MODAL_TOKEN_ID='ak-clean123'
export MODAL_TOKEN_SECRET="$(printf 'as-dirty456\n')"
_ensure_modal_credentials
grep -q 'token_secret = "as-dirty456"' "$HOME/.modal.toml" || { echo FILE_DIRTY; exit 1; }
[ "$MODAL_TOKEN_SECRET" = "as-dirty456" ] || { echo ENV_DIRTY; exit 1; }
echo SANITIZED_OK
"""
    env = {**os.environ, "BENCHMARK_LIB": str(BENCHMARK_LIB), "HOME": str(home)}
    res = subprocess.run(
        ["bash", "-c", script], env=env, text=True, capture_output=True
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "SANITIZED_OK" in res.stdout


def test_agentic_generation_invokes_mini_swe_agent(tmp_path):
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "mini-extra").write_text(
        "#!/bin/bash\n"
        'echo "MINI_ARGV: $*" >> ' + str(shim / "argv.log") + "\n"
        'out=""; prev=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        'mkdir -p "$out"\n'
        'printf \'{"i1": {"instance_id": "i1", "model_name_or_path": "m", "model_patch": "d"}}\' > "$out/preds.json"\n'
    )
    (shim / "mini-extra").chmod(0o755)
    default_yaml = shim / "default.yaml"
    default_yaml.write_text("agent: {}\n")
    (shim / "python3").write_text(
        "#!/bin/bash\n"
        f'if [[ "$*" == *minisweagent* ]]; then echo "This is mini-swe-agent version 2.4.5."; echo "Check the v2 migration guide"; echo {default_yaml}; else exec /usr/bin/python3 "$@"; fi\n'
    )
    (shim / "python3").chmod(0o755)

    gen_dir = tmp_path / "gen"
    gen_dir.mkdir()
    script = r"""
source "$BENCHMARK_LIB" 2>/dev/null
_install_swebench_agent_deps() { :; }
_ensure_modal_credentials() { :; }
export EVAL_LIMIT=10 MODEL_NAME=test-model SWEBENCH_SANDBOX_SWEEP=0 SWEBENCH_WATCHDOG_POLL=1
_run_swebench_agentic_generation "$GEN_DIR" --port 8899 || exit 1
[ -s "$GEN_DIR/agent_out/preds.json" ] || { echo NO_PREDS; exit 1; }
grep -q 'api_base: http://0.0.0.0:8899/v1' "$GEN_DIR/mini_swebench_overrides.yaml" || { echo BAD_PORT; exit 1; }
grep -q 'openai/test-model' "$GEN_DIR/mini_swebench_overrides.yaml" || { echo BAD_MODEL; exit 1; }
grep -q 'additional_critical_guidance' "$GEN_DIR/mini_swebench_overrides.yaml" || { echo NO_GUIDANCE; exit 1; }
grep -q 'BEFORE submitting you MUST run the test' "$GEN_DIR/mini_swebench_overrides.yaml" || { echo NO_VERIFY_RULE; exit 1; }
grep -q 'runtime_timeout: 3600' "$GEN_DIR/mini_swebench_overrides.yaml" || { echo NO_RUNTIME_TIMEOUT; exit 1; }
echo AGENTIC_GEN_OK
"""
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "GEN_DIR": str(gen_dir),
        "PATH": f"{shim}:{os.environ['PATH']}",
    }
    res = subprocess.run(
        ["bash", "-c", script], env=env, text=True, capture_output=True
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "AGENTIC_GEN_OK" in res.stdout
    argv = (shim / "argv.log").read_text()
    assert "--slice 0:10" in argv
    assert "--environment-class swerex_modal" in argv
    assert "--subset lite" in argv


def _agentic_shim(tmp_path, mini_body):
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "mini-extra").write_text("#!/bin/bash\n" + mini_body)
    (shim / "mini-extra").chmod(0o755)
    default_yaml = shim / "default.yaml"
    default_yaml.write_text("agent: {}\n")
    (shim / "python3").write_text(
        "#!/bin/bash\n"
        f'if [[ "$*" == *minisweagent* ]]; then echo {default_yaml}; else exec /usr/bin/python3 "$@"; fi\n'
    )
    (shim / "python3").chmod(0o755)
    gen_dir = tmp_path / "gen"
    gen_dir.mkdir()
    return shim, gen_dir


def _run_agentic(shim, gen_dir, extra_env=None):
    script = r"""
source "$BENCHMARK_LIB" 2>/dev/null
_install_swebench_agent_deps() { :; }
_ensure_modal_credentials() { :; }
_run_swebench_agentic_generation "$GEN_DIR" --port 8899
echo "GEN_RC=$?"
"""
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "GEN_DIR": str(gen_dir),
        "MODEL_NAME": "test-model",
        "SWEBENCH_SANDBOX_SWEEP": "0",
        "SWEBENCH_WATCHDOG_POLL": "1",
        "PATH": f"{shim}:{os.environ['PATH']}",
        **(extra_env or {}),
    }
    return subprocess.run(
        ["bash", "-c", script], env=env, text=True, capture_output=True
    )


def test_agentic_watchdog_kills_hung_mini(tmp_path):
    shim, gen_dir = _agentic_shim(
        tmp_path,
        'out=""; prev=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        'mkdir -p "$out"\n'
        'printf \'{"i1": {"instance_id": "i1", "model_patch": "d"}}\' > "$out/preds.json"\n'
        "exec sleep 600 </dev/null >/dev/null 2>&1\n",
    )
    res = _run_agentic(
        shim, gen_dir, {"EVAL_LIMIT": "1", "SWEBENCH_AGENT_EXIT_GRACE": "2"}
    )
    assert "GEN_RC=0" in res.stdout, res.stdout + res.stderr
    assert "hung after completing all instances" in res.stdout + res.stderr


def test_agentic_salvage_partial_preds_on_failure(tmp_path):
    shim, gen_dir = _agentic_shim(
        tmp_path,
        'out=""; prev=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        'mkdir -p "$out"\n'
        'printf \'{"i1": {"instance_id": "i1", "model_patch": "d"}}\' > "$out/preds.json"\n'
        "exit 7\n",
    )
    res = _run_agentic(shim, gen_dir, {"EVAL_LIMIT": "2"})
    assert "GEN_RC=0" in res.stdout, res.stdout + res.stderr
    assert "scoring the partial set" in res.stdout + res.stderr


def test_agentic_no_preds_still_fails(tmp_path):
    shim, gen_dir = _agentic_shim(tmp_path, "exit 7\n")
    res = _run_agentic(shim, gen_dir, {"EVAL_LIMIT": "2"})
    assert "GEN_RC=7" in res.stdout, res.stdout + res.stderr


def test_agentic_eval_limit_defaults_to_full_split(tmp_path):
    shim, gen_dir = _agentic_shim(
        tmp_path,
        'echo "MINI_ARGV: $*" >> ' + "ARGVLOG" + "\n"
        'out=""; prev=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        'mkdir -p "$out"\n'
        'printf \'{"i1": {"instance_id": "i1", "model_patch": "d"}}\' > "$out/preds.json"\n',
    )
    body = (shim / "mini-extra").read_text().replace("ARGVLOG", str(shim / "argv.log"))
    (shim / "mini-extra").write_text(body)
    res = _run_agentic(shim, gen_dir)
    argv = (shim / "argv.log").read_text()
    assert "--slice" not in argv, argv
    assert "GEN_RC=0" in res.stdout, res.stdout + res.stderr


def test_agentic_eval_limit_full_runs_whole_split(tmp_path):
    shim, gen_dir = _agentic_shim(
        tmp_path,
        'echo "MINI_ARGV: $*" >> ' + "ARGVLOG" + "\n"
        'out=""; prev=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        'mkdir -p "$out"\n'
        'printf \'{"i1": {"instance_id": "i1", "model_patch": "d"}}\' > "$out/preds.json"\n',
    )
    body = (shim / "mini-extra").read_text().replace("ARGVLOG", str(shim / "argv.log"))
    (shim / "mini-extra").write_text(body)
    res = _run_agentic(shim, gen_dir, {"EVAL_LIMIT": "full"})
    argv = (shim / "argv.log").read_text()
    assert "--slice" not in argv, argv
    assert "GEN_RC=0" in res.stdout, res.stdout + res.stderr


def test_multinode_eval_artifact_names_are_bounded_and_distinct() -> None:
    workflow = yaml.safe_load(MULTINODE_WORKFLOW.read_text())
    upload = next(
        step
        for step in workflow["jobs"]["benchmark"]["steps"]
        if step.get("name") == "Upload eval results (if any)"
    )
    expression = upload["with"]["name"]
    assert expression.startswith("eval_")
    assert "RESULT_FILENAME" not in expression

    targets = [
        {
            "EXP_NAME": "kimik3_p2x16ep32dpa_d0x16ep32dpa_conc12",
            "PRECISION": "fp4",
            "FRAMEWORK": "vllm",
            "PREFILL_NUM_WORKERS": "2",
            "PREFILL_TP": "16",
            "PREFILL_PP_SIZE": "1",
            "SPEC_DECODING": "mtp",
            "PREFILL_DCP_SIZE": "1",
            "PREFILL_PCP_SIZE": "1",
            "PREFILL_EP": "32",
            "PREFILL_DP_ATTN": "true",
            "DECODE_NUM_WORKERS": "0",
            "DECODE_TP": "16",
            "DECODE_PP_SIZE": "1",
            "DECODE_DCP_SIZE": "1",
            "DECODE_PCP_SIZE": "1",
            "DECODE_EP": "32",
            "DECODE_DP_ATTN": "true",
            "KV_OFFLOADING": "none",
            "KV_OFFLOAD_BACKEND": "",
            "conc-list": ["1", "12", "16"],
            "runner.name": "h200-dgxc-slurm_00",
        },
        {
            "EXP_NAME": "kimik3_p4x8ep32dpa_d0x8ep32dpa_conc12",
            "PRECISION": "fp4",
            "FRAMEWORK": "vllm",
            "PREFILL_NUM_WORKERS": "4",
            "PREFILL_TP": "8",
            "PREFILL_PP_SIZE": "1",
            "PREFILL_DCP_SIZE": "1",
            "PREFILL_PCP_SIZE": "1",
            "PREFILL_EP": "32",
            "PREFILL_DP_ATTN": "true",
            "DECODE_NUM_WORKERS": "0",
            "SPEC_DECODING": "mtp",
            "DECODE_TP": "8",
            "DECODE_PP_SIZE": "1",
            "DECODE_DCP_SIZE": "1",
            "DECODE_PCP_SIZE": "1",
            "DECODE_EP": "32",
            "DECODE_DP_ATTN": "true",
            "KV_OFFLOADING": "none",
            "KV_OFFLOAD_BACKEND": "",
            "conc-list": ["1", "12", "16"],
            "runner.name": "h200-dgxc-slurm_01",
        },
        {
            "EXP_NAME": "kimik3_p4x8ep32dpa_d0x8ep32dpa_conc12_kvdram-vllm-simple",
            "PRECISION": "fp4",
            "FRAMEWORK": "vllm",
            "PREFILL_NUM_WORKERS": "4",
            "PREFILL_TP": "8",
            "PREFILL_PP_SIZE": "1",
            "PREFILL_DCP_SIZE": "1",
            "PREFILL_PCP_SIZE": "1",
            "PREFILL_EP": "32",
            "PREFILL_DP_ATTN": "true",
            "DECODE_NUM_WORKERS": "0",
            "DECODE_TP": "8",
            "DECODE_PP_SIZE": "1",
            "DECODE_DCP_SIZE": "1",
            "DECODE_PCP_SIZE": "1",
            "DECODE_EP": "32",
            "SPEC_DECODING": "mtp",
            "DECODE_DP_ATTN": "true",
            "KV_OFFLOADING": "dram",
            "KV_OFFLOAD_BACKEND": "vllm-simple",
            "conc-list": ["1", "12", "16"],
            "runner.name": "h200-dgxc-slurm_02",
        },
        {
            "EXP_NAME": "kimik3_p1x8_d0x8_conc16",
            "PRECISION": "fp4",
            "FRAMEWORK": "dynamo-vllm",
            "PREFILL_NUM_WORKERS": "1",
            "PREFILL_TP": "8",
            "PREFILL_PP_SIZE": "2",
            "PREFILL_DCP_SIZE": "1",
            "PREFILL_PCP_SIZE": "1",
            "PREFILL_EP": "1",
            "PREFILL_DP_ATTN": "false",
            "DECODE_NUM_WORKERS": "0",
            "DECODE_TP": "8",
            "DECODE_PP_SIZE": "2",
            "DECODE_DCP_SIZE": "1",
            "DECODE_PCP_SIZE": "1",
            "DECODE_EP": "1",
            "DECODE_DP_ATTN": "false",
            "SPEC_DECODING": "mtp",
            "KV_OFFLOADING": "none",
            "KV_OFFLOAD_BACKEND": "",
            "conc-list": ["1", "16"],
            "runner.name": "b200-dgxc_00",
        },
        {
            "EXP_NAME": "kimik3_p1x16ep16_d0x16ep16_conc16",
            "PRECISION": "fp4",
            "FRAMEWORK": "dynamo-vllm",
            "PREFILL_NUM_WORKERS": "1",
            "PREFILL_TP": "16",
            "PREFILL_PP_SIZE": "1",
            "PREFILL_DCP_SIZE": "1",
            "PREFILL_PCP_SIZE": "1",
            "PREFILL_EP": "16",
            "PREFILL_DP_ATTN": "false",
            "DECODE_NUM_WORKERS": "0",
            "DECODE_TP": "16",
            "DECODE_PP_SIZE": "1",
            "DECODE_DCP_SIZE": "1",
            "DECODE_PCP_SIZE": "1",
            "DECODE_EP": "16",
            "DECODE_DP_ATTN": "false",
            "KV_OFFLOADING": "none",
            "SPEC_DECODING": "mtp",
            "KV_OFFLOAD_BACKEND": "",
            "conc-list": ["16"],
            "runner.name": "gb200-nv_00",
        },
        {
            "EXP_NAME": "kimik3_p1x16_d0x16_conc1",
            "PRECISION": "fp4",
            "FRAMEWORK": "dynamo-vllm",
            "PREFILL_NUM_WORKERS": "1",
            "PREFILL_TP": "16",
            "PREFILL_PP_SIZE": "1",
            "PREFILL_DCP_SIZE": "1",
            "PREFILL_PCP_SIZE": "1",
            "PREFILL_EP": "1",
            "PREFILL_DP_ATTN": "false",
            "DECODE_NUM_WORKERS": "0",
            "DECODE_TP": "16",
            "DECODE_PP_SIZE": "1",
            "DECODE_DCP_SIZE": "1",
            "DECODE_PCP_SIZE": "1",
            "DECODE_EP": "1",
            "DECODE_DP_ATTN": "false",
            "KV_OFFLOADING": "none",
            "KV_OFFLOAD_BACKEND": "",
            "SPEC_DECODING": "mtp",
            "conc-list": ["1"],
            "runner.name": "gb200-nv_01",
        },
    ]
    non_mtp_twin = {**targets[3], "SPEC_DECODING": "none"}
    disagg_twin = {**targets[3], "DISAGG": "true"}
    recipe_twin = {
        **targets[3],
        "EVAL_ARTIFACT_RECIPE": "0123456789abcdef",
    }
    suite_twin = {**targets[3], "EVAL_SUITE": "bfcl_vllm_kimi"}
    conc_twin = {**targets[3], "conc-list": ["2", "16"]}

    def render(values: dict[str, object]) -> str:
        name = expression
        defaults: dict[str, object] = {
            "DISAGG": "false",
            "EVAL_ARTIFACT_RECIPE": "",
            "EVAL_FRAMEWORK": "bfcl",
            "EVAL_SUITE": "bfcl_smoke",
        }
        defaults["EVAL_ARTIFACT_CONC"] = hashlib.sha256(
            " ".join(values["conc-list"]).encode()
        ).hexdigest()[:12]
        for key, value in {**defaults, **values}.items():
            if key != "conc-list":
                name = name.replace(f"${{{{ env.{key} }}}}", str(value))
        name = name.replace("${{ runner.name }}", str(values["runner.name"]))
        name = name.replace("${{ github.run_attempt }}", "1")
        assert "${{" not in name
        return name

    variants = [
        *targets,
        non_mtp_twin,
        disagg_twin,
        recipe_twin,
        suite_twin,
        conc_twin,
    ]
    names = [render(target) for target in variants]
    assert len(names) == len(set(names)) == len(variants)
    assert all(name.startswith("eval_") and len(name.encode()) <= 256 for name in names)


def test_single_node_eval_artifact_name_includes_suite_identity() -> None:
    workflow = yaml.safe_load(SINGLE_NODE_WORKFLOW.read_text())
    upload = next(
        step
        for step in workflow["jobs"]["benchmark"]["steps"]
        if step.get("name") == "Upload eval results (if any)"
    )
    expression = upload["with"]["name"]
    assert "EVAL_FRAMEWORK" in expression
    assert "EVAL_SUITE" in expression
    assert "github.run_attempt" in expression


_GENMODE_SCRIPT = r"""
source "$BENCHMARK_LIB" 2>/dev/null
_install_swebench_agent_deps() { :; }
_ensure_modal_credentials() { :; }
_run_swebench_agentic_generation() {
    echo "GEN=agentic"
    echo "SUITE=$EVAL_SUITE"
    return 42
}
run_lm_eval() {
    echo "GEN=single-shot"
    echo "SUITE=$EVAL_SUITE"
    return 42
}
run_swebench_eval --port 8888
echo "RC=$?"
"""


def _gen_mode(
    tmp_path: Path,
    *,
    is_agentic,
    gen_mode=None,
    eval_suite=None,
) -> str:
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "KV_OFFLOADING": "none",
        "IS_AGENTIC": is_agentic,
        "EVAL_RESULT_DIR": str(tmp_path / "out"),
    }
    env.pop("SWEBENCH_GEN_MODE", None)
    env.pop("SCENARIO_TYPE", None)
    env.pop("EVAL_SUITE", None)
    if gen_mode is not None:
        env["SWEBENCH_GEN_MODE"] = gen_mode
    if eval_suite is not None:
        env["EVAL_SUITE"] = eval_suite
    res = subprocess.run(
        ["bash", "-c", _GENMODE_SCRIPT],
        env=env,
        text=True,
        capture_output=True,
        cwd=BENCHMARK_LIB.parents[1],
    )
    assert "RC=42" in res.stdout, res.stdout + res.stderr
    return res.stdout


def test_gen_mode_defaults_to_agentic(tmp_path):
    output = _gen_mode(tmp_path, is_agentic="1")
    assert "GEN=agentic" in output
    assert "SUITE=swebench_lite" in output


def test_gen_mode_agentic_even_without_agentic_scenario(tmp_path):
    assert "GEN=agentic" in _gen_mode(tmp_path, is_agentic="0")


def test_explicit_single_shot_escape_hatch(tmp_path):
    output = _gen_mode(tmp_path, is_agentic="1", gen_mode="single-shot")
    assert "GEN=single-shot" in output
    assert "SUITE=swebench_lite" in output


def test_swebench_generation_modes_preserve_explicit_suite(tmp_path):
    for gen_mode in ("agentic", "single-shot"):
        output = _gen_mode(
            tmp_path / gen_mode,
            is_agentic="1",
            gen_mode=gen_mode,
            eval_suite="explicit_swebench",
        )
        assert "SUITE=explicit_swebench" in output


def test_agent_sandbox_cpu_knob(tmp_path):
    shim, gen_dir = _agentic_shim(
        tmp_path,
        'out=""; prev=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        'mkdir -p "$out"\n'
        'printf \'{"i1": {"instance_id": "i1", "model_patch": "d"}}\' > "$out/preds.json"\n',
    )
    res = _run_agentic(
        shim, gen_dir, {"EVAL_LIMIT": "1", "SWEBENCH_AGENT_SANDBOX_CPU": "1"}
    )
    assert "GEN_RC=0" in res.stdout, res.stdout + res.stderr
    cfg = (gen_dir / "mini_swebench_overrides.yaml").read_text()
    assert "modal_sandbox_kwargs" in cfg and "cpu: 1" in cfg, cfg

    gen_dir2 = tmp_path / "gen2"
    gen_dir2.mkdir()
    res2 = _run_agentic(shim, gen_dir2, {"EVAL_LIMIT": "1"})
    assert "GEN_RC=0" in res2.stdout, res2.stdout + res2.stderr
    cfg2 = (gen_dir2 / "mini_swebench_overrides.yaml").read_text()
    assert "modal_sandbox_kwargs" not in cfg2, cfg2


def test_eval_limit_rejects_non_positive_integer(tmp_path):
    shim, gen_dir = _agentic_shim(
        tmp_path,
        'out=""; prev=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        'mkdir -p "$out"\n'
        'printf \'{"i1": {"instance_id": "i1", "model_patch": "d"}}\' > "$out/preds.json"\n',
    )
    for bad in ("-5", "abc", "3.5"):
        gd = tmp_path / f"gen_{bad.replace('-', 'neg').replace('.', '_')}"
        gd.mkdir()
        res = _run_agentic(shim, gd, {"EVAL_LIMIT": bad})
        assert "GEN_RC=1" in res.stdout, (
            f"EVAL_LIMIT={bad!r} should fail: {res.stdout}{res.stderr}"
        )
        assert "must be a positive integer" in res.stdout + res.stderr


def test_eval_limit_full_and_zero_accepted(tmp_path):
    shim, gen_dir = _agentic_shim(
        tmp_path,
        'echo "MINI_ARGV: $*" >> ' + "ARGVLOG" + "\n"
        'out=""; prev=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        'mkdir -p "$out"\n'
        'printf \'{"i1": {"instance_id": "i1", "model_patch": "d"}}\' > "$out/preds.json"\n',
    )
    body = (shim / "mini-extra").read_text().replace("ARGVLOG", str(shim / "argv.log"))
    (shim / "mini-extra").write_text(body)
    for sentinel in ("full", "0"):
        gd = tmp_path / f"gen_{sentinel}"
        gd.mkdir()
        res = _run_agentic(shim, gd, {"EVAL_LIMIT": sentinel})
        assert "GEN_RC=0" in res.stdout, (
            f"EVAL_LIMIT={sentinel!r}: {res.stdout}{res.stderr}"
        )
    argv = (shim / "argv.log").read_text()
    assert "--slice" not in argv


def test_qwen_sglang_launchers_expose_structured_tool_calls() -> None:
    for launcher in QWEN_SGLANG_MTP_LAUNCHERS:
        command = launcher.read_text()
        assert "--reasoning-parser qwen3" in command
        assert "--tool-call-parser qwen3_coder" in command


def test_multinode_agentic_waits_for_openai_endpoint_before_requests(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    bin_dir = tmp_path / "bin"
    events_path = tmp_path / "events"
    (workspace / "benchmarks").mkdir(parents=True)
    bin_dir.mkdir()
    (workspace / "benchmarks/benchmark_lib.sh").write_text(
        """
PORT=8765
check_env_vars() { :; }
resolve_trace_source() { echo resolve >> "$EVENTS"; }
AIPERF_PYTHON=python3
install_agentic_deps() { echo deps >> "$EVENTS"; }
build_replay_cmd() { echo build >> "$EVENTS"; }
run_agentic_replay_and_write_outputs() { echo replay >> "$EVENTS"; }
""",
        encoding="utf-8",
    )
    curl = bin_dir / "curl"
    curl.write_text(
        """#!/usr/bin/env bash
printf 'curl %s\n' "$*" >> "$EVENTS"
printf '{"data":[{"id":"test-model"}]}\n'
""",
        encoding="utf-8",
    )
    curl.chmod(curl.stat().st_mode | stat.S_IXUSR)

    subprocess.run(
        ["bash", str(MULTINODE_AGENTIC_SCRIPT)],
        env={
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "INFMAX_CONTAINER_WORKSPACE": str(workspace),
            "EVENTS": str(events_path),
            "MODEL": "test-model",
            "MODEL_PREFIX": "test-prefix",
            "FRAMEWORK": "dynamo-vllm",
            "PRECISION": "fp4",
            "CONC": "1",
            "RESULT_FILENAME": "result",
            "RESULT_DIR": str(tmp_path / "results"),
            "DURATION": "1",
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert events_path.read_text().splitlines() == [
        "resolve",
        "deps",
        "curl -fsS --max-time 10 http://localhost:8765/v1/models",
        "build",
        "replay",
    ]


def test_agentic_eval_workflow_forwards_runner_contract() -> None:
    workflow = yaml.safe_load(E2E_WORKFLOW.read_text())
    forwarded = workflow["jobs"]["test-sweep-agentic-evals"]["with"]

    assert forwarded["spec-decoding"] == "${{ matrix.config.spec-decoding }}"
    assert forwarded["eval-framework"] == "${{ inputs.eval-framework }}"
    assert forwarded["eval-suite"] == "${{ inputs.eval-suite }}"
    assert forwarded["kv-offload-backend"] == (
        "${{ matrix.config['kv-offload-backend'].name }}"
    )
    assert forwarded["kv-offload-backend-metadata"] == (
        "${{ matrix.config['kv-offload-backend'] && "
        "toJson(matrix.config['kv-offload-backend']) || '' }}"
    )


def test_fixed_eval_workflows_forward_provider_contract() -> None:
    workflow = yaml.safe_load(E2E_WORKFLOW.read_text())
    for job_name in ("test-sweep-evals", "test-sweep-multi-node-evals"):
        forwarded = workflow["jobs"][job_name]["with"]
        assert forwarded["eval-framework"] == "${{ inputs.eval-framework }}"
        assert forwarded["eval-suite"] == "${{ inputs.eval-suite }}"

    reusable_workflow = yaml.safe_load(SINGLE_NODE_WORKFLOW.read_text())
    assert reusable_workflow["env"]["EVAL_FRAMEWORK"] == "${{ inputs.eval-framework }}"
    assert reusable_workflow["env"]["EVAL_SUITE"] == "${{ inputs.eval-suite }}"
    assert "*_report.json" in SINGLE_NODE_WORKFLOW.read_text()
    assert "*_results.jsonl" in SINGLE_NODE_WORKFLOW.read_text()
    assert "*_artifacts.tar.gz" in SINGLE_NODE_WORKFLOW.read_text()
    assert "bfcl_vllm_minimax_m3" in SINGLE_NODE_WORKFLOW.read_text()
    assert "bfcl_vllm_kimi" in SINGLE_NODE_WORKFLOW.read_text()


def test_multinode_agentic_eval_workflow_forwards_runner_contract() -> None:
    workflow = yaml.safe_load(E2E_WORKFLOW.read_text())
    forwarded = workflow["jobs"]["test-sweep-multi-node-agentic-evals"]["with"]
    reusable_workflow = yaml.safe_load(MULTINODE_WORKFLOW.read_text())

    assert forwarded["eval-framework"] == "${{ inputs.eval-framework }}"
    assert forwarded["eval-suite"] == "${{ inputs.eval-suite }}"
    assert reusable_workflow["env"]["EVAL_FRAMEWORK"] == "${{ inputs.eval-framework }}"
    assert reusable_workflow["env"]["EVAL_SUITE"] == "${{ inputs.eval-suite }}"
    assert "*_report.json" in MULTINODE_WORKFLOW.read_text()
    assert "*_results.jsonl" in MULTINODE_WORKFLOW.read_text()
    assert "*_artifacts.tar.gz" in MULTINODE_WORKFLOW.read_text()

    assert "bfcl_vllm_minimax_m3" in MULTINODE_WORKFLOW.read_text()
    assert "bfcl_vllm_kimi" in MULTINODE_WORKFLOW.read_text()


def test_trusted_changelog_matrix_keeps_multinode_agentic_evals() -> None:
    workflow = yaml.safe_load(E2E_WORKFLOW.read_text())
    get_jobs = next(
        step
        for step in workflow["jobs"]["get-jobs"]["steps"]
        if step.get("id") == "get-jobs"
    )
    flatten_command = next(
        line for line in get_jobs["run"].splitlines() if "rows.extend" in line
    )

    assert '"multinode_agentic_evals"' in flatten_command
    get_jobs_command = get_jobs["run"]
    assert "EVALS=$(" in get_jobs_command
    assert "score_matrix eval" in get_jobs_command


def test_env_can_force_bfcl_on_agentic_eval() -> None:
    output = _dispatch(is_agentic="1", eval_only="true", env_fw="bfcl")

    assert "DISPATCH=bfcl" in output
    assert "STAGED=summary" in output


def test_cli_can_force_bfcl_on_fixed_seqlen_eval() -> None:
    output = _dispatch(is_agentic="0", cli_fw="bfcl")

    assert "DISPATCH=bfcl" in output
    assert "STAGED=summary" not in output


def test_bfcl_defaults_suite_dispatches_once_without_context_loading() -> None:
    script = r"""
source "$BENCHMARK_LIB"
unset EVAL_MAX_MODEL_LEN
compute_eval_context_length() { echo "UNEXPECTED_CONTEXT_LOAD"; return 99; }
BFCL_DISPATCH_COUNT=0
run_bfcl_eval() {
    BFCL_DISPATCH_COUNT=$((BFCL_DISPATCH_COUNT + 1))
    printf 'DISPATCH=bfcl SUITE=%s ARGS=<%s>\n' "$EVAL_SUITE" "$*"
}
append_lm_eval_summary() { printf 'STAGED=%s\n' "$EVAL_COMPLETED_SUITE"; }
export EVAL_CONCURRENT_REQUESTS=""
export EVAL_ONLY=false
export IS_AGENTIC=0
run_eval --framework bfcl --port 9999
printf 'DISPATCH_COUNT=%s\n' "$BFCL_DISPATCH_COUNT"
printf 'COMPLETED_SUITE=%s\n' "$EVAL_COMPLETED_SUITE"
"""
    env = {**os.environ, "BENCHMARK_LIB": str(BENCHMARK_LIB), "MODEL": "served-model"}
    for key in ("EVAL_FRAMEWORK", "EVAL_SUITE", "EVAL_COMPLETED_SUITE"):
        env.pop(key, None)
    result = subprocess.run(
        ["bash", "-c", script],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "DISPATCH=bfcl SUITE=bfcl_smoke ARGS=<--port 9999>" in result.stdout
    assert "DISPATCH_COUNT=1" in result.stdout
    assert "COMPLETED_SUITE=bfcl_smoke" in result.stdout
    assert "STAGED=bfcl_smoke" not in result.stdout
    assert "UNEXPECTED_CONTEXT_LOAD" not in result.stdout


def test_bfcl_rejects_suite_from_another_provider() -> None:
    result = _run_invalid_call(
        "EVAL_CONCURRENT_REQUESTS='' "
        "EVAL_SUITE=minimax_m3_smoke "
        "run_eval --framework bfcl"
    )

    assert result.returncode == 2
    assert "unsupported BFCL suite 'minimax_m3_smoke'" in result.stderr


def test_bfcl_suite_is_rejected_by_mismatched_framework() -> None:
    result = _run_invalid_call(
        "EVAL_CONCURRENT_REQUESTS='' "
        "EVAL_SUITE=bfcl_smoke "
        "run_eval --framework minimax-vendor"
    )

    assert result.returncode == 2
    assert "unsupported MiniMax Provider Verifier suite 'bfcl_smoke'" in result.stderr


def test_bfcl_rejects_unknown_suite() -> None:
    result = _run_invalid_call("EVAL_SUITE=not_a_bfcl_suite run_bfcl_eval")

    assert result.returncode == 2
    assert "unsupported BFCL suite 'not_a_bfcl_suite'" in result.stderr


def test_bfcl_full_suite_thresholds_are_diagnostic_and_namespaced() -> None:
    thresholds = yaml.safe_load(
        (REPO_ROOT / "utils/evals/thresholds.yaml").read_text()
    )["default"]
    full_suite_tasks = (
        "bfcl_vllm_minimax_m3",
        "bfcl_vllm_minimax_m3_simple_python",
        "bfcl_vllm_minimax_m3_multiple",
        "bfcl_vllm_minimax_m3_parallel",
        "bfcl_vllm_minimax_m3_parallel_multiple",
        "bfcl_vllm_kimi",
        "bfcl_vllm_kimi_simple_python",
        "bfcl_vllm_kimi_multiple",
        "bfcl_vllm_kimi_parallel",
        "bfcl_vllm_kimi_parallel_multiple",
        "bfcl_vllm_kimi_multi_turn",
        "bfcl_vllm_kimi_multi_turn_base",
        "bfcl_vllm_kimi_multi_turn_miss_func",
        "bfcl_vllm_kimi_multi_turn_miss_param",
        "bfcl_vllm_kimi_multi_turn_long_context",
    )

    assert thresholds["bfcl_smoke"] == 0.75
    assert all(thresholds[task] == 0.0 for task in full_suite_tasks)


def test_bfcl_dependency_timeout_uses_integration_error_and_stages(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    python_dir = tmp_path / "python"
    results_dir.mkdir()
    stale_result = results_dir / "results_bfcl_previous.json"
    stale_result.write_text('{"stale": true}\n')
    script = r"""
source "$BENCHMARK_LIB"
export EVAL_SUITE=bfcl_vllm_kimi
unset VENDOR_VERIFIER_PYTHON VENDOR_VERIFIER_PYTHON_CLEANUP_DIR
selected_python() {
    printf 'ADAPTER_ARG=<%s>\n' "$@"
    touch "$RESULTS_DIR/bfcl_report.json" "$RESULTS_DIR/results_bfcl.json"
    return 1
}
_prepare_vendor_verifier_python() {
    mkdir "$PYTHON_DIR"
    VENDOR_VERIFIER_PYTHON=selected_python
    VENDOR_VERIFIER_PYTHON_CLEANUP_DIR="$PYTHON_DIR"
    export VENDOR_VERIFIER_PYTHON VENDOR_VERIFIER_PYTHON_CLEANUP_DIR
}
_prepare_bfcl_runtime() { return 124; }
python3() { echo "UNEXPECTED_SYSTEM_PYTHON"; return 99; }
append_lm_eval_summary() { printf 'STAGED=<%s>\n' "$EVAL_RESULT_DIR"; }
export EVAL_CONCURRENT_REQUESTS=""
export EVAL_ONLY=false
export IS_AGENTIC=0
eval_rc=0
run_eval --framework bfcl --results-dir "$RESULTS_DIR" || eval_rc=$?
printf 'EVAL_RC=%s\n' "$eval_rc"
exit "$eval_rc"
"""
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "RESULTS_DIR": str(results_dir),
        "PYTHON_DIR": str(python_dir),
        "MODEL": "test-model",
    }
    env.pop("EVAL_FRAMEWORK", None)
    result = subprocess.run(
        ["bash", "-c", script],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    output = result.stdout + result.stderr

    assert result.returncode == 124
    assert "EVAL_RC=124" in output
    assert f"ADAPTER_ARG=<{REPO_ROOT / 'utils/evals/bfcl_adapter.py'}>" in output
    assert "ADAPTER_ARG=<test-model>" in output
    assert f"ADAPTER_ARG=<{results_dir}>" in output
    assert "ADAPTER_ARG=<--integration-error>" in output
    assert "ADAPTER_ARG=<--suite>" in output
    assert "ADAPTER_ARG=<bfcl_vllm_kimi>" in output
    assert (
        "ADAPTER_ARG=<BFCL dependency installation failed with exit code 124>" in output
    )
    assert output.count(f"STAGED=<{results_dir}>") == 1
    assert (results_dir / "bfcl_report.json").exists()
    assert (results_dir / "results_bfcl.json").exists()
    assert not stale_result.exists()
    assert "failed to write BFCL failure artifact" not in output
    assert "UNEXPECTED_SYSTEM_PYTHON" not in output
    assert not (results_dir / "bfcl_upstream_artifacts.tar.gz").exists()
    assert not python_dir.exists()


def _run_bfcl_adapter_command(
    tmp_path: Path,
    *,
    adapter_rc: int = 0,
    suite: str = "",
    archive_rc: int = 0,
) -> tuple[subprocess.CompletedProcess[str], tuple[Path, Path, Path, Path]]:
    results_dir = tmp_path / "results"
    runtime_dir = tmp_path / "runtime"
    python_dir = tmp_path / "python"
    project_root = tmp_path / "bfcl-project"
    script = r"""
source "$BENCHMARK_LIB"
selected_python() {
    printf 'ADAPTER_ARG=<%s>\n' "$@"
    local arg
    for arg in "$@"; do
        if [ "$arg" = "--integration-error" ]; then
            touch "$RESULTS_DIR/bfcl_report.json" "$RESULTS_DIR/results_bfcl.json"
            return 1
        fi
    done
    if [ "$TEST_ADAPTER_RC" -eq 0 ]; then
        touch "$RESULTS_DIR/bfcl_report.json" "$RESULTS_DIR/results_bfcl.json"
    fi
    return "$TEST_ADAPTER_RC"
}
_prepare_vendor_verifier_python() {
    printf 'PREPARE_ARG=<%s>\n' "$@"
    mkdir "$PYTHON_DIR"
    VENDOR_VERIFIER_PYTHON=selected_python
    VENDOR_VERIFIER_PYTHON_CLEANUP_DIR="$PYTHON_DIR"
    export VENDOR_VERIFIER_PYTHON VENDOR_VERIFIER_PYTHON_CLEANUP_DIR
}
_prepare_bfcl_runtime() {
    mkdir "$RUNTIME_DIR"
    printf '%s\n' "$RUNTIME_DIR"
}
mktemp() {
    printf 'MKTEMP_ARG=<%s>\n' "$@" >&2
    mkdir "$PROJECT_ROOT"
    printf '%s\n' "$PROJECT_ROOT"
}
_archive_bfcl_upstream_artifacts() {
    printf 'ARCHIVE_PROJECT_ROOT=<%s>\n' "$1"
    printf 'ARCHIVE_PATH=<%s>\n' "$2"
    if [ "$TEST_ARCHIVE_RC" -eq 0 ]; then
        touch "$2"
    fi
    return "$TEST_ARCHIVE_RC"
}
timeout() {
    printf 'TIMEOUT_ARG=<%s>\n' "$1"
    shift
    "$@"
}
append_lm_eval_summary() { printf 'STAGED=<%s>\n' "$EVAL_RESULT_DIR"; }
if [ -n "$TEST_SUITE" ]; then
    export EVAL_SUITE="$TEST_SUITE"
else
    unset EVAL_SUITE
fi
unset EVAL_RESULT_DIR EVAL_COMPLETED_SUITE EVAL_MAX_MODEL_LEN
compute_eval_context_length() { echo "UNEXPECTED_CONTEXT_LOAD"; return 99; }
export EVAL_CONCURRENT_REQUESTS=""
export EVAL_ONLY=false
export IS_AGENTIC=0
eval_rc=0
run_eval --framework bfcl --port 9999 --results-dir "$RESULTS_DIR" || eval_rc=$?
printf 'EVAL_RC=%s\n' "$eval_rc"
printf 'EVAL_COMPLETED_SUITE=%s\n' "$EVAL_COMPLETED_SUITE"
printf 'EVAL_RESULT_DIR=%s\n' "$EVAL_RESULT_DIR"
exit "$eval_rc"
"""
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "RESULTS_DIR": str(results_dir),
        "RUNTIME_DIR": str(runtime_dir),
        "PYTHON_DIR": str(python_dir),
        "PROJECT_ROOT": str(project_root),
        "MODEL": "repository/model",
        "MODEL_NAME": "served-model",
        "OPENAI_API_KEY": "must-not-be-forwarded",
        "TEST_ADAPTER_RC": str(adapter_rc),
        "TEST_SUITE": suite,
        "TEST_ARCHIVE_RC": str(archive_rc),
    }
    for key in (
        "EVAL_FRAMEWORK",
        "EVAL_SUITE",
        "EVAL_RESULT_DIR",
        "EVAL_COMPLETED_SUITE",
        "VENDOR_VERIFIER_PYTHON",
        "VENDOR_VERIFIER_PYTHON_CLEANUP_DIR",
    ):
        env.pop(key, None)
    result = subprocess.run(
        ["bash", "-c", script],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, (results_dir, runtime_dir, python_dir, project_root)


def test_bfcl_runner_uses_fixed_adapter_contract_and_cleans_runtime(
    tmp_path: Path,
) -> None:
    result, paths = _run_bfcl_adapter_command(tmp_path)
    results_dir, runtime_dir, python_dir, project_root = paths
    output = result.stdout + result.stderr

    assert result.returncode == 0, result.stderr
    for value in (
        str(REPO_ROOT / "utils/evals/bfcl_adapter.py"),
        "--base-url",
        "http://127.0.0.1:9999/v1",
        "--api-key",
        "EMPTY",
        "--model",
        "served-model",
        "--output-dir",
        str(results_dir),
        "--bfcl-project-root",
        str(project_root),
        "--num-threads",
        "4",
    ):
        assert f"ADAPTER_ARG=<{value}>" in output
    assert "PREPARE_ARG=<BFCL>" in output
    assert "PREPARE_ARG=<bfcl-python>" in output
    assert "PREPARE_ARG=<true>" in output
    assert "PREPARE_ARG=<10>" in output
    assert "TIMEOUT_ARG=<900>" in output
    assert "ADAPTER_ARG=<--suite>" not in output
    assert "ARCHIVE_PROJECT_ROOT=" not in output
    assert not (results_dir / "bfcl_upstream_artifacts.tar.gz").exists()
    assert "ADAPTER_ARG=<must-not-be-forwarded>" not in output
    assert "UNEXPECTED_CONTEXT_LOAD" not in output
    assert f"STAGED=<{results_dir}>" not in output
    assert "EVAL_RC=0" in output
    assert "EVAL_COMPLETED_SUITE=bfcl_smoke" in output
    assert f"EVAL_RESULT_DIR={results_dir}" in output
    assert results_dir.exists()
    assert not runtime_dir.exists()
    assert not python_dir.exists()
    assert not project_root.exists()


def test_bfcl_full_suites_use_suite_specific_runtime_and_archive_before_cleanup(
    tmp_path: Path,
) -> None:
    suite_contracts = (
        ("bfcl_vllm_minimax_m3", "8"),
        ("bfcl_vllm_kimi", "16"),
    )

    for suite, expected_threads in suite_contracts:
        suite_tmp_path = tmp_path / suite
        suite_tmp_path.mkdir()
        result, paths = _run_bfcl_adapter_command(
            suite_tmp_path,
            suite=suite,
        )
        results_dir, runtime_dir, python_dir, project_root = paths
        output = result.stdout + result.stderr

        assert result.returncode == 0, result.stderr
        assert "TIMEOUT_ARG=<7200>" in output
        assert "ADAPTER_ARG=<--suite>" in output
        assert f"ADAPTER_ARG=<{suite}>" in output
        assert f"EVAL_COMPLETED_SUITE={suite}" in output
        assert "ADAPTER_ARG=<--num-threads>" in output
        assert f"ADAPTER_ARG=<{expected_threads}>" in output
        assert f"ARCHIVE_PROJECT_ROOT=<{project_root}>" in output
        assert (
            f"ARCHIVE_PATH=<{results_dir / 'bfcl_upstream_artifacts.tar.gz'}>" in output
        )
        assert (results_dir / "bfcl_upstream_artifacts.tar.gz").exists()
        assert not runtime_dir.exists()
        assert not python_dir.exists()
        assert not project_root.exists()


def test_bfcl_full_suite_archive_failure_preserves_scores_and_cleans_runtime(
    tmp_path: Path,
) -> None:
    result, paths = _run_bfcl_adapter_command(
        tmp_path,
        suite="bfcl_vllm_kimi",
        archive_rc=73,
    )
    results_dir, runtime_dir, python_dir, project_root = paths
    output = result.stdout + result.stderr

    assert result.returncode == 73
    assert "failed to archive BFCL upstream artifacts (exit code 73)" in output
    assert (results_dir / "bfcl_report.json").exists()
    assert (results_dir / "results_bfcl.json").exists()
    assert not (results_dir / "bfcl_upstream_artifacts.tar.gz").exists()
    assert not runtime_dir.exists()
    assert not python_dir.exists()
    assert not project_root.exists()


def test_bfcl_adapter_timeout_writes_reports_stages_and_propagates(
    tmp_path: Path,
) -> None:
    result, paths = _run_bfcl_adapter_command(tmp_path, adapter_rc=124)
    results_dir, runtime_dir, python_dir, project_root = paths
    output = result.stdout + result.stderr

    assert result.returncode == 124
    assert "EVAL_RC=124" in output
    assert output.count(f"STAGED=<{results_dir}>") == 1
    assert "ADAPTER_ARG=<--integration-error>" in output
    assert "ADAPTER_ARG=<--suite>" in output
    assert "ADAPTER_ARG=<bfcl_smoke>" in output
    assert "ADAPTER_ARG=<BFCL evaluation failed with exit code 124>" in output
    assert (results_dir / "bfcl_report.json").exists()
    assert (results_dir / "results_bfcl.json").exists()
    assert "failed to write BFCL failure artifact" not in output
    assert "run_eval failed with exit code 124" in result.stderr
    assert not runtime_dir.exists()
    assert not python_dir.exists()
    assert not project_root.exists()


def test_bfcl_upstream_archive_is_deterministic_and_survives_cleanup(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    result_path = project_root / "result/run/BFCL_v4_simple_python_result.json"
    score_path = project_root / "score/run/BFCL_v4_simple_python_score.json"
    id_path = project_root / "test_case_ids_to_generate.json"
    for path, content in (
        (result_path, '{"id":"simple_python_0"}\n'),
        (score_path, '{"accuracy":1.0}\n'),
        (id_path, '{"simple_python":["simple_python_0"]}\n'),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    first_archive = tmp_path / "first.tar.gz"
    second_archive = tmp_path / "second.tar.gz"
    script = r"""
source "$BENCHMARK_LIB"
VENDOR_VERIFIER_PYTHON="$PYTHON"
_archive_bfcl_upstream_artifacts "$PROJECT_ROOT" "$FIRST_ARCHIVE"
_archive_bfcl_upstream_artifacts "$PROJECT_ROOT" "$SECOND_ARCHIVE"
_cleanup_vendor_eval "$PROJECT_ROOT"
"""
    subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "BENCHMARK_LIB": str(BENCHMARK_LIB),
            "PYTHON": sys.executable,
            "PROJECT_ROOT": str(project_root),
            "FIRST_ARCHIVE": str(first_archive),
            "SECOND_ARCHIVE": str(second_archive),
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert first_archive.read_bytes() == second_archive.read_bytes()
    with tarfile.open(first_archive, "r:gz") as archive:
        assert archive.getnames() == [
            "result",
            "result/run",
            "result/run/BFCL_v4_simple_python_result.json",
            "score",
            "score/run",
            "score/run/BFCL_v4_simple_python_score.json",
            "test_case_ids_to_generate.json",
        ]
    assert not project_root.exists()


def test_bfcl_upstream_archive_rejects_symbolic_links(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text('{"secret":true}\n')
    (project_root / "escape.json").symlink_to(outside)
    archive = tmp_path / "unsafe.tar.gz"
    script = r"""
source "$BENCHMARK_LIB"
VENDOR_VERIFIER_PYTHON="$PYTHON"
_archive_bfcl_upstream_artifacts "$PROJECT_ROOT" "$ARCHIVE"
"""

    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "BENCHMARK_LIB": str(BENCHMARK_LIB),
            "PYTHON": sys.executable,
            "PROJECT_ROOT": str(project_root),
            "ARCHIVE": str(archive),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "refusing to archive symbolic link: escape.json" in result.stderr
    assert not archive.exists()
    assert not (tmp_path / ".unsafe.tar.gz.tmp").exists()


def test_bfcl_installer_uses_verified_wheel_in_selected_venv(tmp_path: Path) -> None:
    script = r"""
source "$BENCHMARK_LIB"
selected_python() { printf 'PYTHON_ARG=<%s>\n' "$@"; }
timeout() {
    printf 'TIMEOUT_ARG=<%s>\n' "$1"
    shift
    "$@"
}
VENDOR_VERIFIER_PYTHON=selected_python
_install_bfcl_eval_deps "$DOWNLOAD_DIR"
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "BENCHMARK_LIB": str(BENCHMARK_LIB),
            "DOWNLOAD_DIR": str(tmp_path),
        },
        text=True,
        capture_output=True,
        check=True,
    )
    wheel_path = tmp_path / "bfcl_eval-2026.3.23-py3-none-any.whl"

    for value in (
        "https://files.pythonhosted.org/packages/ba/41/"
        "ed458527c770c50225b60bae3b0c3444b26804ee455fa2d8f187018d2cb2/"
        "bfcl_eval-2026.3.23-py3-none-any.whl",
        "3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a",
        str(wheel_path),
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "soundfile==0.13.1",
    ):
        assert f"PYTHON_ARG=<{value}>" in result.stdout
    assert "TIMEOUT_ARG=<600>" in result.stdout
    assert "--break-system-packages" not in result.stdout
    assert "--target" not in result.stdout


def test_bfcl_python_preparation_exposes_system_site_packages(
    tmp_path: Path,
) -> None:
    python_root = tmp_path / "bfcl-python"
    script = r"""
source "$BENCHMARK_LIB"
python3() {
    if [ "$1" = "-c" ]; then
        printf 'VERSION_CHECK_ARG=<%s>\n' "$@"
        return 0
    fi
    printf 'SYSTEM_PYTHON_ARG=<%s>\n' "$@"
    venv_dir="${!#}"
    mkdir -p "$venv_dir/bin"
    printf '#!/usr/bin/env bash\n' > "$venv_dir/bin/python"
    chmod +x "$venv_dir/bin/python"
}
mktemp() {
    mkdir "$PYTHON_ROOT"
    printf '%s\n' "$PYTHON_ROOT"
}
_prepare_vendor_verifier_python "BFCL" "bfcl-python" true 10
cleanup_dir="$VENDOR_VERIFIER_PYTHON_CLEANUP_DIR"
printf 'SELECTED_PYTHON=<%s>\n' "$VENDOR_VERIFIER_PYTHON"
_cleanup_vendor_eval "$cleanup_dir"
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "BENCHMARK_LIB": str(BENCHMARK_LIB),
            "PYTHON_ROOT": str(python_root),
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert "SYSTEM_PYTHON_ARG=<-m>" in result.stdout
    assert "SYSTEM_PYTHON_ARG=<venv>" in result.stdout
    assert "SYSTEM_PYTHON_ARG=<--system-site-packages>" in result.stdout
    assert "VERSION_CHECK_ARG=<10>" in result.stdout
    assert f"SELECTED_PYTHON=<{python_root / 'venv/bin/python'}>" in result.stdout
    assert "--prefix" not in result.stdout
    assert not python_root.exists()
