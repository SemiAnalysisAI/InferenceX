
from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_LIB = REPO_ROOT / "benchmarks" / "benchmark_lib.sh"
MULTINODE_WORKFLOW = REPO_ROOT / ".github/workflows/benchmark-multinode-tmpl.yml"
E2E_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "e2e-tests.yml"

_SCRIPT = r'''
source "$BENCHMARK_LIB"
run_lm_eval()       { echo "DISPATCH=lm-eval"; }
run_swebench_eval() { echo "DISPATCH=swebench"; }
run_kimi_vendor_eval() { echo "DISPATCH=kimi-vendor"; }
append_lm_eval_summary() { echo "STAGED=summary"; }
export EVAL_MAX_MODEL_LEN=16384
export EVAL_CONCURRENT_REQUESTS=""
run_eval ${CLI_FW:+--framework "$CLI_FW"} --port 8888
'''


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



def test_explicit_framework_arg_overrides_scenario():
    assert "DISPATCH=lm-eval" in _dispatch(is_agentic="1", cli_fw="lm-eval")


def test_env_framework_overrides_scenario():
    assert "DISPATCH=lm-eval" in _dispatch(is_agentic="1", env_fw="lm-eval")


def test_env_can_force_swebench_on_fixed_seqlen():
    assert "DISPATCH=swebench" in _dispatch(is_agentic="0", env_fw="swebench")




def test_env_can_force_kimi_vendor_on_agentic_eval() -> None:
    assert "DISPATCH=kimi-vendor" in _dispatch(
        is_agentic="1",
        eval_only="true",
        env_fw="kimi-vendor",
    )


def test_kimi_vendor_skips_unused_model_context_loading() -> None:
    script = r'''
source "$BENCHMARK_LIB"
unset EVAL_MAX_MODEL_LEN
compute_eval_context_length() { echo "UNEXPECTED_CONTEXT_LOAD"; return 99; }
run_kimi_vendor_eval() { echo "DISPATCH=kimi-vendor"; }
export EVAL_FRAMEWORK=kimi-vendor
export EVAL_CONCURRENT_REQUESTS=""
export EVAL_ONLY=false
export IS_AGENTIC=0
run_eval --port 8888
'''
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
    script = r'''
set -u
source "$BENCHMARK_LIB"
run_kimi_vendor_eval() { return 7; }
export EVAL_FRAMEWORK=kimi-vendor
export EVAL_CONCURRENT_REQUESTS=""
export EVAL_MAX_MODEL_LEN=16384
export IS_AGENTIC=0
unset EVAL_ONLY
run_eval --port 8888
'''
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
    result = _run_invalid_call(
        "EVAL_SUITE=gpqa_diamond run_eval --framework lm-eval"
    )

    assert result.returncode == 2
    assert "only supported with EVAL_FRAMEWORK=kimi-vendor" in result.stderr


def test_run_eval_scopes_runner_selected_suite_to_one_call() -> None:
    script = r'''
source "$BENCHMARK_LIB"
run_kimi_vendor_eval() {
    export EVAL_SUITE=kimi_tool_call_schema
    echo "DISPATCH=kimi-vendor SUITE=$EVAL_SUITE"
}
run_lm_eval() { echo "DISPATCH=lm-eval SUITE=${EVAL_SUITE:-unset}"; }
export EVAL_MAX_MODEL_LEN=16384
export EVAL_CONCURRENT_REQUESTS=""
export EVAL_ONLY=false
export IS_AGENTIC=0
unset EVAL_SUITE
export EVAL_FRAMEWORK=kimi-vendor
run_eval --port 8888
export EVAL_FRAMEWORK=lm-eval
run_eval --port 8888
printf 'FINAL_SUITE=%s\n' "${EVAL_SUITE-unset}"
'''
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "BENCHMARK_LIB": str(BENCHMARK_LIB)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "DISPATCH=kimi-vendor SUITE=kimi_tool_call_schema" in result.stdout
    assert "DISPATCH=lm-eval SUITE=unset" in result.stdout
    assert "FINAL_SUITE=unset" in result.stdout


def test_kimi_default_suite_reaches_eval_only_metadata() -> None:
    script = r'''
source "$BENCHMARK_LIB"
run_kimi_vendor_eval() { echo "DISPATCH=$EVAL_SUITE"; }
append_lm_eval_summary() { echo "METADATA=$EVAL_SUITE"; }
export EVAL_FRAMEWORK=kimi-vendor
export EVAL_ONLY=true
export IS_AGENTIC=1
export EVAL_CONCURRENT_REQUESTS=""
unset EVAL_SUITE
run_eval --port 8888
'''
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


def test_kimi_vendor_rejects_batched_concurrency() -> None:
    result = _run_invalid_call(
        "EVAL_MAX_MODEL_LEN=16384 "
        "EVAL_CONCURRENT_REQUESTS='1 4' "
        "run_eval --framework kimi-vendor"
    )
    assert result.returncode == 1
    assert "batched eval concurrency is only supported for lm-eval" in result.stderr


def test_kimi_vendor_rejects_unsupported_suite() -> None:
    result = _run_invalid_call(
        "EVAL_SUITE=gsm8k run_kimi_vendor_eval"
    )
    assert result.returncode == 2
    assert "unsupported Kimi Vendor Verifier suite 'gsm8k'" in result.stderr



def test_kimi_vendor_setup_failure_writes_compatibility_result(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    script = r'''
source "$BENCHMARK_LIB"
_require_kimi_vendor_python() { :; }
_prepare_kimi_vendor_runtime() { return 12; }
run_kimi_vendor_eval --results-dir "$RESULTS_DIR"
printf 'SETUP_RC=%s\n' "$?"
'''
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "RESULTS_DIR": str(results_dir),
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
    assert message in result.stderr
    assert "failed to write Kimi verifier failure artifact" not in result.stderr
    assert len(score_files) == 1
    score_result = json.loads(score_files[0].read_text())
    assert (
        score_result["results"]["kimi_tool_call_schema"][
            "exact_match,strict-match"
        ]
        == 0.0
    )
    assert score_result["integration_error"]["message"] == message
    assert not (results_dir / "kimi_vendor_report.json").exists()


def test_kimi_vendor_dependency_install_is_isolated(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    script = r'''
source "$BENCHMARK_LIB"
python3() { printf 'PYTHON_ARG=<%s>\n' "$@"; }
_install_kimi_vendor_eval_deps "$RUNTIME_DIR"
'''
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
    assert "--break-system-packages" not in result.stdout


def test_kimi_vendor_surfaces_failure_artifact_error(tmp_path: Path) -> None:
    script = r'''
source "$BENCHMARK_LIB"
_require_kimi_vendor_python() { return 12; }
_write_kimi_vendor_integration_error() { return 23; }
run_kimi_vendor_eval --results-dir "$RESULTS_DIR"
printf 'EVAL_RC=%s\n' "$?"
'''
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
    runtime_dir.mkdir()
    verifier_dir.mkdir()
    script = r'''
source "$BENCHMARK_LIB"
_require_kimi_vendor_python() { :; }
_prepare_kimi_vendor_runtime() { printf '%s\n' "$RUNTIME_DIR"; }
_prepare_kimi_vendor_verifier() {
    printf 'CHECKOUT=%s@%s\n' "$1" "$2" >&2
    printf '%s\n' "$VERIFIER_DIR"
}
python3() {
    printf 'PYTHONPATH=<%s>\n' "$PYTHONPATH"
    printf 'PYTHON_ARG=<%s>\n' "$@"
}
run_kimi_vendor_eval --port 9999 --results-dir "$RESULTS_DIR"
printf 'EVAL_SUITE=%s\n' "$EVAL_SUITE"
printf 'EVAL_RESULT_DIR=%s\n' "$EVAL_RESULT_DIR"
'''
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "RESULTS_DIR": str(results_dir),
        "VERIFIER_DIR": str(verifier_dir),
        "MODEL": "test-model",
        "RUNTIME_DIR": str(runtime_dir),
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
    for value in (
        adapter,
        verifier_dir,
        "http://127.0.0.1:9999/v1",
        "EMPTY",
        "test-model",
        results_dir,
    ):
        assert f"PYTHON_ARG=<{value}>" in output
    assert "must-not-be-forwarded" not in output
    assert "EVAL_SUITE=kimi_tool_call_schema" in output
    assert f"EVAL_RESULT_DIR={results_dir}" in output
    assert not (tmp_path / "runtime").exists()
    assert not verifier_dir.exists()


def test_run_lm_eval_rejects_missing_option_value():
    result = _run_invalid_call("run_lm_eval --port")
    assert result.returncode == 2
    assert "--port requires a value" in result.stderr


def test_lm_patch_copy_resolves_outside_repo(tmp_path):
    script = r'''
source "$BENCHMARK_LIB"
cd "$OTHER_CWD"
_patch_lm_eval
patch_dir=${PYTHONPATH%%:*}
cmp "$(_eval_patches_dir)/lm_eval_sitecustomize.py" "$patch_dir/sitecustomize.py"
'''
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "OTHER_CWD": str(tmp_path),
        "KV_OFFLOADING": "none",
    }
    subprocess.run(["bash", "-c", script], env=env, check=True)



_EVAL_LIMIT_SCRIPT = r'''
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
'''


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
    script = r'''
source "$BENCHMARK_LIB"
cd "$WORK_DIR"
append_lm_eval_summary >/dev/null
'''
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "WORK_DIR": str(work_dir),
        "EVAL_RESULT_DIR": str(results_dir),
        "MODEL": "test-model",
        "CONC": "7",
        "KV_OFFLOADING": "none",
    }
    for key in ("EVAL_SUITE", "EVAL_TASKS_DIR"):
        env.pop(key, None)
    env.update(overrides)
    subprocess.run(["bash", "-c", script], env=env, check=True)
    return json.loads((work_dir / "meta_env.json").read_text())


def test_summary_metadata_preserves_lm_eval_gsm8k_defaults(tmp_path: Path) -> None:
    meta = _summary_metadata(tmp_path)

    assert meta["eval_suite"] == "gsm8k"
    assert meta["conc"] == 7


def test_run_lm_eval_exports_cli_task_path(tmp_path: Path) -> None:
    script = r'''
source "$BENCHMARK_LIB"
python3() { :; }
export EVAL_MAX_MODEL_LEN=16384
export INFERENCEX_LM_EVAL_RUNTIME_READY=true
run_lm_eval --task custom.yaml --results-dir "$RESULTS_DIR"
printf 'EVAL_TASKS_DIR=%s\n' "$EVAL_TASKS_DIR"
'''
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


def test_env_is_true_is_case_insensitive_and_unset_safe() -> None:
    script = r'''
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
'''
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



_MODAL_CREDS_SCRIPT = r'''
source "$BENCHMARK_LIB"
_ensure_modal_credentials
echo "HOME_AFTER=$HOME"
if [ -f "$HOME/.modal.toml" ]; then
    echo "TOML_EXISTS=true"
    PERMS=$(stat -c '%a' "$HOME/.modal.toml" 2>/dev/null || stat -f '%A' "$HOME/.modal.toml" 2>/dev/null)
    echo "TOML_PERMS=$PERMS"
fi
'''


def _run_modal_creds(tmp_path: Path, *, home: str, token_id="tok-id", token_secret="tok-secret") -> str:
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
        assert "HOME_AFTER=/tmp/inferencex-modal-home" in out, f"Expected HOME remap:\n{out}"
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
        assert "HOME_AFTER=/tmp/inferencex-modal-home" in out, f"Expected HOME remap:\n{out}"
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



_INCLUDE_PATH_SCRIPT = r'''
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
'''


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
    script = r'''
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
'''
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
    res = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True)
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
        "printf '{\"i1\": {\"instance_id\": \"i1\", \"model_name_or_path\": \"m\", \"model_patch\": \"d\"}}' > \"$out/preds.json\"\n"
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
    env = {**os.environ,
           "BENCHMARK_LIB": str(BENCHMARK_LIB),
           "GEN_DIR": str(gen_dir),
           "PATH": f"{shim}:{os.environ['PATH']}"}
    res = subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True)
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
    env = {**os.environ,
           "BENCHMARK_LIB": str(BENCHMARK_LIB),
           "GEN_DIR": str(gen_dir),
           "MODEL_NAME": "test-model",
           "SWEBENCH_SANDBOX_SWEEP": "0",
           "SWEBENCH_WATCHDOG_POLL": "1",
           "PATH": f"{shim}:{os.environ['PATH']}",
           **(extra_env or {})}
    return subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True)


def test_agentic_watchdog_kills_hung_mini(tmp_path):
    shim, gen_dir = _agentic_shim(tmp_path,
        'out=""; prev=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        'mkdir -p "$out"\n'
        "printf '{\"i1\": {\"instance_id\": \"i1\", \"model_patch\": \"d\"}}' > \"$out/preds.json\"\n"
        "exec sleep 600 </dev/null >/dev/null 2>&1\n"
    )
    res = _run_agentic(shim, gen_dir, {"EVAL_LIMIT": "1", "SWEBENCH_AGENT_EXIT_GRACE": "2"})
    assert "GEN_RC=0" in res.stdout, res.stdout + res.stderr
    assert "hung after completing all instances" in res.stdout + res.stderr


def test_agentic_salvage_partial_preds_on_failure(tmp_path):
    shim, gen_dir = _agentic_shim(tmp_path,
        'out=""; prev=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        'mkdir -p "$out"\n'
        "printf '{\"i1\": {\"instance_id\": \"i1\", \"model_patch\": \"d\"}}' > \"$out/preds.json\"\n"
        "exit 7\n"
    )
    res = _run_agentic(shim, gen_dir, {"EVAL_LIMIT": "2"})
    assert "GEN_RC=0" in res.stdout, res.stdout + res.stderr
    assert "scoring the partial set" in res.stdout + res.stderr


def test_agentic_no_preds_still_fails(tmp_path):
    shim, gen_dir = _agentic_shim(tmp_path, "exit 7\n")
    res = _run_agentic(shim, gen_dir, {"EVAL_LIMIT": "2"})
    assert "GEN_RC=7" in res.stdout, res.stdout + res.stderr


def test_agentic_eval_limit_defaults_to_full_split(tmp_path):
    shim, gen_dir = _agentic_shim(tmp_path,
        'echo "MINI_ARGV: $*" >> ' + "ARGVLOG" + '\n'
        'out=""; prev=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        'mkdir -p "$out"\n'
        "printf '{\"i1\": {\"instance_id\": \"i1\", \"model_patch\": \"d\"}}' > \"$out/preds.json\"\n"
    )
    body = (shim / "mini-extra").read_text().replace("ARGVLOG", str(shim / "argv.log"))
    (shim / "mini-extra").write_text(body)
    res = _run_agentic(shim, gen_dir)
    argv = (shim / "argv.log").read_text()
    assert "--slice" not in argv, argv
    assert "GEN_RC=0" in res.stdout, res.stdout + res.stderr


def test_agentic_eval_limit_full_runs_whole_split(tmp_path):
    shim, gen_dir = _agentic_shim(tmp_path,
        'echo "MINI_ARGV: $*" >> ' + "ARGVLOG" + '\n'
        'out=""; prev=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        'mkdir -p "$out"\n'
        "printf '{\"i1\": {\"instance_id\": \"i1\", \"model_patch\": \"d\"}}' > \"$out/preds.json\"\n"
    )
    body = (shim / "mini-extra").read_text().replace("ARGVLOG", str(shim / "argv.log"))
    (shim / "mini-extra").write_text(body)
    res = _run_agentic(shim, gen_dir, {"EVAL_LIMIT": "full"})
    argv = (shim / "argv.log").read_text()
    assert "--slice" not in argv, argv
    assert "GEN_RC=0" in res.stdout, res.stdout + res.stderr



_GENMODE_SCRIPT = r'''
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
'''


def _gen_mode(
    tmp_path: Path,
    *,
    is_agentic,
    gen_mode=None,
    eval_suite=None,
) -> str:
    env = {**os.environ,
           "BENCHMARK_LIB": str(BENCHMARK_LIB),
           "KV_OFFLOADING": "none",
           "IS_AGENTIC": is_agentic,
           "EVAL_RESULT_DIR": str(tmp_path / "out")}
    env.pop("SWEBENCH_GEN_MODE", None)
    env.pop("SCENARIO_TYPE", None)
    env.pop("EVAL_SUITE", None)
    if gen_mode is not None:
        env["SWEBENCH_GEN_MODE"] = gen_mode
    if eval_suite is not None:
        env["EVAL_SUITE"] = eval_suite
    res = subprocess.run(["bash", "-c", _GENMODE_SCRIPT], env=env,
                         text=True, capture_output=True,
                         cwd=BENCHMARK_LIB.parents[1])
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
    shim, gen_dir = _agentic_shim(tmp_path,
        'out=""; prev=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        'mkdir -p "$out"\n'
        "printf '{\"i1\": {\"instance_id\": \"i1\", \"model_patch\": \"d\"}}' > \"$out/preds.json\"\n"
    )
    res = _run_agentic(shim, gen_dir, {"EVAL_LIMIT": "1", "SWEBENCH_AGENT_SANDBOX_CPU": "1"})
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
    shim, gen_dir = _agentic_shim(tmp_path,
        'out=""; prev=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        'mkdir -p "$out"\n'
        "printf '{\"i1\": {\"instance_id\": \"i1\", \"model_patch\": \"d\"}}' > \"$out/preds.json\"\n"
    )
    for bad in ("-5", "abc", "3.5"):
        gd = tmp_path / f"gen_{bad.replace('-','neg').replace('.','_')}"
        gd.mkdir()
        res = _run_agentic(shim, gd, {"EVAL_LIMIT": bad})
        assert "GEN_RC=1" in res.stdout, f"EVAL_LIMIT={bad!r} should fail: {res.stdout}{res.stderr}"
        assert "must be a positive integer" in res.stdout + res.stderr


def test_eval_limit_full_and_zero_accepted(tmp_path):
    shim, gen_dir = _agentic_shim(tmp_path,
        'echo "MINI_ARGV: $*" >> ' + "ARGVLOG" + '\n'
        'out=""; prev=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        'mkdir -p "$out"\n'
        "printf '{\"i1\": {\"instance_id\": \"i1\", \"model_patch\": \"d\"}}' > \"$out/preds.json\"\n"
    )
    body = (shim / "mini-extra").read_text().replace("ARGVLOG", str(shim / "argv.log"))
    (shim / "mini-extra").write_text(body)
    for sentinel in ("full", "0"):
        gd = tmp_path / f"gen_{sentinel}"
        gd.mkdir()
        res = _run_agentic(shim, gd, {"EVAL_LIMIT": sentinel})
        assert "GEN_RC=0" in res.stdout, f"EVAL_LIMIT={sentinel!r}: {res.stdout}{res.stderr}"
    argv = (shim / "argv.log").read_text()
    assert "--slice" not in argv


def test_agentic_eval_workflow_forwards_runner_contract() -> None:
    workflow = yaml.safe_load(E2E_WORKFLOW.read_text())
    forwarded = workflow["jobs"]["test-sweep-agentic-evals"]["with"]

    assert forwarded["spec-decoding"] == "${{ matrix.config.spec-decoding }}"
    assert forwarded["eval-framework"] == "${{ inputs.eval-framework }}"
    assert forwarded["eval-suite"] == "${{ inputs.eval-suite }}"


def test_multinode_agentic_eval_workflow_forwards_runner_contract() -> None:
    workflow = yaml.safe_load(E2E_WORKFLOW.read_text())
    forwarded = workflow["jobs"]["test-sweep-multi-node-agentic-evals"]["with"]
    reusable_workflow = yaml.safe_load(MULTINODE_WORKFLOW.read_text())

    assert forwarded["eval-framework"] == "${{ inputs.eval-framework }}"
    assert forwarded["eval-suite"] == "${{ inputs.eval-suite }}"
    assert reusable_workflow["env"]["EVAL_FRAMEWORK"] == "${{ inputs.eval-framework }}"
    assert reusable_workflow["env"]["EVAL_SUITE"] == "${{ inputs.eval-suite }}"
    assert "*_vendor_report.json" in MULTINODE_WORKFLOW.read_text()



def test_trusted_changelog_matrix_keeps_multinode_agentic_evals() -> None:
    workflow = yaml.safe_load(E2E_WORKFLOW.read_text())
    get_jobs = next(
        step
        for step in workflow["jobs"]["get-jobs"]["steps"]
        if step.get("id") == "get-jobs"
    )
    flatten_command = next(
        line
        for line in get_jobs["run"].splitlines()
        if "rows.extend" in line
    )

    assert '"multinode_agentic_evals"' in flatten_command
    get_jobs_command = get_jobs["run"]
    assert "EVALS=$(" in get_jobs_command
    assert "score_matrix eval" in get_jobs_command