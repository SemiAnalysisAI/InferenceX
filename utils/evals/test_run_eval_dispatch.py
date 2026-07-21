
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

BENCHMARK_LIB = Path(__file__).resolve().parents[2] / "benchmarks" / "benchmark_lib.sh"

_SCRIPT = r'''
source "$BENCHMARK_LIB"
run_lm_eval()       { echo "DISPATCH=lm-eval"; }
run_swebench_eval() { echo "DISPATCH=swebench"; }
append_lm_eval_summary() { echo "STAGED=summary"; }
export EVAL_MAX_MODEL_LEN=16384
unset EVAL_CONCURRENT_REQUESTS
run_eval ${CLI_FW:+--framework "$CLI_FW"} --port 8888
'''


def _dispatch(*, is_agentic: str = "0", eval_only: str = "false", cli_fw=None, env_fw=None) -> str:
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



def test_agentic_scenario_defaults_to_swebench():
    out = _dispatch(is_agentic="1")
    assert "DISPATCH=swebench" in out
    assert "DISPATCH=lm-eval" not in out


def test_fixed_seqlen_scenario_defaults_to_lm_eval():
    assert "DISPATCH=lm-eval" in _dispatch(is_agentic="0")

def test_agentic_eval_only_stages_summary():
    assert "STAGED=summary" in _dispatch(is_agentic="1", eval_only="true")


def test_fixed_seqlen_eval_only_leaves_staging_to_recipe():
    assert "STAGED=summary" not in _dispatch(is_agentic="0", eval_only="true")



def test_explicit_framework_arg_overrides_scenario():
    assert "DISPATCH=lm-eval" in _dispatch(is_agentic="1", cli_fw="lm-eval")


def test_env_framework_overrides_scenario():
    assert "DISPATCH=lm-eval" in _dispatch(is_agentic="1", env_fw="lm-eval")


def test_fixed_seqlen_rejects_swebench():
    with pytest.raises(subprocess.CalledProcessError):
        _dispatch(is_agentic="0", env_fw="swebench")


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


def test_default_graded_eval_suite_is_registered():
    out = _run_lm_eval_with_include_path()
    assert "--include_path utils/evals" in out, (
        f"Expected repo-local eval tasks to be registered:\n{out}"
    )
    assert "--tasks gsm8k gpqa_diamond_cot_n_shot aime26" in out, (
        f"Expected all graded eval tasks in the default suite:\n{out}"
    )


def test_eval_workflow_separates_agentic_lm_eval_and_swebench():
    repo_root = BENCHMARK_LIB.parents[1]
    e2e_workflow = (repo_root / ".github/workflows/e2e-tests.yml").read_text()
    benchmark_workflow = (
        repo_root / ".github/workflows/benchmark-tmpl.yml"
    ).read_text()

    assert "test-sweep-agentic-lm-evals:" in e2e_workflow
    assert "eval-framework: lm-eval" in e2e_workflow
    assert "eval-framework: swebench" in e2e_workflow
    assert "eval-tasks: ${{ inputs.eval-tasks }}" in e2e_workflow
    assert "inputs.agentic-eval-mode != 'lm-eval'" in e2e_workflow
    assert "inputs.agentic-eval-mode != 'swebench'" in e2e_workflow
    assert "EVAL_FRAMEWORK: ${{ inputs.eval-framework }}" in benchmark_workflow
    assert "EVAL_TASKS_DIR: ${{ inputs.eval-tasks }}" in benchmark_workflow
    assert "EVAL_ARTIFACT_SUFFIX:" in benchmark_workflow


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
_run_swebench_agentic_generation() { echo "GEN=agentic"; return 42; }
run_lm_eval() { echo "GEN=single-shot"; return 42; }
run_swebench_eval --port 8888
echo "RC=$?"
'''


def _gen_mode(tmp_path, *, is_agentic, gen_mode=None) -> str:
    env = {**os.environ,
           "BENCHMARK_LIB": str(BENCHMARK_LIB),
           "KV_OFFLOADING": "none",
           "IS_AGENTIC": is_agentic,
           "EVAL_RESULT_DIR": str(tmp_path / "out")}
    env.pop("SWEBENCH_GEN_MODE", None)
    env.pop("SCENARIO_TYPE", None)
    if gen_mode is not None:
        env["SWEBENCH_GEN_MODE"] = gen_mode
    res = subprocess.run(["bash", "-c", _GENMODE_SCRIPT], env=env,
                         text=True, capture_output=True,
                         cwd=BENCHMARK_LIB.parents[1])
    assert "RC=42" in res.stdout, res.stdout + res.stderr
    return res.stdout


def test_gen_mode_defaults_to_agentic(tmp_path):
    assert "GEN=agentic" in _gen_mode(tmp_path, is_agentic="1")


def test_gen_mode_agentic_even_without_agentic_scenario(tmp_path):
    assert "GEN=agentic" in _gen_mode(tmp_path, is_agentic="0")


def test_explicit_single_shot_escape_hatch(tmp_path):
    assert "GEN=single-shot" in _gen_mode(tmp_path, is_agentic="1", gen_mode="single-shot")


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
