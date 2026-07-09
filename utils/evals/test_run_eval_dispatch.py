"""run_eval framework dispatch.

Scenario picks the default framework (agentic-coding -> swebench, fixed-seq-len
-> lm-eval); an explicit EVAL_FRAMEWORK env or --framework arg overrides it.
"""

import os
import stat
import subprocess
import tempfile
from pathlib import Path

BENCHMARK_LIB = Path(__file__).resolve().parents[2] / "benchmarks" / "benchmark_lib.sh"

# Stub the framework runners so dispatch is observable without a server/Docker,
# and pin EVAL_MAX_MODEL_LEN so run_eval skips context computation. CLI_FW is
# only forwarded as --framework when set (so we can test the no-arg path).
_SCRIPT = r'''
source "$BENCHMARK_LIB"
run_lm_eval()       { echo "DISPATCH=lm-eval"; }
run_swebench_eval() { echo "DISPATCH=swebench"; }
export EVAL_MAX_MODEL_LEN=16384
unset EVAL_CONCURRENT_REQUESTS
run_eval ${CLI_FW:+--framework "$CLI_FW"} --port 8888
'''


def _dispatch(*, is_agentic: str = "0", cli_fw=None, env_fw=None) -> str:
    # AgentX v1.0 added a source-time guard in benchmark_lib.sh that requires
    # KV_OFFLOADING to be set whenever the scenario is agentic (IS_AGENTIC=1 /
    # SCENARIO_TYPE=agentic-coding). KV_OFFLOADING=none satisfies it without
    # affecting framework dispatch, which only reads scenario + framework knobs.
    env = {
        **os.environ,
        "BENCHMARK_LIB": str(BENCHMARK_LIB),
        "IS_AGENTIC": is_agentic,
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


# --- scenario default ------------------------------------------------------

def test_agentic_scenario_defaults_to_swebench():
    assert "DISPATCH=swebench" in _dispatch(is_agentic="1")


def test_fixed_seqlen_scenario_defaults_to_lm_eval():
    assert "DISPATCH=lm-eval" in _dispatch(is_agentic="0")


# --- explicit overrides win over the scenario default ----------------------

def test_explicit_framework_arg_overrides_scenario():
    # agentic, but recipe passed --framework lm-eval -> lm-eval.
    assert "DISPATCH=lm-eval" in _dispatch(is_agentic="1", cli_fw="lm-eval")


def test_env_framework_overrides_scenario():
    assert "DISPATCH=lm-eval" in _dispatch(is_agentic="1", env_fw="lm-eval")


def test_env_can_force_swebench_on_fixed_seqlen():
    assert "DISPATCH=swebench" in _dispatch(is_agentic="0", env_fw="swebench")


def test_recipe_lm_eval_arg_still_lm_eval_on_fixed_seqlen():
    # The existing fixed-seq-len recipes call `run_eval --framework lm-eval`.
    assert "DISPATCH=lm-eval" in _dispatch(is_agentic="0", cli_fw="lm-eval")


# --- EVAL_LIMIT smoke-test knob --------------------------------------------
#
# Use a shim python3 on PATH that echoes its args and exits 0 so we can
# observe the constructed lm_eval command line without a real server.

_EVAL_LIMIT_SCRIPT = r'''
set -e
# Build a tiny python3 shim that just echoes its args to stdout
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

# Intercept _install_lm_eval_deps and _patch_lm_eval so they are no-ops
_install_lm_eval_deps() { :; }
_patch_lm_eval() { :; }

PATH="$SHIM_DIR:$PATH" run_lm_eval --port 9999 2>&1
'''


def _run_lm_eval_cmdline(*, eval_limit=None) -> str:
    """Run run_lm_eval with a python3 shim and return captured stdout."""
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


# --- Modal credential HOME hardening tests ---------------------------------
#
# Tests for _ensure_modal_credentials HOME-remap logic (Change B).

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
    """When HOME is a writable directory, no remap happens and .modal.toml is written there."""
    home = str(tmp_path / "writable_home")
    Path(home).mkdir()
    out = _run_modal_creds(tmp_path, home=home)
    assert f"HOME_AFTER={home}" in out, f"HOME should not be remapped:\n{out}"
    assert "TOML_EXISTS=true" in out
    toml_path = Path(home) / ".modal.toml"
    assert toml_path.exists()
    # Check mode 600
    mode = oct(stat.S_IMODE(toml_path.stat().st_mode))
    assert mode == "0o600", f"Expected 0o600 got {mode}"


def test_modal_creds_remaps_home_when_not_writable_parent(tmp_path):
    """When HOME is nested under a read-only dir (mkdir -p fails), HOME is remapped."""
    # Create a read-only parent so mkdir -p "$HOME" inside the function will fail.
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
    """When HOME exists but is not writable, HOME is remapped."""
    readonly_home = tmp_path / "readonly_home"
    readonly_home.mkdir(mode=0o555)
    try:
        out = _run_modal_creds(tmp_path, home=str(readonly_home))
        assert "HOME_AFTER=/tmp/inferencex-modal-home" in out, f"Expected HOME remap:\n{out}"
        assert "TOML_EXISTS=true" in out
    finally:
        # Restore write permission so tmp_path cleanup can remove it
        readonly_home.chmod(0o755)


def test_modal_creds_no_remap_when_disabled(tmp_path):
    """When SWEBENCH_USE_MODAL != true, _ensure_modal_credentials is a no-op."""
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


# --- SWEBENCH_NAMESPACE ns_args construction tests -------------------------
#
# Tests that the ns_args array construction in benchmark_lib.sh handles the
# three cases correctly: unset (0 args), empty (2 args with empty value),
# and set-with-value (2 args with the value).

_NS_ARGS_SNIPPET = r'''
# Replicate the ns_args construction from benchmark_lib.sh (wrapped in a
# function so `local` is valid, matching the original context).
_test_ns_args() {
    local ns_args=()
    if [ "${SWEBENCH_NAMESPACE+set}" = "set" ]; then ns_args=(--namespace "$SWEBENCH_NAMESPACE"); fi
    echo "COUNT=${#ns_args[@]}"
    if [ "${#ns_args[@]}" -gt 0 ]; then
        echo "ARG0=${ns_args[0]}"
        echo "ARG1=${ns_args[1]}"
    fi
}
_test_ns_args
'''


def _run_ns_args(*, namespace_set: bool, namespace_value: str = "") -> dict:
    """Run the ns_args snippet and return a dict of parsed KEY=VALUE outputs."""
    if namespace_set:
        env_extra = {"SWEBENCH_NAMESPACE": namespace_value}
    else:
        env_extra = {}
    env = {k: v for k, v in os.environ.items() if k != "SWEBENCH_NAMESPACE"}
    env.update(env_extra)
    script = "set -e\n" + _NS_ARGS_SNIPPET
    res = subprocess.run(
        ["bash", "-c", script],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    parsed = {}
    for line in res.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            parsed[k] = v
    return parsed


def test_ns_args_unset_produces_zero_args():
    """When SWEBENCH_NAMESPACE is unset, ns_args should be empty (0 args)."""
    result = _run_ns_args(namespace_set=False)
    assert result["COUNT"] == "0", f"Expected COUNT=0, got: {result}"


def test_ns_args_empty_produces_two_args():
    """When SWEBENCH_NAMESPACE is set but empty, ns_args should be (--namespace '')."""
    result = _run_ns_args(namespace_set=True, namespace_value="")
    assert result["COUNT"] == "2", f"Expected COUNT=2, got: {result}"
    assert result["ARG0"] == "--namespace"
    assert result["ARG1"] == ""


def test_ns_args_value_produces_two_args():
    """When SWEBENCH_NAMESPACE has a value, ns_args should be (--namespace <value>)."""
    result = _run_ns_args(namespace_set=True, namespace_value="my-namespace")
    assert result["COUNT"] == "2", f"Expected COUNT=2, got: {result}"
    assert result["ARG0"] == "--namespace"
    assert result["ARG1"] == "my-namespace"


def test_benchmark_lib_no_longer_uses_old_namespace_pattern():
    """Static assertion: benchmark_lib.sh must not contain the old word-split pattern."""
    content = BENCHMARK_LIB.read_text()
    assert "${SWEBENCH_NAMESPACE+--namespace" not in content, (
        "benchmark_lib.sh still contains the old ${SWEBENCH_NAMESPACE+--namespace ...} pattern"
    )
    assert "ns_args" in content, (
        "benchmark_lib.sh does not contain the ns_args fix"
    )


# --- include_path wiring for swebench generation ---------------------------
#
# The pinned lm-eval (0.4.9.2, ref b315ef3) crashes with
# KeyError: '<task_name>' in pretty_print_task when --tasks receives a path to
# an external YAML whose task: name is not in lm-eval's bundled registry.
# The fix is to invoke with --include_path <dir> --tasks <task-name> instead.
#
# Two tests:
#   1. Dynamic: drive run_lm_eval directly via the shim with
#      EVAL_INCLUDE_PATH=utils/evals and EVAL_TASKS_DIR=swebench_lite; assert
#      --include_path utils/evals and --tasks swebench_lite appear in argv and
#      that argv contains no .yaml path in the --tasks position.
#   2. Default (EVAL_INCLUDE_PATH unset): --include_path must be absent and
#      --tasks must carry the default utils/evals/gsm8k.yaml path unchanged.
#   3. Static: run_swebench_eval's source must contain the EVAL_INCLUDE_PATH
#      wiring (EVAL_INCLUDE_PATH= and dirname), proving the include-path form
#      is wired for the swebench generation call.

# Reuse the shim-based _EVAL_LIMIT_SCRIPT infrastructure.
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
    """Run run_lm_eval with the shim and optional EVAL_INCLUDE_PATH/EVAL_TASKS_DIR."""
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
    """When EVAL_INCLUDE_PATH is set, --include_path <dir> appears before --tasks."""
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
    # Must NOT pass a .yaml path to --tasks
    assert ".yaml" not in out.split("--tasks")[1].split()[0], (
        f"--tasks must not contain a .yaml path when include_path is set:\n{out}"
    )


def test_include_path_absent_when_eval_include_path_unset():
    """When EVAL_INCLUDE_PATH is unset, --include_path must not appear and --tasks carries the default yaml path."""
    out = _run_lm_eval_with_include_path()  # both env vars unset
    assert "--include_path" not in out, (
        f"Expected no '--include_path' in output:\n{out}"
    )
    assert "--tasks utils/evals/gsm8k.yaml" in out, (
        f"Expected '--tasks utils/evals/gsm8k.yaml' in output:\n{out}"
    )


def test_swebench_eval_source_contains_include_path_wiring():
    """Static: run_swebench_eval source must wire EVAL_INCLUDE_PATH and use dirname."""
    content = BENCHMARK_LIB.read_text()
    assert "EVAL_INCLUDE_PATH=" in content, (
        "benchmark_lib.sh run_swebench_eval does not set EVAL_INCLUDE_PATH"
    )
    assert 'dirname "$yaml_path"' in content or "dirname \"$yaml_path\"" in content, (
        "benchmark_lib.sh run_swebench_eval does not derive EVAL_INCLUDE_PATH via dirname"
    )


def test_modal_credentials_sanitizes_whitespace_contaminated_tokens(tmp_path):
    """CI secrets pasted with a trailing newline must be stripped before use
    (a contaminated token fails Modal validation: 'Token validation failed')."""
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
    """SWEBENCH_GEN_MODE=agentic: mini-extra called with slice/workers/config;
    preds.json produced; config carries the local endpoint + model."""
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
        # emulate mini's import-time version banner (regression: the banner must
        # not end up in the captured config path)
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
    """Shared scaffolding for watchdog/salvage tests: shim dir with a scripted
    mini-extra, a python3 that answers mini's config-path probe, and a gen dir."""
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
    """mini-extra hangs after writing all expected preds (observed at
    workers=144): the watchdog must kill it and count generation as success."""
    shim, gen_dir = _agentic_shim(tmp_path,
        'out=""; prev=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        'mkdir -p "$out"\n'
        "printf '{\"i1\": {\"instance_id\": \"i1\", \"model_patch\": \"d\"}}' > \"$out/preds.json\"\n"
        # hang-on-exit emulation; exec + detached stdio so the pytest capture
        # pipe isn't held open by an orphan after the watchdog kills us
        "exec sleep 600 </dev/null >/dev/null 2>&1\n"
    )
    res = _run_agentic(shim, gen_dir, {"EVAL_LIMIT": "1", "SWEBENCH_AGENT_EXIT_GRACE": "2"})
    assert "GEN_RC=0" in res.stdout, res.stdout + res.stderr
    assert "hung after completing all instances" in res.stdout + res.stderr


def test_agentic_salvage_partial_preds_on_failure(tmp_path):
    """Generation dies mid-run with some preds written: salvage and proceed
    (rc 0) instead of discarding real work."""
    shim, gen_dir = _agentic_shim(tmp_path,
        'out=""; prev=""\n'
        'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
        'mkdir -p "$out"\n'
        "printf '{\"i1\": {\"instance_id\": \"i1\", \"model_patch\": \"d\"}}' > \"$out/preds.json\"\n"
        "exit 7\n"  # crash after 1 of 2 expected instances
    )
    res = _run_agentic(shim, gen_dir, {"EVAL_LIMIT": "2"})
    assert "GEN_RC=0" in res.stdout, res.stdout + res.stderr
    assert "scoring the partial set" in res.stdout + res.stderr


def test_agentic_no_preds_still_fails(tmp_path):
    """Generation fails with zero preds: must still fail (nothing to salvage)."""
    shim, gen_dir = _agentic_shim(tmp_path, "exit 7\n")
    res = _run_agentic(shim, gen_dir, {"EVAL_LIMIT": "2"})
    assert "GEN_RC=7" in res.stdout, res.stdout + res.stderr
