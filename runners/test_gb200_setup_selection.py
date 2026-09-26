"""Run the pool launcher through submission with native infrastructure stubbed."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECIPE = "glm5.2/sglang/gb200-fp4/agentx/disagg-mtp-variants.yaml"
C48_RECIPE = "glm5.2/sglang/gb200-fp4/agentx/disagg-dep8-mtp-variants.yaml:override_1p4d_tp4_c48"
C45_RECIPE = "glm5.2/sglang/gb200-fp4/agentx/disagg-dep8-mtp-variants.yaml:override_1p6d_tp4_c45"
C128_RECIPE = "glm5.2/sglang/gb200-fp4/agentx/disagg-dep8-mtp-variants.yaml:override_2p1d_dep16_c128"


@pytest.mark.parametrize("model,recipe,eval_only,expected", [
    ("glm5.2", f"recipes/{RECIPE}:zip_override_mtp_agentx_frontier[0]", "false", "glm52-gb200-nixl-prefill.sh"),
    ("glm5.2", f"benchmarks/multi_node/srt-slurm-recipes/{RECIPE}:base", "false", "glm52-gb200-nixl-prefill.sh"),
    ("glm5.2", f"recipes/{RECIPE}:base", "true", "install-torchao.sh"),
    ("glm5.2", "recipes/glm5.2/sglang/gb200-fp4/agentx/agg.yaml:base", "false", "install-torchao.sh"),
    ("glm5.2", "recipes/glm5.2/sglang/gb200-fp4/agentx/disagg-mtp-nightly.yaml:base", "false", "install-torchao.sh"),
    ("glm5.2", f"recipes/{C48_RECIPE}", "false", "glm52-gb200-nixl-prefill.sh"),
    ("glm5.2", f"benchmarks/multi_node/srt-slurm-recipes/{C48_RECIPE}", "false", "glm52-gb200-nixl-prefill.sh"),
    ("glm5.2", f"recipes/{C48_RECIPE}", "true", "install-torchao.sh"),
    ("glm5.2", f"recipes/{C45_RECIPE}", "false", "glm52-gb200-nixl-prefill.sh"),
    ("glm5.2", f"benchmarks/multi_node/srt-slurm-recipes/{C45_RECIPE}", "false", "glm52-gb200-nixl-prefill.sh"),
    ("glm5.2", f"recipes/{C45_RECIPE}", "true", "install-torchao.sh"),
    ("glm5.2", f"recipes/{C128_RECIPE}", "false", "glm52-gb200-nixl-prefill.sh"),
    ("glm5.2", f"benchmarks/multi_node/srt-slurm-recipes/{C128_RECIPE}", "false", "glm52-gb200-nixl-prefill.sh"),
    ("glm5.2", f"recipes/{C128_RECIPE}", "true", "install-torchao.sh"),
    ("glm5.2", "recipes/glm5.2/sglang/gb200-fp4/agentx/agg-mtp-variants.yaml:override_c2", "false", "install-torchao.sh"),
    ("glm5.2", "benchmarks/multi_node/srt-slurm-recipes/glm5.2/sglang/gb200-fp4/agentx/agg-mtp-variants.yaml:override_c4", "false", "install-torchao.sh"),
    ("glm5.2", "recipes/glm5.2/sglang/gb200-fp4/agentx/agg-mtp-variants.yaml:override_c8", "false", "install-torchao.sh"),
    ("dsr1", f"recipes/{RECIPE}:base", "false", "install-torchao.sh"),
])
def test_launcher_submits_scoped_setup(tmp_path, model, recipe, eval_only, expected):
    checkout = tmp_path / "checkout"
    (checkout / "runners").mkdir(parents=True)
    (checkout / "benchmarks").symlink_to(ROOT / "benchmarks", target_is_directory=True)
    shutil.copy2(ROOT / "runners/launch_gb200-nv.sh", checkout / "runners")
    # These collaborators import images, install on the native fleet, submit to
    # Slurm and collect remote results. Keep the launcher's routing/argv real.
    (checkout / "runners/slurm_utils.sh").write_text('''
setup_srt_slurm() {
    cd "$TEST_NATIVE" || return 1
}
install_srt_slurm() { return 0; }
prepare_srt_power() { SRTCTL_RECIPE_ARGS=(); }
write_srt_cluster_config() { echo 'fixture: native' > "$2"; }
run_srt_setup() { return 0; }
apply_srt_recipe() {
    printf '%s\\0' "$@" > "$TEST_CAPTURE"
    printf '%s\\n' "$RUNNER_NAME" > "$TEST_CAPTURE.runner"
    echo 'Job 42'
}
stream_slurm_job_log() { return 0; }
copy_agentic_results() { return 0; }
copy_eval_artifacts() { return 0; }
bundle_server_logs() { return 0; }
''')
    native = tmp_path / "native"
    config = native / recipe.split(":")[0]
    config.parent.mkdir(parents=True)
    config.write_text("name: fixture\n")
    (native / "outputs/42/logs").mkdir(parents=True)
    (native / ".venv/bin").mkdir(parents=True)
    (native / ".venv/bin/activate").write_text(":\n")
    home = tmp_path / "home"
    (home / ".local/bin").mkdir(parents=True)
    (home / ".local/bin/env").write_text(":\n")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, body in {
        "curl": "echo ':'",
        "uv": ":",
        "srtctl": ":",
        "squeue": ":",
        "scancel": ":",
        "findmnt": "echo lustre",
        "chmod": ":",
        "grep": "echo 42",
    }.items():
        path = binaries / name
        path.write_text(f"#!/usr/bin/env bash\n{body}\n")
        path.chmod(0o755)
    # Cache directory operations are fleet I/O; local test paths stay real.
    mkdir = binaries / "mkdir"
    mkdir.write_text(f"#!{sys.executable}\n" + '''import os, sys
for path in sys.argv[1:]:
    if not path.startswith(("-", "/mnt/")):
        os.makedirs(path, exist_ok=True)
''')
    mkdir.chmod(0o755)
    sed = binaries / "sed"
    sed.write_text('''#!/usr/bin/env bash
if [[ "$1" == -i ]]; then exit 0; fi
exec /usr/bin/sed "$@"
''')
    sed.chmod(0o755)
    # Image-cache locks live on the native fleet, outside this CPU test filesystem.
    bootstrap = tmp_path / "bootstrap.sh"
    bootstrap.write_text('''
trap 'if declare -F import_squash >/dev/null; then
    import_squash() { return 0; }
    trap - DEBUG
fi' DEBUG
''')
    capture = tmp_path / "argv"
    env = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}", "BASH_ENV": str(bootstrap),
        "HOME": str(home), "GITHUB_WORKSPACE": str(checkout),
        "TEST_NATIVE": str(native), "TEST_CAPTURE": str(capture),
        "EVAL_ONLY": eval_only, "IS_AGENTIC": "1", "IS_MULTINODE": "true",
        "RUN_EVAL": "false", "SALLOC_TIME_LIMIT": "10", "MODEL_PREFIX": model,
        "FRAMEWORK": "dynamo-sglang", "PRECISION": "fp4", "SPEC_DECODING": "mtp",
        "IMAGE": "fixture:tag", "MODEL": "fixture/model", "CONFIG_FILE": recipe,
        "GITHUB_RUN_ID": "1", "GITHUB_RUN_ATTEMPT": "1", "RUNNER_NAME": "gb200-nv_fixture",
        "ISL": "0", "OSL": "0", "RESULT_FILENAME": "result", "USER": "fixture",
    }
    result = subprocess.run(
        ["bash", str(checkout / "runners/launch_gb200-nv.sh")],
        cwd=checkout, env=env, capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    argv = capture.read_bytes().decode().rstrip("\0").split("\0")
    assert argv[argv.index("--setup-script") + 1] == expected
    assert argv.count("--setup-script") == 1
    assert argv[argv.index("-f") + 1] == recipe
    assert Path(str(capture) + ".runner").read_text().strip() == "inferencex-gb200-nv_fixture"
    if "/agentx/disagg-dep8-mtp-variants.yaml" in recipe or "/agentx/agg-mtp-variants.yaml" in recipe:
        assert 'name="inferencex-gb200-nv_fixture"' in argv


@pytest.mark.parametrize("role,install_rc,patch_rc,expected_rc,expected_steps", [
    ("1", 0, 0, 0, ["torchao", "nixl"]),
    ("0", 0, 0, 0, ["torchao"]),
    ("1", 7, 0, 7, ["torchao"]),
    ("1", 0, 9, 9, ["torchao", "nixl"]),
    ("bad", 0, 0, 1, ["torchao"]),
])
def test_composite_setup_orders_dependencies_and_propagates_failures(
    tmp_path, role, install_rc, patch_rc, expected_rc, expected_steps,
):
    setup = tmp_path / "glm52-gb200-nixl-prefill.sh"
    shutil.copy2(ROOT / "benchmarks/multi_node/srt-slurm-recipes/configs" / setup.name, setup)
    (tmp_path / "install-torchao.sh").write_text(
        'echo torchao >> "$TEST_STEPS"\nexit "$TEST_INSTALL_RC"\n'
    )
    # /infmax-workspace is a container mount absent on the CPU test host.
    bootstrap = tmp_path / "bootstrap.sh"
    bootstrap.write_text('''
source() {
    if [[ "$1" == /infmax-workspace/* ]]; then
        local path="$TEST_ROOT/${1#/infmax-workspace/}"
        shift
        builtin source "$path" "$@"
    else
        builtin source "$@"
    fi
}
python3() {
    printf 'nixl\\n' >> "$TEST_STEPS"
    [[ "$1" == /infmax-workspace/runners/patch_glm52_nixl_sync.py ]] || return 99
    return "$TEST_PATCH_RC"
}
''')
    steps = tmp_path / "steps"
    result = subprocess.run(["bash", str(setup)], env={
        **os.environ, "BASH_ENV": str(bootstrap), "TEST_ROOT": str(ROOT),
        "TEST_STEPS": str(steps), "TEST_INSTALL_RC": str(install_rc),
        "TEST_PATCH_RC": str(patch_rc), "INFX_GLM52_NIXL_SYNC_PATCH": role,
    }, capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == expected_rc, result.stdout + result.stderr
    assert steps.read_text().splitlines() == expected_steps
