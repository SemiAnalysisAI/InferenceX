"""Exercise launcher routing and exporter imports without Slurm or network access."""

import json
import os
import shlex
import subprocess
import sys
import tarfile
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = REPO_ROOT / "runners/launch_gb300-nv.sh"
MASTER_CONFIG_PATH = REPO_ROOT / "configs/nvidia-master.yaml"
# Controlled routing inputs, deliberately independent of the deployed pins.
FORK_URL = "https://example.test/power-producer.git"
PRODUCER_PIN = "a" * 40
AGENTX_PRODUCER_PIN = "b" * 40


def _launcher_routing_source(launcher_path: Path = LAUNCHER_PATH) -> str:
    """Extract the real clone-routing chain, not a copy of its implementation."""
    launcher = launcher_path.read_text()
    if launcher_path.name == "launch_gb200-nv.sh":
        gate_start = launcher.index("USES_DCGM_POWER=0")
        gate_end = launcher.index('if [[ "$USES_DCGM_POWER" == "1" ]]; then\n    DCGM_EXPORTER_IMAGE=', gate_start)
        route_start = launcher.index('if [[ "$IS_AGENTIC" == "1" && "$MODEL_PREFIX" == "glm5.2"')
        route_end = launcher.index('\necho "Installing srtctl..."', route_start)
        return launcher[gate_start:gate_end] + launcher[route_start:route_end]
    route_start = launcher.index(
        'if [[ "$IS_AGENTIC" == "1" && $FRAMEWORK == "dynamo-sglang" '
        '&& $MODEL_PREFIX == "qwen3.5" ]]; then'
    )
    route_end_marker = '\nfi\n\necho "Installing srtctl..."'
    route_end = launcher.index(route_end_marker, route_start) + len("\nfi")
    return launcher[route_start:route_end]


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


def _run_dsv4_route(
    tmp_path: Path, uses_dcgm_power: bool, *, reported_head: str = "",
    launcher_path: Path = LAUNCHER_PATH, model_prefix: str = "dsv4",
    is_agentic: bool = False, recipe_name: str | None = None,
) -> tuple[list[str], Path, Path, Path]:
    """Execute only the real launcher routing region in a temporary checkout."""
    workspace = tmp_path / "workspace"
    stub_bin = tmp_path / "bin"
    recipe_directory = ("sglang/glm5.2/gb200-fp4/agentic" if model_prefix == "glm5.2"
                        else "sglang/deepseek-v4/8k1k")
    recipe_name = recipe_name or ("glm5.2-agentx-agg.yaml" if model_prefix == "glm5.2" else "recipe.yaml")
    source = workspace / "benchmarks/multi_node/srt-slurm-recipes" / recipe_directory
    source.mkdir(parents=True)
    (source / "overlay-marker.txt").write_text("from-workspace\n")
    (source / recipe_name).write_text(
        "telemetry:\n  enabled: true\n  provider: dcgm-power\n"
        if uses_dcgm_power else "benchmark:\n  type: custom\n"
    )
    stub_bin.mkdir()

    route_log = tmp_path / "route.log"
    _write_executable(
        stub_bin / "git",
        """#!/bin/bash
set -e
printf 'git %s\\n' "$*" >> "$ROUTE_LOG"
case "$1" in
  clone)
    for arg in "$@"; do destination="$arg"; done
    /bin/mkdir -p "$destination"
    ;;
  checkout)
    printf '%s\\n' "$2" > .stub-head
    ;;
  rev-parse)
    if [[ -n "$STUB_HEAD" ]]; then
      printf '%s\\n' "$STUB_HEAD"
    else
      /bin/cat .stub-head
    fi
    ;;
  *)
    printf 'unexpected git command: %s\\n' "$*" >&2
    exit 64
    ;;
esac
""",
    )
    _write_executable(
        stub_bin / "mkdir",
        """#!/bin/bash
printf 'mkdir %s\\n' "$*" >> "$ROUTE_LOG"
exec /bin/mkdir "$@"
""",
    )
    _write_executable(
        stub_bin / "cp",
        """#!/bin/bash
set -e
printf 'cp %s\\n' "$*" >> "$ROUTE_LOG"
if [[ "$1" != "-rT" || "$#" != 3 ]]; then
  printf 'unexpected cp command: %s\\n' "$*" >&2
  exit 64
fi
/bin/mkdir -p "$3"
exec /bin/cp -R "$2"/. "$3"
""",
    )

    routing = _launcher_routing_source(launcher_path)
    repo_dir = workspace / "srt-slurm-route-test"
    harness = tmp_path / "route.sh"
    harness.write_text(
        f"""#!/bin/bash
set -eo pipefail
POWER_SRT_SLURM_URL={FORK_URL}
POWER_SRT_SLURM_PIN={PRODUCER_PIN}
AGENTX_POWER_SRT_SLURM_PIN={AGENTX_PRODUCER_PIN}
IS_AGENTIC={int(is_agentic)}
FRAMEWORK=dynamo-sglang
MODEL_PREFIX={model_prefix}
PRECISION=fp4
SPEC_DECODING=
USES_DCGM_POWER={int(uses_dcgm_power)}
CONFIG_FILE=recipes/{recipe_directory}/{recipe_name}
GITHUB_WORKSPACE={workspace!s}
SRT_REPO_DIR={repo_dir!s}
{routing}
"""
    )
    env = os.environ.copy()
    env["PATH"] = f"{stub_bin}:/usr/bin:/bin"
    env["ROUTE_LOG"] = str(route_log)
    env["STUB_HEAD"] = reported_head
    subprocess.run(["/bin/bash", str(harness)], env=env, check=True)

    marker = repo_dir / "recipes" / recipe_directory / "overlay-marker.txt"
    return route_log.read_text().splitlines(), workspace, repo_dir, marker


def _config_file_values(value: object) -> Iterator[str]:
    """Yield every CONFIG_FILE value reachable below a config search space."""
    if isinstance(value, dict):
        for child in value.values():
            yield from _config_file_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _config_file_values(child)
    elif isinstance(value, str) and value.startswith("CONFIG_FILE="):
        yield value.removeprefix("CONFIG_FILE=").split(":", 1)[0]


def _workspace_recipe_path(config_file: str) -> Path:
    assert config_file.startswith("recipes/")
    relative = config_file.removeprefix("recipes/")
    return REPO_ROOT / "benchmarks/multi_node/srt-slurm-recipes" / relative


@pytest.mark.parametrize(
    ("launcher_name", "indent", "cache_directory"),
    [
        ("launch_gb300-nv.sh", "", "/data/home/sa-shared/gharunners/squash/"),
        ("launch_h200-dgxc-slurm.sh", "    ", "/data/gharunners/containers/"),
    ],
)
def test_exporter_cold_import_uses_nvidia_registry(
    tmp_path: Path, launcher_name: str, indent: str, cache_directory: str
) -> None:
    launcher = (REPO_ROOT / "runners" / launcher_name).read_text()
    start = launcher.index(
        f'{indent}if [[ "$USES_DCGM_POWER" == "1" ]]; then\n'
        f'{indent}    DCGM_EXPORTER_IMAGE='
    )
    end_marker = f"\n{indent}fi"
    end = launcher.index(end_marker, start) + len(end_marker)
    source = launcher[start:end].replace(cache_directory, f"{tmp_path}/")
    source = source.replace("${HOME}/.cache/enroot", "${GITHUB_WORKSPACE}/enroot-cache")
    if "import_squash() {" in launcher:
        helper_start = launcher.index("import_squash() {")
        helper_end = launcher.index("\n}\n", helper_start) + len("\n}")
        source = launcher[helper_start:helper_end] + "\n" + source

    # Run the real launcher code, including the command passed through srun.
    # Only cluster/container tools are replaced; all files stay in tmp_path.
    harness = """
set -euo pipefail
srun() {
    while [[ "$1" != bash ]]; do shift; done
    "$@"
}
flock() { :; }
unsquashfs() { test -s "$2"; }
enroot() {
    printf '%s\\n' "$@" > "$IMPORT_ARGS"
    printf 'collector image\\n' > "$3"
}
sha256sum() { printf 'fixture-hash  %s\\n' "$1"; }
export -f flock unsquashfs enroot sha256sum
"""
    import_args = tmp_path / "import-args"
    env = os.environ.copy()
    env.update(
        GITHUB_WORKSPACE=str(tmp_path),
        IMPORT_ARGS=str(import_args),
        USES_DCGM_POWER="1",
        SLURM_ACCOUNT="test",
        SLURM_PARTITION="test",
        RUNNER_NAME="exporter-import-test",
    )
    subprocess.run(
        ["/bin/bash"],
        input=harness + source,
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        check=True,
    )

    command, output_flag, image_path, reference = import_args.read_text().splitlines()
    assert (command, output_flag) == ("import", "-o")
    assert reference.startswith("docker://nvcr.io#nvidia/k8s/dcgm-exporter:")
    assert reference.count("#") == 1
    assert Path(image_path).read_text() == "collector image\n"


def test_dsv4_power_route_executes_pinned_producer_and_overlay(tmp_path):
    log, workspace, repo_dir, marker = _run_dsv4_route(tmp_path, uses_dcgm_power=True)

    assert f"git clone {FORK_URL} {repo_dir}" in log
    assert f"git checkout {PRODUCER_PIN}" in log
    assert (workspace / "power-producer-sha.txt").read_text() == f"{PRODUCER_PIN}\n"
    assert marker.read_text() == "from-workspace\n"


def test_dsv4_non_power_route_uses_upstream_without_power_stamp(tmp_path):
    log, workspace, repo_dir, marker = _run_dsv4_route(tmp_path, uses_dcgm_power=False)

    assert f"git clone https://github.com/NVIDIA/srt-slurm.git {repo_dir}" in log
    assert all(FORK_URL not in entry and PRODUCER_PIN not in entry for entry in log)
    assert not (workspace / "power-producer-sha.txt").exists()
    assert marker.read_text() == "from-workspace\n"


def test_dsv4_power_route_rejects_unexpected_checkout_before_publishing_stamp(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        _run_dsv4_route(tmp_path, uses_dcgm_power=True, reported_head="b" * 40)

    assert not (tmp_path / "workspace/power-producer-sha.txt").exists()


@pytest.mark.parametrize(
    ("model_prefix", "is_agentic", "expected_pin"),
    [("dsv4", False, PRODUCER_PIN), ("glm5.2", True, AGENTX_PRODUCER_PIN)],
)
def test_gb200_routes_fixed_sequence_and_opted_in_agentx_power(
    tmp_path: Path, model_prefix: str, is_agentic: bool, expected_pin: str
) -> None:
    log, workspace, repo_dir, marker = _run_dsv4_route(
        tmp_path, True, launcher_path=REPO_ROOT / "runners/launch_gb200-nv.sh",
        model_prefix=model_prefix, is_agentic=is_agentic,
    )
    assert f"git clone {FORK_URL} {repo_dir}" in log
    assert f"git checkout {expected_pin}" in log
    assert (workspace / "power-producer-sha.txt").read_text() == expected_pin + "\n"
    assert marker.read_text() == "from-workspace\n"


@pytest.mark.parametrize("recipe_name,reported_head", [
    ("glm5.2-agentx-disagg.yaml", ""), ("glm5.2-agentx-agg.yaml", "c" * 40),
])
def test_gb200_rejects_unqualified_recipe_or_wrong_agentx_producer(
    tmp_path: Path, recipe_name: str, reported_head: str
) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        _run_dsv4_route(
            tmp_path, True, launcher_path=REPO_ROOT / "runners/launch_gb200-nv.sh",
            model_prefix="glm5.2", is_agentic=True, recipe_name=recipe_name,
            reported_head=reported_head,
        )
    assert not (tmp_path / "workspace/power-producer-sha.txt").exists()


@pytest.mark.parametrize("replies,expected_rc,attempts", [
    ("42|RUNNING|0:0\n42|COMPLETED|0:0\n", 0, 2),
    ("42|FAILED|1:0\n", 1, 1),
    ("", 1, 3),
])
def test_gb200_native_status_waits_for_terminal_and_fails_closed(
    tmp_path: Path, replies: str, expected_rc: int, attempts: int
) -> None:
    launcher = (REPO_ROOT / "runners/launch_gb200-nv.sh").read_text()
    start = launcher.index('    mkdir -p "$LOGS_DIR/power"', launcher.index("AGENTX_POWER_RC=0"))
    end = launcher.index('    copy_agentic_results ', start)
    reply_file = tmp_path / "replies.txt"
    reply_file.write_text(replies)
    env = os.environ.copy()
    env.update(JOB_ID="42", LOGS_DIR=str(tmp_path / "logs"), REPLIES=str(reply_file))
    result = subprocess.run(["/bin/bash"], input=(
        "set -euo pipefail\nAGENTX_POWER_RC=0\nn=0\n"
        'sacct() { n=$((n+1)); sed -n "${n}p" "$REPLIES"; }\n'
        "sleep() { :; }\n" + launcher[start:end] + 'exit "$AGENTX_POWER_RC"\n'
    ), text=True, capture_output=True, env=env)
    assert result.returncode == expected_rc, result.stderr
    assert (tmp_path / "logs/power/native-job-status-attempts.txt").read_text() == f"{attempts}\n"


def test_gb200_agentx_window_injection_and_failure_artifacts(tmp_path: Path) -> None:
    launcher = (REPO_ROOT / "runners/launch_gb200-nv.sh").read_text()
    injection_end = launcher.index("# Don't leak the login-node venv")
    injection_start = launcher.rfind('if [[ "$USES_AGENTX_POWER" == "1" ]]; then', 0, injection_end)
    collection_start = launcher.index("AGENTX_POWER_RC=0")
    collection_end = launcher.index('\nif [[ "${EVAL_ONLY:-false}" != "true" ]]; then', collection_start)
    workspace, compute, producer = (tmp_path / name for name in ("workspace", "compute", "producer"))
    for path in (workspace, compute, producer):
        path.mkdir()
    (workspace / "runners").symlink_to(REPO_ROOT / "runners", target_is_directory=True)
    recipe = producer / "recipe.yaml"
    recipe.write_text("benchmark:\n  type: custom\n  concurrencies: [99]\n")
    (compute / "result_conc1.json").write_text(json.dumps({
        "disagg": False, "num_prefill_gpu": 8, "num_decode_gpu": 0,
        "power_valid": 1, "avg_power_w": 999,
    }))
    logs = producer / "outputs/42/logs"
    (logs / "agentic/conc_1").mkdir(parents=True)
    (logs / "sweep_42.log").write_text("job failed before a formal window\n")
    (workspace / "power-producer-sha.txt").write_text(AGENTX_PRODUCER_PIN)
    (workspace / "exporter-image.sha256").write_text("fixture-exporter-hash\n")
    env = os.environ.copy()
    env.update(GITHUB_WORKSPACE=str(workspace), INFMAX_WORKSPACE=str(compute),
               PYTHONPATH=str(REPO_ROOT), USES_AGENTX_POWER="1", USES_DCGM_POWER="1",
               CONFIG_PATH=str(recipe), CONC_LIST="1", RESULT_FILENAME="result", EVAL_ONLY="false",
               JOB_ID="42", LOGS_DIR="outputs/42/logs", LOG_FILE="outputs/42/logs/sweep_42.log",
               AGENTX_POWER_SRT_SLURM_PIN=AGENTX_PRODUCER_PIN)
    result = subprocess.run(["/bin/bash"], input=(
        "set -euo pipefail\n"
        f"source {shlex.quote(str(REPO_ROOT / 'runners/slurm_utils.sh'))}\n"
        f"python3() {{ {shlex.quote(sys.executable)} \"$@\"; }}\n"
        "stream_slurm_job_log() { return 1; }\n"
        "sacct() { printf '42|FAILED|1:0\\n'; }\n"
        + launcher[injection_start:injection_end]
        + launcher[collection_start:collection_end]
    ), text=True, capture_output=True, cwd=producer, env=env)
    assert result.returncode == 1, result.stderr
    assert yaml.safe_load(recipe.read_text())["benchmark"]["concurrencies"] == [1]
    aggregate = json.loads((workspace / "result_conc1.json").read_text())
    assert aggregate["power_valid"] == 0
    assert "avg_power_w" not in aggregate
    validation = json.loads((workspace / "LOGS/agentic/conc_1/power_validation.json").read_text())
    assert validation["reasons"] == ["formal_benchmark_result_missing"]
    assert (workspace / "LOGS/power/power-producer-sha.txt").read_text() == AGENTX_PRODUCER_PIN
    assert (workspace / "LOGS/power/native-job-status.txt").read_text() == "42|FAILED|1:0\n"
    with tarfile.open(workspace / "multinode_server_logs.tar.gz") as archive:
        assert "./agentic/conc_1/power_validation.json" in archive.getnames()


def test_gb300_dsv4_recipe_images_match_their_master_configs():
    master = yaml.safe_load(MASTER_CONFIG_PATH.read_text())
    configs = {
        key: config
        for key, config in master.items()
        if isinstance(config, dict)
        and config.get("runner") == "gb300"
        and config.get("framework") == "dynamo-sglang"
        and config.get("model-prefix") == "dsv4"
    }
    assert configs
    for key, config in configs.items():
        config_files = set(_config_file_values(config["scenarios"]))
        assert config_files, key
        for config_file in config_files:
            recipe_path = _workspace_recipe_path(config_file)
            assert recipe_path.is_file(), (key, config_file)
            recipe_image = yaml.safe_load(recipe_path.read_text())["model"]["container"]
            assert recipe_image == config["image"], (key, config_file)
