"""Exercise launcher routing and exporter imports without Slurm or network access."""

import os
import subprocess
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


def _launcher_routing_source() -> str:
    """Extract the real clone-routing chain, not a copy of its implementation."""
    launcher = LAUNCHER_PATH.read_text()
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
    tmp_path: Path, uses_dcgm_power: bool, *, reported_head: str = ""
) -> tuple[list[str], Path, Path, Path]:
    """Execute only the real launcher routing region in a temporary checkout."""
    workspace = tmp_path / "workspace"
    stub_bin = tmp_path / "bin"
    source = (
        workspace
        / "benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4/8k1k"
    )
    source.mkdir(parents=True)
    (source / "overlay-marker.txt").write_text("from-workspace\n")
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

    routing = _launcher_routing_source()
    repo_dir = workspace / "srt-slurm-route-test"
    harness = tmp_path / "route.sh"
    harness.write_text(
        f"""#!/bin/bash
set -eo pipefail
POWER_SRT_SLURM_URL={FORK_URL}
POWER_SRT_SLURM_PIN={PRODUCER_PIN}
IS_AGENTIC=0
FRAMEWORK=dynamo-sglang
MODEL_PREFIX=dsv4
PRECISION=fp4
SPEC_DECODING=
USES_DCGM_POWER={int(uses_dcgm_power)}
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

    marker = repo_dir / "recipes/sglang/deepseek-v4/8k1k/overlay-marker.txt"
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
