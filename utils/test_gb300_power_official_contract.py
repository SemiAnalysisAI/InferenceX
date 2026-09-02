"""Contracts for the official GB300 dcgm-power lane.

Shares helpers with the GB200 contract; GB300 specifics are the literal
shared squash cache path (no SQUASH_DIR var in this launcher), the 19401
exporter port, and the preserved v1.0.25/sa-submission non-power refs. The
DSV4 routing tests execute the launcher's real first-match branch chain with
filesystem-local command stubs so branch ordering is covered without Slurm or
Docker side effects.
"""

import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

import yaml

from test_gb200_power_official_contract import (
    FORK_URL,
    PRODUCER_PIN,
    REPO_ROOT,
    assert_exporter_provisioning,
    assert_pinned_clone_contract,
    assert_recipe_driven_detection,
)

RECIPE_PATH = REPO_ROOT / "benchmarks/multi_node/srt-slurm-recipes/sglang/qwen3.5/gb300-fp8/8k1k/1p1d-tp4-tp4.yaml"
LAUNCHER_PATH = REPO_ROOT / "runners/launch_gb300-nv.sh"
MASTER_CONFIG_PATH = REPO_ROOT / "configs/nvidia-master.yaml"
OPERATIONS_GUIDE_PATH = REPO_ROOT / ".github/AGENT_OPERATIONS.md"
DSV4_GB300_CONFIG_KEY = "dsv4-fp4-gb300-dynamo-sglang"
DEAD_DSV4_IMAGES = {
    "lmsysorg/sglang:nightly-dev-cu13-20260707-b4155233",
    "lmsysorg/sglang:nightly-dev-cu13-20260721-8905cbd4",
}


def _launcher_routing_source() -> tuple[str, str]:
    """Extract the literal constants and complete clone-routing chain."""
    launcher = LAUNCHER_PATH.read_text()
    constant_lines = [
        line
        for line in launcher.splitlines()
        if line.startswith(("POWER_SRT_SLURM_URL=", "POWER_SRT_SLURM_PIN="))
    ]
    assert len(constant_lines) == 2

    route_start = launcher.index(
        'if [[ "$IS_AGENTIC" == "1" && $FRAMEWORK == "dynamo-sglang" '
        '&& $MODEL_PREFIX == "qwen3.5" ]]; then'
    )
    route_end_marker = '\nfi\n\necho "Installing srtctl..."'
    route_end = launcher.index(route_end_marker, route_start) + len("\nfi")
    return "\n".join(constant_lines), launcher[route_start:route_end]


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


def _run_dsv4_route(
    tmp_path: Path, uses_dcgm_power: bool
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
    /bin/cat .stub-head
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

    constants, routing = _launcher_routing_source()
    repo_dir = workspace / "srt-slurm-route-test"
    harness = tmp_path / "route.sh"
    harness.write_text(
        f"""#!/bin/bash
set -eo pipefail
{constants}
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


def test_recipe_declares_enabled_dcgm_power_lane():
    recipe = yaml.safe_load(RECIPE_PATH.read_text())

    telemetry = recipe["telemetry"]
    assert telemetry["enabled"] is True
    assert telemetry["provider"] == "dcgm-power"
    assert telemetry["required"] is True
    assert telemetry["dcgm_exporter"]["container_image"] == "dcgm-exporter"
    # 9401 is already bound by the cluster-level exporter on im-gb300 nodes.
    assert telemetry["dcgm_exporter"]["port"] == 19401

    assert recipe["benchmark"]["concurrencies"] == "1x2x4x8x16x32x64x128"
    assert recipe["resources"]["prefill_nodes"] == 1
    assert recipe["resources"]["decode_nodes"] == 1
    assert recipe["resources"]["gpus_per_node"] == 4


def test_launcher_detects_power_lane_from_recipe():
    assert_recipe_driven_detection(LAUNCHER_PATH.read_text())


def test_launcher_provisions_exporter_through_shared_squash_path():
    launcher = LAUNCHER_PATH.read_text()
    assert_exporter_provisioning(launcher)
    # No SQUASH_DIR var here; the /data/ mount avoids the /home NFS ELOOP bug.
    assert 'DCGM_EXPORTER_SQSH="/data/home/sa-shared/gharunners/squash/' in launcher
    assert 'srun --partition=$SLURM_PARTITION --exclusive --time=30 bash -c "unsquashfs -l' in launcher


def test_launcher_pins_power_producer():
    assert_pinned_clone_contract(LAUNCHER_PATH.read_text())


def test_non_power_lane_keeps_existing_ref_logic():
    launcher = LAUNCHER_PATH.read_text()
    assert 'git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"' in launcher
    assert "git checkout v1.0.25" in launcher
    assert "git checkout sa-submission-q2-2026" in launcher


def test_dsv4_power_route_executes_pinned_producer_and_overlay(tmp_path):
    log, workspace, repo_dir, marker = _run_dsv4_route(tmp_path, uses_dcgm_power=True)

    assert f"git clone {FORK_URL} {repo_dir}" in log
    assert f"git checkout {PRODUCER_PIN}" in log
    assert log.count("git rev-parse HEAD") == 2
    assert (workspace / "power-producer-sha.txt").read_text() == f"{PRODUCER_PIN}\n"
    assert marker.read_text() == "from-workspace\n"


def test_dsv4_non_power_route_keeps_upstream_v1_0_25_without_stamp(tmp_path):
    log, workspace, repo_dir, marker = _run_dsv4_route(tmp_path, uses_dcgm_power=False)

    assert f"git clone https://github.com/NVIDIA/srt-slurm.git {repo_dir}" in log
    assert "git checkout v1.0.25" in log
    assert all(FORK_URL not in entry and PRODUCER_PIN not in entry for entry in log)
    assert not (workspace / "power-producer-sha.txt").exists()
    assert marker.read_text() == "from-workspace\n"


def test_dsv4_gb300_master_image_matches_every_reachable_fixed_seq_recipe():
    config = yaml.safe_load(MASTER_CONFIG_PATH.read_text())[DSV4_GB300_CONFIG_KEY]
    fixed_seq = config["scenarios"]["fixed-seq-len"]
    config_files = list(_config_file_values(fixed_seq))

    assert len(config_files) == len(set(config_files)) == 7

    master_image = config["image"]
    problems = []
    if master_image in DEAD_DSV4_IMAGES:
        problems.append(f"master image is unavailable: {master_image}")

    for config_file in config_files:
        recipe_path = _workspace_recipe_path(config_file)
        assert recipe_path.is_file(), config_file
        recipe_image = yaml.safe_load(recipe_path.read_text())["model"]["container"]
        if recipe_image != master_image:
            problems.append(
                f"{config_file}: model.container={recipe_image!r}, "
                f"master image={master_image!r}"
            )

    assert problems == []


def test_operations_guide_describes_framework_gated_power_lanes():
    guide = OPERATIONS_GUIDE_PATH.read_text()

    assert (
        "Eligible recipe-gated `dynamo-sglang` dcgm-power lanes are validated."
        in guide
    )
    assert "Only `PRECISION=fp8` dcgm-power lanes are validated." not in guide
