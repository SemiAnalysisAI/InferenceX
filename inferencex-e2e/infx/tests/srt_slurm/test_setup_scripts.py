"""Setup scripts that install a master-versioned component, run with a recording python3."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from infx.srt_slurm.workload import INSTALLERS, VERSION_ENV
from infx.tests.bench.stubs import executable

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "benchmarks/multi_node/srt-slurm-recipes/configs"
PYTHON3 = f"""#!{sys.executable}
import json, os, sys
with open(os.environ["STUB_PYTHON3_TRACE"], "a") as trace:
    trace.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1:] == ["-m", "pip", "install", "--help"]:
    print("  --break-system-packages")
"""
MOUNT = 'source() { builtin source "${1/#\\/infmax-workspace/$STUB_WORKSPACE}" "${@:2}"; }\n'


def run(tmp_path: Path, script: str, env: dict[str, str]) -> tuple[subprocess.CompletedProcess, list]:
    """Run ``script`` as srtctl does; return the result and the pip installs it ran."""
    (tmp_path / "bin").mkdir()
    executable(tmp_path / "bin/python3", PYTHON3)
    (tmp_path / "mount.sh").write_text(MOUNT)
    trace = tmp_path / "python3.jsonl"
    result = subprocess.run(
        ["bash", str(CONFIGS / script)],
        env={
            "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
            "BASH_ENV": str(tmp_path / "mount.sh"),
            "STUB_WORKSPACE": str(ROOT),
            "STUB_PYTHON3_TRACE": str(trace),
            **env,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    calls = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    return result, [call for call in calls if call[:3] == ["-m", "pip", "install"] and "--help" not in call]


@pytest.mark.parametrize("script", sorted(INSTALLERS))
def test_an_installer_fails_without_the_version_the_binder_writes(tmp_path, script):
    result, installs = run(tmp_path, script, {})

    assert result.returncode == 1
    assert f"  - {VERSION_ENV[INSTALLERS[script][0]]}\n" in result.stdout
    assert installs == []


@pytest.mark.parametrize(("script", "env", "expected"), [
    ("vllm-router.sh", {"ROUTER_VERSION": "9.8.7"}, {"vllm-router==9.8.7"}),
    ("vllm-mooncake.sh", {"KV_OFFLOAD_BACKEND_VERSION": "9.8.7"},
     {"mooncake-transfer-engine-cuda13==9.8.7"}),
    ("glm5.3-tilert-rocm.sh", {"ROUTER_VERSION": "9.8.7", "TILERT_ROLE": "prefill"}, {"tilert==9.8.7"}),
])  # fmt: skip
def test_an_installer_installs_exactly_the_bound_version(tmp_path, script, env, expected):
    result, installs = run(tmp_path, script, env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert expected <= set(installs[0])


def b300_mooncake(command: str, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Source the B300 Mooncake setup script, which then only defines its functions, and run ``command``."""
    return subprocess.run(
        ["bash", "-c", f'source "$1"; shift; {command}', "_", str(CONFIGS / "kimik3-b300-mooncake.sh"), *args],
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        check=False,
    )


def rdma_port(root: Path, name: str, driver: str, state: str, link_layer: str = "InfiniBand") -> None:
    """One sysfs ``infiniband`` class entry: driver symlink, port 1 state and link layer."""
    (root / name / "device").mkdir(parents=True)
    (root / name / "device/driver").symlink_to(f"../../../bus/pci/drivers/{driver}")
    (root / name / "ports/1").mkdir(parents=True)
    (root / name / "ports/1/state").write_text(f"{state}\n")
    (root / name / "ports/1/link_layer").write_text(f"{link_layer}\n")


def select_rail(root: Path) -> subprocess.CompletedProcess:
    return b300_mooncake('select_mooncake_rdma_device "$1"; echo "$MOONCAKE_RAIL"', str(root))


def test_kimik3_b300_rail_is_the_first_active_mellanox_port_never_efa(tmp_path):
    rdma_port(tmp_path, "a_efa", "efa", "4: ACTIVE")
    rdma_port(tmp_path, "ibp198s0f0", "mlx5_core", "1: DOWN")
    rdma_port(tmp_path, "ibp199s0f0", "mlx5_core", "4: ACTIVE")
    rdma_port(tmp_path, "rdmap86s0", "efa", "4: ACTIVE")

    result = select_rail(tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "ibp199s0f0\n"


def test_kimik3_b300_rail_skips_an_active_ethernet_mellanox_port(tmp_path):
    rdma_port(tmp_path, "mlx5_0", "mlx5_core", "4: ACTIVE", "Ethernet")
    rdma_port(tmp_path, "mlx5_1", "mlx5_core", "4: ACTIVE")

    result = select_rail(tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "mlx5_1\n"


def test_kimik3_b300_rail_fails_instead_of_falling_back_to_efa(tmp_path):
    rdma_port(tmp_path, "ibp198s0f0", "mlx5_core", "1: DOWN")
    rdma_port(tmp_path, "mlx5_bond_0", "mlx5_core", "4: ACTIVE", "Ethernet")
    rdma_port(tmp_path, "rdmap86s0", "efa", "4: ACTIVE")

    result = select_rail(tmp_path)

    assert result.returncode == 1
    assert result.stdout == ""
    assert "no ACTIVE InfiniBand mlx5_core port" in result.stderr
    assert "rdmap86s0: driver=../../../bus/pci/drivers/efa state=4: ACTIVE link_layer=InfiniBand" in result.stderr


RECIPE = ROOT / "benchmarks/single_node/srt-slurm-recipes/kimik3/vllm/b300-fp4-mtp/agentic.yaml"


def host_with_mlx5_provider(root: Path) -> None:
    """The host's rdma-core layout: libibverbs/libmlx5-rdmav34.so links to ../libmlx5.so.1.*."""
    lib = root / "usr/lib/x86_64-linux-gnu"
    (lib / "libibverbs").mkdir(parents=True)
    (lib / "libmlx5.so.1.25.58.0").write_bytes(b"\x7fELF")
    (lib / "libibverbs/libmlx5-rdmav34.so").symlink_to("../libmlx5.so.1.25.58.0")


def bind(host: Path, container: Path, spec: str) -> None:
    """Bind ``host:container[:flags]``: the container sees the tree, links unresolved, at its own path."""
    source, target = spec.split(":")[:2]
    shutil.copytree(host / source.lstrip("/"), container / target.lstrip("/"), symlinks=True)


def check_provider(container: Path, drivers: str) -> subprocess.CompletedProcess:
    return b300_mooncake(
        'check_mooncake_verbs_provider; echo "$MOONCAKE_VERBS_PROVIDER"',
        env={"RDMAV_DRIVERS": f"{container}{drivers}"},
    )


def test_kimik3_b300_recipe_mount_resolves_the_host_mlx5_provider(tmp_path):
    base = yaml.safe_load(RECIPE.read_text())["base"]
    host_with_mlx5_provider(tmp_path / "host")
    for spec in base["extra_mount"]:
        bind(tmp_path / "host", tmp_path / "container", spec)

    result = check_provider(tmp_path / "container", base["roles"]["agg"]["env"]["RDMAV_DRIVERS"])

    assert result.returncode == 0, result.stderr
    assert result.stdout.endswith("/host-usr-lib/libmlx5.so.1.25.58.0\n")


def test_kimik3_b300_provider_check_fails_when_only_the_provider_directory_is_mounted(tmp_path):
    host_with_mlx5_provider(tmp_path / "host")
    bind(tmp_path / "host", tmp_path / "container", "/usr/lib/x86_64-linux-gnu/libibverbs:/host-usr-lib/libibverbs:ro")

    result = check_provider(tmp_path / "container", "/host-usr-lib/libibverbs/libmlx5")

    assert result.returncode == 1
    assert "no verbs provider resolves" in result.stderr
    assert "libmlx5-rdmav34.so -> ../libmlx5.so.1.25.58.0" in result.stderr


def test_kimik3_b300_setup_rejects_a_roce_gid_index_before_installing(tmp_path):
    result, installs = run(tmp_path, "kimik3-b300-mooncake.sh", {
        "KV_OFFLOAD_BACKEND_VERSION": "9.8.7",
        "MOONCAKE_CONFIG_PATH": str(tmp_path / "mooncake_store_config.json"),
        "MC_GID_INDEX": "3",
        "RDMAV_DRIVERS": str(tmp_path / "libmlx5"),
    })  # fmt: skip

    assert result.returncode == 1
    assert "MC_GID_INDEX=3; the InfiniBand Mooncake rail needs GID index 0" in result.stderr
    assert installs == []
