"""Exercise rail selection with Linux sysfs layouts, including DSXE names."""
import subprocess
from pathlib import Path

import pytest

LIB = Path(__file__).resolve().parents[1] / "benchmarks/benchmark_lib.sh"


def add_device(
    root: Path, name: str, *, driver: str = "mlx5_core",
    state: str = "4: ACTIVE", layer: str = "InfiniBand",
) -> None:
    device = root / name
    port = device / "ports/1"
    port.mkdir(parents=True)
    (port / "state").write_text(state + "\n")
    (port / "link_layer").write_text(layer + "\n")
    (device / "device").mkdir()
    (device / "device/driver").symlink_to("/sys/bus/pci/drivers/" + driver)


def select(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-ec", 'source "$1"; select_mooncake_rdma_device "$2"; '
         'printf "%s %s\n" "$MOONCAKE_RAIL" "$MC_GID_INDEX"',
         "bash", str(LIB), str(root)],
        text=True, capture_output=True, timeout=5,
    )


def test_renamed_infiniband_device_skips_efa_and_down_port(tmp_path: Path) -> None:
    add_device(tmp_path, "a_efa", driver="efa", layer="Unknown")
    add_device(tmp_path, "ibp198s0f0", state="1: DOWN")
    add_device(tmp_path, "ibp199s0f0")
    result = select(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ibp199s0f0 0\n"


def test_roce_keeps_gid_three(tmp_path: Path) -> None:
    add_device(tmp_path, "mlx5_0", state="1: DOWN", layer="Ethernet")
    add_device(tmp_path, "mlx5_1", layer="Ethernet")
    result = select(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "mlx5_1 3\n"


@pytest.mark.parametrize("device", ["missing", "down", "efa", "unknown-layer"])
def test_no_usable_rail_fails(tmp_path: Path, device: str) -> None:
    if device == "down":
        add_device(tmp_path, "mlx5_0", state="1: DOWN")
    elif device == "efa":
        add_device(tmp_path, "rdmap86s0", driver="efa", layer="Unknown")
    elif device == "unknown-layer":
        add_device(tmp_path, "mlx5_0", layer="Unknown")
    result = select(tmp_path)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "no active Mellanox RDMA rail" in result.stderr
