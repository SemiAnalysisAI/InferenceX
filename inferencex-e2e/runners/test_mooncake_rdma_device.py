"""Exercise rail selection with Linux sysfs layouts, including DSXE names."""
import json
import subprocess
from pathlib import Path

LIB = Path(__file__).resolve().parents[1] / "benchmarks/benchmark_lib.sh"
PATCH_CONFIG = r"""
import json, sys
path, rail = sys.argv[1:]
if not rail.strip():
    raise SystemExit("Error: refusing to write empty Mooncake device_name")
with open(path) as handle:
    config = json.load(handle)
config["device_name"] = rail
with open(path, "w") as handle:
    json.dump(config, handle, indent=2)
written = json.load(open(path))
if not str(written.get("device_name") or "").strip():
    raise SystemExit(f"Error: mooncake store config still has empty device_name: {written!r}")
print(written["device_name"])
"""


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
    # Match kimik3-b300-mooncake.sh: the helper must resolve under --validation-only.
    return subprocess.run(
        [
            "bash",
            "-ec",
            (
                'source "$1" --validation-only; select_mooncake_rdma_device "$2"; '
                'printf "%s %s\n" "$MOONCAKE_RAIL" "$MC_GID_INDEX"'
            ),
            "bash",
            str(LIB),
            str(root),
        ],
        text=True,
        capture_output=True,
        timeout=5,
    )


def patch_config(path: Path, rail: str) -> subprocess.CompletedProcess[str]:
    """Same write+verify contract as kimik3-b300-mooncake.sh."""
    return subprocess.run(
        ["python3", "-", str(path), rail],
        input=PATCH_CONFIG,
        text=True,
        capture_output=True,
        timeout=5,
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


def test_no_usable_rail_fails(tmp_path: Path) -> None:
    add_device(tmp_path, "mlx5_0", state="1: DOWN")
    add_device(tmp_path, "rdmap86s0", driver="efa", layer="Unknown")
    result = select(tmp_path)
    assert result.returncode != 0
    assert result.stdout == ""


def test_select_then_patch_replaces_empty_device_name(tmp_path: Path) -> None:
    add_device(tmp_path, "ibp198s0f0")
    config = tmp_path / "mooncake_store_config.json"
    config.write_text(json.dumps({
        "mode": "embedded",
        "protocol": "rdma",
        "device_name": "",
        "enable_offload": False,
    }))
    selected = select(tmp_path)
    assert selected.returncode == 0, selected.stderr
    rail, gid = selected.stdout.strip().split()
    assert rail == "ibp198s0f0"
    assert gid == "0"
    patched = patch_config(config, rail)
    assert patched.returncode == 0, patched.stderr
    assert patched.stdout.strip() == "ibp198s0f0"
    written = json.loads(config.read_text())
    assert written["device_name"] == "ibp198s0f0"
    assert written["protocol"] == "rdma"


def test_empty_rail_cannot_silently_patch_device_name(tmp_path: Path) -> None:
    config = tmp_path / "mooncake_store_config.json"
    config.write_text(json.dumps({
        "mode": "embedded",
        "protocol": "rdma",
        "device_name": "",
    }))
    for empty in ("", "   "):
        result = patch_config(config, empty)
        assert result.returncode != 0, result.stdout
        assert "empty" in (result.stderr + result.stdout).lower()
        assert json.loads(config.read_text())["device_name"] == ""
