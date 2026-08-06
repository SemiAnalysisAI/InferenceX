#!/usr/bin/env python3
"""Allocation and network checks used by CollectiveX launchers."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path


def default_route_interface(route_path: Path = Path("/proc/net/route")) -> str:
    for line in route_path.read_text().splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 4 and fields[1] == "00000000" and int(fields[3], 16) & 1:
            return fields[0]
    return ""


def prepare_cache(parent_path: str) -> str:
    path = Path(parent_path).resolve() / f".collectivex-backend-cache-{os.getuid()}"
    # parents=True: the parent is the operator's squash_dir, which on an established cluster
    # was created by the first container import long ago -- but this probe runs BEFORE that
    # import, so on a fresh pool the parent does not exist yet and a bare mkdir throws
    # FileNotFoundError (every b200-nscale leg of run 31092445934 died on exactly this, four
    # seconds in). The 0o700 applies to the cache dir; the squash dir itself stays
    # default-permissioned, as the import path's own mkdir -p would leave it.
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return str(path)


def validate_cuda_context(expected: int) -> None:
    cuda = ctypes.CDLL("libcuda.so.1")
    count = ctypes.c_int()
    if cuda.cuInit(0) != 0 or cuda.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value != expected:
        raise SystemExit(1)


_GPU_HEALTH_FIELDS = ("index", "clocks_event_reasons.sw_thermal_slowdown",
                      "clocks_event_reasons.hw_thermal_slowdown", "temperature.gpu")


def gpu_health_faults(output: str, max_temperature_c: int = 90) -> list[str]:
    """Throttled or overheating GPUs in an `nvidia-smi --format=csv,noheader` block.

    Split out from the I/O so the parsing is testable without hardware; see
    tests/test_runtime.py::GpuHealthProbe. Returns [] for anything it cannot read, because the
    caller treats an unreadable probe as healthy rather than blocking a leg on it.
    """
    faults = []
    for line in output.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != len(_GPU_HEALTH_FIELDS):
            continue
        index, software, hardware, temperature = cells
        # "Not Active" is the healthy reading, so compare exactly rather than searching for
        # "Active" -- a substring test passes the fault straight through.
        throttled = "Active" in (software, hardware)
        try:
            too_hot = int(temperature.split()[0]) > max_temperature_c
        except (IndexError, ValueError):
            too_hot = False
        if throttled or too_hot:
            faults.append(
                f"gpu {index}: sw_thermal={software} hw_thermal={hardware} temp={temperature}"
            )
    return faults


def gpu_temperature_spread(output: str) -> tuple[int, int, int] | None:
    """`(hottest, median, spread)` GPU temperature, or None if unreadable.

    Reported, NOT gated on. The absolute threshold in `gpu_health_faults` is architecture
    dependent and can be unreachable: an H100 engages its software thermal slowdown at ~86-87 C,
    so a clamped H100 never crosses a 90 C limit even under load, and at pre-flight it is idle
    anyway. In the one fault measured end to end, the only signal visible at pre-flight time was
    the RELATIVE one -- the sick GPU idled at 55 C against ~30 C for every sibling, then clamped
    under load. Recording the spread every run is how that becomes a gate with evidence behind its
    threshold instead of a heuristic picked from a single incident. Healthy references so far:
    50-66 C under load on h100 (~16 C spread), 34-39 C on b200.
    """
    temperatures = []
    for line in output.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != len(_GPU_HEALTH_FIELDS):
            continue
        try:
            temperatures.append(int(cells[3].split()[0]))
        except (IndexError, ValueError):
            continue
    if not temperatures:
        return None
    temperatures.sort()
    median = temperatures[len(temperatures) // 2]
    return temperatures[-1], median, temperatures[-1] - median


def validate_gpu_health(max_temperature_c: int = 90) -> None:
    """Reject an allocation holding a thermally throttled GPU.

    Every collective is a barrier across all ranks, so one clamped device paces the whole leg. A
    B200 with GPU 7 held at 120 MHz against 1965 MHz on its siblings ran a case 17x slower and was
    killed by the wall-clock guard twice, at 900s and again at 1800s, looking exactly like a code
    pathology. Rejecting the allocation costs seconds; diagnosing it cost two 30-minute burns.

    The signal is the throttle FLAG, not the clock: the allocation is idle at this point and an idle
    B200 also reads 120 MHz, so a clock threshold cannot tell health from idleness. Temperature is a
    second, independent signal because the flag can clear between samples while the fault persists.

    Fails OPEN on anything unexpected -- no `nvidia-smi`, non-zero exit, unparseable output. A check
    that blocks legs when it cannot read the hardware is worse than the fault it looks for.
    """
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(_GPU_HEALTH_FIELDS)}",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return
    faults = gpu_health_faults(output, max_temperature_c)
    for fault in faults:
        _emit(f"gpu-health-fault {fault}")
    if faults:
        raise SystemExit(1)
    # Positive control. Without it a gate that has gone BLIND -- no visible devices, or a driver
    # old enough to spell these fields `clocks_throttle_reasons.*` -- writes an empty log and is
    # indistinguishable from one that inspected eight healthy GPUs. Recording the count is what
    # makes "the gate ran and saw N devices" checkable per cluster instead of assumed.
    spread = gpu_temperature_spread(output)
    detail = "" if spread is None else f" hottest={spread[0]}C median={spread[1]}C spread={spread[2]}C"
    _emit(
        f"gpu-health-checked gpus={sum(1 for line in output.splitlines() if line.strip())}{detail}"
    )


def _emit(marker: str) -> None:
    # collx_validate_network_profile_on_job (runtime/common.sh) greps these exact strings
    # out of the per-node probe log to derive COLLX_SOCKET_IFNAME / COLLX_RDMA_LINK_LAYER and to
    # diagnose failures. The marker vocabulary is a string contract with that function —
    # keep the two halves in lockstep (see tests/test_runtime.py::NetworkProfileContract).
    print(f"[collectivex-private] {marker}")


def _check_port(port_path: Path, ordinal: int, gid_index: str, profile: str):
    # Return the port's link layer ("roce"/"infiniband") when it is active, carries a
    # non-empty GID at the pinned index, and agrees with any already-seen link layer;
    # otherwise emit the matching rdma-port-<ordinal>=<reason> marker and return None.
    if not port_path.is_dir():
        _emit(f"rdma-port-{ordinal}=missing"); return None
    state = port_path / "state"
    if not state.is_file() or state.read_text().split()[:1] != ["4:"]:
        _emit(f"rdma-port-{ordinal}=inactive"); return None
    if gid_index:
        gid = port_path / "gids" / gid_index
        if not gid.is_file():
            _emit(f"rdma-port-{ordinal}=gid-missing"); return None
        if not "".join(c for c in gid.read_text() if c not in ":0" and not c.isspace()):
            _emit(f"rdma-port-{ordinal}=gid-empty"); return None
    link = port_path / "link_layer"
    if not link.is_file():
        _emit(f"rdma-port-{ordinal}=link-layer-missing"); return None
    layer = {"Ethernet": "roce", "InfiniBand": "infiniband"}.get(link.read_text().strip())
    if layer is None:
        _emit(f"rdma-port-{ordinal}=link-layer-invalid"); return None
    if profile and profile != layer:
        _emit(f"rdma-port-{ordinal}=link-layer-mixed"); return None
    return layer


def validate_network_profile(socket_names: str, rdma_devices: str, gid_index: str,
                             sys_root: Path = Path("/sys"),
                             route_path: Path = Path("/proc/net/route")) -> None:
    # Prove the operator-pinned scale-out fabric on this node: resolve the cross-node socket
    # interface (operator selector, else this node's default route), confirm it is live, and
    # confirm every pinned RDMA port is active with a consistent link layer. On success emit
    # the socket-interface-selected and rdma-link-layer markers the launcher consumes; on any
    # failure emit the diagnostic marker and exit non-zero. The seam carries exactly ONE socket
    # interface: the launcher's marker-extraction regex has no comma, so a multi-interface
    # selector could never survive past this probe — fail it loudly here instead.
    interface = socket_names or default_route_interface(route_path)
    if not interface:
        _emit("socket-interface-1=default-route-missing")
        raise SystemExit(1)
    _emit(f"socket-interface-selected={interface}")
    net = sys_root / "class" / "net" / interface
    if not net.is_dir():
        _emit("socket-interface-1=missing"); raise SystemExit(1)
    operstate = net / "operstate"
    state = operstate.read_text().strip() if operstate.is_file() else ""
    if state not in ("up", "unknown"):
        _emit("socket-interface-1=down"); raise SystemExit(1)
    profile = ""
    for ordinal, selector in enumerate((s for s in rdma_devices.split(",") if s), start=1):
        device, _, configured_port = selector.partition(":")
        ports = sys_root / "class" / "infiniband" / device / "ports"
        if not ports.is_dir():
            _emit(f"rdma-device-{ordinal}=missing"); raise SystemExit(1)
        if configured_port:
            layer = _check_port(ports / configured_port, ordinal, gid_index, profile)
            if layer is None: raise SystemExit(1)
            profile = layer
        else:
            active = False
            for port_path in sorted(p for p in ports.iterdir() if p.is_dir()):
                layer = _check_port(port_path, ordinal, gid_index, profile)
                if layer is not None:
                    profile, active = layer, True
            if not active: raise SystemExit(1)
    if not profile: raise SystemExit(1)
    _emit(f"rdma-link-layer={profile}")


def main() -> None:
    parser = argparse.ArgumentParser(); commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("default-route-interface")
    command = commands.add_parser("prepare-cache"); command.add_argument("parent")
    command = commands.add_parser("cuda-context"); command.add_argument("expected", type=int)
    commands.add_parser("gpu-health")
    command = commands.add_parser("network-profile"); command.add_argument("socket_names"); command.add_argument("rdma_devices"); command.add_argument("gid_index")
    args = parser.parse_args()
    if args.command == "default-route-interface": print(default_route_interface(), end="")
    elif args.command == "prepare-cache": print(prepare_cache(args.parent), end="")
    elif args.command == "cuda-context": validate_cuda_context(args.expected)
    elif args.command == "gpu-health": validate_gpu_health()
    else: validate_network_profile(args.socket_names, args.rdma_devices, args.gid_index)


if __name__ == "__main__": main()
