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
    path.mkdir(mode=0o700, exist_ok=True)
    os.chmod(path, 0o700)
    return str(path)


def validate_cuda_context(expected: int) -> None:
    cuda = ctypes.CDLL("libcuda.so.1")
    count = ctypes.c_int()
    if cuda.cuInit(0) != 0 or cuda.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value != expected:
        raise SystemExit(1)


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
    command = commands.add_parser("network-profile"); command.add_argument("socket_names"); command.add_argument("rdma_devices"); command.add_argument("gid_index")
    args = parser.parse_args()
    if args.command == "default-route-interface": print(default_route_interface(), end="")
    elif args.command == "prepare-cache": print(prepare_cache(args.parent), end="")
    elif args.command == "cuda-context": validate_cuda_context(args.expected)
    else: validate_network_profile(args.socket_names, args.rdma_devices, args.gid_index)


if __name__ == "__main__": main()
