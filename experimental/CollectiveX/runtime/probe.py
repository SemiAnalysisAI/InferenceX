#!/usr/bin/env python3
"""Allocation and network checks used by CollectiveX launchers."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import socket


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


def verify_cache_mount(root: str) -> None:
    if root != "/cx-cache" or not Path(root).is_dir(): raise SystemExit(1)


def validate_cuda_context(expected: int) -> None:
    cuda = ctypes.CDLL("libcuda.so.1")
    count = ctypes.c_int()
    if cuda.cuInit(0) != 0 or cuda.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value != expected:
        raise SystemExit(1)


def validate_network_profile(socket_names: str, rdma_devices: str, gid_index: str) -> None:
    for name in filter(None, socket_names.split(",")):
        socket.if_nametoindex(name)
    for selector in filter(None, rdma_devices.split(",")):
        name, _, port = selector.partition(":")
        port = port or "1"
        state = Path(f"/sys/class/infiniband/{name}/ports/{port}/state")
        if not state.read_text().strip().startswith("4:"): raise SystemExit(1)
        if gid_index and not Path(f"/sys/class/infiniband/{name}/ports/{port}/gids/{gid_index}").exists():
            raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(); commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("default-route-interface")
    command = commands.add_parser("prepare-cache"); command.add_argument("parent")
    command = commands.add_parser("verify-cache-mount"); command.add_argument("root")
    command = commands.add_parser("cuda-context"); command.add_argument("expected", type=int)
    command = commands.add_parser("network-profile"); command.add_argument("socket_names"); command.add_argument("rdma_devices"); command.add_argument("gid_index")
    args = parser.parse_args()
    if args.command == "default-route-interface": print(default_route_interface(), end="")
    elif args.command == "prepare-cache": print(prepare_cache(args.parent), end="")
    elif args.command == "verify-cache-mount": verify_cache_mount(args.root)
    elif args.command == "cuda-context": validate_cuda_context(args.expected)
    else: validate_network_profile(args.socket_names, args.rdma_devices, args.gid_index)


if __name__ == "__main__": main()
