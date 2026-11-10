import argparse
import ctypes as ct
import json
import os
import time
from pathlib import Path
from typing import Any

from infx.bench.env import BenchError


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, "monotonic": time.monotonic(), **fields}), flush=True)


def api(lib: Any, name: str, result: Any, args: list[Any]) -> Any:
    fn = getattr(lib, name)
    fn.restype, fn.argtypes = result, args
    return fn


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Register every pool segment on every requested NIC"
    )
    parser.add_argument(
        "--arena",
        action="append",
        required=True,
        help="NUMA_NODE:LOGICAL_BYTES; repeat for each actual arena",
    )
    parser.add_argument(
        "--nic",
        action="append",
        required=True,
        help="Exact ibv device name; repeat for all actual devices",
    )
    parser.add_argument("--segment-bytes", type=int, required=True)
    parser.add_argument("--page-bytes", type=int, required=True)
    parser.add_argument("--access-flags", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    arenas = [tuple(map(int, item.split(":"))) for item in args.arena]
    if (
        args.page_bytes != 2097152
        or args.segment_bytes <= 0
        or args.segment_bytes % args.page_bytes
    ):
        raise BenchError("Require 2 MiB pages and a positive, page-aligned MR limit")
    if len(args.nic) != len(set(args.nic)) or not all(
        node >= 0 and size > 0 for node, size in arenas
    ):
        raise BenchError("Require unique NICs and positive arena sizes")
    segments = [
        (i, node, offset, min(args.segment_bytes, size - offset))
        for i, (node, size) in enumerate(arenas)
        for offset in range(0, size, args.segment_bytes)
    ]
    emit(
        "plan",
        arenas=arenas,
        nics=args.nic,
        segments=segments,
        registrations=len(segments) * len(args.nic),
        flags=args.access_flags,
        scope="all arena segments x all requested NICs, all successful MRs retained concurrently",
        execute=args.execute,
    )
    if not args.execute:
        return 0
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
    emit(
        "environment",
        hostname=os.uname().nodename,
        memlock_before=[soft, hard],
        limits=Path("/proc/self/limits").read_text(),
        hugepages={
            str(node): {
                p.name: p.read_text().strip()
                for p in (
                    Path(f"/sys/devices/system/node/node{node}/hugepages/hugepages-2048kB")
                ).glob("*")
                if p.is_file()
            }
            for node, _ in arenas
        },
    )
    libc = ct.CDLL(None, use_errno=True)
    verbs = ct.CDLL("libibverbs.so.1", use_errno=True)
    numa = ct.CDLL("libnuma.so.1", use_errno=True)
    mmap = api(
        libc,
        "mmap",
        ct.c_void_p,
        [ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_int, ct.c_int, ct.c_long],
    )
    munmap = api(libc, "munmap", ct.c_int, [ct.c_void_p, ct.c_size_t])
    mbind = api(
        numa,
        "mbind",
        ct.c_long,
        [ct.c_void_p, ct.c_ulong, ct.c_int, ct.POINTER(ct.c_ulong), ct.c_ulong, ct.c_uint],
    )
    getlist = api(verbs, "ibv_get_device_list", ct.POINTER(ct.c_void_p), [ct.POINTER(ct.c_int)])
    freelist = api(verbs, "ibv_free_device_list", None, [ct.POINTER(ct.c_void_p)])
    getname = api(verbs, "ibv_get_device_name", ct.c_char_p, [ct.c_void_p])
    opendev = api(verbs, "ibv_open_device", ct.c_void_p, [ct.c_void_p])
    closedev = api(verbs, "ibv_close_device", ct.c_int, [ct.c_void_p])
    allocpd = api(verbs, "ibv_alloc_pd", ct.c_void_p, [ct.c_void_p])
    deallocpd = api(verbs, "ibv_dealloc_pd", ct.c_int, [ct.c_void_p])
    regmr = api(verbs, "ibv_reg_mr", ct.c_void_p, [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int])
    deregmr = api(verbs, "ibv_dereg_mr", ct.c_int, [ct.c_void_p])
    count = ct.c_int()
    devices = getlist(ct.byref(count))
    if not devices:
        raise OSError(ct.get_errno(), "ibv_get_device_list")
    contexts, pds, buffers, mrs = [], {}, [], []
    failures = 0
    try:
        available = {getname(devices[i]).decode(): devices[i] for i in range(count.value)}
        if set(args.nic) - available.keys():
            raise RuntimeError(f"Requested NIC absent: {set(args.nic) - available.keys()}")
        for name in args.nic:
            context = opendev(available[name])
            if not context:
                raise OSError(ct.get_errno(), f"open device {name}")
            contexts.append(context)
            pd = allocpd(context)
            if not pd:
                raise OSError(ct.get_errno(), f"allocate PD {name}")
            pds[name] = pd
        for node, size in arenas:
            mapped = (size + args.page_bytes - 1) // args.page_bytes * args.page_bytes
            ptr = mmap(None, mapped, 3, 2 | 32 | 0x40000 | (21 << 26), -1, 0)
            if ptr == ct.c_void_p(-1).value:
                raise OSError(ct.get_errno(), f"HugeTLB mmap {mapped}")
            buffers.append((ptr, mapped))
            bits = ct.sizeof(ct.c_ulong) * 8
            mask = (ct.c_ulong * (node // bits + 1))()
            mask[node // bits] = 1 << (node % bits)
            if mbind(ptr, mapped, 2, mask, node + 1, 1) != 0:
                raise OSError(ct.get_errno(), f"mbind node {node}")
            for offset in range(0, mapped, args.page_bytes):
                ct.c_ubyte.from_address(ptr + offset).value = 0
            emit(
                "arena_ready", numa=node, address=hex(ptr), logical_bytes=size, mapped_bytes=mapped
            )
        for arena, node, offset, size in segments:
            ptr = buffers[arena][0] + offset
            for name, pd in pds.items():
                ct.set_errno(0)
                started = time.monotonic()
                mr = regmr(pd, ptr, size, args.access_flags)
                error = ct.get_errno() if not mr else 0
                emit(
                    "registration",
                    nic=name,
                    arena=arena,
                    numa=node,
                    offset=offset,
                    address=hex(ptr),
                    bytes=size,
                    flags=args.access_flags,
                    ok=bool(mr),
                    errno=error,
                    error=os.strerror(error),
                    seconds=time.monotonic() - started,
                )
                if mr:
                    mrs.append(mr)
                else:
                    failures += 1
    finally:
        for mr in reversed(mrs):
            if deregmr(mr):
                failures += 1
                emit("cleanup_error", operation="ibv_dereg_mr", errno=ct.get_errno())
        for pd in pds.values():
            if deallocpd(pd):
                failures += 1
                emit("cleanup_error", operation="ibv_dealloc_pd", errno=ct.get_errno())
        for context in contexts:
            if closedev(context):
                failures += 1
                emit("cleanup_error", operation="ibv_close_device", errno=ct.get_errno())
        freelist(devices)
        for ptr, size in buffers:
            if munmap(ptr, size):
                failures += 1
                emit("cleanup_error", operation="munmap", errno=ct.get_errno())
    emit("result", failures=failures, successful=len(mrs), expected=len(segments) * len(args.nic))
    return 1 if failures else 0
