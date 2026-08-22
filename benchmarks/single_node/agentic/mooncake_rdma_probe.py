#!/usr/bin/env python3
"""Diagnose Mooncake RDMA segment registration on the CI compute node.

Runs before the engine loads weights so a full hypothesis sweep costs seconds
instead of a six-minute CI cycle. Every cell spawns real worker processes so
the eight-rank contention is reproduced rather than approximated.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

GB = 1024 * 1024 * 1024


def sh(cmd):
    try:
        out = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=60
        )
        return (out.stdout or "") + (out.stderr or "")
    except Exception as exc:  # noqa: BLE001
        return f"<{exc}>"


def nics():
    try:
        return sorted(os.listdir("/sys/class/infiniband"))
    except OSError:
        return []


def dump_environment():
    print("=" * 72, flush=True)
    print("PROBE: environment", flush=True)
    print("=" * 72, flush=True)
    print("nics:", nics(), flush=True)
    print("ulimit -l:", sh("bash -c 'ulimit -l; ulimit -Hl'").strip(), flush=True)
    print(sh("grep -E 'Huge|MemFree|MemAvailable' /proc/meminfo").strip(), flush=True)
    for var in (
        "MC_STORE_USE_HUGEPAGE",
        "MC_STORE_HUGEPAGE_SIZE",
        "MC_MAX_MR_SIZE",
        "MC_GID_INDEX",
        "MC_ENABLE_PARALLEL_REG_MR",
        "MC_STORE_MEMCPY",
    ):
        print(f"env {var}={os.environ.get(var)}", flush=True)

    if shutil.which("ibv_devinfo"):
        text = sh("ibv_devinfo -v -d rdma0")
        keep = (
            "max_mr",
            "max_mr_size",
            "max_pd",
            "max_qp",
            "phys_port_cnt",
            "hca_id",
            "state",
            "link_layer",
            "page_size_cap",
        )
        for line in text.splitlines():
            if any(k in line for k in keep):
                print("ibv:", line.strip(), flush=True)
    else:
        print("ibv_devinfo not present in image", flush=True)

    try:
        from mooncake.store import MooncakeDistributedStore  # noqa: PLC0415
        import inspect  # noqa: PLC0415

        print(
            "setup signature:",
            inspect.signature(MooncakeDistributedStore.setup),
            flush=True,
        )
    except Exception:  # noqa: BLE001
        print("could not introspect setup():", flush=True)
        traceback.print_exc()


def preload_gpu(index, gib):
    """Reproduce the worker's ROCm state before Mooncake registers anything."""
    import torch

    torch.cuda.set_device(index % torch.cuda.device_count())
    blocks = []
    for _ in range(gib):
        blocks.append(torch.empty(1024**3, dtype=torch.uint8, device="cuda"))
    torch.cuda.synchronize()
    print(f"WORKER gpu_reserved={torch.cuda.memory_allocated() // 1024**3}GiB", flush=True)
    return blocks


def preload_host(gib):
    buffers = []
    for _ in range(gib):
        buf = bytearray(1024**3)
        buf[::4096] = b"\x01" * (len(buf) // 4096)
        buffers.append(buf)
    print(f"WORKER host_touched={gib}GiB", flush=True)
    return buffers


def worker(args):
    """Register one segment and report the return code."""
    held = []
    if args.gpu_gib:
        held.append(preload_gpu(args.port % 8, args.gpu_gib))
    if args.host_gib:
        held.append(preload_host(args.host_gib))

    os.environ["MC_STORE_USE_HUGEPAGE"] = "1" if args.hugepage else "0"
    if args.hugepage:
        os.environ["MC_STORE_HUGEPAGE_SIZE"] = "2MB"
    if args.max_mr_size:
        os.environ["MC_MAX_MR_SIZE"] = str(args.max_mr_size)
    else:
        os.environ.pop("MC_MAX_MR_SIZE", None)

    cfg = {
        "mode": "embedded",
        "metadata_server": "P2PHANDSHAKE",
        "master_server_address": args.master,
        "global_segment_size": f"{args.segment_mb}MB",
        "local_buffer_size": "128MB",
        "protocol": "rdma",
        "device_name": args.device,
        "enable_offload": False,
    }
    handle, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(handle, "w") as fh:
        json.dump(cfg, fh)
    os.environ["MOONCAKE_CONFIG_PATH"] = path

    def do_setup():
        from mooncake.store import MooncakeDistributedStore

        store = MooncakeDistributedStore()
        return store.setup(
            f"127.0.0.1:{args.port}",
            "P2PHANDSHAKE",
            args.segment_mb * 1024 * 1024,
            128 * 1024 * 1024,
            "rdma",
            args.device,
            args.master,
        )

    # vLLM spawns its workers from the engine core, and registering memory in a
    # child after the parent has initialised the GPU is the one structural
    # difference left between this probe and the failing run.
    if args.fork:
        pid = os.fork()
        if pid == 0:
            try:
                rc = do_setup()
                print(f"WORKER forked rc={rc}", flush=True)
                os._exit(0 if rc == 0 else 1)
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                os._exit(1)
        return 1 if os.waitpid(pid, 0)[1] else 0

    rc = do_setup()
    print(f"WORKER rc={rc}", flush=True)
    return 0 if rc == 0 else 1


def run_cell(
    name,
    procs,
    segment_mb,
    hugepage,
    per_rank_nic,
    master,
    max_mr_size,
    gpu_gib=0,
    host_gib=0,
    fork=False,
):
    available = nics() or ["rdma0"]
    children = []
    for idx in range(procs):
        device = available[idx % len(available)] if per_rank_nic else available[0]
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--worker",
            "--segment-mb",
            str(segment_mb),
            "--device",
            device,
            "--master",
            master,
            "--port",
            str(20000 + idx),
        ]
        if hugepage:
            cmd.append("--hugepage")
        if max_mr_size:
            cmd += ["--max-mr-size", str(max_mr_size)]
        if gpu_gib:
            cmd += ["--gpu-gib", str(gpu_gib)]
        if host_gib:
            cmd += ["--host-gib", str(host_gib)]
        if fork:
            cmd.append("--fork")
        children.append(
            subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        )

    failures = 0
    first_error = ""
    interesting = (
        "NUMA-segmented",
        "Mounting segment",
        "Failed to register",
        "rc=",
        "Allocated NUMA",
        "Using specified RDMA devices",
        "gpu_reserved",
    )
    for pos, child in enumerate(children):
        out = child.communicate(timeout=600)[0].decode(errors="replace")
        if pos == 0:
            for line in out.splitlines():
                if any(k in line for k in interesting):
                    print(f"    rank0| {line.strip()[:180]}", flush=True)
        if child.returncode != 0:
            failures += 1
            if not first_error:
                for line in out.splitlines():
                    if "Failed to register" in line or "Error" in line:
                        first_error = line.strip()
                        break
                first_error = first_error or (out.strip().splitlines() or [""])[-1]

    verdict = "OK" if failures == 0 else f"FAIL {failures}/{procs}"
    print(f"PROBE_RESULT {name:<38} {verdict}", flush=True)
    if first_error:
        print(f"    first error: {first_error[:200]}", flush=True)
    return failures == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--segment-mb", type=int, default=2048)
    parser.add_argument("--device", default="rdma0")
    parser.add_argument("--master", default="127.0.0.1:20888")
    parser.add_argument("--port", type=int, default=20000)
    parser.add_argument("--hugepage", action="store_true")
    parser.add_argument("--max-mr-size", type=int, default=0)
    parser.add_argument("--gpu-gib", type=int, default=0)
    parser.add_argument("--host-gib", type=int, default=0)
    parser.add_argument("--fork", action="store_true")
    args = parser.parse_args()

    if args.worker:
        try:
            return worker(args)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            return 1

    dump_environment()
    print("=" * 72, flush=True)
    print("PROBE: sweep", flush=True)
    print("=" * 72, flush=True)

    big_mr = 34359738368
    # The bare sweep (segment size, page size, NIC binding, process count,
    # max_mr_size) passed every cell in CI run 32597763849, so the failure needs
    # the worker's process state: ROCm initialised with the weights resident.
    cells = [
        ("8proc 2GB baseline", 8, 2048, True, False, big_mr, 0, 0, False),
        ("8proc 2GB forked child", 8, 2048, True, False, big_mr, 0, 0, True),
        ("8proc 2GB + 180GiB gpu forked", 8, 2048, True, False, big_mr, 180, 0, True),
        ("8proc 8GB forked child", 8, 8192, True, False, big_mr, 0, 0, True),
        ("8proc 8GB + 180GiB gpu forked", 8, 8192, True, False, big_mr, 180, 0, True),
    ]
    for name, procs, segment_mb, hugepage, per_rank, max_mr, gpu, host, fork in cells:
        print(sh("grep HugePages_Free /proc/meminfo").strip(), flush=True)
        try:
            run_cell(
                name,
                procs,
                segment_mb,
                hugepage,
                per_rank,
                args.master,
                max_mr,
                gpu_gib=gpu,
                host_gib=host,
                fork=fork,
            )
        except Exception:  # noqa: BLE001
            print(f"PROBE_RESULT {name:<38} EXCEPTION", flush=True)
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
