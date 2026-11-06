#!/usr/bin/env python3
"""Run GPU RMA through libfabric before the engine loads, and report which setup works.

A broken EFA or CUDA stack otherwise shows up only after model load and warmup, as requests that return
nothing. Each setup runs in its own process because libfabric reads its environment, and the process maps
libcuda, only once. Exit status 1 means NIXL reported a failed GPU transfer in the workers' environment and
pinning the host's libcuda didn't help. A probe that couldn't run or finish doesn't stop the engine. The
workers' LD_LIBRARY_PATH starts with PIN_DIR, which stays absent unless that pin is needed.
"""

import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

PIN_DIR = Path("/opt/host-cuda")
TRY_DIR = Path("/opt/host-cuda.try")
DRIVER_LIBS = Path("/usr/lib/x86_64-linux-gnu")
SIZE = 1 << 20
TRANSFER_TIMEOUT_S = 20
SETUP_TIMEOUT_S = 240
SYNC_MEMOPS = 6
MARK = "PREFLIGHT "


def sync_memops(cuda, ptr: int) -> int:
    value = ctypes.c_int(1)
    return cuda.cuPointerSetAttribute(
        ctypes.byref(value), SYNC_MEMOPS, ctypes.c_ulonglong(ptr)
    )


def transfer(
    np,
    agent,
    peer: str,
    op: str,
    mem: str,
    local,
    remote,
    local_dev: int,
    remote_dev: int,
) -> str:
    """Move SIZE bytes between two tensors, then check they arrived."""
    source, sink = (remote, local) if op == "READ" else (local, remote)
    source.fill_(0x5A)
    sink.fill_(0)
    try:
        local_list = np.array([[local.data_ptr(), SIZE, local_dev]], dtype=np.int64)
        remote_list = np.array([[remote.data_ptr(), SIZE, remote_dev]], dtype=np.int64)
        local_side = agent.prep_xfer_dlist("NIXL_INIT_AGENT", local_list, mem)
        remote_side = agent.prep_xfer_dlist(peer, remote_list, mem)
        handle = agent.make_prepped_xfer(op, local_side, [0], remote_side, [0])
        state = agent.transfer(handle)
        deadline = time.monotonic() + TRANSFER_TIMEOUT_S
        while state not in ("DONE", "ERR"):
            if time.monotonic() > deadline:
                return "timeout"
            state = agent.check_xfer_state(handle)
    except Exception as error:  # noqa: BLE001 - any NIXL error is a result to report
        return f"{type(error).__name__}: {str(error)[:120]}"
    if state == "ERR":
        return "error"
    return "ok" if bool((sink == 0x5A).all()) else "corrupt"


def report(result: dict) -> None:
    print(MARK + json.dumps(result), flush=True)


def run_setup() -> None:
    """Runs in a child process. Reports through MARK lines, so a NIXL crash keeps the CUDA results."""
    import numpy as np
    import torch

    torch.cuda.init()
    cuda = ctypes.CDLL("libcuda.so.1")
    version = ctypes.c_int()
    cuda.cuDriverGetVersion(ctypes.byref(version))
    with open("/proc/self/maps", encoding="utf-8") as maps:
        libcuda = sorted({line.split()[-1] for line in maps if "libcuda.so" in line})
    result = {"cuda_driver": version.value, "libcuda": libcuda}

    local_dev, remote_dev = 0, min(1, torch.cuda.device_count() - 1)
    gpu_local = torch.empty(SIZE, dtype=torch.uint8, device=f"cuda:{local_dev}")
    gpu_remote = torch.empty(SIZE, dtype=torch.uint8, device=f"cuda:{remote_dev}")
    result["sync_memops_main"] = sync_memops(cuda, gpu_local.data_ptr())
    # Mooncake and NIXL post from their own threads, which have no current CUDA context.
    worker: dict = {}
    thread = threading.Thread(
        target=lambda: worker.update(rc=sync_memops(cuda, gpu_remote.data_ptr()))
    )
    thread.start()
    thread.join()
    result["sync_memops_thread"] = worker.get("rc")
    report(result)

    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError:
        from nixl_cu13._api import nixl_agent, nixl_agent_config
    cpu_local, cpu_remote = (
        torch.empty(SIZE, dtype=torch.uint8),
        torch.empty(SIZE, dtype=torch.uint8),
    )
    try:
        config = nixl_agent_config(backends=["LIBFABRIC"])
        local_agent = nixl_agent("preflight-local", config)
        remote_agent = nixl_agent("preflight-remote", config)
        local_agent.register_memory([gpu_local])
        local_agent.register_memory([cpu_local])
        remote_agent.register_memory([gpu_remote])
        remote_agent.register_memory([cpu_remote])
        peer = local_agent.add_remote_agent(remote_agent.get_agent_metadata())
        remote_agent.add_remote_agent(local_agent.get_agent_metadata())
    except Exception as error:  # noqa: BLE001 - any NIXL error is a result to report
        result["nixl_setup"] = f"{type(error).__name__}: {str(error)[:120]}"
        report(result)
        return
    for op in ("READ", "WRITE"):
        result[f"cpu_{op}"] = transfer(
            np, local_agent, peer, op, "DRAM", cpu_local, cpu_remote, 0, 0
        )
        result[f"gpu_{op}"] = transfer(
            np,
            local_agent,
            peer,
            op,
            "VRAM",
            gpu_local,
            gpu_remote,
            local_dev,
            remote_dev,
        )
        report(result)


def works(result: dict) -> bool:
    return result.get("gpu_READ") == "ok" and result.get("gpu_WRITE") == "ok"


def failed(result: dict) -> bool:
    """NIXL reported a failed GPU transfer, as opposed to a probe that couldn't run or finish."""
    return any(
        str(result.get(key, "")).startswith(("error", "corrupt", "nixlBackendError"))
        for key in ("gpu_READ", "gpu_WRITE")
    )


def try_setup(name: str, extra_env: dict) -> dict:
    """Run one setup over the NICs, without the shm provider. Print its result and libfabric's warnings."""
    env = {
        **os.environ,
        "FI_LOG_LEVEL": "warn",
        "FI_EFA_ENABLE_SHM_TRANSFER": "0",
        **extra_env,
    }
    try:
        done = subprocess.run(
            [sys.executable, __file__, "--child"],
            check=False,
            env=env,
            capture_output=True,
            text=True,
            timeout=SETUP_TIMEOUT_S,
        )
        output, status = done.stdout + done.stderr, done.returncode
    except subprocess.TimeoutExpired as expired:
        output, status = f"{expired.stdout or ''}{expired.stderr or ''}", "timeout"
    results = [
        line[len(MARK) :] for line in output.splitlines() if line.startswith(MARK)
    ]
    result = json.loads(results[-1]) if results else {}
    if status != 0:
        result["exit"] = status
    print(f"EFA preflight [{name}] {json.dumps(result)}", flush=True)
    problems = Counter(
        re.sub(r"0x[0-9a-f]+|\d+", "N", line)[:200]
        for line in output.splitlines()
        if "libfabric:" in line
        or "post failed" in line
        or "CQ error" in line
        or "Error" in line
    )
    for line, count in problems.most_common(12):
        print(f"EFA preflight [{name}] {count}x {line}", flush=True)
    return result


def stage_host_cuda() -> dict | None:
    """Prepare TRY_DIR with the libcuda matching the kernel module, or return None if it isn't mounted."""
    version = Path("/sys/module/nvidia/version").read_text(encoding="utf-8").strip()
    libcuda = DRIVER_LIBS / f"libcuda.so.{version}"
    if not libcuda.exists():
        print(f"EFA preflight: no {libcuda} in the container", flush=True)
        return None
    shutil.rmtree(TRY_DIR, ignore_errors=True)
    TRY_DIR.mkdir(parents=True)
    (TRY_DIR / "libcuda.so.1").symlink_to(libcuda)
    jit = DRIVER_LIBS / f"libnvidia-ptxjitcompiler.so.{version}"
    if jit.exists():
        (TRY_DIR / "libnvidia-ptxjitcompiler.so.1").symlink_to(jit)
    return {"LD_LIBRARY_PATH": f"{TRY_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}"}


def main() -> int:
    if "--child" in sys.argv:
        run_setup()
        os._exit(0)  # NIXL teardown can hang after a failed transfer

    shutil.rmtree(PIN_DIR, ignore_errors=True)
    workers = try_setup("workers", {})
    # Same stack with the options other EFA recipes set, to tell whether they matter.
    try_setup(
        "fi-options",
        {
            "FI_PROVIDER": "efa",
            "FI_EFA_USE_DEVICE_RDMA": "1",
            "FI_EFA_FORK_SAFE": "1",
            "FI_EFA_USE_UNSOLICITED_WRITE_RECV": "0",
        },
    )
    try_setup("no-direct-path", {"FI_EFA_USE_DATA_PATH_DIRECT": "0"})
    host_env = stage_host_cuda()
    host = try_setup("host-libcuda", host_env) if host_env else {}

    if works(workers):
        print("EFA preflight: passed with the workers' environment", flush=True)
        return 0
    if works(host):
        TRY_DIR.rename(PIN_DIR)
        print(
            f"EFA preflight: passed only with the host's libcuda, pinned through {PIN_DIR}",
            flush=True,
        )
        return 0
    if failed(workers):
        print(
            "EFA preflight: GPU RMA failed; not starting the engine",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print("EFA preflight: inconclusive; starting the engine", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
