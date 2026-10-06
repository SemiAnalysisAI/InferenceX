"""Exercise the failed MORI GPU MR path without loading Qwen.

The pinned MORI RegisterMemory only records a descriptor. CreateSession asks the
peer for its MR and reaches ibv_reg_mr, which is where job 47614 aborted.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import resource
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

FAILED_REGION_BYTES = 5_306_354_176
CONTROL_REGION_BYTES = 64 * 1024 * 1024
EXPECTED_SGLANG_SHA = "fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1"
EXPECTED_MORI_SHA = "f7e6ac6863c53821bc7afb91a578cc6ce38fcad0"
ROUND_DEADLINE_S = 165
TRANSFER_BYTES = 4096


def emit(event: str, **details: object) -> None:
    print(json.dumps({"event": event, "pid": os.getpid(), **details}), flush=True)


def git_head(path: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", path, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def package_commit(name: str) -> str | None:
    direct_url = metadata.distribution(name).read_text("direct_url.json")
    if direct_url:
        return json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    return None


def runtime_metadata(device: int) -> dict[str, object]:
    import mori
    import torch

    sglang_sha = git_head("/sgl-workspace/sglang") or package_commit("sglang")
    mori_sha = git_head("/sgl-workspace/mori") or package_commit("amd-mori")
    return {
        "hostname": os.uname().nodename,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "sglang_git_commit": sglang_sha,
        "mori_git_commit": mori_sha,
        "mori_version": metadata.version("amd-mori"),
        "mori_module": getattr(mori, "__file__", None),
        "gpu_count_visible": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_device": device,
        "gpu_properties": str(torch.cuda.get_device_properties(device)),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "memlock_bytes_soft_hard": resource.getrlimit(resource.RLIMIT_MEMLOCK),
        "mori_rdma_devices": os.environ.get("MORI_RDMA_DEVICES"),
        "mori_shmem_mode": os.environ.get("MORI_SHMEM_MODE"),
        "mori_io_tc": os.environ.get("MORI_IO_TC"),
        "mori_rdma_tc": os.environ.get("MORI_RDMA_TC"),
        "mori_disable_auto_xgmi": os.environ.get("MORI_DISABLE_AUTO_XGMI"),
    }


def endpoint(
    role: str, device: int, size: int, cpus: tuple[int, ...], pipe: object
) -> None:
    os.sched_setaffinity(0, cpus)
    import torch
    from mori.io import (
        BackendType,
        EngineDesc,
        IOEngine,
        IOEngineConfig,
        MemoryDesc,
        PollCqMode,
        RdmaBackendConfig,
    )
    from sglang.srt.utils.network import get_local_ip_auto

    runtime = runtime_metadata(device)
    emit("runtime", role=role, bytes=size, **runtime)
    if runtime["sglang_git_commit"] != EXPECTED_SGLANG_SHA:
        raise RuntimeError("SGLang commit differs from job 47614 fingerprint")
    if runtime["mori_git_commit"] != EXPECTED_MORI_SHA:
        raise RuntimeError("MORI commit differs from job 47614 fingerprint")
    if runtime["gpu_count_visible"] < 2:
        raise RuntimeError("probe needs two disjoint visible GPUs")
    if runtime["mori_disable_auto_xgmi"] != "1":
        raise RuntimeError("MORI_DISABLE_AUTO_XGMI=1 is required for same-node RDMA")

    torch.cuda.set_device(device)
    tensor = torch.empty(size, dtype=torch.uint8, device=f"cuda:{device}")
    tensor[:TRANSFER_BYTES].fill_(0 if role == "target" else 0x5A)
    torch.cuda.synchronize(device)
    engine = IOEngine(
        f"powerx-rdma-{role}-{os.getpid()}",
        IOEngineConfig(host=get_local_ip_auto(), port=0),
    )
    # SGLang v0.5.16 _init_engine uses these five arguments for job 47614.
    engine.create_backend(
        BackendType.RDMA, RdmaBackendConfig(4, -1, 4, PollCqMode.POLLING, False)
    )
    memory = engine.register_torch_tensor(tensor)
    pipe.send((engine.get_engine_desc().pack(), memory.pack()))
    peer_engine, peer_memory = pipe.recv()
    engine.register_remote_engine(EngineDesc.unpack(peer_engine))
    pipe.send("connected")

    if role == "initiator":
        if pipe.recv() != "transfer":
            raise RuntimeError("unexpected coordinator command")
        remote_memory = MemoryDesc.unpack(peer_memory)
        emit("create_session_start", local_bytes=size, target_bytes=remote_memory.size)
        started = time.monotonic()
        session = engine.create_session(memory, remote_memory)
        if session is None:
            raise RuntimeError("RDMA session unavailable")
        emit("create_session_success", elapsed_s=time.monotonic() - started)
        transfer_id = engine.allocate_transfer_uid()
        status = session.write(0, 0, TRANSFER_BYTES, transfer_id)
        status.Wait()
        if not status.Succeeded():
            raise RuntimeError(f"RDMA write failed: {status.Message()}")
        pipe.send("transfer_succeeded")
    else:
        if pipe.recv() != "verify":
            raise RuntimeError("unexpected coordinator command")
        torch.cuda.synchronize(device)
        if not bool(torch.all(tensor[:TRANSFER_BYTES] == 0x5A).item()):
            raise RuntimeError("target did not receive the 4 KiB RDMA transfer")
        pipe.send("verified")
    emit("endpoint_success", role=role, bytes=size)


def receive(pipe: object, process: mp.Process, deadline: float) -> object:
    while time.monotonic() < deadline:
        if pipe.poll(0.25):
            return pipe.recv()
        if process.exitcode is not None:
            raise RuntimeError(
                f"{process.name} exited before response: {process.exitcode}"
            )
    raise TimeoutError(f"{process.name} did not respond before round deadline")


def run_round(size: int) -> dict[str, object]:
    context = mp.get_context("spawn")
    allowed_cpus = sorted(os.sched_getaffinity(0))
    if len(allowed_cpus) < 2:
        raise RuntimeError("two disjoint CPU affinity sets are required")
    midpoint = len(allowed_cpus) // 2
    initiator_cpus = tuple(allowed_cpus[:midpoint])
    target_cpus = tuple(allowed_cpus[midpoint:])
    initiator_pipe, initiator_child = context.Pipe()
    target_pipe, target_child = context.Pipe()
    target = context.Process(
        target=endpoint,
        name="target",
        args=("target", 1, size, target_cpus, target_child),
    )
    initiator = context.Process(
        target=endpoint,
        name="initiator",
        args=("initiator", 0, CONTROL_REGION_BYTES, initiator_cpus, initiator_child),
    )
    started = time.monotonic()
    deadline = started + ROUND_DEADLINE_S
    result: dict[str, object] = {
        "target_bytes": size,
        "initiator_cpu_affinity": initiator_cpus,
        "target_cpu_affinity": target_cpus,
    }
    try:
        target.start()
        initiator.start()
        target_desc, target_memory = receive(target_pipe, target, deadline)
        initiator_desc, initiator_memory = receive(initiator_pipe, initiator, deadline)
        target_pipe.send((initiator_desc, initiator_memory))
        initiator_pipe.send((target_desc, target_memory))
        receive(target_pipe, target, deadline)
        receive(initiator_pipe, initiator, deadline)
        initiator_pipe.send("transfer")
        if receive(initiator_pipe, initiator, deadline) != "transfer_succeeded":
            raise RuntimeError("initiator did not complete transfer")
        target_pipe.send("verify")
        if receive(target_pipe, target, deadline) != "verified":
            raise RuntimeError("target did not verify transfer")
        result["status"] = "success"
    except (EOFError, OSError, RuntimeError, TimeoutError) as exc:
        result.update(status="failure", error=str(exc))
    finally:
        for process in (initiator, target):
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        result["initiator_exit_code"] = initiator.exitcode
        result["target_exit_code"] = target.exitcode
        if result.get("status") == "success" and any(
            process.exitcode != 0 for process in (initiator, target)
        ):
            result.update(status="failure", error="endpoint did not exit cleanly")
        result["elapsed_s"] = time.monotonic() - started
        emit("round_result", **result)
    return result


def main() -> int:
    output = Path("speedbench_results/mori-rdma-preflight.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for size in (CONTROL_REGION_BYTES, FAILED_REGION_BYTES):
        result = run_round(size)
        results.append(result)
        output.write_text(json.dumps({"rounds": results}, indent=2) + "\n")
        if result["status"] != "success":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
