"""sglang collective backend: tries vLLM CustomAllreduce, MSCCL++, and torch
symmetric memory for allreduce, then picks the fastest per shape.

For non-AR collectives sglang doesn't ship its own kernels — it falls back to
torch.distributed/NCCL, which is already covered by the `torch` backend.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

from operatorx.core import BackendImpl, Op, UnsupportedOpError, lookup_versions


def versions() -> dict[str, str]:
    return lookup_versions("sglang", "torch")


_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _ensure_dist() -> None:
    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))


# Cache communicator instances per process — each holds IPC handles and pinned
# scratch buffers; constructing them is expensive (registers buffers in NCCL).
_COMMS: dict[str, object] = {}


def _get_communicators() -> dict[str, object]:
    """Construct one of each available all-reduce communicator. Returns whatever
    succeeded; failures (e.g. missing peer access, unsupported topology) are
    silently dropped so we benchmark only what's actually usable."""
    if _COMMS:
        return _COMMS
    group = dist.group.WORLD
    device = torch.cuda.current_device()

    try:
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )
        # 256 MB cap covers the largest testlist shapes (64M elems × 4 bytes).
        _COMMS["custom_ar"] = CustomAllreduce(group, device, max_size=256 * 1024 * 1024)
    except Exception:
        pass

    try:
        from sglang.srt.distributed.device_communicators.pymscclpp import (
            PyMscclppCommunicator,
        )
        _COMMS["mscclpp"] = PyMscclppCommunicator(group, device, max_bytes=256 * 1024 * 1024)
    except Exception:
        pass

    try:
        from sglang.srt.distributed.device_communicators.torch_symm_mem import (
            TorchSymmMemCommunicator,
        )
        _COMMS["symm_mem"] = TorchSymmMemCommunicator(group, device)
    except Exception:
        pass

    return _COMMS


def _supports(name: str, comm, t: torch.Tensor) -> bool:
    """Each communicator has a size/topology gate; respect it instead of
    blindly invoking (which would return None or fall back to NCCL)."""
    if name == "custom_ar":
        return bool(comm.should_custom_ar(t))
    if name == "mscclpp":
        return bool(comm.should_mscclpp_allreduce(t))
    if name == "symm_mem":
        return bool(comm.should_torch_symm_mem_allreduce(t))
    return False


def _allreduce_with(name: str, comm, t: torch.Tensor) -> torch.Tensor:
    if name == "custom_ar":
        return comm.custom_all_reduce(t)
    if name == "mscclpp":
        return comm.all_reduce(t)
    if name == "symm_mem":
        return comm.all_reduce(t)
    raise UnsupportedOpError(f"unknown comm={name!r}")


def _prepare_allreduce(op: Op) -> dict:
    a = op.args
    if a["dtype"] not in _DTYPES:
        raise UnsupportedOpError(f"sglang_comm allreduce: dtype={a['dtype']!r} not supported")
    _ensure_dist()
    dt = _DTYPES[a["dtype"]]
    t = torch.randn(a["num_elements"], dtype=dt, device="cuda")

    comms = _get_communicators()
    eligible = [(name, c) for name, c in comms.items() if _supports(name, c, t)]
    if not eligible:
        raise UnsupportedOpError(
            f"sglang_comm allreduce: no variant supports this tensor "
            f"(n={a['num_elements']}, dtype={a['dtype']!r}, world_size={a['world_size']})"
        )

    # Mini-benchmark to pick the fastest variant for this exact shape; runs
    # entirely outside the timed window. CUDA events on the current stream
    # give sub-microsecond resolution and account for cross-rank sync.
    pinned: list = []  # keep the tensor alive across the closure call below
    pinned.append(t)
    times = {}
    for name, c in eligible:
        try:
            for _ in range(3):
                _allreduce_with(name, c, t)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(5):
                _allreduce_with(name, c, t)
            end.record()
            torch.cuda.synchronize()
            times[name] = start.elapsed_time(end) / 5.0
        except Exception:
            continue

    if not times:
        raise UnsupportedOpError(
            "sglang_comm allreduce: all variants failed during selection"
        )

    winner_name = min(times, key=times.get)
    winner_comm = dict(eligible)[winner_name]

    def _call(_pin=pinned):
        return _allreduce_with(winner_name, winner_comm, t)

    return {"_call": _call, "_winner": winner_name, "_times_us": times}


def _kernel_allreduce(ctx: dict) -> None:
    ctx["_call"]()


def _symm_mem_ag_rs_available() -> bool:
    try:
        import torch.distributed._symmetric_memory as ts  # noqa: F401
        return True
    except Exception:
        return False


def _ensure_symm_mem_group():
    """Symmetric memory needs the group to be opted in. Idempotent."""
    import torch.distributed._symmetric_memory as ts
    group = dist.group.WORLD
    name = group.group_name
    if not ts.is_symm_mem_enabled_for_group(name):
        ts.enable_symm_mem_for_group(name)
    return name


def _prepare_allgather(op: Op) -> dict:
    a = op.args
    if a["dtype"] not in _DTYPES:
        raise UnsupportedOpError(f"sglang_comm allgather: dtype={a['dtype']!r} not supported")
    if not _symm_mem_ag_rs_available():
        raise UnsupportedOpError("sglang_comm allgather: torch symm_mem not available")
    _ensure_dist()
    import torch.distributed._symmetric_memory as ts
    group_name = _ensure_symm_mem_group()
    dt = _DTYPES[a["dtype"]]
    n_per = int(a["num_elements_per_rank"])

    # symm_mem path: input must be allocated in symm_mem. Allocate via
    # ts.empty + rendezvous so peer pointers exist.
    inp = ts.empty(n_per, dtype=dt, device="cuda")
    ts.rendezvous(inp, group_name)
    pinned = [inp]

    def _call(_pin=pinned):
        return ts._low_contention_all_gather(inp, group_name)

    return {"_call": _call}


def _kernel_allgather(ctx: dict) -> None:
    ctx["_call"]()


def _prepare_reduce_scatter(op: Op) -> dict:
    a = op.args
    if a["dtype"] not in _DTYPES:
        raise UnsupportedOpError(f"sglang_comm reduce_scatter: dtype={a['dtype']!r} not supported")
    if not _symm_mem_ag_rs_available():
        raise UnsupportedOpError("sglang_comm reduce_scatter: torch symm_mem not available")
    _ensure_dist()
    import torch.distributed._symmetric_memory as ts
    group_name = _ensure_symm_mem_group()
    dt = _DTYPES[a["dtype"]]
    n = int(a["num_elements"])
    ws = int(a["world_size"])
    if n % ws != 0:
        raise UnsupportedOpError(
            f"sglang_comm reduce_scatter: num_elements={n} not divisible by world_size={ws}"
        )

    inp = ts.empty(n, dtype=dt, device="cuda")
    ts.rendezvous(inp, group_name)
    pinned = [inp]

    def _call(_pin=pinned):
        return ts._low_contention_reduce_scatter(inp, "sum", group_name)

    return {"_call": _call}


def _kernel_reduce_scatter(ctx: dict) -> None:
    ctx["_call"]()


IMPLS = [
    BackendImpl(op_type="allreduce", prepare=_prepare_allreduce, kernel=_kernel_allreduce),
    BackendImpl(op_type="allgather", prepare=_prepare_allgather, kernel=_kernel_allgather),
    BackendImpl(op_type="reduce_scatter", prepare=_prepare_reduce_scatter, kernel=_kernel_reduce_scatter),
]
