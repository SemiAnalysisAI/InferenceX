"""flashinfer collective backend: TRT-LLM custom_all_reduce via
flashinfer.comm.allreduce_fusion with backend='auto' (AUTO internally picks
between Lamport one-shot and two-shot ring per shape).

We cache the IPC workspace per (tokens, hidden, dtype) — workspace creation
does a cross-rank IPC handle exchange that costs tens of seconds, and the
workspace itself owns shared-memory buffers that we don't want to multiply.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

from operatorx.core import BackendImpl, Op, UnsupportedOpError, lookup_versions


def versions() -> dict[str, str]:
    return lookup_versions("flashinfer", "torch")


_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _ensure_dist() -> None:
    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))


# Cache workspaces by (tokens, hidden, dtype) — IPC setup is expensive.
_WS: dict[tuple, object] = {}


def _factor(n: int, max_hidden: int = 8192, align: int = 128) -> tuple[int, int]:
    """Factor n into (tokens, hidden) with hidden ≤ max_hidden and hidden % align == 0."""
    if n % align != 0:
        raise UnsupportedOpError(
            f"flashinfer_comm allreduce: num_elements={n} must be a multiple of {align}"
        )
    if n <= max_hidden:
        return 1, n
    hidden = max_hidden
    while n % hidden != 0 and hidden > align:
        hidden -= align
    if n % hidden != 0:
        raise UnsupportedOpError(
            f"flashinfer_comm allreduce: cannot factor n={n} with hidden ≤ {max_hidden}"
        )
    return n // hidden, hidden


def _get_workspace(tokens: int, hidden: int, dt: torch.dtype, ws: int, rank: int):
    key = (tokens, hidden, dt, ws)
    if key in _WS:
        return _WS[key]
    from flashinfer.comm import create_allreduce_fusion_workspace
    # backend="trtllm" avoids the "auto" probe path that lazy-imports mpi4py
    # (which we don't have); trtllm IPC works on standard NVLink/NVSwitch.
    workspace = create_allreduce_fusion_workspace(
        backend="trtllm", world_size=ws, rank=rank,
        max_token_num=tokens, hidden_dim=hidden, dtype=dt,
    )
    _WS[key] = workspace
    return workspace


def _prepare_allreduce(op: Op) -> dict:
    a = op.args
    if a["dtype"] not in _DTYPES:
        raise UnsupportedOpError(f"flashinfer_comm allreduce: dtype={a['dtype']!r} not supported")
    _ensure_dist()
    dt = _DTYPES[a["dtype"]]
    ws = int(a["world_size"])
    rank = dist.get_rank()
    n = int(a["num_elements"])
    tokens, hidden = _factor(n)

    from flashinfer.comm import AllReduceFusionPattern, allreduce_fusion

    workspace = _get_workspace(tokens, hidden, dt, ws, rank)
    x = torch.randn(tokens, hidden, dtype=dt, device="cuda")
    y = torch.empty_like(x)

    def _call(_pin=(x, y, workspace)):
        return allreduce_fusion(
            input=x, workspace=workspace,
            pattern=AllReduceFusionPattern.kAllReduce,
            output=y, use_oneshot=None,  # let AUTO pick oneshot/twoshot
            launch_with_pdl=False,
        )

    return {"_call": _call}


def _kernel_allreduce(ctx: dict) -> None:
    ctx["_call"]()


IMPLS = [
    BackendImpl(op_type="allreduce", prepare=_prepare_allreduce, kernel=_kernel_allreduce),
]
