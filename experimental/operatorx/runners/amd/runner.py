"""ROCm GEMM timing with HIP events exposed through torch.cuda."""

from __future__ import annotations

import torch

from operatorx.core import Op, Result, UnsupportedOpError
from operatorx.runners.amd.backends import torch as backend

_L2_BUF: dict[int, torch.Tensor] = {}
_WARMUP = 5
_ITERS = 10


def run(op: Op) -> Result:
    if op.backend != "torch" or op.type != "gemm":
        raise UnsupportedOpError(f"amd/{op.backend} has no impl for {op.type!r}")
    if not torch.version.hip:
        raise RuntimeError("AMD measurements require a ROCm PyTorch build")
    device = torch.cuda.current_device()
    if device not in _L2_BUF:
        size = torch.cuda.get_device_properties(device).L2_cache_size
        if size <= 0:
            raise RuntimeError("ROCm did not report a positive L2 cache size")
        _L2_BUF[device] = torch.empty(size, dtype=torch.int8, device="cuda")
    ctx = backend.prepare(op)
    for _ in range(_WARMUP):
        backend.kernel(ctx)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(_ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(_ITERS)]
    for start, end in zip(starts, ends):
        _L2_BUF[device].zero_()
        start.record()
        backend.kernel(ctx)
        end.record()
    torch.cuda.synchronize()
    times = sorted(start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends))
    return Result(op=op, metrics={"latency_us": times[_ITERS // 2]})
