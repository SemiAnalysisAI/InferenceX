"""ROCm operator timing with HIP events exposed through torch.cuda."""

from __future__ import annotations

from importlib import import_module

import torch

from operatorx.core import Op, Result, UnsupportedOpError

_L2_BUF: dict[int, torch.Tensor] = {}
_WARMUP = 5
_ITERS = 10


def run(op: Op) -> Result:
    if op.backend not in {"torch", "aiter", "vllm"}:
        raise UnsupportedOpError(f"unknown AMD backend: {op.backend}")
    backend = import_module(f"operatorx.runners.amd.backends.{op.backend}")
    impl = next((item for item in backend.IMPLS if item.op_type == op.type), None)
    if impl is None:
        raise UnsupportedOpError(f"amd/{op.backend} has no impl for {op.type!r}")
    if not torch.version.hip:
        raise RuntimeError("AMD measurements require a ROCm PyTorch build")
    device = torch.cuda.current_device()
    if device not in _L2_BUF:
        size = torch.cuda.get_device_properties(device).L2_cache_size
        if size <= 0:
            raise RuntimeError("ROCm did not report a positive L2 cache size")
        _L2_BUF[device] = torch.empty(size, dtype=torch.int8, device="cuda")
    ctx = impl.prepare(op)
    for _ in range(_WARMUP):
        impl.kernel(ctx)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(_ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(_ITERS)]
    for start, end in zip(starts, ends):
        _L2_BUF[device].zero_()
        start.record()
        impl.kernel(ctx)
        end.record()
    torch.cuda.synchronize()
    times = sorted(start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends))
    return Result(op=op, metrics={"latency_us": times[_ITERS // 2]})
