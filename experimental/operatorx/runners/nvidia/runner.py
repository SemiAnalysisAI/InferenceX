from __future__ import annotations

from importlib import import_module

import torch

from operatorx.core import BackendImpl, Op, Result, UnsupportedOpError

_BACKENDS = ["torch", "deepgemm", "flashinfer", "deepep", "sglang", "flashinfer_comm", "sglang_comm"]
_DISPATCH: dict[tuple[str, str], BackendImpl] = {}
_L2_BUF: dict[int, torch.Tensor] = {}


def _load() -> None:
    if _DISPATCH:
        return
    for name in _BACKENDS:
        try:
            mod = import_module(f"operatorx.runners.nvidia.backends.{name}")
        except ImportError:
            continue
        for impl in getattr(mod, "IMPLS", []):
            _DISPATCH[(impl.op_type, name)] = impl


def _clear_l2() -> None:
    """Flush L2 by writing zeros to a buffer sized to the device's L2 cache."""
    dev = torch.cuda.current_device()
    buf = _L2_BUF.get(dev)
    if buf is None:
        l2 = torch.cuda.get_device_properties(dev).L2_cache_size
        buf = torch.empty(l2, dtype=torch.int8, device=dev)
        _L2_BUF[dev] = buf
    buf.zero_()


_WARMUP = 5
_ITERS = 10
_NUM_BUFFER_SETS = 1

def run(op: Op) -> Result:
    _load()
    impl = _DISPATCH.get((op.type, op.backend))
    if impl is None:
        raise UnsupportedOpError(
            f"nvidia/{op.backend} has no impl for op_type={op.type!r}"
        )

    ctxs = [impl.prepare(op) for _ in range(_NUM_BUFFER_SETS)]

    for i in range(_WARMUP):
        impl.kernel(ctxs[i % _NUM_BUFFER_SETS])
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(_ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(_ITERS)]
    for i in range(_ITERS):
        _clear_l2()
        starts[i].record()
        impl.kernel(ctxs[i % _NUM_BUFFER_SETS])
        ends[i].record()
    torch.cuda.synchronize()

    times = sorted(starts[i].elapsed_time(ends[i]) * 1000.0 for i in range(_ITERS))
    median_us = times[_ITERS // 2]
    return Result(op=op, metrics={"latency_us": median_us})
