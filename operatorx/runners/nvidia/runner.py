from __future__ import annotations

import os
import pkgutil
import time
from importlib import import_module

import torch

from operatorx.core import BackendImpl, Op, Result, UnsupportedOpError
from operatorx.runners.common import profiling, ranks, telemetry, timing

_COOLDOWN_RATIO = float(os.environ.get("OPERATORX_COOLDOWN_RATIO", "4"))
_COOLDOWN_MAX_S = float(os.environ.get("OPERATORX_COOLDOWN_MAX_S", "1.0"))

_DISPATCH: dict[tuple[str, str], BackendImpl] = {}
_L2_BUF: dict[int, torch.Tensor] = {}


def _load() -> None:
    if _DISPATCH:
        return
    from operatorx.runners.nvidia import backends
    for info in pkgutil.iter_modules(backends.__path__):
        if info.name.startswith("_"):
            continue
        try:
            mod = import_module(f"operatorx.runners.nvidia.backends.{info.name}")
        except ImportError:
            continue
        for impl in getattr(mod, "IMPLS", []):
            _DISPATCH[(impl.op_type, info.name)] = impl


def _clear_l2() -> None:
    dev = torch.cuda.current_device()
    if dev not in _L2_BUF:
        size = torch.cuda.get_device_properties(dev).L2_cache_size
        _L2_BUF[dev] = torch.empty(size, dtype=torch.int8, device=dev)
    _L2_BUF[dev].zero_()


def run(op: Op) -> Result:
    _load()
    impl = _DISPATCH.get((op.type, op.backend))
    if impl is None:
        raise UnsupportedOpError(f"nvidia/{op.backend} has no impl for op_type={op.type!r}")
    ctx = impl.prepare(op)

    fn, cuda_graph = impl.launcher(ctx) if impl.launcher else ((lambda: impl.kernel(ctx)), False)
    median_us, telem = telemetry.measure(op, lambda sleep_s: timing.time_op(fn, _clear_l2, sleep_s))

    if _COOLDOWN_RATIO > 0.0:
        busy_s = median_us * 1e-6 * (timing.ITERS + timing.WARMUP)
        time.sleep(min(busy_s * _COOLDOWN_RATIO, _COOLDOWN_MAX_S))

    metrics = {"latency_us": median_us, "cuda_graph": cuda_graph, "telemetry": telem}
    if isinstance(ctx, dict) and ctx.get("meta"):
        metrics["backend_meta"] = ctx["meta"]
    prof = profiling.profile_op(fn)
    if prof is not None:
        metrics["profile"] = prof
    per_rank = ranks.summary(metrics)  # every rank's timeline and telemetry, on rank 0
    if per_rank is not None:
        metrics["ranks"] = per_rank
        if per_rank["capped_ranks"] and metrics.get("telemetry"):
            metrics["telemetry"]["capped"] = True  # any rank capped caps the op
    return Result(op=op, metrics=metrics)
