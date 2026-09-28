from __future__ import annotations

import time
from importlib import import_module

import jax

from operatorx.core import BackendImpl, Op, Result, UnsupportedOpError

_BACKENDS = ["jax", "vllm", "sglang"]
_DISPATCH: dict[tuple[str, str], BackendImpl] = {}


def _load() -> None:
    if _DISPATCH:
        return
    for name in _BACKENDS:
        try:
            mod = import_module(f"operatorx.runners.tpu.backends.{name}")
        except ImportError:
            continue
        for impl in getattr(mod, "IMPLS", []):
            _DISPATCH[(impl.op_type, name)] = impl


_WARMUP = 3
_ITERS = 10


def run(op: Op) -> Result:
    _load()
    impl = _DISPATCH.get((op.type, op.backend))
    if impl is None:
        raise UnsupportedOpError(
            f"tpu/{op.backend} has no impl for op_type={op.type!r}"
        )

    # Per-iter prepare + per-iter timer. Prepare runs outside the timed region
    # so each kernel call sees a fresh input buffer. This defeats two JAX-side
    # caching pitfalls that bias measurements:
    #   1. `jax.device_get` caches the host numpy buffer on its source array,
    #      so repeated D2H against the same `jax.Array` reads the cache (we
    #      observed 7us flat for 2GB transfers).
    #   2. XLA's ConstantFolding pass can elide work when the same value flows
    #      through identical inputs across iterations.
    # Trade-off: re-allocates input buffers every iter; relies on refcount-
    # driven HBM cleanup when ctx is rebound. Per-iter timer noise is amortized
    # over _ITERS samples.
    for _ in range(_WARMUP):
        ctx = impl.prepare(op)
        impl.kernel(ctx)
        jax.block_until_ready(ctx.get("out"))

    times = []
    for _ in range(_ITERS):
        ctx = impl.prepare(op)
        for v in ctx.values():
            if hasattr(v, "block_until_ready"):
                v.block_until_ready()
        t0 = time.perf_counter()
        impl.kernel(ctx)
        jax.block_until_ready(ctx.get("out"))
        times.append((time.perf_counter() - t0) * 1e6)

    times.sort()
    med = times[_ITERS // 2]
    metrics = {"latency_us": med}

    # Single-shot host timing is dominated by dispatch overhead (~50-500us)
    # once the kernel itself is sub-millisecond, so small (e.g. decode-shape)
    # ops read as launch latency rather than kernel time. Re-time those in a
    # pipelined loop: enqueue K calls back-to-back and block once — JAX
    # dispatch is async, so the host runs ahead and the device executes the
    # queue contiguously. Per-call time then approaches true device latency
    # (still an upper bound if host enqueue can't keep up).
    if med < 1000.0:
        K = min(512, max(16, int(4000.0 / max(med, 1.0))))
        ctx = impl.prepare(op)
        for v in ctx.values():
            if hasattr(v, "block_until_ready"):
                v.block_until_ready()
        impl.kernel(ctx)  # one unmeasured call so the executable is hot
        jax.block_until_ready(ctx.get("out"))
        t0 = time.perf_counter()
        for _ in range(K):
            impl.kernel(ctx)
        jax.block_until_ready(ctx.get("out"))
        pipelined = (time.perf_counter() - t0) * 1e6 / K
        metrics["latency_us_single_shot"] = med
        metrics["latency_us_pipelined"] = pipelined
        # Best per-call kernel-latency estimate.
        metrics["latency_us"] = min(med, pipelined)

    return Result(op=op, metrics=metrics)
