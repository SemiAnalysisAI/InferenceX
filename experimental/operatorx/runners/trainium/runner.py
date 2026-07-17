from __future__ import annotations

import os
import time
from importlib import import_module

# Force LNC=2 (4 logical NCs per Trainium3 chip — each LNC=2 logical NC owns
# one of the chip's 4 HBM banks and consists of 2 paired physical NC-v4 cores
# sharing that bank's address space). This is the unit operatorx counts as
# "world_size=1" on Trainium: one HBM bank + 2 tensor engines + ~64 MiB SBUF.
# Going below this (LNC=1 / single physical NC) is rarely useful since the 2
# cores in a pair share HBM anyway; going above means crossing into another
# bank which requires CC-Core collectives.
#
# Per AWS Neuron docs, the *compiler* must produce a NEFF matching the
# runtime's LNC mode — otherwise the runtime rejects the binary at load time
# with "Cannot run Neff [...] with Logical Core Size of N". Pass
# `--logical-nc-config=2` to neuronx-cc via NEURON_CC_FLAGS.
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
_cc_flags = os.environ.get("NEURON_CC_FLAGS", "")
if "--logical-nc-config" not in _cc_flags:
    os.environ["NEURON_CC_FLAGS"] = (_cc_flags + " --logical-nc-config=2").strip()

# Default Neuron compile cache lives at /var/tmp/neuron-compile-cache, which
# is often owned by a different user on shared dev boxes. Point it at a
# per-user path under $HOME so writes don't fail with permission denied.
os.environ.setdefault(
    "NEURON_COMPILE_CACHE_URL",
    os.path.join(os.path.expanduser("~"), ".cache", "neuron-compile-cache"),
)

import torch_xla.core.xla_model as xm

from operatorx.core import BackendImpl, Op, Result, UnsupportedOpError

_BACKENDS = ["torch", "nkilib", "nxd"]
_DISPATCH: dict[tuple[str, str], BackendImpl] = {}
_WARMUP = 3
_ITERS = 10
_NUM_BUFFER_SETS = 1


def _load() -> None:
    if _DISPATCH:
        return
    for name in _BACKENDS:
        try:
            mod = import_module(f"operatorx.runners.trainium.backends.{name}")
        except ImportError:
            continue
        for impl in getattr(mod, "IMPLS", []):
            _DISPATCH[(impl.op_type, name)] = impl


def _sync(ctx: dict) -> None:
    """Force device synchronization. Under PJRT-Neuron + LNC=2,
    xm.wait_device_ops() / torch_xla.sync() / xm.mark_step() all return
    BEFORE the Neuron device actually finishes — they only flush the queue.
    The only reliable way to wait is a CPU readback, so the backend's kernel
    function must place its output tensor under ctx["out"] and we tear off
    one element from it to force the sync."""
    out = ctx.get("out")
    if out is None:
        # Last-resort fallback if a backend doesn't expose ctx["out"].
        # This still measures dispatch only — flagged loudly in description.
        xm.wait_device_ops()
        return
    # Single-element readback is enough to force the entire output (and
    # everything it depends on) to materialize. Far cheaper than .cpu() on
    # the full tensor and dominates per-iter wall time only minimally.
    _ = out.flatten()[0].item()


def run(op: Op) -> Result:
    _load()
    impl = _DISPATCH.get((op.type, op.backend))
    if impl is None:
        raise UnsupportedOpError(
            f"trainium/{op.backend} has no impl for op_type={op.type!r}"
        )

    ctxs = [impl.prepare(op) for _ in range(_NUM_BUFFER_SETS)]

    # Backends that self-benchmark (e.g. nkilib via SpikeModel.benchmark) populate
    # ctx["latency_us"] in prepare and the runner trusts that number.
    if "latency_us" in ctxs[0]:
        return Result(op=op, metrics={"latency_us": ctxs[0]["latency_us"]})

    for i in range(_WARMUP):
        impl.kernel(ctxs[i % _NUM_BUFFER_SETS])
    _sync(ctxs[0])

    times = []
    for i in range(_ITERS):
        ctx = ctxs[i % _NUM_BUFFER_SETS]
        t0 = time.perf_counter()
        impl.kernel(ctx)
        _sync(ctx)
        times.append((time.perf_counter() - t0) * 1e6)

    times.sort()
    return Result(op=op, metrics={"latency_us": times[_ITERS // 2]})
