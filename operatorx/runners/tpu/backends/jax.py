from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from operatorx.core import BackendImpl, Op, UnsupportedOpError, lookup_versions


def versions() -> dict[str, str]:
    return lookup_versions("jax", "jaxlib")

_DTYPES = {
    "bf16": jnp.bfloat16,
    "fp16": jnp.float16,
    "fp32": jnp.float32,
    # jax has fp8 dtypes; on TPU v6e there's no FP8 MXU so the runtime upcasts
    # to bf16 — kernel still runs, timing reflects the bf16 path.
    "fp8":  jnp.float8_e4m3fn,
}

_AOT_CACHE: dict[tuple, object] = {}


def _aot(fn, *args):
    """Lower + compile once per (fn, arg shapes), then reuse the Executable."""
    key = (id(fn),) + tuple(
        ("arr", a.shape, str(a.dtype)) if hasattr(a, "shape") else ("id", id(a))
        for a in args
    )
    exe = _AOT_CACHE.get(key)
    if exe is None:
        exe = fn.lower(*args).compile()
        _AOT_CACHE[key] = exe
    return exe


def _resolve(dtype: str) -> jnp.dtype:
    if dtype not in _DTYPES:
        raise UnsupportedOpError(f"jax backend doesn't support dtype={dtype!r}")
    return _DTYPES[dtype]


_gemm_jit = jax.jit(jnp.dot)


def _prepare_gemm(op: Op) -> dict:
    # TPU v6e MXU is natively bf16; fp8 has no hardware path on Trillium and would
    # silently upcast to bf16, so we don't expose an fp8 gemm here.
    a = op.args
    if a["dtype_b"] != a["dtype_a"]:
        raise UnsupportedOpError("jax gemm requires dtype_a == dtype_b")
    dt = _resolve(a["dtype_a"])
    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, (a["m"], a["k"]), dtype=dt)
    B = jax.random.normal(key, (a["k"], a["n"]), dtype=dt)
    return {"A": A, "B": B, "fn": _aot(_gemm_jit, A, B)}


def _kernel_gemm(ctx: dict) -> None:
    ctx["out"] = ctx["fn"](ctx["A"], ctx["B"])






@functools.lru_cache(maxsize=8)
def _allreduce_fn(world_size: int):
    devices = jax.devices()[:world_size]

    def _reduce(x):
        return jax.lax.psum(x, "i")

    return jax.pmap(_reduce, axis_name="i", devices=devices)


def _prepare_allreduce(op: Op) -> dict:
    a = op.args
    ws = a["world_size"]
    if ws > jax.device_count():
        raise NotImplementedError(
            f"world_size={ws} exceeds available TPU devices ({jax.device_count()})"
        )
    dt = _resolve(a["dtype"])
    # pmap convention: leading dim = world_size; each rank sees its own row.
    x = jax.random.normal(jax.random.PRNGKey(0), (ws, a["num_elements"]), dtype=dt)
    fn = _allreduce_fn(ws)
    return {"x": x, "fn": _aot(fn, x)}


def _kernel_allreduce(ctx: dict) -> None:
    ctx["out"] = ctx["fn"](ctx["x"])


@functools.lru_cache(maxsize=8)
def _allgather_fn(world_size: int):
    devices = jax.devices()[:world_size]

    def _gather(x):
        return jax.lax.all_gather(x, "i", tiled=True)

    return jax.pmap(_gather, axis_name="i", devices=devices)


def _prepare_allgather(op: Op) -> dict:
    a = op.args
    ws = a["world_size"]
    if ws > jax.device_count():
        raise NotImplementedError(f"world_size={ws} > available {jax.device_count()}")
    dt = _resolve(a["dtype"])
    # pmap convention: leading dim = world_size; each rank sees its own row of (per_rank,).
    x = jax.random.normal(jax.random.PRNGKey(0), (ws, a["num_elements_per_rank"]), dtype=dt)
    fn = _allgather_fn(ws)
    return {"x": x, "fn": _aot(fn, x)}


def _kernel_allgather(ctx: dict) -> None:
    ctx["out"] = ctx["fn"](ctx["x"])


@functools.lru_cache(maxsize=8)
def _reduce_scatter_fn(world_size: int):
    devices = jax.devices()[:world_size]
    mesh = Mesh(devices, ("i",))

    def _rs(x):
        return jax.lax.psum_scatter(x, "i", tiled=True)

    return jax.jit(shard_map(_rs, mesh=mesh, in_specs=P(), out_specs=P("i")))


def _prepare_reduce_scatter(op: Op) -> dict:
    a = op.args
    ws = a["world_size"]
    if ws > jax.device_count():
        raise NotImplementedError(f"world_size={ws} > available {jax.device_count()}")
    if a["num_elements"] % ws != 0:
        raise ValueError("num_elements must be divisible by world_size for reduce_scatter")
    dt = _resolve(a["dtype"])
    x = jax.random.normal(jax.random.PRNGKey(0), (a["num_elements"],), dtype=dt)
    fn = _reduce_scatter_fn(ws)
    return {"x": x, "fn": _aot(fn, x)}


def _kernel_reduce_scatter(ctx: dict) -> None:
    ctx["out"] = ctx["fn"](ctx["x"])


@functools.lru_cache(maxsize=8)
def _alltoall_fn(world_size: int):
    devices = jax.devices()[:world_size]
    mesh = Mesh(devices, ("i",))

    def _ata(x):
        return jax.lax.all_to_all(x, "i", split_axis=0, concat_axis=0, tiled=True)

    return jax.jit(shard_map(_ata, mesh=mesh, in_specs=P("i"), out_specs=P("i")))


def _prepare_alltoall(op: Op) -> dict:
    a = op.args
    ws = a["world_size"]
    if ws > jax.device_count():
        raise NotImplementedError(f"world_size={ws} > available {jax.device_count()}")
    if a["num_elements_per_rank"] % ws != 0:
        raise ValueError("num_elements_per_rank must be divisible by world_size for alltoall")
    dt = _resolve(a["dtype"])
    # x is sharded P('i') — each rank sees (num_elements_per_rank,) which holds ws chunks.
    x = jax.random.normal(jax.random.PRNGKey(0), (ws * a["num_elements_per_rank"],), dtype=dt)
    fn = _alltoall_fn(ws)
    return {"x": x, "fn": _aot(fn, x)}


def _kernel_alltoall(ctx: dict) -> None:
    ctx["out"] = ctx["fn"](ctx["x"])


# moe_forward is not implemented in the jax backend — see runners/tpu/backends/maxtext.py
# for the SOTA TPU implementation via Google's MaxText library.


# ---------- dispatch / combine (modeled as all-to-all of routed tokens) ----------
#
# Real DeepEP dispatch is "send each token to its top_k experts across world_size
# ranks, where (N-1)/N of the data crosses the interconnect." A faithful TPU
# analog using `all_to_all` of size (num_tokens * top_k * hidden) per rank gets
# the same network cost. The kernel does no routing/permutation logic — pure
# comm — which matches the way bytes are scored in ops/collective.py.


@functools.lru_cache(maxsize=8)
def _dispatch_fn(world_size: int):
    devices = jax.devices()[:world_size]
    mesh = Mesh(devices, ("i",))

    def _d(x):
        return jax.lax.all_to_all(x, "i", split_axis=0, concat_axis=0, tiled=True)

    return jax.jit(shard_map(_d, mesh=mesh, in_specs=P("i"), out_specs=P("i")))


def _prepare_dispatch(op: Op) -> dict:
    a = op.args
    ws = a["world_size"]
    if ws > jax.device_count():
        raise UnsupportedOpError(
            f"world_size={ws} > available TPU devices ({jax.device_count()})"
        )
    dt = _resolve(a["dtype"])
    per_rank_elems = a["num_tokens"] * a["top_k"] * a["hidden"]
    if per_rank_elems % ws != 0:
        raise UnsupportedOpError(
            f"per-rank elems {per_rank_elems} must divide world_size {ws} for all-to-all"
        )
    x = jax.random.normal(jax.random.PRNGKey(0), (ws * per_rank_elems,), dtype=dt)
    fn = _dispatch_fn(ws)
    return {"x": x, "fn": _aot(fn, x)}


def _kernel_dispatch(ctx: dict) -> None:
    ctx["out"] = ctx["fn"](ctx["x"])


def _prepare_combine(op: Op) -> dict:
    # Symmetric to dispatch in network cost — reuse the same all-to-all.
    return _prepare_dispatch(op)


def _kernel_combine(ctx: dict) -> None:
    ctx["out"] = ctx["fn"](ctx["x"])


IMPLS = [
    BackendImpl(op_type="gemm", prepare=_prepare_gemm, kernel=_kernel_gemm),
    BackendImpl(op_type="allreduce", prepare=_prepare_allreduce, kernel=_kernel_allreduce),
    BackendImpl(op_type="allgather", prepare=_prepare_allgather, kernel=_kernel_allgather),
    BackendImpl(op_type="reduce_scatter", prepare=_prepare_reduce_scatter, kernel=_kernel_reduce_scatter),
    BackendImpl(op_type="alltoall", prepare=_prepare_alltoall, kernel=_kernel_alltoall),
    BackendImpl(op_type="dispatch", prepare=_prepare_dispatch, kernel=_kernel_dispatch),
    BackendImpl(op_type="combine", prepare=_prepare_combine, kernel=_kernel_combine),
]
