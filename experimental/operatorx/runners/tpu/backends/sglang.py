"""SGLang-JAX backend for operatorx on TPU.

Canonical entrypoints from https://github.com/sgl-project/sglang-jax
(install from git main — the 0.0.2 pip release has a different EPMoE API and a
dense-einsum single-device path that main has since replaced with megablox
gmm), matching sgl-jax's production defaults:

  gemm          -> the computation inside `sgl_jax.srt.layers.linear.LinearBase`
                   (`lax.dot_general(..., preferred_element_type)`) over the
                   module's own weights. No custom kernel for unquantized dense.
  moe_forward   -> `sgl_jax.srt.layers.moe.EPMoE` — sgl-jax's default MoE
                   (`--moe-backend epmoe`): shard_map + megablox Pallas gmm,
                   used for all ep_size including 1. Routing (softmax + topk +
                   renormalize, the qwen3 norm_topk_prob semantics) happens
                   caller-side on main, so we compute it inline in the timed
                   region from a LinearBase router — matching how the vllm
                   backend's kernels time their internal routing.
  moe_gemm      -> same EPMoE with ep_size=1 and per-rank-equivalent shapes
                   (num_tokens*top_k/EP tokens over num_experts/EP experts,
                   topk=1) — expert compute only, via the same gmm kernel.
  attention_mha -> `sgl_jax.srt.kernels.ragged_paged_attention.
                   ragged_paged_attention_v3.ragged_paged_attention` — the
                   Pallas RPA-v3 kernel behind sgl-jax's default "fa"
                   attention backend (fused KV-cache variant).

sgl-jax has no standalone collective kernels — those stay with the `jax`
backend.
"""
from __future__ import annotations

import math

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from operatorx.core import BackendImpl, Op, UnsupportedOpError, lookup_versions


def versions() -> dict[str, str]:
    return lookup_versions("sglang-jax", "jax", "jaxlib", "libtpu", "flax")


try:
    from sgl_jax.srt.layers.moe import EPMoE as _EPMoE
    _MOE_AVAILABLE = True
    _MOE_ERR: Exception | None = None
except Exception as e:
    _EPMoE = None  # type: ignore
    _MOE_AVAILABLE = False
    _MOE_ERR = e

try:
    from sgl_jax.srt.layers.linear import LinearBase as _LinearBase
    _LINEAR_AVAILABLE = True
    _LINEAR_ERR: Exception | None = None
except Exception as e:
    _LinearBase = None  # type: ignore
    _LINEAR_AVAILABLE = False
    _LINEAR_ERR = e

try:
    from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
        ragged_paged_attention as _rpa,
    )
    _RPA_AVAILABLE = True
    _RPA_ERR: Exception | None = None
except Exception as e:
    _rpa = None  # type: ignore
    _RPA_AVAILABLE = False
    _RPA_ERR = e

try:
    from sgl_jax.srt.kernels.mla.v2.kernel import (
        mla_ragged_paged_attention as _mla,
        get_kv_cache_shape as _mla_kv_cache_shape,
    )
    _MLA_AVAILABLE = True
    _MLA_ERR: Exception | None = None
except Exception as e:
    _mla = None  # type: ignore
    _mla_kv_cache_shape = None  # type: ignore
    _MLA_AVAILABLE = False
    _MLA_ERR = e


_DTYPES = {
    "bf16": jnp.bfloat16,
    "fp16": jnp.float16,
    "fp32": jnp.float32,
}


def _resolve_dtype(name: str) -> jnp.dtype:
    if name not in _DTYPES:
        raise UnsupportedOpError(
            f"sglang backend doesn't support dtype={name!r} on TPU "
            "(supported: bf16/fp16/fp32)"
        )
    return _DTYPES[name]


_MODULE_CACHE: dict[tuple, object] = {}
_AOT_CACHE: dict[tuple, object] = {}


def _cache_module(key: tuple, module):
    if key not in _MODULE_CACHE:
        _MODULE_CACHE.clear()
        _AOT_CACHE.clear()
        _MODULE_CACHE[key] = module
    return _MODULE_CACHE[key]


def _aot(fn, *args):
    key = (id(fn),) + tuple(
        ("arr", a.shape, str(a.dtype)) if hasattr(a, "shape") else ("v", a)
        for a in args
    )
    exe = _AOT_CACHE.get(key)
    if exe is None:
        exe = fn.lower(*args).compile()
        _AOT_CACHE[key] = exe
    return exe


# EPMoE derives its internal (expert, tensor) moe_mesh from the mesh we pass;
# it expects the standard sgl-jax ("data", "tensor") axis names. Cached per
# ep_size so jit closures keyed on mesh identity stay stable across prepare().
_MESH_CACHE: dict[int, Mesh] = {}


def _make_mesh(ep_size: int) -> Mesh:
    if ep_size > jax.device_count():
        raise UnsupportedOpError(
            f"ep_size={ep_size} > available TPU devices ({jax.device_count()})"
        )
    mesh = _MESH_CACHE.get(ep_size)
    if mesh is None:
        devices = np.asarray(jax.devices()[:ep_size]).reshape(ep_size, 1)
        # sgl-jax builds its meshes with Explicit axis types (see
        # srt/utils/mesh_utils.py) — nnx eager weight sharding requires it.
        mesh = Mesh(devices, ("data", "tensor"),
                    axis_types=(jax.sharding.AxisType.Explicit,) * 2)
        _MESH_CACHE[ep_size] = mesh
    return mesh


# ---------- gemm via LinearBase -----------------------------------------------


def _build_linear(K: int, N: int, dtype: jnp.dtype, mesh: Mesh | None = None):
    if not _LINEAR_AVAILABLE:
        raise UnsupportedOpError(
            f"sgl_jax LinearBase not importable: {_LINEAR_ERR!r}"
        )
    # The router must live on the SAME mesh as the module it feeds — mixing
    # arrays from different meshes in one jit raises incompatible-devices.
    mesh = mesh if mesh is not None else _make_mesh(1)
    key = ("linear", K, N, str(dtype), tuple(mesh.shape.items()))
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    # main-branch signature: (input_size, output_size, mesh, use_bias,
    # skip_bias_add, params_dtype, kernel_axes, scope_name) — no rngs.
    # kernel_axes must be explicit: the default None crashes in
    # `P(*kernel_axes)` upstream. (None, None) = replicated weight. Init
    # resolves that PartitionSpec eagerly, which requires an active mesh
    # context (same requirement as EPMoE).
    with jax.sharding.set_mesh(mesh):
        module = _LinearBase(
            input_size=K, output_size=N, mesh=mesh,
            use_bias=False, params_dtype=dtype,
            kernel_axes=(None, None),
        )
    return _cache_module(key, module)


# LinearBase.__call__ computes
# `lax.dot_general(x, weight, (((x.ndim-1,), (0,)), ((), ())),
#  preferred_element_type=params_dtype)` (see sgl_jax/srt/layers/linear.py).
# We reproduce that exact call over the module's own weight array, AOT-compiled
# over plain arrays — passing the nnx module through the compiled call would
# flatten its pytree per invocation (~1ms python-side), polluting the timing.
def _make_gemm_call(dt):
    @jax.jit
    def _gemm_call(x, w):
        return jax.lax.dot_general(
            x, w, (((x.ndim - 1,), (0,)), ((), ())),
            preferred_element_type=dt,
        )
    return _gemm_call


_GEMM_CALL_CACHE: dict[str, object] = {}


def _prepare_gemm(op: Op) -> dict:
    a = op.args
    if a["dtype_a"] != a["dtype_b"]:
        raise UnsupportedOpError("sglang gemm requires dtype_a == dtype_b")
    dt = _resolve_dtype(a["dtype_a"])

    module = _build_linear(a["k"], a["n"], dt)
    W = module.weight.value  # library-initialized weight, library layout+dtype
    A = jax.random.normal(jax.random.PRNGKey(0), (a["m"], a["k"]), dtype=dt)

    fn = _GEMM_CALL_CACHE.setdefault(str(dt), _make_gemm_call(dt))
    exe = _aot(fn, A, W)
    return {"fn": exe, "A": A, "W": W}


def _kernel_gemm(ctx: dict) -> None:
    ctx["out"] = ctx["fn"](ctx["A"], ctx["W"])


# ---------- EPMoE (git main API) ------------------------------------------------


def _build_epmoe(num_experts: int, num_experts_per_tok: int, ep_size: int,
                 hidden: int, intermediate: int, dtype: jnp.dtype, mesh: Mesh):
    if not _MOE_AVAILABLE:
        raise UnsupportedOpError(
            f"sgl_jax EPMoE not importable: {_MOE_ERR!r}"
        )
    key = ("epmoe", num_experts, num_experts_per_tok, ep_size,
           hidden, intermediate, str(dtype), tuple(mesh.shape.items()))
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    with jax.sharding.set_mesh(mesh):
        module = _EPMoE(
            hidden_size=hidden,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            ep_size=ep_size,
            mesh=mesh,
            intermediate_dim=intermediate,
            weight_dtype=dtype,
            dtype=dtype,
            activation="silu",
        )
    return _cache_module(key, module)


# Compiled MoE executables cached across per-iteration prepare() calls. The
# nnx modules are split once (graphdef static / state pytree of arrays); only
# the cheap state pytrees flow through the compiled call — passing the module
# itself would flatten its full pytree per invocation (~1ms python overhead in
# the timed region) and a fresh nnx.jit closure per prepare would recompile
# every iteration.
_MOE_EXE_CACHE: dict[tuple, tuple] = {}


def _moe_executable(cache_key, module, router, tokens, top_k: int,
                    shared=None):
    """shared: optional (sh_gu [H, 2*I_sh], sh_d [I_sh, H]) dense shared-expert
    MLP weights, composed with the routed output the way model code does
    (n parallel shared MLPs summed == one MLP with n*intermediate, via the
    library's lax.dot_general path)."""
    from flax import nnx as _nnx
    hit = _MOE_EXE_CACHE.get(cache_key)
    if hit is not None:
        return hit
    gd_m, st_m = _nnx.split(module)
    gd_r, st_r = _nnx.split(router)

    def _routed(ms, rs, x):
        moe = _nnx.merge(gd_m, ms)
        rt = _nnx.merge(gd_r, rs)
        logits, _ = rt(x)
        # main EPMoE takes pre-routed (topk_weights, topk_ids); reproduce the
        # qwen3 routing: softmax (fp32) -> top_k -> renormalize.
        probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        topk_w, topk_ids = jax.lax.top_k(probs, top_k)
        topk_w = topk_w / jnp.sum(topk_w, axis=-1, keepdims=True)
        return moe(x, topk_w.astype(moe.dtype), topk_ids)

    if shared is not None:
        def _call(ms, rs, x, sh_gu, sh_d):
            out = _routed(ms, rs, x)
            x2 = x.reshape(-1, x.shape[-1])
            dn = (((1,), (0,)), ((), ()))
            proj = jax.lax.dot_general(x2, sh_gu, dn,
                                       preferred_element_type=x.dtype)
            gate, up = jnp.split(proj, 2, axis=-1)
            sh = jax.lax.dot_general(jax.nn.silu(gate) * up, sh_d, dn,
                                     preferred_element_type=x.dtype)
            return out + sh.reshape(out.shape)

        exe = jax.jit(_call).lower(st_m, st_r, tokens, *shared).compile()
    else:
        exe = jax.jit(_routed).lower(st_m, st_r, tokens).compile()
    _MOE_EXE_CACHE[cache_key] = (exe, st_m, st_r)
    return exe, st_m, st_r


def _prepare_moe_forward(op: Op) -> dict:
    a = op.args
    dt = _resolve_dtype(a["dtype_act"])
    if a["dtype_weight"] != a["dtype_act"]:
        raise UnsupportedOpError(
            "sglang moe_forward requires dtype_act == dtype_weight"
        )
    ep = a.get("expert_parallel_size", a["world_size"])
    if a["num_experts"] % ep != 0:
        raise UnsupportedOpError(
            f"num_experts={a['num_experts']} not divisible by EP={ep}"
        )

    mesh = _make_mesh(ep)
    module = _build_epmoe(
        num_experts=a["num_experts"],
        num_experts_per_tok=a["top_k"],
        ep_size=ep,
        hidden=a["hidden"], intermediate=a["intermediate"],
        dtype=dt, mesh=mesh,
    )
    router = _build_linear(a["hidden"], a["num_experts"], dt, mesh=mesh)

    key = jax.random.PRNGKey(0)
    tokens = jax.random.normal(key, (a["num_tokens"], a["hidden"]), dtype=dt)

    n_shared = a.get("n_shared_experts", 0)
    shared_tp = a.get("shared_tensor_parallel_size", 1)
    shared = None
    if n_shared > 0:
        if (n_shared * a["intermediate"]) % shared_tp != 0:
            raise UnsupportedOpError(
                f"n_shared*intermediate not divisible by shared_tp={shared_tp}")
        i_sh = n_shared * a["intermediate"] // shared_tp
        shared = (
            jax.random.normal(jax.random.fold_in(key, 4),
                              (a["hidden"], 2 * i_sh), dtype=dt),
            jax.random.normal(jax.random.fold_in(key, 5),
                              (i_sh, a["hidden"]), dtype=dt),
        )

    cache_key = ("fwd", a["num_experts"], a["top_k"], ep,
                 a["hidden"], a["intermediate"], str(dt), tokens.shape,
                 n_shared, shared_tp)
    exe, st_m, st_r = _moe_executable(cache_key, module, router, tokens,
                                      top_k=a["top_k"], shared=shared)
    ctx = {"fn": exe, "st_m": st_m, "st_r": st_r, "tokens": tokens}
    if shared is not None:
        ctx["sh_gu"], ctx["sh_d"] = shared
    return ctx


def _kernel_moe_forward(ctx: dict) -> None:
    if "sh_gu" in ctx:
        ctx["out"] = ctx["fn"](ctx["st_m"], ctx["st_r"], ctx["tokens"],
                               ctx["sh_gu"], ctx["sh_d"])
    else:
        ctx["out"] = ctx["fn"](ctx["st_m"], ctx["st_r"], ctx["tokens"])


def _prepare_moe_gemm(op: Op) -> dict:
    a = op.args
    dt = _resolve_dtype(a["dtype_act"])
    if a["dtype_weight"] != a["dtype_act"]:
        raise UnsupportedOpError(
            "sglang moe_gemm requires dtype_act == dtype_weight"
        )
    ep = a.get("expert_parallel_size", 1)
    if a["num_experts"] % ep != 0:
        raise UnsupportedOpError(
            f"num_experts={a['num_experts']} not divisible by EP={ep}"
        )
    if (a["num_tokens"] * a["top_k"]) % ep != 0:
        raise UnsupportedOpError(
            f"num_tokens*top_k={a['num_tokens']*a['top_k']} not divisible by EP={ep}"
        )

    local_tokens = (a["num_tokens"] * a["top_k"]) // ep
    local_experts = a["num_experts"] // ep

    mesh = _make_mesh(1)
    module = _build_epmoe(
        num_experts=local_experts,
        num_experts_per_tok=1,
        ep_size=1,
        hidden=a["hidden"], intermediate=a["intermediate"],
        dtype=dt, mesh=mesh,
    )
    router = _build_linear(a["hidden"], local_experts, dt, mesh=mesh)
    key = jax.random.PRNGKey(0)
    tokens = jax.random.normal(key, (local_tokens, a["hidden"]), dtype=dt)

    n_shared = a.get("n_shared_experts", 0)
    shared_tp = a.get("shared_tensor_parallel_size", 1)
    shared = None
    if n_shared > 0:
        if (n_shared * a["intermediate"]) % shared_tp != 0:
            raise UnsupportedOpError(
                f"n_shared*intermediate not divisible by shared_tp={shared_tp}")
        i_sh = n_shared * a["intermediate"] // shared_tp
        shared = (
            jax.random.normal(jax.random.fold_in(key, 4),
                              (a["hidden"], 2 * i_sh), dtype=dt),
            jax.random.normal(jax.random.fold_in(key, 5),
                              (i_sh, a["hidden"]), dtype=dt),
        )

    cache_key = ("gemm", local_experts, 1, 1,
                 a["hidden"], a["intermediate"], str(dt), tokens.shape,
                 n_shared, shared_tp)
    exe, st_m, st_r = _moe_executable(cache_key, module, router, tokens,
                                      top_k=1, shared=shared)
    ctx = {"fn": exe, "st_m": st_m, "st_r": st_r, "tokens": tokens}
    if shared is not None:
        ctx["sh_gu"], ctx["sh_d"] = shared
    return ctx


def _kernel_moe_gemm(ctx: dict) -> None:
    if "sh_gu" in ctx:
        ctx["out"] = ctx["fn"](ctx["st_m"], ctx["st_r"], ctx["tokens"],
                               ctx["sh_gu"], ctx["sh_d"])
    else:
        ctx["out"] = ctx["fn"](ctx["st_m"], ctx["st_r"], ctx["tokens"])


# ---------- attention_mha via RPA v3 (fused-KV variant) ---------------------------

_RPA_PAGE_SIZE = 64


def _align(x: int, n: int) -> int:
    return (x + n - 1) // n * n


def _prepare_attention_mha(op: Op) -> dict:
    if not _RPA_AVAILABLE:
        raise UnsupportedOpError(
            f"sgl_jax ragged_paged_attention v3 not importable: {_RPA_ERR!r}")
    a = op.args
    dt = _resolve_dtype(a.get("dtype_q", "bf16"))
    B = a["batch_size"]
    Sq, Skv = a["seq_len_q"], a["seq_len_kv"]
    H = a["num_heads"]
    Hkv = a.get("num_heads_kv", H)
    D = a["head_dim"]
    if Sq > Skv:
        raise UnsupportedOpError("attention requires seq_len_q <= seq_len_kv")

    max_num_seqs = _align(B, 8)
    pages_per_seq = -(-Skv // _RPA_PAGE_SIZE)
    total_pages = max_num_seqs * pages_per_seq + 1
    T = _align(B * Sq, 128)

    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (T, H, D), dtype=dt)
    k = jax.random.normal(jax.random.fold_in(key, 1), (T, Hkv, D), dtype=dt)
    v = jax.random.normal(jax.random.fold_in(key, 2), (T, Hkv, D), dtype=dt)
    # Fused cache is 5-D packed (see prepare_kv_cache_fused):
    # [pages, page_size, (Hkv*2)//packing, packing, D] where packing is the
    # number of elements per 32-bit word (bf16 -> 2, fp32 -> 1).
    packing = 4 // jnp.dtype(dt).itemsize
    kv_cache = jax.random.normal(
        jax.random.fold_in(key, 3),
        (total_pages, _RPA_PAGE_SIZE, (Hkv * 2) // packing, packing, D),
        dtype=dt)

    kv_lens = jnp.asarray([Skv] * B + [0] * (max_num_seqs - B), dtype=jnp.int32)
    page_indices = jnp.asarray(
        np.arange(max_num_seqs * pages_per_seq, dtype=np.int32))
    cu_q = np.minimum(np.arange(max_num_seqs + 1) * Sq, B * Sq)
    cu_q_lens = jnp.asarray(cu_q, dtype=jnp.int32)
    # cu_kv_lens = cumsum of page_size-aligned kv lens.
    aligned_kv = _align(Skv, _RPA_PAGE_SIZE)
    cu_kv = np.minimum(np.arange(max_num_seqs + 1) * aligned_kv, B * aligned_kv)
    cu_kv_lens = jnp.asarray(cu_kv, dtype=jnp.int32)
    if Sq == 1:
        dist = (B, B, B)
    elif Sq == Skv:
        dist = (0, B, B)
    else:
        dist = (0, 0, B)
    distribution = jnp.asarray(dist, dtype=jnp.int32)

    sm_scale = 1.0 / math.sqrt(D)

    # Their jit donates q/k/v/kv_cache_fused — donation would invalidate our
    # buffers across repeated kernel() calls, so re-jit without donation.
    base = getattr(_rpa, "__wrapped__", None)
    if base is not None:
        fn = jax.jit(
            lambda q_, k_, v_, c_, kl_, pi_, cq_, ckv_, d_: base(
                q_, k_, v_, c_, kl_, pi_, cq_, ckv_, d_, None,
                sm_scale=sm_scale, causal=1),
        )
    else:
        def fn(q_, k_, v_, c_, kl_, pi_, cq_, ckv_, d_):
            return _rpa(q_, k_, v_, c_, kl_, pi_, cq_, ckv_, d_, None,
                        sm_scale=sm_scale, causal=1)

    exe = _aot(fn, q, k, v, kv_cache, kv_lens, page_indices,
               cu_q_lens, cu_kv_lens, distribution)
    return {"fn": exe, "q": q, "k": k, "v": v, "kv_cache": kv_cache,
            "kv_lens": kv_lens, "page_indices": page_indices,
            "cu_q_lens": cu_q_lens, "cu_kv_lens": cu_kv_lens,
            "distribution": distribution}


def _kernel_attention_mha(ctx: dict) -> None:
    out, _cache = ctx["fn"](
        ctx["q"], ctx["k"], ctx["v"], ctx["kv_cache"], ctx["kv_lens"],
        ctx["page_indices"], ctx["cu_q_lens"], ctx["cu_kv_lens"],
        ctx["distribution"])
    ctx["out"] = out


# ---------- attention_mla via the MLA v2 Pallas kernel ---------------------------
#
# Absorbed/latent MLA formulation, same as vLLM-TPU's kernel but the sgl v2
# variant additionally takes cu_kv_lens (page-aligned cumulative kv lens).


def _prepare_attention_mla(op: Op) -> dict:
    if not _MLA_AVAILABLE:
        raise UnsupportedOpError(
            f"sgl_jax mla_ragged_paged_attention not importable: {_MLA_ERR!r}")
    a = op.args
    dt = _resolve_dtype(a.get("dtype_q", "bf16"))
    B = a["batch_size"]
    Sq, Skv = a["seq_len_q"], a["seq_len_kv"]
    H = a["num_heads"]
    lkv = a["kv_lora_rank"]
    r = a["head_dim_qk_rope"]
    if Sq > Skv:
        raise UnsupportedOpError("attention requires seq_len_q <= seq_len_kv")

    max_num_seqs = _align(B, 8)
    pages_per_seq = -(-Skv // _RPA_PAGE_SIZE)
    total_pages = max_num_seqs * pages_per_seq + 1
    T = _align(B * Sq, 128)

    key = jax.random.PRNGKey(0)
    ql_nope = jax.random.normal(key, (T, H, lkv), dtype=dt)
    q_pe = jax.random.normal(jax.random.fold_in(key, 1), (T, H, r), dtype=dt)
    new_kv_c = jax.random.normal(jax.random.fold_in(key, 2), (T, lkv), dtype=dt)
    new_k_pe = jax.random.normal(jax.random.fold_in(key, 3), (T, r), dtype=dt)
    cache_shape = _mla_kv_cache_shape(total_pages, _RPA_PAGE_SIZE, lkv + r, dt)
    cache_kv = jax.random.normal(jax.random.fold_in(key, 4), cache_shape, dtype=dt)

    kv_lens = jnp.asarray([Skv] * B + [0] * (max_num_seqs - B), dtype=jnp.int32)
    page_indices = jnp.asarray(
        np.arange(max_num_seqs * pages_per_seq, dtype=np.int32))
    cu = np.minimum(np.arange(max_num_seqs + 1) * Sq, B * Sq)
    cu_q_lens = jnp.asarray(cu, dtype=jnp.int32)
    aligned_kv = _align(Skv, _RPA_PAGE_SIZE)
    cu_kv = np.minimum(np.arange(max_num_seqs + 1) * aligned_kv, B * aligned_kv)
    cu_kv_lens = jnp.asarray(cu_kv, dtype=jnp.int32)
    if Sq == 1:
        dist = (B, B, B)
    elif Sq == Skv:
        dist = (0, B, B)
    else:
        dist = (0, 0, B)
    distribution = jnp.asarray(dist, dtype=jnp.int32)

    sm_scale = 1.0 / math.sqrt(a["head_dim_qk_nope"] + r)

    base = getattr(_mla, "__wrapped__", None)
    if base is not None:
        fn = jax.jit(
            lambda qn_, qp_, kc_, kp_, c_, kl_, pi_, cq_, ckv_, d_: base(
                qn_, qp_, kc_, kp_, c_, kl_, pi_, cq_, ckv_, d_,
                sm_scale=sm_scale))
    else:
        def fn(qn_, qp_, kc_, kp_, c_, kl_, pi_, cq_, ckv_, d_):
            return _mla(qn_, qp_, kc_, kp_, c_, kl_, pi_, cq_, ckv_, d_,
                        sm_scale=sm_scale)

    exe = _aot(fn, ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv,
               kv_lens, page_indices, cu_q_lens, cu_kv_lens, distribution)
    return {"fn": exe, "ql_nope": ql_nope, "q_pe": q_pe, "new_kv_c": new_kv_c,
            "new_k_pe": new_k_pe, "cache_kv": cache_kv, "kv_lens": kv_lens,
            "page_indices": page_indices, "cu_q_lens": cu_q_lens,
            "cu_kv_lens": cu_kv_lens, "distribution": distribution}


def _kernel_attention_mla(ctx: dict) -> None:
    out, _cache = ctx["fn"](
        ctx["ql_nope"], ctx["q_pe"], ctx["new_kv_c"], ctx["new_k_pe"],
        ctx["cache_kv"], ctx["kv_lens"], ctx["page_indices"],
        ctx["cu_q_lens"], ctx["cu_kv_lens"], ctx["distribution"])
    ctx["out"] = out


IMPLS = [
    BackendImpl(op_type="gemm",          prepare=_prepare_gemm,          kernel=_kernel_gemm),
    BackendImpl(op_type="moe_forward",   prepare=_prepare_moe_forward,   kernel=_kernel_moe_forward),
    BackendImpl(op_type="moe_gemm",      prepare=_prepare_moe_gemm,      kernel=_kernel_moe_gemm),
    BackendImpl(op_type="attention_mha", prepare=_prepare_attention_mha, kernel=_kernel_attention_mha),
    BackendImpl(op_type="attention_mla", prepare=_prepare_attention_mla, kernel=_kernel_attention_mla),
]
