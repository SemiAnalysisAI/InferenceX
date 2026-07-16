"""vLLM-TPU (tpu-inference) backend for operatorx on TPU.

Canonical entrypoints from https://github.com/vllm-project/tpu-inference
(pip: `tpu-inference`, requires vllm installed as a peer dep), matching what
vLLM-TPU actually selects in production serving:

  gemm          -> the computation inside `JaxLinear`/`JaxEinsum`
                   (`jnp.einsum("mn,np->mp")`) over the module's own weights.
                   Unquantized dense linears have no custom-kernel path.
  moe_forward   -> `layers/common/fused_moe_gmm.fused_moe_func` — the GMM_TP /
                   GMM_EP backend (megablox gmm_v2 + SparseCore ragged_gather),
                   which is vLLM-TPU's production DEFAULT for bf16 MoE.
                   GMM_EP when the shape declares expert_parallel_size>1
                   (vLLM's --enable-expert-parallel), else GMM_TP.
                   Set OPERATORX_VLLM_MOE_KERNEL=fused to benchmark the
                   opt-in `fused_ep_moe` Pallas kernel instead (vLLM-TPU's
                   USE_MOE_EP_KERNEL=1 path).
  moe_gemm      -> same entrypoints on a 1-device mesh with shapes rescaled to
                   the per-rank equivalent of an EP deployment (num_tokens *
                   top_k / EP tokens over num_experts / EP experts, topk=1) —
                   no cross-device comms, expert compute only.
  attention_mha -> `kernels/ragged_paged_attention/v3/kernel.py`
                   `ragged_paged_attention` — the production-default Pallas
                   attention kernel (RPA v3).

tpu-inference ships no plain reduce_scatter/allreduce kernels (only the fused
all_gather_matmul, a different op shape) — plain collectives stay with the
`jax` backend.
"""
from __future__ import annotations

import math
import os

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from operatorx.core import BackendImpl, Op, UnsupportedOpError, lookup_versions


def versions() -> dict[str, str]:
    return lookup_versions("tpu-inference", "vllm", "jax", "jaxlib", "libtpu", "flax")


try:
    from tpu_inference.layers.common.fused_moe_gmm import fused_moe_func as _fused_moe_func
    _GMM_MOE_AVAILABLE = True
    _GMM_MOE_ERR: Exception | None = None
except Exception as e:
    _fused_moe_func = None  # type: ignore
    _GMM_MOE_AVAILABLE = False
    _GMM_MOE_ERR = e

try:
    from tpu_inference.kernels.fused_moe.v1.kernel import fused_ep_moe as _fused_ep_moe
    _FUSED_MOE_AVAILABLE = True
    _FUSED_MOE_ERR: Exception | None = None
except Exception as e:
    _fused_ep_moe = None  # type: ignore
    _FUSED_MOE_AVAILABLE = False
    _FUSED_MOE_ERR = e

try:
    from tpu_inference.layers.jax.linear import JaxLinear as _JaxLinear
    _LINEAR_AVAILABLE = True
    _LINEAR_ERR: Exception | None = None
except Exception as e:
    _JaxLinear = None  # type: ignore
    _LINEAR_AVAILABLE = False
    _LINEAR_ERR = e

try:
    from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
        ragged_paged_attention as _rpa,
        get_kv_cache_shape as _rpa_kv_cache_shape,
    )
    _RPA_AVAILABLE = True
    _RPA_ERR: Exception | None = None
except Exception as e:
    _rpa = None  # type: ignore
    _rpa_kv_cache_shape = None  # type: ignore
    _RPA_AVAILABLE = False
    _RPA_ERR = e

# Production MLA is the v2 kernel — layers/common/attention_interface.py calls
# it with hardcoded tuned block sizes (see below); v1 requires explicit block
# sizes and is not what serving uses.
try:
    from tpu_inference.kernels.mla.v2.kernel import (
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

# Mesh axis names as the library defines them (layers/common/sharding.py).
# fused_moe_func's shard_map specs reference these names, so our bench mesh
# must carry them. Fallback literals match the library's current values but
# are validated on-device.
try:
    from tpu_inference.layers.common.sharding import ShardingAxisName as _AxN
    _AX_MLP_DATA = getattr(_AxN, "MLP_DATA", "data")
    _AX_MLP_TENSOR = getattr(_AxN, "MLP_TENSOR", "model")
    _AX_EXPERT = getattr(_AxN, "EXPERT", "expert")
except Exception:
    _AX_MLP_DATA, _AX_MLP_TENSOR, _AX_EXPERT = "data", "model", "expert"


_DTYPES = {
    "bf16": jnp.bfloat16,
    "fp16": jnp.float16,
    "fp32": jnp.float32,
}


def _resolve_dtype(name: str) -> jnp.dtype:
    if name not in _DTYPES:
        raise UnsupportedOpError(
            f"vllm backend doesn't support dtype={name!r} on TPU "
            "(supported: bf16/fp16/fp32)"
        )
    return _DTYPES[name]


# Size-1 module cache (nnx modules hold device buffers) + AOT executable cache.
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


# ---------- gemm (JaxLinear computation over library weights) ------------------


def _build_linear(K: int, N: int, dtype: jnp.dtype):
    if not _LINEAR_AVAILABLE:
        raise UnsupportedOpError(
            f"tpu-inference JaxLinear not importable: {_LINEAR_ERR!r}"
        )
    from flax import nnx
    key = ("linear", K, N, str(dtype))
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]
    module = _JaxLinear(
        input_size=K, output_size=N, rngs=nnx.Rngs(0),
        use_bias=False, quant_config=None, param_dtype=dtype,
    )
    return _cache_module(key, module)


# JaxLinear.__call__ computes `jnp.einsum("mn,np->mp", x, weight)` (see
# tpu_inference/layers/jax/linear.py). We reproduce that exact computation over
# the module's own weight array, AOT-compiled over plain arrays. Passing the
# nnx module itself through the compiled call would flatten its pytree on every
# invocation (~1ms python-side), polluting the kernel timing — in real serving
# the module is traversed once per model step, not per layer call.
@jax.jit
def _gemm_call(x, w):
    return jnp.einsum("mn,np->mp", x, w)


def _prepare_gemm(op: Op) -> dict:
    a = op.args
    if a["dtype_a"] != a["dtype_b"]:
        raise UnsupportedOpError("vllm gemm requires dtype_a == dtype_b")
    dt = _resolve_dtype(a["dtype_a"])

    module = _build_linear(a["k"], a["n"], dt)
    W = module.weight.value  # library-initialized weight, library layout+dtype
    A = jax.random.normal(jax.random.PRNGKey(0), (a["m"], a["k"]), dtype=dt)

    exe = _aot(_gemm_call, A, W)
    return {"fn": exe, "A": A, "W": W}


def _kernel_gemm(ctx: dict) -> None:
    ctx["out"] = ctx["fn"](ctx["A"], ctx["W"])


# ---------- MoE meshes ----------------------------------------------------------

_MESH_CACHE: dict[tuple, Mesh] = {}


def _moe_mesh(n_devices: int, use_ep: bool) -> Mesh:
    """Mesh whose axis names satisfy fused_moe_func's shard_map specs.

    In tpu-inference's ShardingAxisName, EXPERT and MLP_TENSOR are the SAME
    physical axis ("model") and MLP_DATA/ATTN_DATA are "data" — production
    runs a single 2-D (data, model) mesh and GMM_TP vs GMM_EP merely change
    which tensor dimension is sharded along "model". So both paths use the
    same (data=1, model=n) mesh here (verified on tpu7x, tpu-inference
    0.22.1: MLP_DATA='data', MLP_TENSOR='model', EXPERT='model').
    """
    if n_devices > jax.device_count():
        raise UnsupportedOpError(
            f"world_size={n_devices} > available TPU devices ({jax.device_count()})"
        )
    del use_ep  # same mesh either way; kept in signature for call-site clarity
    key = n_devices
    mesh = _MESH_CACHE.get(key)
    if mesh is None:
        devices = np.asarray(jax.devices()[:n_devices]).reshape(1, n_devices)
        mesh = Mesh(devices, (_AX_MLP_DATA, _AX_MLP_TENSOR))
        _MESH_CACHE[key] = mesh
    return mesh


# fused_ep_moe (the opt-in Pallas kernel) needs a 2-D mesh with a "data" axis
# present and all non-EP axes size 1.
def _fused_kernel_mesh(ep_size: int) -> Mesh:
    if ep_size > jax.device_count():
        raise UnsupportedOpError(
            f"ep_size={ep_size} > available TPU devices ({jax.device_count()})"
        )
    key = (ep_size, "fused")
    mesh = _MESH_CACHE.get(key)
    if mesh is None:
        devices = np.asarray(jax.devices()[:ep_size]).reshape(1, ep_size)
        mesh = Mesh(devices, ("data", "model"))
        _MESH_CACHE[key] = mesh
    return mesh


def _moe_kernel_choice() -> str:
    """'gmm' (production default) or 'fused' (USE_MOE_EP_KERNEL=1 analog)."""
    return os.environ.get("OPERATORX_VLLM_MOE_KERNEL", "gmm").lower()


# ---------- MoE weights ---------------------------------------------------------


def _gmm_weights(num_experts, hidden, intermediate, dtype, key):
    """fused_moe_func layout: w1 [E, H, 2*I] (up|gate fused), w2 [E, I, H]."""
    w1 = jax.random.normal(
        jax.random.fold_in(key, 1), (num_experts, hidden, 2 * intermediate), dtype=dtype)
    w2 = jax.random.normal(
        jax.random.fold_in(key, 2), (num_experts, intermediate, hidden), dtype=dtype)
    return w1, w2


def _fused_weights(num_experts, hidden, intermediate, dtype, key):
    """fused_ep_moe layout: w1 [E, 2, H, I], w2 [E, I, H]."""
    w1 = jax.random.normal(
        jax.random.fold_in(key, 1), (num_experts, 2, hidden, intermediate), dtype=dtype)
    w2 = jax.random.normal(
        jax.random.fold_in(key, 2), (num_experts, intermediate, hidden), dtype=dtype)
    return w1, w2


# ---------- MoE call builders ----------------------------------------------------

_MOE_FN_CACHE: dict[tuple, object] = {}


def _build_moe_call(kchoice: str, mesh: Mesh, topk: int, use_ep: bool,
                    has_shared: bool):
    """Composed MoE call: routed experts via the selected library kernel,
    plus (optionally) the shared-expert dense MLP summed into the output —
    the same composition production model code uses (shared experts are
    ordinary MLPs at model level in both vLLM-TPU and sgl-jax; n parallel
    shared MLPs summed == one MLP with n*intermediate, so we fuse them).
    Cached per config so per-iteration prepare() reuses one jitted fn.
    """
    key = (kchoice, id(mesh), topk, use_ep, has_shared)
    fn = _MOE_FN_CACHE.get(key)
    if fn is not None:
        return fn

    if kchoice == "fused":
        if not _FUSED_MOE_AVAILABLE:
            raise UnsupportedOpError(
                f"tpu-inference fused_ep_moe not importable: {_FUSED_MOE_ERR!r}")

        def _routed(tokens, w1, w2, gating):
            return _fused_ep_moe(
                mesh=mesh, tokens=tokens, w1=w1, w2=w2,
                gating_output=gating, top_k=topk,
                act_fn="silu", scoring_fn="softmax",
                ep_axis_name="model",
            )
    else:
        if not _GMM_MOE_AVAILABLE:
            raise UnsupportedOpError(
                f"tpu-inference fused_moe_func not importable: {_GMM_MOE_ERR!r}")

        def _routed(tokens, w1, w2, gating):
            return _fused_moe_func(
                tokens, w1, w2,
                None, None,      # w1_scale, w2_scale (unquantized)
                None, None,      # w1_bias, w2_bias
                gating,
                topk=topk,
                renormalize=True,
                mesh=mesh,
                use_ep=use_ep,
                activation="silu",
                scoring_fn="softmax",
            )

    if has_shared:
        def _call(tokens, w1, w2, gating, sh_gu, sh_d):
            out = _routed(tokens, w1, w2, gating)
            # Shared-expert dense MLP over all tokens (jnp.einsum — the same
            # XLA path vLLM-TPU dense MLPs use), summed with routed output.
            proj = jnp.einsum("mn,np->mp", tokens, sh_gu)
            gate, up = jnp.split(proj, 2, axis=-1)
            shared = jnp.einsum("mn,np->mp", jax.nn.silu(gate) * up, sh_d)
            return out + shared
    else:
        def _call(tokens, w1, w2, gating):
            return _routed(tokens, w1, w2, gating)

    fn = jax.jit(_call)
    _MOE_FN_CACHE[key] = fn
    return fn


def _moe_setup(num_tokens: int, num_experts: int, hidden: int, intermediate: int,
               top_k: int, n_devices: int, use_ep: bool, dt,
               n_shared: int = 0, shared_tp: int = 1) -> dict:
    """Shared prepare for moe_forward / moe_gemm across both kernel choices."""
    kchoice = _moe_kernel_choice()
    key = jax.random.PRNGKey(0)
    tokens = jax.random.normal(key, (num_tokens, hidden), dtype=dt)
    gating = jax.random.normal(
        jax.random.fold_in(key, 3), (num_tokens, num_experts), dtype=dt)

    if kchoice == "fused":
        w1, w2 = _fused_weights(num_experts, hidden, intermediate, dt, key)
        mesh = _fused_kernel_mesh(n_devices)
    else:
        # gmm kernel constraint (fused_moe_gmm.py): (num_tokens*topk) % 16 == 0.
        # Production doesn't reject such batches — vLLM-TPU bucket-pads token
        # counts (VLLM_TPU_BUCKET_PADDING_GAP); mirror that by padding tokens
        # up to the smallest count satisfying the constraint.
        if (num_tokens * top_k) % 16 != 0:
            step = 16 // math.gcd(top_k, 16)
            padded = -(-num_tokens // step) * step
            tokens = jax.random.normal(key, (padded, hidden), dtype=dt)
            gating = jax.random.normal(
                jax.random.fold_in(key, 3), (padded, num_experts), dtype=dt)
        w1, w2 = _gmm_weights(num_experts, hidden, intermediate, dt, key)
        mesh = _moe_mesh(n_devices, use_ep)

    has_shared = n_shared > 0
    fn = _build_moe_call(kchoice, mesh, top_k, use_ep, has_shared)

    if has_shared:
        if (n_shared * intermediate) % shared_tp != 0:
            raise UnsupportedOpError(
                f"n_shared*intermediate not divisible by shared_tp={shared_tp}")
        i_sh = n_shared * intermediate // shared_tp  # per-rank shard
        sh_gu = jax.random.normal(
            jax.random.fold_in(key, 4), (hidden, 2 * i_sh), dtype=dt)
        sh_d = jax.random.normal(
            jax.random.fold_in(key, 5), (i_sh, hidden), dtype=dt)
        exe = _aot(fn, tokens, w1, w2, gating, sh_gu, sh_d)
        return {"fn": exe, "tokens": tokens, "w1": w1, "w2": w2,
                "gating": gating, "sh_gu": sh_gu, "sh_d": sh_d}

    exe = _aot(fn, tokens, w1, w2, gating)
    return {"fn": exe, "tokens": tokens, "w1": w1, "w2": w2, "gating": gating}


def _kernel_moe(ctx: dict) -> None:
    if "sh_gu" in ctx:
        ctx["out"] = ctx["fn"](ctx["tokens"], ctx["w1"], ctx["w2"],
                               ctx["gating"], ctx["sh_gu"], ctx["sh_d"])
    else:
        ctx["out"] = ctx["fn"](ctx["tokens"], ctx["w1"], ctx["w2"], ctx["gating"])


# ---------- moe_forward ----------------------------------------------------------


def _prepare_moe_forward(op: Op) -> dict:
    a = op.args
    dt = _resolve_dtype(a["dtype_act"])
    if a["dtype_weight"] != a["dtype_act"]:
        raise UnsupportedOpError(
            "vllm moe_forward requires dtype_act == dtype_weight on this path")
    ep = a.get("expert_parallel_size", 1)
    ws = a.get("world_size", max(ep, 1))
    if a["num_experts"] % max(ep, 1) != 0:
        raise UnsupportedOpError(
            f"num_experts={a['num_experts']} not divisible by EP={ep}")
    # vLLM semantics: EP>1 => --enable-expert-parallel => GMM_EP; otherwise the
    # devices form a TP group over the MoE weights => GMM_TP.
    use_ep = ep > 1
    return _moe_setup(a["num_tokens"], a["num_experts"], a["hidden"],
                      a["intermediate"], a["top_k"], ws, use_ep, dt,
                      n_shared=a.get("n_shared_experts", 0),
                      shared_tp=a.get("shared_tensor_parallel_size", 1))


# ---------- moe_gemm (per-rank expert compute, no comms) --------------------------


def _prepare_moe_gemm(op: Op) -> dict:
    a = op.args
    dt = _resolve_dtype(a["dtype_act"])
    if a["dtype_weight"] != a["dtype_act"]:
        raise UnsupportedOpError(
            "vllm moe_gemm requires dtype_act == dtype_weight on this path")
    ep = a.get("expert_parallel_size", 1)
    if a["num_experts"] % max(ep, 1) != 0:
        raise UnsupportedOpError(
            f"num_experts={a['num_experts']} not divisible by EP={ep}")
    if (a["num_tokens"] * a["top_k"]) % max(ep, 1) != 0:
        raise UnsupportedOpError(
            f"num_tokens*top_k not divisible by EP={ep}")

    local_tokens = (a["num_tokens"] * a["top_k"]) // max(ep, 1)
    local_experts = a["num_experts"] // max(ep, 1)
    return _moe_setup(local_tokens, local_experts, a["hidden"],
                      a["intermediate"], 1, 1, False, dt,
                      n_shared=a.get("n_shared_experts", 0),
                      shared_tp=a.get("shared_tensor_parallel_size", 1))


# ---------- attention_mha via RPA v3 ----------------------------------------------

_RPA_PAGE_SIZE = 64


def _align(x: int, n: int) -> int:
    return (x + n - 1) // n * n


def _prepare_attention_mha(op: Op) -> dict:
    if not _RPA_AVAILABLE:
        raise UnsupportedOpError(
            f"tpu-inference ragged_paged_attention v3 not importable: {_RPA_ERR!r}")
    a = op.args
    dt_name = a.get("dtype_q", "bf16")
    dt = _resolve_dtype(dt_name)
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

    cache_shape = _rpa_kv_cache_shape(total_pages, _RPA_PAGE_SIZE, Hkv, D, dt)
    # Pre-populate so the "existing context" (Skv - Sq tokens) reads real data.
    kv_cache = jax.random.normal(jax.random.fold_in(key, 3), cache_shape, dtype=dt)

    kv_lens = jnp.asarray(
        [Skv] * B + [0] * (max_num_seqs - B), dtype=jnp.int32)
    page_indices = jnp.asarray(
        np.arange(max_num_seqs * pages_per_seq, dtype=np.int32))
    cu = np.minimum(np.arange(max_num_seqs + 1) * Sq, B * Sq)
    cu_q_lens = jnp.asarray(cu, dtype=jnp.int32)
    if Sq == 1:
        dist = (B, B, B)          # decode-only
    elif Sq == Skv:
        dist = (0, B, B)          # pure prefill
    else:
        dist = (0, 0, B)          # extend/mixed
    distribution = jnp.asarray(dist, dtype=jnp.int32)

    sm_scale = 1.0 / math.sqrt(D)

    # The library jit donates q/k/v/kv_cache — donation would invalidate our
    # buffers across the runner's repeated kernel() calls (pipelined timing),
    # so we re-jit the unwrapped function without donation when possible.
    base = getattr(_rpa, "__wrapped__", None)
    if base is not None:
        fn = jax.jit(
            lambda q_, k_, v_, c_, kl_, pi_, cq_, d_: base(
                q_, k_, v_, c_, kl_, pi_, cq_, d_,
                sm_scale=sm_scale, use_causal_mask=True),
        )
    else:
        def fn(q_, k_, v_, c_, kl_, pi_, cq_, d_):
            return _rpa(q_, k_, v_, c_, kl_, pi_, cq_, d_,
                        sm_scale=sm_scale, use_causal_mask=True)

    exe = _aot(fn, q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution)
    return {"fn": exe, "q": q, "k": k, "v": v, "kv_cache": kv_cache,
            "kv_lens": kv_lens, "page_indices": page_indices,
            "cu_q_lens": cu_q_lens, "distribution": distribution}


def _kernel_attention_mha(ctx: dict) -> None:
    out, _cache = ctx["fn"](
        ctx["q"], ctx["k"], ctx["v"], ctx["kv_cache"], ctx["kv_lens"],
        ctx["page_indices"], ctx["cu_q_lens"], ctx["distribution"])
    ctx["out"] = out


# ---------- attention_mla via the MLA Pallas kernel -------------------------------
#
# `mla_ragged_paged_attention` implements the absorbed/latent MLA formulation:
# queries are pre-projected into the kv_lora_rank latent space (ql_nope
# [T, H, lkv]) plus the rope part (q_pe [T, H, r]); the paged cache stores
# latent+rope per token, and the output is in latent space (the o-projection
# happens outside, scored separately as gemm). This is exactly the kernel
# vLLM-TPU serves DeepSeek-style models with.


def _prepare_attention_mla(op: Op) -> dict:
    if not _MLA_AVAILABLE:
        raise UnsupportedOpError(
            f"tpu-inference mla_ragged_paged_attention not importable: {_MLA_ERR!r}")
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
    # v2 kernel takes ql_nope HEAD-major [H, T, lkv] (production call site:
    # "q_NTA: (num_query_heads, tokens_query, q_lora_rank) # head-major");
    # q_pe stays token-major [T, H, r].
    ql_nope = jax.random.normal(key, (H, T, lkv), dtype=dt)
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
    if Sq == 1:
        dist = (B, B, B)
    elif Sq == Skv:
        dist = (0, B, B)
    else:
        dist = (0, 0, B)
    distribution = jnp.asarray(dist, dtype=jnp.int32)

    # Softmax over the full qk dim (nope + rope), DeepSeek convention.
    sm_scale = 1.0 / math.sqrt(a["head_dim_qk_nope"] + r)

    # Block sizes exactly as vLLM-TPU's production call site hardcodes them
    # (layers/common/attention_interface.py — "TODO: use auto tuner").
    kw = dict(sm_scale=sm_scale,
              num_kv_pages_per_block=(3, 1, 1),
              num_queries_per_block=(1, 16, 16),
              decode_batch_size=4)
    base = getattr(_mla, "__wrapped__", None) or _mla

    # Production runs MLA TP-sharded: query heads split across cores, the
    # latent KV cache REPLICATED on every rank (that's MLA's memory trade).
    # The kernel's internal transpose stack-allocates [H_local, T, lkv] in
    # VMEM, which overflows for full H=128/lkv=512 on one core — so we mirror
    # the production sharding whenever the head count divides the chip's
    # cores; else fall back to single-core.
    n_dev = jax.device_count()
    if H % n_dev == 0 and n_dev > 1:
        from jax.sharding import PartitionSpec as P
        devices = np.asarray(jax.devices()).reshape(n_dev)
        mesh = Mesh(devices, ("model",))

        def _sharded(qn_, qp_, kc_, kp_, c_, kl_, pi_, cq_, d_):
            return jax.shard_map(
                lambda *args: base(*args, **kw),
                mesh=mesh,
                in_specs=(P("model", None, None),  # ql_nope [H, T, lkv]
                          P(None, "model", None),  # q_pe    [T, H, r]
                          P(None, None),           # new_kv_c (replicated)
                          P(None, None),           # new_k_pe (replicated)
                          P(*(None,) * len(c_.shape)),  # cache (replicated)
                          P(None), P(None), P(None), P(None)),
                out_specs=(P(None, "model", None), P(*(None,) * len(c_.shape))),
                check_vma=False,
            )(qn_, qp_, kc_, kp_, c_, kl_, pi_, cq_, d_)

        fn = jax.jit(_sharded)
    else:
        def fn(qn_, qp_, kc_, kp_, c_, kl_, pi_, cq_, d_):
            return base(qn_, qp_, kc_, kp_, c_, kl_, pi_, cq_, d_, **kw)
        fn = jax.jit(fn)

    exe = _aot(fn, ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv,
               kv_lens, page_indices, cu_q_lens, distribution)
    return {"fn": exe, "ql_nope": ql_nope, "q_pe": q_pe, "new_kv_c": new_kv_c,
            "new_k_pe": new_k_pe, "cache_kv": cache_kv, "kv_lens": kv_lens,
            "page_indices": page_indices, "cu_q_lens": cu_q_lens,
            "distribution": distribution}


def _kernel_attention_mla(ctx: dict) -> None:
    out, _cache = ctx["fn"](
        ctx["ql_nope"], ctx["q_pe"], ctx["new_kv_c"], ctx["new_k_pe"],
        ctx["cache_kv"], ctx["kv_lens"], ctx["page_indices"],
        ctx["cu_q_lens"], ctx["distribution"])
    ctx["out"] = out


IMPLS = [
    BackendImpl(op_type="gemm",          prepare=_prepare_gemm,          kernel=_kernel_gemm),
    BackendImpl(op_type="moe_forward",   prepare=_prepare_moe_forward,   kernel=_kernel_moe),
    BackendImpl(op_type="moe_gemm",      prepare=_prepare_moe_gemm,      kernel=_kernel_moe),
    BackendImpl(op_type="attention_mha", prepare=_prepare_attention_mha, kernel=_kernel_attention_mha),
    BackendImpl(op_type="attention_mla", prepare=_prepare_attention_mla, kernel=_kernel_attention_mla),
]
