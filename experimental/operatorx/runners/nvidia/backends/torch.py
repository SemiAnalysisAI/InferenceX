from __future__ import annotations

import torch
import torch.nn.functional as F

from operatorx.core import BackendImpl, Op, UnsupportedOpError, lookup_versions


def versions() -> dict[str, str]:
    return lookup_versions("torch")

_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    # FP8 dtypes exist in torch but most kernels (torch.matmul, SDPA, NCCL collectives)
    # don't accept them directly. We let them through here for tensor construction; ops
    # that can't actually run on fp8 will fail at the kernel and be recorded as errors.
    "fp8":  getattr(torch, "float8_e4m3fn", None),
}


def _resolve(dtype: str) -> torch.dtype:
    if dtype not in _DTYPES or _DTYPES[dtype] is None:
        raise UnsupportedOpError(f"torch backend doesn't support dtype={dtype!r}")
    return _DTYPES[dtype]


def _prepare_gemm(op: Op) -> dict:
    a = op.args
    if a.get("activation") is not None:
        raise UnsupportedOpError(
            f"torch gemm has no native activation fusion; got activation={a['activation']!r}"
        )
    if a["dtype_b"] != a["dtype_a"]:
        raise UnsupportedOpError(
            f"torch gemm requires dtype_a == dtype_b; got {a['dtype_a']}/{a['dtype_b']}"
        )
    dt = _resolve(a["dtype_a"])
    M, N, K = a["m"], a["n"], a["k"]
    has_bias = bool(a.get("bias"))
    out_dt = _DTYPES.get(a.get("dtype_out") or "bf16") or torch.bfloat16

    if dt == _DTYPES.get("fp8"):
        # B200 fp8 path: torch._scaled_mm. Expects A:(M,K) row-major, B:(K,N) column-major,
        # so build B as (N,K) and transpose. Scales = 1.0 (no quantization simulated here).
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda").to(dt)
        B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda").to(dt).t()
        scale_a = torch.tensor(1.0, device="cuda")
        scale_b = torch.tensor(1.0, device="cuda")
        bias = torch.randn(N, dtype=out_dt, device="cuda") if has_bias else None
        return {"A": A, "B": B, "scale_a": scale_a, "scale_b": scale_b,
                "out_dtype": out_dt, "bias": bias, "fp8": True}

    A = torch.randn(M, K, dtype=dt, device="cuda")
    B = torch.randn(K, N, dtype=dt, device="cuda")
    bias = torch.randn(N, dtype=dt, device="cuda") if has_bias else None
    return {"A": A, "B": B, "bias": bias, "fp8": False}


def _kernel_gemm(ctx: dict) -> None:
    if ctx["fp8"]:
        # torch._scaled_mm accepts bias natively as a kwarg.
        ctx["C"] = torch._scaled_mm(
            ctx["A"], ctx["B"], ctx["scale_a"], ctx["scale_b"],
            out_dtype=ctx["out_dtype"], bias=ctx["bias"],
        )
    elif ctx["bias"] is not None:
        # torch.addmm(bias, A, B) → bias[None,:] + A @ B, native fused bias.
        ctx["C"] = torch.addmm(ctx["bias"], ctx["A"], ctx["B"])
    else:
        ctx["C"] = torch.matmul(ctx["A"], ctx["B"])


def _prepare_attention_mha(op: Op) -> dict:
    a = op.args
    if a.get("kv_layout", "contig") != "contig":
        raise UnsupportedOpError("torch attention_mha here only handles contiguous KV")
    # F.scaled_dot_product_attention doesn't accept fp8 — and we use one dtype for Q/K/V.
    for k in ("dtype_q", "dtype_k", "dtype_v"):
        if a[k] == "fp8":
            raise UnsupportedOpError(
                f"torch SDPA doesn't support fp8 ({k}={a[k]}); use flashinfer for fp8 attention"
            )
    if not (a["dtype_q"] == a["dtype_k"] == a["dtype_v"]):
        raise UnsupportedOpError(
            f"torch SDPA needs uniform Q/K/V dtype "
            f"(got {a['dtype_q']}/{a['dtype_k']}/{a['dtype_v']})"
        )

    B, S_q, S_kv = a["batch_size"], a["seq_len_q"], a["seq_len_kv"]
    H, H_kv, D = a["num_heads"], a["num_heads_kv"], a["head_dim"]
    dt = _resolve(a["dtype_q"])
    causal = a.get("causal", True) and S_q == S_kv

    q = torch.randn(B, H, S_q, D, dtype=dt, device="cuda")
    k = torch.randn(B, H_kv, S_kv, D, dtype=dt, device="cuda")
    v = torch.randn(B, H_kv, S_kv, D, dtype=dt, device="cuda")

    if H_kv != H:
        rep = H // H_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    return {"q": q, "k": k, "v": v, "causal": causal}


def _kernel_attention_mha(ctx: dict) -> None:
    ctx["out"] = F.scaled_dot_product_attention(
        ctx["q"], ctx["k"], ctx["v"], is_causal=ctx["causal"]
    )


def _prepare_attention_mla(op: Op) -> dict:
    """Inner-kernel MLA: pre-materialise (Q, K, V) outside the timed region,
    then time only F.scaled_dot_product_attention. The c_kv -> K_nope / V
    up-projection is a plain gemm and is benchmarked separately via the
    ``gemm`` op."""
    a = op.args
    for k in ("dtype_q", "dtype_kv"):
        if a[k] == "fp8":
            raise UnsupportedOpError(
                f"torch SDPA doesn't support fp8 ({k}={a[k]})"
            )
    if a["dtype_q"] != a["dtype_kv"]:
        raise UnsupportedOpError(
            f"torch SDPA needs uniform Q/KV dtype (got {a['dtype_q']} vs {a['dtype_kv']})"
        )
    B, S_q, S_kv = a["batch_size"], a["seq_len_q"], a["seq_len_kv"]
    H = a["num_heads"]
    qk_dim = a["head_dim_qk_nope"] + a["head_dim_qk_rope"]
    D_v = a["head_dim_v"]
    dt = _resolve(a["dtype_q"])

    q = torch.randn(B, H, S_q, qk_dim, dtype=dt, device="cuda")
    k = torch.randn(B, H, S_kv, qk_dim, dtype=dt, device="cuda")
    v = torch.randn(B, H, S_kv, D_v, dtype=dt, device="cuda")
    causal = a.get("causal", True) and S_q == S_kv
    return {"q": q, "k": k, "v": v, "causal": causal}


def _kernel_attention_mla(ctx: dict) -> None:
    ctx["out"] = F.scaled_dot_product_attention(
        ctx["q"], ctx["k"], ctx["v"], is_causal=ctx["causal"]
    )






def _ensure_dist() -> None:
    import os

    import torch.distributed as dist
    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))


def _reject_fp8_collective(op: Op, name: str) -> None:
    if op.args["dtype"] == "fp8":
        raise UnsupportedOpError(
            f"torch.distributed/{name} (NCCL) doesn't support fp8 reductions"
        )


def _prepare_allreduce(op: Op) -> dict:
    _reject_fp8_collective(op, "allreduce")
    _ensure_dist()
    dt = _resolve(op.args["dtype"])
    t = torch.randn(op.args["num_elements"], dtype=dt, device="cuda")
    return {"t": t}


def _kernel_allreduce(ctx: dict) -> None:
    import torch.distributed as dist
    dist.all_reduce(ctx["t"])


def _prepare_allgather(op: Op) -> dict:
    _reject_fp8_collective(op, "allgather")
    _ensure_dist()
    dt = _resolve(op.args["dtype"])
    n_per = op.args["num_elements_per_rank"]
    ws = op.args["world_size"]
    in_t = torch.randn(n_per, dtype=dt, device="cuda")
    out_t = torch.empty(n_per * ws, dtype=dt, device="cuda")
    return {"in": in_t, "out": out_t}


def _kernel_allgather(ctx: dict) -> None:
    import torch.distributed as dist
    dist.all_gather_into_tensor(ctx["out"], ctx["in"])


def _prepare_reduce_scatter(op: Op) -> dict:
    _reject_fp8_collective(op, "reduce_scatter")
    _ensure_dist()
    dt = _resolve(op.args["dtype"])
    n = op.args["num_elements"]
    ws = op.args["world_size"]
    if n % ws != 0:
        raise UnsupportedOpError(
            f"reduce_scatter requires num_elements divisible by world_size ({n} % {ws} != 0)"
        )
    in_t = torch.randn(n, dtype=dt, device="cuda")
    out_t = torch.empty(n // ws, dtype=dt, device="cuda")
    return {"in": in_t, "out": out_t}


def _kernel_reduce_scatter(ctx: dict) -> None:
    import torch.distributed as dist
    dist.reduce_scatter_tensor(ctx["out"], ctx["in"])


def _prepare_alltoall(op: Op) -> dict:
    # NCCL all_to_all_single technically accepts fp8 byte buffers (no reduction), but
    # torch's interface still rejects the dtype on most builds. Mark unsupported.
    _reject_fp8_collective(op, "alltoall")
    _ensure_dist()
    dt = _resolve(op.args["dtype"])
    n_per = op.args["num_elements_per_rank"]
    ws = op.args["world_size"]
    if n_per % ws != 0:
        raise UnsupportedOpError(
            f"alltoall requires num_elements_per_rank divisible by world_size ({n_per} % {ws} != 0)"
        )
    in_t = torch.randn(n_per, dtype=dt, device="cuda")
    out_t = torch.empty(n_per, dtype=dt, device="cuda")
    return {"in": in_t, "out": out_t}


def _kernel_alltoall(ctx: dict) -> None:
    import torch.distributed as dist
    dist.all_to_all_single(ctx["out"], ctx["in"])


# --- MoE collectives (dispatch/combine) ---
# Reference torch baseline: token routing modeled as plain all_to_all_single.
# Each rank's hidden-state batch is shuffled across the world; volume per rank is
# num_tokens * top_k * hidden / world_size. Matches the bus-bytes formula.

def _moe_routed_buffer(op: Op) -> torch.Tensor:
    _ensure_dist()
    a = op.args
    dt = _resolve(a["dtype"])
    nt, k, h = a["num_tokens"], a["top_k"], a["hidden"]
    ws = a["world_size"]
    # Total per-rank send/recv volume = nt * k * h elements (split evenly across ws).
    return torch.randn(nt * k * h, dtype=dt, device="cuda")


def _prepare_dispatch(op: Op) -> dict:
    buf_in = _moe_routed_buffer(op)
    buf_out = torch.empty_like(buf_in)
    return {"in": buf_in, "out": buf_out}


def _kernel_dispatch(ctx: dict) -> None:
    import torch.distributed as dist
    dist.all_to_all_single(ctx["out"], ctx["in"])


def _prepare_combine(op: Op) -> dict:
    return _prepare_dispatch(op)  # symmetric


def _kernel_combine(ctx: dict) -> None:
    _kernel_dispatch(ctx)


# moe_forward is owned by the sglang and flashinfer backends (production MoE
# kernels). The torch backend doesn't expose its own MoE module.


IMPLS = [
    BackendImpl(op_type="gemm", prepare=_prepare_gemm, kernel=_kernel_gemm),
    BackendImpl(op_type="attention_mha", prepare=_prepare_attention_mha, kernel=_kernel_attention_mha),
    BackendImpl(op_type="attention_mla", prepare=_prepare_attention_mla, kernel=_kernel_attention_mla),
    BackendImpl(op_type="allreduce", prepare=_prepare_allreduce, kernel=_kernel_allreduce),
    BackendImpl(op_type="allgather", prepare=_prepare_allgather, kernel=_kernel_allgather),
    BackendImpl(op_type="reduce_scatter", prepare=_prepare_reduce_scatter, kernel=_kernel_reduce_scatter),
    BackendImpl(op_type="alltoall", prepare=_prepare_alltoall, kernel=_kernel_alltoall),
    BackendImpl(op_type="dispatch", prepare=_prepare_dispatch, kernel=_kernel_dispatch),
    BackendImpl(op_type="combine", prepare=_prepare_combine, kernel=_kernel_combine),
]
