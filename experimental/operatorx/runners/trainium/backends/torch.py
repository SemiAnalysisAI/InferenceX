from __future__ import annotations

import torch
import torch.nn.functional as F
import torch_neuronx  # noqa: F401  registers the neuron backend
import torch_xla.core.xla_model as xm

from operatorx.core import BackendImpl, Op, UnsupportedOpError, lookup_versions


def versions() -> dict[str, str]:
    return lookup_versions("torch", "torch_neuronx", "torch_xla")

_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _resolve(dtype: str) -> torch.dtype:
    if dtype not in _DTYPES or _DTYPES[dtype] is None:
        raise UnsupportedOpError(f"trainium torch doesn't support dtype={dtype!r}")
    return _DTYPES[dtype]


def _device():
    return xm.xla_device()


def _prepare_gemm(op: Op) -> dict:
    a = op.args
    if a["dtype_b"] != a["dtype_a"]:
        raise UnsupportedOpError("trainium torch gemm requires dtype_a == dtype_b")
    dt = _resolve(a["dtype_a"])
    A = torch.randn(a["m"], a["k"], dtype=dt, device=_device())
    B = torch.randn(a["k"], a["n"], dtype=dt, device=_device())
    return {"A": A, "B": B}


def _kernel_gemm(ctx: dict) -> None:
    ctx["out"] = torch.matmul(ctx["A"], ctx["B"])
    xm.mark_step()


def _prepare_attention_mha(op: Op) -> dict:
    a = op.args
    if a.get("kv_layout", "contig") != "contig":
        raise NotImplementedError("trainium torch attention_mha here only handles contiguous KV")

    B, S_q, S_kv = a["batch_size"], a["seq_len_q"], a["seq_len_kv"]
    H, H_kv, D = a["num_heads"], a["num_heads_kv"], a["head_dim"]
    dt = _resolve(a["dtype_q"])
    causal = a.get("causal", True) and S_q == S_kv

    q = torch.randn(B, H, S_q, D, dtype=dt, device=_device())
    k = torch.randn(B, H_kv, S_kv, D, dtype=dt, device=_device())
    v = torch.randn(B, H_kv, S_kv, D, dtype=dt, device=_device())

    if H_kv != H:
        rep = H // H_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    return {"q": q, "k": k, "v": v, "causal": causal}


def _kernel_attention_mha(ctx: dict) -> None:
    ctx["out"] = F.scaled_dot_product_attention(
        ctx["q"], ctx["k"], ctx["v"], is_causal=ctx["causal"]
    )
    xm.mark_step()






def _prepare_allreduce(op: Op) -> dict:
    a = op.args
    dt = _resolve(a["dtype"])
    t = torch.randn(a["num_elements"], dtype=dt, device=_device())
    return {"t": t}


def _kernel_allreduce(ctx: dict) -> None:
    xm.all_reduce(xm.REDUCE_SUM, [ctx["t"]])
    ctx["out"] = ctx["t"]  # for runner._sync readback
    xm.mark_step()


def _prepare_allgather(op: Op) -> dict:
    a = op.args
    dt = _resolve(a["dtype"])
    t = torch.randn(a["num_elements_per_rank"], dtype=dt, device=_device())
    return {"t": t}


def _kernel_allgather(ctx: dict) -> None:
    ctx["out"] = xm.all_gather(ctx["t"], dim=0)
    xm.mark_step()


def _prepare_reduce_scatter(op: Op) -> dict:
    a = op.args
    dt = _resolve(a["dtype"])
    ws = a["world_size"]
    if a["num_elements"] % ws != 0:
        raise UnsupportedOpError(
            f"reduce_scatter requires num_elements divisible by world_size ({a['num_elements']} % {ws} != 0)"
        )
    t = torch.randn(a["num_elements"], dtype=dt, device=_device())
    return {"t": t, "shard_count": ws}


def _kernel_reduce_scatter(ctx: dict) -> None:
    ctx["out"] = xm.reduce_scatter(
        xm.REDUCE_SUM, ctx["t"], scale=1.0, scatter_dim=0, shard_count=ctx["shard_count"],
    )
    xm.mark_step()


def _prepare_alltoall(op: Op) -> dict:
    a = op.args
    dt = _resolve(a["dtype"])
    ws = a["world_size"]
    n_per = a["num_elements_per_rank"]
    if n_per % ws != 0:
        raise UnsupportedOpError(
            f"alltoall requires num_elements_per_rank divisible by world_size ({n_per} % {ws} != 0)"
        )
    t = torch.randn(n_per, dtype=dt, device=_device())
    return {"t": t, "split_count": ws}


def _kernel_alltoall(ctx: dict) -> None:
    ctx["out"] = xm.all_to_all(
        ctx["t"], split_dimension=0, concat_dimension=0, split_count=ctx["split_count"],
    )
    xm.mark_step()


# Token routing — reference baseline modeled as a flat all_to_all.
def _moe_routed_buffer(op: Op) -> torch.Tensor:
    a = op.args
    dt = _resolve(a["dtype"])
    nt, k, h = a["num_tokens"], a["top_k"], a["hidden"]
    return torch.randn(nt * k * h, dtype=dt, device=_device())


def _prepare_dispatch(op: Op) -> dict:
    return {"t": _moe_routed_buffer(op), "split_count": op.args["world_size"]}


def _kernel_dispatch(ctx: dict) -> None:
    ctx["out"] = xm.all_to_all(
        ctx["t"], split_dimension=0, concat_dimension=0, split_count=ctx["split_count"],
    )
    xm.mark_step()


def _prepare_combine(op: Op) -> dict:
    return _prepare_dispatch(op)


def _kernel_combine(ctx: dict) -> None:
    _kernel_dispatch(ctx)


# MoE forward (compute-only reference): top_k SwiGLU passes with expert-0 weights.
def _prepare_moe_forward(op: Op) -> dict:
    a = op.args
    dt_act = _resolve(a["dtype_act"])
    dt_w = _resolve(a["dtype_weight"])
    nt, h, im = a["num_tokens"], a["hidden"], a["intermediate"]
    x = torch.randn(nt, h, dtype=dt_act, device=_device())
    gate = torch.randn(h, im, dtype=dt_w, device=_device())
    up = torch.randn(h, im, dtype=dt_w, device=_device())
    down = torch.randn(im, h, dtype=dt_w, device=_device())
    return {"x": x, "gate": gate, "up": up, "down": down, "k": a["top_k"]}


def _kernel_moe_forward(ctx: dict) -> None:
    x = ctx["x"]
    out = torch.zeros_like(x)
    for _ in range(ctx["k"]):
        gate = x @ ctx["gate"]
        up = x @ ctx["up"]
        mid = F.silu(gate) * up
        out = out + mid @ ctx["down"]
    ctx["out"] = out
    xm.mark_step()


IMPLS = [
    BackendImpl(op_type="gemm", prepare=_prepare_gemm, kernel=_kernel_gemm),
    BackendImpl(op_type="attention_mha", prepare=_prepare_attention_mha, kernel=_kernel_attention_mha),
    BackendImpl(op_type="allreduce", prepare=_prepare_allreduce, kernel=_kernel_allreduce),
    BackendImpl(op_type="allgather", prepare=_prepare_allgather, kernel=_kernel_allgather),
    BackendImpl(op_type="reduce_scatter", prepare=_prepare_reduce_scatter, kernel=_kernel_reduce_scatter),
    BackendImpl(op_type="alltoall", prepare=_prepare_alltoall, kernel=_kernel_alltoall),
    BackendImpl(op_type="dispatch", prepare=_prepare_dispatch, kernel=_kernel_dispatch),
    BackendImpl(op_type="combine", prepare=_prepare_combine, kernel=_kernel_combine),
    BackendImpl(op_type="moe_forward", prepare=_prepare_moe_forward, kernel=_kernel_moe_forward),
]
