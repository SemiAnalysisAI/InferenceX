from __future__ import annotations

import deep_ep
import torch
import torch.distributed as dist

from operatorx.core import BackendImpl, Op, lookup_versions


def versions() -> dict[str, str]:
    return lookup_versions("deep_ep", "torch")

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
_BUFFERS: dict[tuple[int, str], deep_ep.Buffer] = {}


def _ensure_dist() -> None:
    import os

    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))


def _get_buffer(
    world_size: int, mode: str, num_experts: int, hidden: int,
    max_tpr: int | None = None,
) -> deep_ep.Buffer:
    key = (world_size, mode, max_tpr) if mode == "low_latency" else (world_size, mode)
    if key in _BUFFERS:
        return _BUFFERS[key]
    deep_ep.Buffer.set_num_sms(24)
    needs_rdma = world_size > 8
    if mode == "low_latency":
        rdma_hint = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            max_tpr, hidden, world_size, num_experts
        )
        buf = deep_ep.Buffer(
            dist.group.WORLD,
            num_nvl_bytes=int(2e9),
            num_rdma_bytes=rdma_hint,
            low_latency_mode=True,
            num_qps_per_rank=num_experts // world_size,
        )
    else:
        buf = deep_ep.Buffer(
            dist.group.WORLD,
            num_nvl_bytes=int(2e9),
            num_rdma_bytes=int(2e9) if needs_rdma else 0,
            low_latency_mode=False,
        )
    _BUFFERS[key] = buf
    return buf


def _make_inputs(op: Op):
    from operatorx.core import UnsupportedOpError
    a = op.args
    if a["dtype"] not in _DTYPES:
        raise UnsupportedOpError(f"deepep doesn't support dtype={a['dtype']!r}")
    nt, ne, k, h = a["num_tokens"], a["num_experts"], a["top_k"], a["hidden"]
    dt = _DTYPES[a["dtype"]]
    # adjust experts to be divisible by world_size
    ws = a["world_size"]
    if ne % ws != 0:
        ne = ((ne + ws - 1) // ws) * ws
    x = torch.randn(nt, h, dtype=dt, device="cuda")
    topk_idx = torch.randint(0, ne, (nt, k), device="cuda", dtype=torch.int64)
    topk_weights = torch.randn(nt, k, device="cuda", dtype=torch.float32).softmax(dim=-1)
    return x, topk_idx, topk_weights, ne, h


def _build_dispatch_call_normal(op: Op):
    a = op.args
    ws = a["world_size"]
    x, topk_idx, topk_weights, ne, h = _make_inputs(op)
    buf = _get_buffer(ws, "normal", ne, h)
    ntpr, ntprr, ntpe, itir, _ = buf.get_dispatch_layout(topk_idx, ne)
    torch.cuda.synchronize()
    cfg = deep_ep.Buffer.get_dispatch_config(ws)
    pinned = [x, topk_idx, topk_weights, ntpr, ntprr, ntpe, itir]

    def _call(_pin=pinned):
        return buf.dispatch(
            x, topk_idx=topk_idx, topk_weights=topk_weights,
            num_tokens_per_rank=ntpr, num_tokens_per_rdma_rank=ntprr,
            is_token_in_rank=itir, num_tokens_per_expert=ntpe,
            config=cfg, async_finish=False,
        )
    return _call


def _build_dispatch_call_ll(op: Op):
    a = op.args
    ws = a["world_size"]
    x, topk_idx, _, ne, h = _make_inputs(op)
    max_tpr = max(1, (a["num_tokens"] * a["top_k"] + ws - 1) // ws * 2)
    buf = _get_buffer(ws, "low_latency", ne, h, max_tpr=max_tpr)
    pinned = [x, topk_idx]

    def _call(_pin=pinned):
        return buf.low_latency_dispatch(
            x, topk_idx, max_tpr, ne, use_fp8=True, async_finish=False,
        )
    return _call


def _pick_fastest(builders: dict, label: str):
    """Build each candidate, time 3+5 iters, return the closure of the winner.
    Builders that throw at construction or call are silently dropped."""
    candidates = {}
    for name, build in builders.items():
        try:
            c = build()
            for _ in range(3):
                c()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(5):
                c()
            end.record()
            torch.cuda.synchronize()
            candidates[name] = (c, start.elapsed_time(end) / 5.0)
        except Exception:
            continue
    if not candidates:
        from operatorx.core import UnsupportedOpError
        raise UnsupportedOpError(f"deepep {label}: no variant supports this shape")
    winner = min(candidates, key=lambda k: candidates[k][1])
    return candidates[winner][0]


def _try_ll() -> bool:
    # DeepEP's low_latency path requires NVSHMEM + IBGDA. On clusters where
    # IBGDA DCT setup fails, nvshmem_init aborts the process — we can't
    # catch it from Python, so we gate the LL probe behind an opt-in env var.
    import os
    return os.environ.get("OPERATORX_DEEPEP_TRY_LL", "0") == "1"


def _prepare_dispatch(op: Op) -> dict:
    _ensure_dist()
    builders = {"normal": lambda: _build_dispatch_call_normal(op)}
    if _try_ll():
        builders["low_latency"] = lambda: _build_dispatch_call_ll(op)
    call = _pick_fastest(builders, "dispatch")
    return {"_call": call}


def _kernel_dispatch(ctx: dict) -> None:
    ctx["_call"]()


def _build_combine_call_normal(op: Op):
    a = op.args
    ws = a["world_size"]
    x, topk_idx, topk_weights, ne, h = _make_inputs(op)
    buf = _get_buffer(ws, "normal", ne, h)
    ntpr, ntprr, ntpe, itir, _ = buf.get_dispatch_layout(topk_idx, ne)
    torch.cuda.synchronize()
    d_cfg = deep_ep.Buffer.get_dispatch_config(ws)
    c_cfg = deep_ep.Buffer.get_combine_config(ws)
    recv_x, _, recv_tw, _, handle, _ = buf.dispatch(
        x, topk_idx=topk_idx, topk_weights=topk_weights,
        num_tokens_per_rank=ntpr, num_tokens_per_rdma_rank=ntprr,
        is_token_in_rank=itir, num_tokens_per_expert=ntpe,
        config=d_cfg, async_finish=False,
    )
    torch.cuda.synchronize()
    expert_out = torch.randn_like(recv_x)
    pinned = [x, topk_idx, topk_weights, recv_x, recv_tw, expert_out, handle]

    def _call(_pin=pinned):
        return buf.combine(
            x=expert_out, handle=handle, topk_weights=recv_tw,
            config=c_cfg, async_finish=False,
        )
    return _call


def _build_combine_call_ll(op: Op):
    a = op.args
    ws = a["world_size"]
    x, topk_idx, topk_weights, ne, h = _make_inputs(op)
    max_tpr = max(1, (a["num_tokens"] * a["top_k"] + ws - 1) // ws * 2)
    buf = _get_buffer(ws, "low_latency", ne, h, max_tpr=max_tpr)
    packed, _expert_count, handle, _, _ = buf.low_latency_dispatch(
        x, topk_idx, max_tpr, ne, use_fp8=True, async_finish=False,
    )
    torch.cuda.synchronize()
    recv_tensor = packed[0] if isinstance(packed, tuple) else packed
    expert_out = torch.randn(*recv_tensor.shape, dtype=torch.bfloat16, device="cuda").contiguous()
    pinned = [x, topk_idx, topk_weights, expert_out, handle]

    def _call(_pin=pinned):
        return buf.low_latency_combine(
            x=expert_out, topk_idx=topk_idx, topk_weights=topk_weights,
            handle=handle, async_finish=False,
        )
    return _call


def _prepare_combine(op: Op) -> dict:
    _ensure_dist()
    builders = {"normal": lambda: _build_combine_call_normal(op)}
    if _try_ll():
        builders["low_latency"] = lambda: _build_combine_call_ll(op)
    call = _pick_fastest(builders, "combine")
    return {"_call": call}


def _kernel_combine(ctx: dict) -> None:
    ctx["_call"]()


def _prepare_moe_forward(op: Op) -> dict:
    _ensure_dist()
    a = op.args
    nt, h, im = a["num_tokens"], a["hidden"], a["intermediate"]
    ne, k = a["num_experts"], a["top_k"]
    ws = a["world_size"]
    from operatorx.core import UnsupportedOpError
    if a["dtype_act"] not in _DTYPES:
        raise UnsupportedOpError(f"deepep doesn't support dtype_act={a['dtype_act']!r}")
    if a["dtype_weight"] not in _DTYPES:
        raise UnsupportedOpError(f"deepep doesn't support dtype_weight={a['dtype_weight']!r}")
    dt_act = _DTYPES[a["dtype_act"]]
    dt_w = _DTYPES[a["dtype_weight"]]

    if ne % ws != 0:
        ne = ((ne + ws - 1) // ws) * ws
    num_local_experts = ne // ws

    x = torch.randn(nt, h, dtype=dt_act, device="cuda")
    gate_w = torch.randn(h, ne, dtype=dt_w, device="cuda")
    gate_proj_w = torch.randn(num_local_experts, h, im, dtype=dt_w, device="cuda")
    up_w = torch.randn(num_local_experts, h, im, dtype=dt_w, device="cuda")
    down_w = torch.randn(num_local_experts, im, h, dtype=dt_w, device="cuda")

    buf = _get_buffer(ws, "normal", ne, h)
    cfg_d = deep_ep.Buffer.get_dispatch_config(ws)
    cfg_c = deep_ep.Buffer.get_combine_config(ws)

    return {
        "x": x, "gate_w": gate_w,
        "gate_proj_w": gate_proj_w, "up_w": up_w, "down_w": down_w,
        "buffer": buf, "k": k, "ne": ne,
        "cfg_d": cfg_d, "cfg_c": cfg_c,
    }


def _kernel_moe_forward(ctx: dict) -> None:
    x = ctx["x"]
    logits = x @ ctx["gate_w"]
    weights, idx = torch.topk(torch.softmax(logits.float(), dim=-1), ctx["k"])
    # DeepEP combine requires fp32 topk_weights; keep weights as float32.

    ntpr, ntprr, ntpe, itir, _ = ctx["buffer"].get_dispatch_layout(idx, ctx["ne"])
    recv_x, _, recv_w, num_recv_per_expert, handle, _ = ctx["buffer"].dispatch(
        x, topk_idx=idx, topk_weights=weights,
        num_tokens_per_rank=ntpr, num_tokens_per_rdma_rank=ntprr,
        is_token_in_rank=itir, num_tokens_per_expert=ntpe,
        config=ctx["cfg_d"], async_finish=False,
    )

    counts = num_recv_per_expert.tolist() if isinstance(num_recv_per_expert, torch.Tensor) else list(num_recv_per_expert)
    expert_out = torch.empty_like(recv_x)
    offset = 0
    for e, n in enumerate(counts):
        if n == 0:
            continue
        x_e = recv_x[offset:offset + n]
        gate_h = x_e @ ctx["gate_proj_w"][e]
        up_h = x_e @ ctx["up_w"][e]
        h_act = torch.nn.functional.silu(gate_h) * up_h
        expert_out[offset:offset + n] = h_act @ ctx["down_w"][e]
        offset += n

    ctx["buffer"].combine(
        x=expert_out, handle=handle,
        topk_weights=recv_w, config=ctx["cfg_c"],
        async_finish=False,
    )


IMPLS = [
    BackendImpl(op_type="dispatch", prepare=_prepare_dispatch, kernel=_kernel_dispatch),
    BackendImpl(op_type="combine", prepare=_prepare_combine, kernel=_kernel_combine),
    BackendImpl(op_type="moe_forward", prepare=_prepare_moe_forward, kernel=_kernel_moe_forward),
]
