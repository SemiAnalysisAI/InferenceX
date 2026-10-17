"""Single-GPU routed expert kernels matching vLLM's generic MoE benchmark shapes.

Routing is prepared before timing. EP/TP describe a local kernel shape, not a
live distributed group. No router, shared experts or communication is timed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping

import torch

from operatorx.core import BackendImpl, Op, UnsupportedOpError, lookup_versions


def versions() -> dict[str, str]:
    return lookup_versions("torch", "vllm", "triton")


def local_dimensions(args: Mapping[str, Any]) -> tuple[int, int]:
    """Validate a routed-only request and derive the per-rank weight dimensions."""
    for key in ("num_tokens", "hidden", "intermediate", "num_experts", "top_k"):
        if type(args[key]) is not int or args[key] <= 0:
            raise UnsupportedOpError(f"MoE requires positive integer {key}")
    ep = args.get("expert_parallel_size", 1)
    tp = args.get("routed_tensor_parallel_size", 1)
    if type(ep) is not int or ep <= 0 or type(tp) is not int or tp <= 0:
        raise UnsupportedOpError("MoE shard factors must be positive integers")
    if args["num_experts"] % ep or args["intermediate"] % tp:
        raise UnsupportedOpError("MoE dimensions must divide evenly into EP/TP shards")
    experts, intermediate = args["num_experts"] // ep, args["intermediate"] // tp
    if args["top_k"] > experts:
        raise UnsupportedOpError("MoE top_k exceeds the local expert count")
    if args.get("n_shared_experts", 0) != 0:
        raise UnsupportedOpError("vllm moe_gemm measures routed experts only")
    if args.get("expert_distribution", "uniform") != "uniform":
        raise UnsupportedOpError("vllm moe_gemm supports uniform-random routing only")
    if (
        args["dtype_act"] not in {"bf16", "fp16"}
        or args["dtype_weight"] != args["dtype_act"]
    ):
        raise UnsupportedOpError(
            "vllm moe_gemm requires matching BF16/FP16 activations and weights"
        )
    return experts, intermediate


def prepare(op: Op) -> dict:
    a = op.args
    experts, intermediate = local_dimensions(a)
    fused_experts = import_module(
        "vllm.model_executor.layers.fused_moe.fused_moe"
    ).fused_experts
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[a["dtype_act"]]
    tokens, hidden, top_k = a["num_tokens"], a["hidden"], a["top_k"]
    x = torch.empty(tokens, hidden, device="cuda", dtype=dtype).normal_(std=0.1)
    w1 = torch.empty(
        experts, 2 * intermediate, hidden, device="cuda", dtype=dtype
    ).normal_(std=0.02)
    w2 = torch.empty(experts, hidden, intermediate, device="cuda", dtype=dtype).normal_(
        std=0.02
    )
    logits = torch.randn(tokens, experts, device="cuda", dtype=torch.float32)
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).contiguous()
    return {
        "x": x,
        "w1": w1,
        "w2": w2,
        "topk_weights": weights,
        "topk_ids": ids.to(torch.int32),
        "fused_experts": fused_experts,
    }


def kernel(ctx: dict) -> None:
    # vLLM's default activation is SiLU-and-multiply, as in benchmark_moe.py.
    ctx["out"] = ctx["fused_experts"](
        ctx["x"],
        ctx["w1"],
        ctx["w2"],
        ctx["topk_weights"],
        ctx["topk_ids"],
        inplace=False,
    )


IMPLS = [BackendImpl(op_type="moe_gemm", prepare=prepare, kernel=kernel)]
