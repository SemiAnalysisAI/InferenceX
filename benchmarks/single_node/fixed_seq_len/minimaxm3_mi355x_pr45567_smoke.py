from __future__ import annotations

import torch

from vllm.model_executor.layers.fused_moe.experts.mxfp8_native_moe import (
    fused_moe_mxfp8_native,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _mxfp8_e4m3_quantize_torch,
    dequant_mxfp8_to_bf16,
)


def reference_moe(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    alpha: float,
    beta: float,
    limit: float,
) -> torch.Tensor:
    tokens, hidden = x.shape
    intermediate = w2.shape[-1]
    top_k = topk_ids.shape[1]
    output = torch.zeros(tokens, hidden, device=x.device, dtype=torch.float32)

    for token in range(tokens):
        for route in range(top_k):
            expert = int(topk_ids[token, route].item())
            gate_up = x[token].float() @ w13[expert].float().T
            gate = gate_up[:intermediate].clamp(max=limit)
            up = gate_up[intermediate:].clamp(min=-limit, max=limit)
            activation = gate * torch.sigmoid(alpha * gate) * (up + beta)
            expert_output = activation @ w2[expert].float().T
            output[token] += topk_weights[token, route].float() * expert_output

    return output.to(x.dtype)


@torch.inference_mode()
def main() -> None:
    arch = torch.cuda.get_device_properties(0).gcnArchName
    if "gfx95" not in arch:
        raise RuntimeError(f"Expected a gfx95x GPU, found {arch}")

    torch.manual_seed(0)
    tokens, hidden, intermediate, experts, top_k = 1, 512, 256, 16, 4
    alpha, beta, limit = 1.702, 1.0, 7.0

    w13_bf16 = (
        torch.randn(
            experts,
            2 * intermediate,
            hidden,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    w2_bf16 = (
        torch.randn(
            experts,
            hidden,
            intermediate,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    w13_fp8, w13_scale = _mxfp8_e4m3_quantize_torch(
        w13_bf16, is_sf_swizzled_layout=False
    )
    w2_fp8, w2_scale = _mxfp8_e4m3_quantize_torch(
        w2_bf16, is_sf_swizzled_layout=False
    )
    w13_dequant = dequant_mxfp8_to_bf16(w13_fp8, w13_scale)
    w2_dequant = dequant_mxfp8_to_bf16(w2_fp8, w2_scale)

    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.5
    logits = torch.randn(tokens, experts, device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = logits.softmax(dim=-1).topk(top_k, dim=-1)

    actual = fused_moe_mxfp8_native(
        x,
        w13_fp8,
        w13_scale,
        w2_fp8,
        w2_scale,
        topk_weights.to(torch.float32),
        topk_ids.to(torch.int32),
        alpha=alpha,
        beta=beta,
        limit=limit,
        global_num_experts=experts,
        expert_map=None,
    )
    expected = reference_moe(
        x,
        w13_dequant,
        w2_dequant,
        topk_weights,
        topk_ids,
        alpha,
        beta,
        limit,
    )
    relative_error = (
        (actual.float() - expected.float()).norm()
        / (expected.float().norm() + 1e-8)
    ).item()
    print(
        "MI355X PR45567 MXFP8 numerical smoke: "
        f"arch={arch} relative_error={relative_error:.6f}"
    )
    if relative_error >= 5e-2:
        raise AssertionError(
            f"MXFP8 MoE relative error {relative_error:.6f} exceeds 0.05"
        )


if __name__ == "__main__":
    main()
