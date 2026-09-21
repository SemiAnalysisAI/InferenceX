"""ROCm adapter for the nightly's stock block-FP8 UE8M0 linear path.

Reuse the existing Triton quantizer and GEMM. The CUDA JIT quantizer selected
by the nightly's generic wrapper cannot compile on HIP. Quantization keeps
the same group size, absmax floor, E4M3 range and upward power-of-two scale.
"""

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    _per_token_group_quant_8bit_colmajor,
    w8a8_block_fp8_matmul_triton,
)


def quantize_ue8m0(x: torch.Tensor, group_size: int):
    assert x.ndim == 2 and x.shape[1] % group_size == 0
    x = x.contiguous()
    rows, width = x.shape
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (width // group_size, rows), device=x.device, dtype=torch.float32
    ).t()
    if rows:
        _per_token_group_quant_8bit_colmajor[(rows * width // group_size,)](
            x,
            q,
            scales,
            group_size,
            width,
            scales.stride(1),
            1e-10,
            bit8_min=-448.0,
            bit8_max=448.0,
            BLOCK=group_size,
            num_warps=1,
            num_stages=1,
            SCALE_UE8M0=True,
        )
    return q, scales


def rocm_v41_block_fp8_linear(
    input, weight, block_size, weight_scale, input_scale=None, bias=None
):
    assert block_size == [32, 32]
    shape = [*input.shape[:-1], weight.shape[0]]
    flat = input.reshape(-1, input.shape[-1])
    if input_scale is None:
        q, scale = quantize_ue8m0(flat, block_size[1])
        dtype = input.dtype
    else:
        q, scale, dtype = flat, input_scale, torch.bfloat16
    output = w8a8_block_fp8_matmul_triton(
        q, weight, scale, weight_scale, block_size, output_dtype=dtype
    )
    if bias is not None:
        output += bias
    return output.to(dtype=dtype).view(*shape)
