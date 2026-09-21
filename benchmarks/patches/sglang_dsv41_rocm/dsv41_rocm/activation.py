"""V4.1 shared-expert activation using the nightly AITER masked kernel."""

import torch
import triton
from aiter.ops.triton._triton_kernels.fusions.fused_clamp_act_mul import (
    _fused_clamp_silu_mul_kernel,
)


def rocm_v41_silu_and_mul_clamp(inp, out, limit):
    rows, twice_width = inp.shape
    width = twice_width // 2
    assert out.shape == (rows, width)
    if rows:
        _fused_clamp_silu_mul_kernel[(rows,)](
            inp,
            out,
            inp,
            inp,
            rows,
            width,
            inp.stride(0),
            inp.stride(1),
            out.stride(0),
            out.stride(1),
            0,
            0,
            0,
            0,
            limit,
            BLOCK_SIZE_N=triton.next_power_of_2(width),
            QUANT_BLOCK_SIZE=32,
            SCALE_FMT="fp32",
            DTYPE_MAX=448.0,
            DTYPE_MIN=-448.0,
            HAVE_WEIGHTS=False,
            WEIGHT_BROADCAST=False,
            HAVE_SWIGLU_CLAMP=True,
            HAS_QUANT=False,
            ACTIVATION="silu",
            SHUFFLE=False,
            SCALE_N_PAD=0,
            num_warps=4,
        )
