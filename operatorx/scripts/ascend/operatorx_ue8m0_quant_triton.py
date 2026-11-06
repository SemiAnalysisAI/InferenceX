import torch
import triton
import triton.language as tl

EPSILON = 1e-10


@triton.jit
def _quant(
    X,
    Q,
    S,
    M: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    P: tl.constexpr,
    BR: tl.constexpr,
    EPS: tl.constexpr,
    WIDE: tl.constexpr,
):
    g = tl.program_id(0) * BR + tl.arange(0, BR)
    j = tl.arange(0, 128)
    if WIDE:
        offsets = g[:, None].to(tl.int64) * 128 + j[None, :]
    else:
        offsets = g[:, None] * 128 + j[None, :]
    mask = g[:, None] < M * G
    x = tl.load(X + offsets, mask, 0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 1), EPS)
    raw_bits = (amax * (1.0 / 448.0)).to(tl.uint32, bitcast=True)
    exponent_code = (raw_bits >> 23) + ((raw_bits & 0x7FFFFF) != 0).to(tl.uint32)
    inverse = ((254 - exponent_code) << 23).to(tl.float32, bitcast=True)
    q = tl.minimum(tl.maximum(x * inverse[:, None], -448.0), 448.0)
    q = q.to(tl.float8e4nv, fp_downcast_rounding="rtne")
    tl.store(Q + offsets, q, mask)
    exponent = exponent_code.to(tl.uint8)
    if G % 2:
        row = g // G
        col = g % G
        tl.store(S + g + row, exponent, g < M * G)
        tl.store(S + row * P + G, 0, (g < M * G) & (col == G - 1))
    else:
        tl.store(S + g, exponent, g < M * G)


def requires_wide_indices(m, k):
    if type(m) is not int or type(k) is not int or m <= 0 or k <= 0:
        raise ValueError("Positive integer dimensions required")
    return m * k > 2**31


def quantize(a, *, block_rows=64, force_wide=True):
    if a.ndim != 2 or a.dtype != torch.bfloat16 or not a.is_contiguous():
        raise ValueError("Require contiguous BF16 [M,K] input")
    m, k = a.shape
    if m <= 0 or k <= 0 or k % 128:
        raise ValueError(
            "Exact public contract requires positive M and K divisible by 128"
        )
    if block_rows not in (16, 32, 64, 128):
        raise ValueError("Unvalidated tile size")
    g = k // 128
    p = (g + 1) // 2 * 2
    q = torch.empty_like(a, dtype=torch.float8_e4m3fn)
    scales = torch.empty((m, p // 2, 2), device=a.device, dtype=torch.uint8)
    _quant[(triton.cdiv(m * g, block_rows),)](
        a,
        q,
        scales,
        m,
        k,
        g,
        p,
        block_rows,
        EPSILON,
        force_wide or requires_wide_indices(m, k),
    )
    return q, scales
