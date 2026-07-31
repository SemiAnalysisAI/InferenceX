"""Independent FP32 numerical oracle for the Kimi-K3 gfx942 A16W4 SiTUv2 MoE.

Deliberately imports NOTHING from aiter. Everything (mxfp4 quant/dequant, e8m0
scales, expert GEMM, SiTUv2) is written here in plain PyTorch so a mismatch
against the kernel cannot be masked by a shared code path.

Frozen shape (do not edit):
    model_dim=3584  inter_dim=512  experts=896  topk=16
    a_dtype=bf16    w_dtype=mxfp4 (float4_e2m1fn_x2)   quant=per_1x32
    activation=SiTUv2
    M in {1, 2, 4096}

Frozen thresholds (never loosen after seeing a result):
    pure FP32 SiTU formula      : max_delta < 1e-5
    BF16 x MXFP4 kernel output  : cosine > 0.999 AND >= 95% of elements
                                  within atol=1.0, rtol=0.05

Layouts produced by ``make_case`` are the CANONICAL (unshuffled) ones:
    inp        bf16  [M, model_dim]
    w1_packed  uint8 [E, 2*inter_dim, model_dim//2]   byte j = (elem 2j+1)<<4 | elem 2j
    w1_scale   uint8 [E, 2*inter_dim, model_dim//32]  biased e8m0
    w2_packed  uint8 [E, model_dim, inter_dim//2]
    w2_scale   uint8 [E, model_dim, inter_dim//32]
    topk_ids   int32 [M, topk]
    topk_w     f32   [M, topk]
A caller feeding an aiter kernel is responsible for any shuffle
(shuffle_weight_a16w4 / e8m0_shuffle); the oracle stays canonical.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------- frozen shape

MODEL_DIM = 3584
INTER_DIM = 512
EXPERTS = 896
TOPK = 16
GROUP = 32  # per_1x32
# moonshotai/Kimi-K3 config.json is the source of truth for these parameters.
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0
# Note:(Wenyao Gao) SiTUv2 is unclamped. The +-7.0 bound belongs to swiglu only:
# in mixed_moe_gemm_2stage.py the constant lives inside _swiglu_mul_vec4 /
# _act_elem's swiglu branch, while _situ_mul_vec4 applies no clamp, and
# aiter.fused_moe.situv2 takes no limit argument. A16W4 does not even forward
# one -- moe_kernels.py passes pass_swiglu_limit=not _is_a16w4. Setting this to
# 7.0 makes the oracle disagree with a correct kernel on large activations.
SITU_LIMIT = None
M_VALUES = (1, 2, 4096)

# frozen gates
SITU_FP32_MAX_DELTA = 1e-5
KERNEL_COSINE_MIN = 0.999
KERNEL_PCT_CLOSE_MIN = 0.95
KERNEL_ATOL = 1.0
KERNEL_RTOL = 0.05

# OCP MXFP4 E2M1 code -> value (index == the 4-bit code)
_MXFP4_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
_MXFP4_MAGS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_MXFP4_MAX = 6.0
_MXFP4_TARGET_MAX_POW2 = 2  # largest power of two <= 6.0 is 4 == 2**2


# ------------------------------------------------------------ mxfp4 / e8m0


def e8m0_to_f32(biased: torch.Tensor) -> torch.Tensor:
    """Biased 8-bit exponent -> fp32 scale. 0 -> 2**-127, 0xFF -> NaN."""
    b = biased.view(torch.uint8).to(torch.int32)
    bits = b << 23
    bits = torch.where(b == 0, torch.full_like(bits, 0x00400000), bits)
    bits = torch.where(b == 0xFF, torch.full_like(bits, 0x7F800001), bits)
    return bits.view(torch.float32)


def unpack_mxfp4(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K//2] -> fp32 [..., K]; low nibble is the even element."""
    p = packed.view(torch.uint8)
    lo = (p & 0x0F).to(torch.long)
    hi = (p >> 4).to(torch.long)
    codes = torch.stack((lo, hi), dim=-1).reshape(*p.shape[:-1], p.shape[-1] * 2)
    table = torch.tensor(_MXFP4_VALUES, dtype=torch.float32, device=p.device)
    return table[codes]


def pack_mxfp4(codes: torch.Tensor) -> torch.Tensor:
    """uint8 codes [..., K] -> packed uint8 [..., K//2]."""
    c = codes.to(torch.uint8)
    lo = c[..., 0::2]
    hi = c[..., 1::2]
    return (hi << 4) | (lo & 0x0F)


def quantize_mxfp4_per_1x32(x: torch.Tensor):
    """fp32 [..., K] (K % 32 == 0) -> (packed uint8 [..., K//2], e8m0 [..., K//32]).

    Scale exponent = floor(log2(amax)) - 2 so the group max lands in [4, 8).
    Elements are then rounded to the nearest representable E2M1 magnitude
    (ties away from zero, which is what a plain nearest search gives on the
    non-uniform grid) and clamped to +-6.
    """
    xf = x.float()
    k = xf.shape[-1]
    assert k % GROUP == 0, f"last dim {k} not a multiple of {GROUP}"
    g = xf.reshape(*xf.shape[:-1], k // GROUP, GROUP)

    amax = g.abs().amax(dim=-1)
    exp = torch.floor(torch.log2(amax.clamp(min=1e-38))) - _MXFP4_TARGET_MAX_POW2
    exp = torch.where(amax == 0, torch.zeros_like(exp), exp)
    biased = (exp + 127.0).clamp(0, 254).to(torch.uint8)

    scale = e8m0_to_f32(biased).unsqueeze(-1)
    q = (g / scale).clamp(-_MXFP4_MAX, _MXFP4_MAX)

    mags = torch.tensor(_MXFP4_MAGS, dtype=torch.float32, device=xf.device)
    # midpoints between consecutive representable magnitudes
    edges = (mags[1:] + mags[:-1]) / 2.0
    idx = torch.bucketize(q.abs(), edges)  # 0..7
    codes = idx.to(torch.uint8) | (q < 0).to(torch.uint8) * 8
    codes = codes.reshape(*xf.shape)
    return pack_mxfp4(codes), biased


def dequant_mxfp4_per_1x32(packed: torch.Tensor, e8m0: torch.Tensor) -> torch.Tensor:
    """(packed [..., K//2], e8m0 [..., K//32]) -> fp32 [..., K]."""
    vals = unpack_mxfp4(packed)
    k = vals.shape[-1]
    scale = e8m0_to_f32(e8m0).unsqueeze(-1)
    return (vals.reshape(*vals.shape[:-1], k // GROUP, GROUP) * scale).reshape(
        *vals.shape[:-1], k
    )


# ------------------------------------------------------------------ SiTUv2


def situ_v2(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float = SITU_BETA,
    linear_beta: float = SITU_LINEAR_BETA,
    limit: float | None = SITU_LIMIT,
) -> torch.Tensor:
    """SiTUv2 as the A16W4 stage1 kernel computes it.

        out = (beta*tanh(gate/beta)*sigmoid(gate)) * (linear_beta*tanh(up/linear_beta))

    ``limit`` defaults to None because the kernel's situv2 path does not clamp;
    it stays a parameter only so the self-test can exercise the clamped branch.
    """
    g = gate.float()
    u = up.float()
    if limit is not None:
        g = g.clamp(max=limit)
        u = u.clamp(min=-limit, max=limit)
    situ_gate = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    up_scaled = linear_beta * torch.tanh(u / linear_beta)
    return situ_gate * up_scaled


def _situ_v2_scalar(g: float, u: float, beta: float, linear_beta: float,
                    limit: float | None) -> float:
    """Independent scalar restatement in float64, used only by the self-test."""
    if limit is not None:
        g = min(g, limit)
        u = max(min(u, limit), -limit)
    sig = 1.0 / (1.0 + math.exp(-g))
    return (beta * math.tanh(g / beta) * sig) * (linear_beta * math.tanh(u / linear_beta))


# --------------------------------------------------------------- test case


@dataclass
class Case:
    m: int
    device: str
    inp: torch.Tensor        # bf16 [M, model_dim]
    w1_packed: torch.Tensor  # uint8 [E, 2*inter, model_dim//2]
    w1_scale: torch.Tensor   # uint8 [E, 2*inter, model_dim//32]
    w2_packed: torch.Tensor  # uint8 [E, model_dim, inter//2]
    w2_scale: torch.Tensor   # uint8 [E, model_dim, inter//32]
    topk_ids: torch.Tensor   # int32 [M, topk]
    topk_w: torch.Tensor     # f32   [M, topk]

    def w1_deq(self, experts: torch.Tensor) -> torch.Tensor:
        return dequant_mxfp4_per_1x32(self.w1_packed[experts], self.w1_scale[experts])

    def w2_deq(self, experts: torch.Tensor) -> torch.Tensor:
        return dequant_mxfp4_per_1x32(self.w2_packed[experts], self.w2_scale[experts])


def make_case(m: int, device: str = "cuda", seed: int = 0,
              experts: int = EXPERTS, expert_chunk: int = 64) -> Case:
    """Seeded, deterministic. Identical bytes for a given (m, seed, experts)."""
    gen = torch.Generator(device=device).manual_seed(seed)

    inp = (
        torch.randn((m, MODEL_DIM), generator=gen, device=device,
                    dtype=torch.float32) / 10.0
    ).to(torch.bfloat16)

    def _quant_weight(rows: int, cols: int):
        packed = torch.empty((experts, rows, cols // 2), dtype=torch.uint8, device=device)
        scale = torch.empty((experts, rows, cols // GROUP), dtype=torch.uint8, device=device)
        for s in range(0, experts, expert_chunk):
            e = min(s + expert_chunk, experts)
            w = torch.randn((e - s, rows, cols), generator=gen, device=device,
                            dtype=torch.float32) / 10.0
            p, sc = quantize_mxfp4_per_1x32(w)
            packed[s:e] = p
            scale[s:e] = sc
            del w, p, sc
        return packed, scale

    w1_packed, w1_scale = _quant_weight(2 * INTER_DIM, MODEL_DIM)
    w2_packed, w2_scale = _quant_weight(MODEL_DIM, INTER_DIM)

    score = torch.randn((m, experts), generator=gen, device=device, dtype=torch.float32)
    probs = torch.softmax(score, dim=-1)
    topk_w, topk_ids = torch.topk(probs, TOPK, dim=-1)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

    return Case(
        m=m, device=device, inp=inp,
        w1_packed=w1_packed, w1_scale=w1_scale,
        w2_packed=w2_packed, w2_scale=w2_scale,
        topk_ids=topk_ids.to(torch.int32), topk_w=topk_w.float(),
    )


# ------------------------------------------------------------- fp32 reference


def ref_stage1(case: Case, beta: float = SITU_BETA,
               linear_beta: float = SITU_LINEAR_BETA,
               limit: float | None = SITU_LIMIT) -> torch.Tensor:
    """-> fp32 [M, topk, inter_dim]. Loops over the experts actually routed to."""
    out = torch.zeros((case.m, TOPK, INTER_DIM), dtype=torch.float32, device=case.device)
    hidden = case.inp.float()
    ids = case.topk_ids.long()
    for e in torch.unique(ids):
        t_idx, k_idx = (ids == e).nonzero(as_tuple=True)
        w1 = dequant_mxfp4_per_1x32(case.w1_packed[e], case.w1_scale[e])  # [2*I, D]
        gu = hidden[t_idx] @ w1.t()                                       # [n, 2*I]
        out[t_idx, k_idx] = situ_v2(gu[:, :INTER_DIM], gu[:, INTER_DIM:],
                                    beta, linear_beta, limit)
        del w1, gu
    return out


def ref_stage2(case: Case, a2: torch.Tensor, doweight: bool = True) -> torch.Tensor:
    """a2 fp32/bf16 [M, topk, inter_dim] -> fp32 [M, model_dim] (summed over topk)."""
    a2f = a2.float().reshape(case.m, TOPK, INTER_DIM)
    out = torch.zeros((case.m, TOPK, MODEL_DIM), dtype=torch.float32, device=case.device)
    ids = case.topk_ids.long()
    for e in torch.unique(ids):
        t_idx, k_idx = (ids == e).nonzero(as_tuple=True)
        w2 = dequant_mxfp4_per_1x32(case.w2_packed[e], case.w2_scale[e])  # [D, I]
        out[t_idx, k_idx] = a2f[t_idx, k_idx] @ w2.t()
        del w2
    if doweight:
        out = out * case.topk_w.view(case.m, TOPK, 1)
    return out.sum(dim=1)


def ref_fused(case: Case, **kw) -> torch.Tensor:
    return ref_stage2(case, ref_stage1(case, **kw))


# ------------------------------------------------------------------ metrics


def compare(got: torch.Tensor, ref: torch.Tensor, label: str = "",
            atol: float = KERNEL_ATOL, rtol: float = KERNEL_RTOL) -> dict:
    g = got.detach().float().reshape(-1)
    r = ref.detach().float().reshape(-1)
    assert g.shape == r.shape, f"{label}: shape {tuple(got.shape)} vs {tuple(ref.shape)}"
    nan = int(torch.isnan(g).sum())
    inf = int(torch.isinf(g).sum())
    finite = torch.isfinite(g) & torch.isfinite(r)
    gf, rf = g[finite], r[finite]
    diff = (gf - rf).abs()
    close = diff <= (atol + rtol * rf.abs())
    denom = rf.norm().item()
    cos = torch.nn.functional.cosine_similarity(gf, rf, dim=0).item() if gf.numel() else float("nan")
    return {
        "label": label,
        "n": int(g.numel()),
        "max_abs_error": float(diff.max()) if diff.numel() else float("nan"),
        "relative_L2": float((gf - rf).norm() / denom) if denom else float("nan"),
        "cosine": cos,
        "percent_close": float(close.float().mean()) if close.numel() else float("nan"),
        "nan_count": nan,
        "inf_count": inf,
    }


def verdict(metrics: dict) -> bool:
    return (
        metrics["nan_count"] == 0
        and metrics["inf_count"] == 0
        and metrics["cosine"] > KERNEL_COSINE_MIN
        and metrics["percent_close"] >= KERNEL_PCT_CLOSE_MIN
    )


def report(metrics: dict) -> str:
    return (
        "[{label}] n={n} max_abs={max_abs_error:.6g} relL2={relative_L2:.6g} "
        "cos={cosine:.9f} pct_close={percent_close:.4f} nan={nan_count} inf={inf_count} "
        "-> {v}".format(v="PASS" if verdict(metrics) else "FAIL", **metrics)
    )


# --------------------------------------------------------------- self-tests


def selftest_situ_fp32(device: str = "cpu") -> float:
    """Vectorized fp32 SiTUv2 vs an independent float64 scalar restatement.

    Gate: max_delta < 1e-5.
    """
    torch.manual_seed(1234)
    g = (torch.rand(4096, dtype=torch.float64) * 40.0 - 20.0)
    u = (torch.rand(4096, dtype=torch.float64) * 40.0 - 20.0)
    worst = 0.0
    for beta, lbeta, limit in (
        (SITU_BETA, SITU_LINEAR_BETA, SITU_LIMIT),
        (1.0, 1.0, None),
        (0.5, 2.0, 7.0),
        (1.5, 0.8, 7.0),
    ):
        # Compare both implementations at the exact same representable inputs.
        # Using the original float64 samples for the scalar path also measures
        # float64-to-float32 input rounding, which is not a formula error and is
        # large enough to make this frozen 1e-5 gate host-dependent.
        g32 = g.to(device=device, dtype=torch.float32)
        u32 = u.to(device=device, dtype=torch.float32)
        vec = situ_v2(g32, u32, beta, lbeta, limit).double().cpu()
        sca = torch.tensor(
            [_situ_v2_scalar(float(a), float(b), beta, lbeta, limit)
             for a, b in zip(
                 g32.double().cpu().tolist(), u32.double().cpu().tolist()
             )],
            dtype=torch.float64,
        )
        worst = max(worst, float((vec - sca).abs().max()))
    assert worst < SITU_FP32_MAX_DELTA, (
        f"SiTU fp32 formula self-test max_delta={worst:.3e} >= {SITU_FP32_MAX_DELTA}"
    )
    return worst


def selftest_quant_roundtrip(device: str = "cpu") -> float:
    """Codec self-consistency.

    1. pack/unpack is bit-exact for all 16 E2M1 codes.
    2. exactly-representable values survive quantize -> dequantize unchanged.
    3. dequantized output always lands on the {grid value * group scale} lattice.
    """
    # 1. pack/unpack round-trip over every code
    codes = torch.arange(16, dtype=torch.uint8, device=device).repeat(2)  # [32]
    vals = unpack_mxfp4(pack_mxfp4(codes))
    table = torch.tensor(_MXFP4_VALUES, dtype=torch.float32, device=device)
    assert torch.equal(vals, table.repeat(2)), "pack/unpack is not a round-trip"

    # 2. an exactly representable group must survive unchanged
    exact = (table * 0.25).repeat(4).reshape(1, 64)  # 2 groups of 32, scale 2**-2
    p, s = quantize_mxfp4_per_1x32(exact)
    deq_exact = dequant_mxfp4_per_1x32(p, s)
    assert torch.equal(deq_exact, exact), "exactly-representable values were altered"

    # 3. general input lands on the lattice, and the error is bounded by the
    #    largest half-gap of the grid (1.0) times the group scale
    torch.manual_seed(7)
    x = torch.randn(64, 256, device=device) / 10.0
    packed, e8 = quantize_mxfp4_per_1x32(x)
    deq = dequant_mxfp4_per_1x32(packed, e8)
    assert float(deq.abs().max()) > 0, "dequant produced all zeros"
    scale = e8m0_to_f32(e8).unsqueeze(-1).expand(-1, -1, GROUP).reshape(x.shape)
    ratio = (deq / scale)
    on_grid = (ratio.abs().unsqueeze(-1) - torch.tensor(_MXFP4_MAGS, device=device)).abs().amin(-1)
    assert float(on_grid.max()) < 1e-5, "dequant left the E2M1 lattice"
    # FLOOR scale selection puts the group max in [4, 8) scale units; anything
    # above 6 is clipped, so the worst case error is just under 2 scale units.
    err = float(((deq - x).abs() / scale).max())
    assert err < 2.0, f"quantization error {err} exceeds the FLOOR-mode bound"
    return err


if __name__ == "__main__":
    # CPU by default, and not because a GPU might be absent: both self-tests
    # check device-independent formulas, but they are seeded, and torch's CUDA
    # RNG draws different numbers than its CPU RNG for the same seed. The
    # quantization bound below is a tight analytic one (measured 1.984 against
    # 2.0), so running it on GPU turns a deterministic gate into a coin flip
    # that would block the experiment it is supposed to protect.
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    d = selftest_situ_fp32(dev)
    print(f"selftest_situ_fp32       max_delta={d:.3e} < {SITU_FP32_MAX_DELTA}  PASS")
    e = selftest_quant_roundtrip(dev)
    print(f"selftest_quant_roundtrip PASS (max err in scale units = {e:.4g} < 2.0)")
    print("oracle constants:", MODEL_DIM, INTER_DIM, EXPERTS, TOPK, M_VALUES)
