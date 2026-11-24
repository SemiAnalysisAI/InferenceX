# pyright: reportAttributeAccessIssue=false
"""Trainium3 Tensor Engine sustained matmul kernels (NKI 0.7), one NeuronCore at LNC=1.

Each kernel is the fixed-stationary burst of the single-core benchmarks inside an on-device loop
(nl.dynamic_range), so one execution streams matmuls for as long as REPS asks, without returning to
the host:

  mm_bf16     nisa.nc_matmul, bf16 x bf16 (../trn3_matmul mm_bf16_fixed): stationary [128, M],
              moving [128, N], K = 128.
  mm_fp8dr    nisa.nc_matmul, perf_mode=double_row, float8_e4m3 (../trn3_double_fp8_matmul
              mm_fp8dr_fixed): stationary [128, 2, M], moving [128, 2, N], K = 256.
  mm_mx       nisa.nc_matmul_mx, float8_e4m3fn_x4 with uint8 e8m0 scales (../trn3_quad_mx_matmul
              mm_mx_fixed): stationary [128, M] x4, moving [128, N] x4, K = 512.
  mm_mxfp4    nisa.nc_matmul_mx, float4_e2m1fn_x4 (four 4-bit values in 16 bits) with uint8 e8m0
              scales: the same tile shapes as mm_mx, K = 512.

One loop iteration is NUM_MM matmuls that all reuse one stationary and one moving tile in SBUF; matmul
i overwrites (accumulate=False) fp32 PSUM tile i % T, T = 8 up to N = 512. neuronx-cc requires every
tensor that is live across a dynamic loop's boundary to be in HBM (NCC_IBIR092 otherwise), so each
iteration also loads the fp32 input from HBM, casts it on VectorE (quantizes it for mm_mx) before its
matmuls, and afterwards copies every PSUM tile to SBUF, sums them and stores the sum, which keeps the
matmuls live. That takes about 20 us per iteration, against 1.8 ms for 8,192 matmuls at N = 512.
NUM_MM, REPS and M_DIM are module globals set by get_kernel.

nisa.quantize_mx only produces the fp8 x4 types, so mm_mxfp4 takes operands that make_input quantized
on the host: a uint16 input whose first M + N columns are the packed fp4 data and whose last
(M + N) / 2 columns hold the M + N uint8 e8m0 scales of each partition. Each iteration loads the data
and scale tiles of both operands straight from HBM with four dma_copy, with no VectorE step.
"""

import neuron_dtypes
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

NUM_MM = 1024  # matmuls per loop iteration; set by get_kernel
REPS = 1  # on-device loop iterations; set by get_kernel
M_DIM = 128  # stationary free size M (output partitions); set by get_kernel
PSUM_BANKS = 8  # per partition, 2 KiB each
Q = 4  # fp8 values packed per float8_e4m3fn_x4 element
FP8 = nl.float8_e4m3
MX = nl.float8_e4m3fn_x4
MXFP4 = nl.float4_e2m1fn_x4
FP4_EMAX = 2  # largest unbiased exponent of float4_e2m1fn
MX_GROUP_PARTITIONS = 8  # one e8m0 scale covers 8 partitions x one x4 element
MX_QUADRANT_PARTITIONS = 32
DOUBLE_ROW = nisa.matmul_perf_mode.double_row
K_PER_PARTITION = {"mm_bf16": 1, "mm_fp8dr": 2, "mm_mx": 4, "mm_mxfp4": 4}


def psum_tiles(N):
    """PSUM tiles to rotate through: each [M, N] fp32 tile takes ceil(4 N / 2 KiB) banks of the 8."""
    return max(1, PSUM_BANKS // -(-4 * N // 2048))


def quantize_mxfp4(x):
    """MXFP4-quantize fp32 [128, 4 * cols] on the host as nisa.quantize_mx does for fp8: per group of 8
    partitions x 4 values, scale 2^(floor(log2(max |x|)) - (emax - 1)). Returns the packed data and the
    e8m0 scales, uint16 and uint8 [128, cols]; the four scales of each 32-partition quadrant sit in its
    first four partitions and the other partitions are zero, the layout nc_matmul_mx reads."""
    P, cols = x.shape[0], x.shape[1] // Q
    groups = np.abs(x).reshape(P // MX_GROUP_PARTITIONS, MX_GROUP_PARTITIONS, cols, Q).max(axis=(1, 3))
    exponent = np.floor(np.log2(groups)).astype(np.int64) - (FP4_EMAX - 1)
    divisor = np.repeat(np.repeat(np.exp2(exponent), MX_GROUP_PARTITIONS, axis=0), Q, axis=1)
    data = neuron_dtypes.static_cast_fp32_to_float4_e2m1fn_x4((x / divisor).astype(np.float32)).view(np.uint16)
    scale = np.zeros((P, cols), dtype=np.uint8)
    per_quadrant = MX_QUADRANT_PARTITIONS // MX_GROUP_PARTITIONS
    for q in range(P // MX_QUADRANT_PARTITIONS):
        rows = slice(q * MX_QUADRANT_PARTITIONS, q * MX_QUADRANT_PARTITIONS + per_quadrant)
        scale[rows] = exponent[q * per_quadrant:(q + 1) * per_quadrant] + 127
    return data, scale


def make_input(name, m, n):
    """The kernel input: fp32 random values whose first M columns feed the stationary and the next N
    the moving tile, or for mm_mxfp4 those values quantized, followed by their scale bytes."""
    rng = np.random.default_rng(0)
    if name == "mm_bf16":
        return rng.standard_normal((128, m + n)).astype(np.float32)
    if name == "mm_fp8dr":
        return rng.standard_normal((128, 2, m + n)).astype(np.float32)
    x = rng.standard_normal((128, Q * (m + n))).astype(np.float32)
    if name == "mm_mx":
        return x
    data, scale = quantize_mxfp4(x)
    return np.concatenate([data, scale.view(np.uint16)], axis=1)


def _psum(M, N):
    ps = []
    for t in range(psum_tiles(N)):
        ps.append(nl.ndarray([M, N], dtype=nl.float32, buffer=nl.psum))
    return ps


def _drain(ps, out_tensor, M, N):
    acc = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.sbuf)
    tmp = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    for t in range(len(ps)):
        nisa.tensor_copy(dst=tmp, src=ps[t], engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)
    nl.store(out_tensor, value=acc)


@nki.jit
def mm_bf16_kernel(in_tensor):
    P = in_tensor.shape[0]
    M = M_DIM
    N = in_tensor.shape[1] - M
    out_tensor = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.shared_hbm)
    for _ in nl.dynamic_range(REPS):
        raw = nl.ndarray([P, M + N], dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=raw, src=in_tensor)
        st = nl.ndarray([P, M], dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=st, src=raw[:, 0:M], engine=nisa.engine.vector)
        mov = nl.ndarray([P, N], dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mov, src=raw[:, M:M + N], engine=nisa.engine.vector)
        ps = _psum(M, N)
        n_ps = len(ps)
        with nl.no_reorder():
            for i in nl.static_range(NUM_MM):
                nisa.nc_matmul(dst=ps[i % n_ps], stationary=st, moving=mov, accumulate=False)
        _drain(ps, out_tensor, M, N)
    return out_tensor


@nki.jit
def mm_fp8dr_kernel(in_tensor):
    P, two = in_tensor.shape[0], in_tensor.shape[1]
    M = M_DIM
    N = in_tensor.shape[2] - M
    out_tensor = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.shared_hbm)
    for _ in nl.dynamic_range(REPS):
        raw = nl.ndarray([P, two, M + N], dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=raw, src=in_tensor)
        st = nl.ndarray([P, two, M], dtype=FP8, buffer=nl.sbuf)
        nisa.tensor_copy(dst=st, src=raw[:, :, 0:M], engine=nisa.engine.vector)
        mov = nl.ndarray([P, two, N], dtype=FP8, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mov, src=raw[:, :, M:M + N], engine=nisa.engine.vector)
        ps = _psum(M, N)
        n_ps = len(ps)
        with nl.no_reorder():
            for i in nl.static_range(NUM_MM):
                nisa.nc_matmul(dst=ps[i % n_ps], stationary=st, moving=mov, accumulate=False, perf_mode=DOUBLE_ROW)
        _drain(ps, out_tensor, M, N)
    return out_tensor


def _quantize(raw, start, cols):
    """Quantize raw[:, Q*start : Q*(start+cols)] (fp32) to an MXFP8 tile [P, cols] and its scales."""
    P = raw.shape[0]
    src = nl.ndarray([P, Q * cols], dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=src, src=raw[:, Q * start:Q * (start + cols)], engine=nisa.engine.vector)
    data = nl.ndarray([P, cols], dtype=MX, buffer=nl.sbuf)
    scale = nl.ndarray([P, cols], dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(dst=data, src=src, dst_scale=scale)
    return data, scale


@nki.jit
def mm_mx_kernel(in_tensor):
    P = in_tensor.shape[0]
    M = M_DIM
    N = in_tensor.shape[1] // Q - M
    out_tensor = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.shared_hbm)
    for _ in nl.dynamic_range(REPS):
        raw = nl.ndarray([P, Q * (M + N)], dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=raw, src=in_tensor)
        st, st_s = _quantize(raw, 0, M)
        mov, mov_s = _quantize(raw, M, N)
        ps = _psum(M, N)
        n_ps = len(ps)
        with nl.no_reorder():
            for i in nl.static_range(NUM_MM):
                nisa.nc_matmul_mx(dst=ps[i % n_ps], stationary=st, moving=mov, stationary_scale=st_s,
                                  moving_scale=mov_s, accumulate=False)
        _drain(ps, out_tensor, M, N)
    return out_tensor


@nki.jit
def mm_mxfp4_kernel(in_tensor):
    P = in_tensor.shape[0]
    M = M_DIM
    N = (in_tensor.shape[1] * 2) // 3 - M
    D = M + N
    out_tensor = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.shared_hbm)
    for _ in nl.dynamic_range(REPS):
        st = nl.ndarray([P, M], dtype=MXFP4, buffer=nl.sbuf)
        nisa.dma_copy(dst=st, src=in_tensor[:, 0:M].view(MXFP4))
        mov = nl.ndarray([P, N], dtype=MXFP4, buffer=nl.sbuf)
        nisa.dma_copy(dst=mov, src=in_tensor[:, M:D].view(MXFP4))
        st_s = nl.ndarray([P, M], dtype=nl.uint8, buffer=nl.sbuf)
        nisa.dma_copy(dst=st_s, src=in_tensor[:, D:D + M // 2].view(nl.uint8))
        mov_s = nl.ndarray([P, N], dtype=nl.uint8, buffer=nl.sbuf)
        nisa.dma_copy(dst=mov_s, src=in_tensor[:, D + M // 2:D + D // 2].view(nl.uint8))
        ps = _psum(M, N)
        n_ps = len(ps)
        with nl.no_reorder():
            for i in nl.static_range(NUM_MM):
                nisa.nc_matmul_mx(dst=ps[i % n_ps], stationary=st, moving=mov, stationary_scale=st_s,
                                  moving_scale=mov_s, accumulate=False)
        _drain(ps, out_tensor, M, N)
    return out_tensor


KERNELS = {"mm_bf16": mm_bf16_kernel, "mm_fp8dr": mm_fp8dr_kernel, "mm_mx": mm_mx_kernel, "mm_mxfp4": mm_mxfp4_kernel}


def get_kernel(name, num_mm, reps, m=128):
    global NUM_MM, REPS, M_DIM
    NUM_MM = num_mm
    REPS = reps
    M_DIM = m
    return KERNELS[name]
