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

One loop iteration is NUM_MM matmuls that all reuse one stationary and one moving tile in SBUF; matmul
i overwrites (accumulate=False) fp32 PSUM tile i % T, T = 8 up to N = 512. neuronx-cc requires every
tensor that is live across a dynamic loop's boundary to be in HBM (NCC_IBIR092 otherwise), so each
iteration also loads the fp32 input from HBM, casts it on VectorE (quantizes it for mm_mx) before its
matmuls, and afterwards copies every PSUM tile to SBUF, sums them and stores the sum, which keeps the
matmuls live. That takes about 20 us per iteration, against 1.8 ms for 8,192 matmuls at N = 512.
NUM_MM, REPS and M_DIM are module globals set by get_kernel.
"""

import nki
import nki.isa as nisa
import nki.language as nl

NUM_MM = 1024  # matmuls per loop iteration; set by get_kernel
REPS = 1  # on-device loop iterations; set by get_kernel
M_DIM = 128  # stationary free size M (output partitions); set by get_kernel
PSUM_BANKS = 8  # per partition, 2 KiB each
Q = 4  # fp8 values packed per float8_e4m3fn_x4 element
FP8 = nl.float8_e4m3
MX = nl.float8_e4m3fn_x4
DOUBLE_ROW = nisa.matmul_perf_mode.double_row
K_PER_PARTITION = {"mm_bf16": 1, "mm_fp8dr": 2, "mm_mx": 4}


def psum_tiles(N):
    """PSUM tiles to rotate through: each [M, N] fp32 tile takes ceil(4 N / 2 KiB) banks of the 8."""
    return max(1, PSUM_BANKS // -(-4 * N // 2048))


def input_shape(name, m, n):
    """Shape of the fp32 kernel input: the first M columns feed the stationary, the next N the moving tile."""
    if name == "mm_bf16":
        return (128, m + n)
    if name == "mm_fp8dr":
        return (128, 2, m + n)
    return (128, Q * (m + n))


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


KERNELS = {"mm_bf16": mm_bf16_kernel, "mm_fp8dr": mm_fp8dr_kernel, "mm_mx": mm_mx_kernel}


def get_kernel(name, num_mm, reps, m=128):
    global NUM_MM, REPS, M_DIM
    NUM_MM = num_mm
    REPS = reps
    M_DIM = m
    return KERNELS[name]
