# pyright: reportAttributeAccessIssue=false
"""Trainium3 Tensor Engine quad MX (MXFP8) matmul throughput kernels (NKI 0.7), one NeuronCore at LNC=1.

nisa.nc_matmul_mx (NeuronCore-v4) multiplies OCP-microscaling tensors: each operand element is a
float8_e4m3fn_x4, four fp8 values packed in 32 bits along the contraction, so every PE row contracts
4 values per cycle and one instruction contracts K = 128 partitions x 4 = 512. Each operand comes with
an e8m0 scale per 32-value group (8 partitions x 4 packed values), applied inside the array. Shapes:
stationary [128, M] x4, moving [128, N] x4, scales uint8 of the same shapes (nisa.quantize_mx's layout,
as nkilib's mlp_tkg_quad_fp8_mx uses), dst [M, N] fp32 PSUM.

  mm_mx_fixed  a burst of NUM_MM nc_matmul_mx, every matmul reusing one stationary (and its scale) and
               one moving tile, all resident in SBUF. Matmul i overwrites (accumulate=False) PSUM tile
               i % T. A tile takes ceil(N / 512) banks of the 8, so T = 8 up to N = 512, 1 at 4096.

  ls_mx_alt    the same, but matmul i uses stationary i % 2 (two tiles), so every matmul needs its
               own LoadStationary (LDWEIGHTS).

The kernel input is a [128, 4 (M + N + W)] fp32 random tile (W = WARMUP_N with a warm-up, else 0). It
is cast to bf16 and quantized to MXFP8 on VectorE (nisa.quantize_mx) before the burst: the first 4M
columns become the stationary, the next 4N the moving tile, the last 4W the warm-up moving tile. After
the burst each PSUM tile is copied to SBUF and summed (not counted) so no matmul is dead, and the sum is
stored to HBM. NUM_MM, M_DIM and WARMUP_MM are module globals set by get_kernel.

Warm-up (WARMUP_MM > 0). TensorE starts every burst under its activity throttle (4/8 of cycles) and is
released to 8/8 only when the array streams moving data on more than ~22% of cycles (see
../trn3_double_fp8_matmul). The warm-up issues WARMUP_MM fixed-stationary matmuls with WARMUP_N = 512
moving columns right before the burst, in the same no_reorder block, into the first 512 columns of the
PSUM tiles (allocated [M, max(N, 512)], still one bank each at N <= 512); the burst writes columns
0..N-1. ../trn3_matmul/parse.py drops the first WARMUP_MM MATMULs (meta.json warmup_mm).
"""

import nki
import nki.isa as nisa
import nki.language as nl

NUM_MM = 1024
WARMUP_MM = 0  # fixed-stationary N=WARMUP_N matmuls before the burst; set by get_kernel
WARMUP_N = 512
M_DIM = 128  # stationary free size M (output partitions); set by get_kernel
PSUM_BANKS = 8  # per partition, 2 KiB each
Q = 4  # fp8 values packed per float8_e4m3fn_x4 element (contraction)
MX = nl.float8_e4m3fn_x4
SCALE = nl.uint8


def psum_tiles(N):
    """PSUM tiles to rotate through: each [M, N] fp32 tile takes ceil(4 N / 2 KiB) banks (NeuronCore-v4
    lets one matmul span banks, N <= 4096 for fp32), so 8 tiles up to N = 512, 1 tile at N = 4096."""
    return max(1, PSUM_BANKS // -(-4 * N // 2048))


def _quantize(raw, start, cols):
    """Quantize raw[:, Q*start : Q*(start+cols)] (fp32) to an MXFP8 tile [P, cols] and its scales."""
    P = raw.shape[0]
    src = nl.ndarray([P, Q * cols], dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=src, src=raw[:, Q * start:Q * (start + cols)], engine=nisa.engine.vector)
    data = nl.ndarray([P, cols], dtype=MX, buffer=nl.sbuf)
    scale = nl.ndarray([P, cols], dtype=SCALE, buffer=nl.sbuf)
    nisa.quantize_mx(dst=data, src=src, dst_scale=scale)
    return data, scale


def _burst(in_tensor, n_stat):
    P = in_tensor.shape[0]
    M = M_DIM
    W = WARMUP_N if WARMUP_MM else 0
    N = in_tensor.shape[1] // Q - M - W
    out_tensor = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.shared_hbm)

    raw = nl.ndarray([P, Q * (M + N + W)], dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=raw, src=in_tensor)
    stats = []
    for s in range(n_stat):
        stats.append(_quantize(raw, 0, M))
    mov, mov_s = _quantize(raw, M, N)
    if W:
        wmov, wmov_s = _quantize(raw, M + N, W)

    width = max(N, W)
    n_ps = psum_tiles(width)
    ps = []
    for t in range(n_ps):
        ps.append(nl.ndarray([M, width], dtype=nl.float32, buffer=nl.psum))

    with nl.no_reorder():
        for i in nl.static_range(WARMUP_MM):
            nisa.nc_matmul_mx(dst=ps[i % n_ps][:, 0:W], stationary=stats[0][0], moving=wmov,
                              stationary_scale=stats[0][1], moving_scale=wmov_s, accumulate=False)
        for i in nl.static_range(NUM_MM):
            st, st_s = stats[i % n_stat]
            nisa.nc_matmul_mx(dst=ps[i % n_ps][:, 0:N], stationary=st, moving=mov, stationary_scale=st_s,
                              moving_scale=mov_s, accumulate=False)

    acc = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.sbuf)
    tmp = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    for t in range(n_ps):
        nisa.tensor_copy(dst=tmp, src=ps[t][:, 0:N], engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)
    nl.store(out_tensor, value=acc)
    return out_tensor


@nki.jit
def mm_mx_fixed_kernel(in_tensor):
    return _burst(in_tensor, 1)


@nki.jit
def ls_mx_alt_kernel(in_tensor):
    return _burst(in_tensor, 2)


KERNELS = {"mm_mx_fixed": mm_mx_fixed_kernel, "ls_mx_alt": ls_mx_alt_kernel}


def get_kernel(name, num_mm, m=128, warmup=0):
    global NUM_MM, M_DIM, WARMUP_MM
    NUM_MM = num_mm
    M_DIM = m
    WARMUP_MM = warmup
    return KERNELS[name]
