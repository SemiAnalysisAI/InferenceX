# pyright: reportAttributeAccessIssue=false
"""Trainium3 Tensor Engine fp8 double_row matmul throughput kernels (NKI 0.7), one NeuronCore at LNC=1.

nisa.nc_matmul with perf_mode=double_row packs two fp8 contraction rows into each of the 128 PE rows,
so one instruction contracts K = 128 partitions x 2 = 256. Both operands carry the pair as the first
free dimension: stationary [128, 2, M], moving [128, 2, N], dst [M, N] fp32 PSUM. The array sums
over every (p, j); the Trainium2 arch guide (which covers Trn3) assigns slot j to half j of K, i.e.
[p, j, :] is contraction row k = 128 j + p. Any assignment gives the same result as long as both
operands use it, and the benchmark only times, so the data here are not laid out by k.

neuronx-cc's BIR verifier (not the docs) also requires, for each double_row input access pattern:
the pair dim has count 2 and a step that is a multiple of 16 elements, and the last dim has step 1
and an even count. With [128, 2, N] tiles the pair step is N (M for the stationary), so N and M must
be multiples of 16.

  mm_fp8dr_fixed  a burst of NUM_MM double_row matmuls, every matmul reusing one [128, 2, M] stationary
                  and one [128, 2, N] moving tile, both resident in SBUF. Matmul i overwrites
                  (accumulate=False) PSUM tile i % T. A tile takes ceil(N / 512) banks of the 8,
                  so T = 8 up to N = 512, 4 at 1024, 2 at 2048, 1 at 4096.

  ls_fp8dr_alt    the same, but matmul i uses stationary i % 2 (two tiles), so every matmul needs its
                  own LoadStationary (LDWEIGHTS).

The kernel input is a [128, 2, M + N (+ WARMUP_N)] fp32 random tile: columns 0..M-1 become the
stationary, the next N the moving tile, and the last WARMUP_N (only with a warm-up) the warm-up moving
tile, all cast to fp8 on VectorE before the burst, so the array sees non-constant data. After the
burst, each PSUM tile is copied to SBUF and summed (not counted) so no matmul is dead, and the sum is
stored to HBM. NUM_MM, M_DIM and WARMUP_MM are module globals set by get_kernel.

Warm-up (WARMUP_MM > 0). TensorE starts every burst under its activity throttle (4/8 of cycles) and
is released to 8/8 only when it is busy enough: the burst above releases when, under the throttle,
the array streams moving columns on more than ~22% of cycles, and never releases below that (fp8
alternating at N <= 64). The warm-up issues WARMUP_MM fixed-stationary matmuls with WARMUP_N = 512
moving columns (~100% busy) right before the burst, in the same no_reorder block, so the throttle is
released before burst matmul 0. The warm-up writes the first 512 columns of the PSUM tiles, which are
then allocated [M, max(N, 512)] (still one bank each at N <= 512, so the rotation is unchanged); the
burst writes columns 0..N-1. parse.py drops the first WARMUP_MM MATMULs (meta.json warmup_mm).
"""

import nki
import nki.isa as nisa
import nki.language as nl

NUM_MM = 1024
WARMUP_MM = 0  # fixed-stationary N=WARMUP_N matmuls before the burst; set by get_kernel
WARMUP_N = 512
M_DIM = 128  # stationary free size M (output partitions); set by get_kernel
PSUM_BANKS = 8  # per partition, 2 KiB each


def psum_tiles(N):
    """PSUM tiles to rotate through: each [M, N] fp32 tile takes ceil(4 N / 2 KiB) banks (NeuronCore-v4
    lets one matmul span banks, N <= 4096 for fp32), so 8 tiles up to N = 512, 1 tile at N = 4096."""
    return max(1, PSUM_BANKS // -(-4 * N // 2048))
FP8 = nl.float8_e4m3
DOUBLE_ROW = nisa.matmul_perf_mode.double_row


def _setup(in_tensor, n_stat):
    """DMA the fp32 input to SBUF and cast n_stat stationary tiles, the moving tile and (with a warm-up)
    the warm-up moving tile to fp8."""
    P, two = in_tensor.shape[0], in_tensor.shape[1]
    M = M_DIM
    W = WARMUP_N if WARMUP_MM else 0
    N = in_tensor.shape[2] - M - W
    raw = nl.ndarray([P, two, M + N + W], dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=raw, src=in_tensor)
    stats = []
    for s in range(n_stat):
        st = nl.ndarray([P, two, M], dtype=FP8, buffer=nl.sbuf)
        nisa.tensor_copy(dst=st, src=raw[:, :, 0:M], engine=nisa.engine.vector)
        stats.append(st)
    mov = nl.ndarray([P, two, N], dtype=FP8, buffer=nl.sbuf)
    nisa.tensor_copy(dst=mov, src=raw[:, :, M:M + N], engine=nisa.engine.vector)
    wmov = None
    if W:
        wmov = nl.ndarray([P, two, W], dtype=FP8, buffer=nl.sbuf)
        nisa.tensor_copy(dst=wmov, src=raw[:, :, M + N:M + N + W], engine=nisa.engine.vector)
    return stats, mov, wmov, M, N


def _drain(ps, M, N):
    """Copy every PSUM tile to SBUF, sum them and store the sum, so no matmul is dead."""
    out_tensor = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.shared_hbm)
    acc = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.sbuf)
    tmp = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    for t in range(len(ps)):
        nisa.tensor_copy(dst=tmp, src=ps[t][:, 0:N], engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)
    nl.store(out_tensor, value=acc)
    return out_tensor


def _burst(in_tensor, n_stat):
    stats, mov, wmov, M, N = _setup(in_tensor, n_stat)
    width = max(N, WARMUP_N) if WARMUP_MM else N
    n_ps = psum_tiles(width)
    ps = []
    for t in range(n_ps):
        ps.append(nl.ndarray([M, width], dtype=nl.float32, buffer=nl.psum))
    with nl.no_reorder():
        for i in nl.static_range(WARMUP_MM):
            nisa.nc_matmul(dst=ps[i % n_ps][:, 0:WARMUP_N], stationary=stats[0], moving=wmov, accumulate=False,
                           perf_mode=DOUBLE_ROW)
        for i in nl.static_range(NUM_MM):
            nisa.nc_matmul(dst=ps[i % n_ps][:, 0:N], stationary=stats[i % n_stat], moving=mov, accumulate=False,
                           perf_mode=DOUBLE_ROW)
    return _drain(ps, M, N)


@nki.jit
def mm_fp8dr_fixed_kernel(in_tensor):
    return _burst(in_tensor, 1)


@nki.jit
def ls_fp8dr_alt_kernel(in_tensor):
    return _burst(in_tensor, 2)


KERNELS = {"mm_fp8dr_fixed": mm_fp8dr_fixed_kernel, "ls_fp8dr_alt": ls_fp8dr_alt_kernel}


def get_kernel(name, num_mm, m=128, warmup=0):
    global NUM_MM, M_DIM, WARMUP_MM
    NUM_MM = num_mm
    M_DIM = m
    WARMUP_MM = warmup
    return KERNELS[name]
