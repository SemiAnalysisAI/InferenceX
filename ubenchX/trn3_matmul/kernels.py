# pyright: reportAttributeAccessIssue=false
"""Trainium3 Tensor Engine matmul throughput kernels (NKI 0.7), one NeuronCore at LNC=1.

  mm_bf16_fixed   a burst of NUM_MM nisa.nc_matmul, bf16 x bf16 -> fp32 PSUM, every matmul reusing one
                  [K=128, M] stationary tile and one [K=128, N] moving tile, both resident in SBUF.
                  Matmul i overwrites (accumulate=False) PSUM tile i % T. A tile takes ceil(N / 512)
                  banks of the 8, so T = 8 up to N = 512, 4 at 1024, 2 at 2048, 1 at 4096.

  ls_bf16_alt     the same, but matmul i uses stationary i % 2 (two tiles), so every matmul needs its
                  own LoadStationary (LDWEIGHTS); with a small N the load dominates.

The kernel input is a [128, M + N] fp32 random tile: columns 0..M-1 become the stationary, the
rest the moving tile (cast to bf16 on VectorE before the burst), so the array sees non-constant data.
After the burst, each PSUM tile is copied to SBUF and summed (not counted) so no matmul is dead,
and the sum is stored to HBM. NUM_MM, M_DIM and WARMUP_MM are module globals set by get_kernel.

Warm-up (WARMUP_MM > 0, same scheme as ../trn3_double_fp8_matmul/kernels.py). TensorE starts under
its activity throttle (4/8 of cycles) and is released to 8/8 only once the array streams moving
data on more than ~22% of cycles, which ls_bf16_alt at N = 8 never does. The warm-up issues
WARMUP_MM fixed-stationary matmuls with WARMUP_N = 512 moving columns from 512 extra input columns,
right before the burst in the same no_reorder block, into the first 512 columns of the PSUM tiles
(allocated [M, max(N, 512)], still one bank each at N <= 512); the burst writes columns 0..N-1.
parse.py drops the first WARMUP_MM MATMULs (meta.json warmup_mm).
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


def _burst(in_tensor, n_stat):
    """n_stat bf16 stationaries (1: fixed, 2: alternating), one moving tile, NUM_MM matmuls."""
    K = in_tensor.shape[0]
    M = M_DIM
    W = WARMUP_N if WARMUP_MM else 0
    N = in_tensor.shape[1] - M - W
    out_tensor = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.shared_hbm)

    raw = nl.ndarray([K, M + N + W], dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=raw, src=in_tensor)
    stats = []
    for s in range(n_stat):
        st = nl.ndarray([K, M], dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=st, src=raw[:, 0:M], engine=nisa.engine.vector)
        stats.append(st)
    mov = nl.ndarray([K, N], dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=mov, src=raw[:, M:M + N], engine=nisa.engine.vector)
    if W:
        wmov = nl.ndarray([K, W], dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=wmov, src=raw[:, M + N:M + N + W], engine=nisa.engine.vector)

    width = max(N, W)
    n_ps = psum_tiles(width)
    ps = []
    for t in range(n_ps):
        ps.append(nl.ndarray([M, width], dtype=nl.float32, buffer=nl.psum))

    with nl.no_reorder():
        for i in nl.static_range(WARMUP_MM):
            nisa.nc_matmul(dst=ps[i % n_ps][:, 0:W], stationary=stats[0], moving=wmov, accumulate=False)
        for i in nl.static_range(NUM_MM):
            nisa.nc_matmul(dst=ps[i % n_ps][:, 0:N], stationary=stats[i % n_stat], moving=mov, accumulate=False)

    acc = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.sbuf)
    tmp = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    for t in range(n_ps):
        nisa.tensor_copy(dst=tmp, src=ps[t][:, 0:N], engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)
    nl.store(out_tensor, value=acc)
    return out_tensor


@nki.jit
def mm_bf16_fixed_kernel(in_tensor):
    return _burst(in_tensor, 1)


@nki.jit
def ls_bf16_alt_kernel(in_tensor):
    """LoadStationary timing: matmul i uses stationary i % 2, so every matmul needs its own
    LDWEIGHTS. Use a small N so the multiply is short and the load dominates."""
    return _burst(in_tensor, 2)


KERNELS = {"mm_bf16_fixed": mm_bf16_fixed_kernel, "ls_bf16_alt": ls_bf16_alt_kernel}


def get_kernel(name, num_mm, m=128, warmup=0):
    global NUM_MM, M_DIM, WARMUP_MM
    NUM_MM = num_mm
    M_DIM = m
    WARMUP_MM = warmup
    return KERNELS[name]
