# pyright: reportAttributeAccessIssue=false
"""Trainium3 Tensor Engine matmul throughput kernels (NKI 0.7), one NeuronCore at LNC=1.

  mm_bf16_fixed   a burst of NUM_MM nisa.nc_matmul, bf16 x bf16 -> fp32 PSUM, every matmul reusing one
                  [K=128, M] stationary tile and one [K=128, N] moving tile, both resident in SBUF.
                  Matmul i overwrites (accumulate=False) PSUM tile i % 8, one 2 KiB bank per tile when
                  N <= 512, so no matmul depends on another.

  ls_bf16_alt     the same, but matmul i uses stationary i % 2 (two tiles), so every matmul needs its
                  own LoadStationary (LDWEIGHTS); with a small N the load dominates.

The kernel input is a [128, M + N] fp32 random tile: columns 0..M-1 become the stationary, the
rest the moving tile (cast to bf16 on VectorE before the burst), so the array sees non-constant data.
After the burst, each PSUM tile is copied to SBUF and summed (not counted) so no matmul is dead,
and the sum is stored to HBM. NUM_MM and M_DIM are module globals set by get_kernel.
"""

import nki
import nki.isa as nisa
import nki.language as nl

NUM_MM = 1024
M_DIM = 128  # stationary free size M (output partitions); set by get_kernel
N_PS = 8  # PSUM banks


@nki.jit
def mm_bf16_fixed_kernel(in_tensor):
    K = in_tensor.shape[0]
    M = M_DIM
    N = in_tensor.shape[1] - M
    out_tensor = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.shared_hbm)

    raw = nl.ndarray([K, M + N], dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=raw, src=in_tensor)
    stat = nl.ndarray([K, M], dtype=nl.bfloat16, buffer=nl.sbuf)
    mov = nl.ndarray([K, N], dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=stat, src=raw[:, 0:M], engine=nisa.engine.vector)
    nisa.tensor_copy(dst=mov, src=raw[:, M:M + N], engine=nisa.engine.vector)

    ps = []
    for t in range(N_PS):
        ps.append(nl.ndarray([M, N], dtype=nl.float32, buffer=nl.psum))

    with nl.no_reorder():
        for i in nl.static_range(NUM_MM):
            nisa.nc_matmul(dst=ps[i % N_PS], stationary=stat, moving=mov, accumulate=False)

    acc = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.sbuf)
    tmp = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    for t in range(N_PS):
        nisa.tensor_copy(dst=tmp, src=ps[t], engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)
    nl.store(out_tensor, value=acc)
    return out_tensor


@nki.jit
def ls_bf16_alt_kernel(in_tensor):
    """LoadStationary timing: matmul i uses stationary i % 2, so every matmul needs its own
    LDWEIGHTS. Use a small N so the multiply is short and the load dominates."""
    K = in_tensor.shape[0]
    M = M_DIM
    N = in_tensor.shape[1] - M
    out_tensor = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.shared_hbm)

    raw = nl.ndarray([K, M + N], dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=raw, src=in_tensor)
    stat0 = nl.ndarray([K, M], dtype=nl.bfloat16, buffer=nl.sbuf)
    stat1 = nl.ndarray([K, M], dtype=nl.bfloat16, buffer=nl.sbuf)
    mov = nl.ndarray([K, N], dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=stat0, src=raw[:, 0:M], engine=nisa.engine.vector)
    nisa.tensor_copy(dst=stat1, src=raw[:, 0:M], engine=nisa.engine.vector)
    nisa.tensor_copy(dst=mov, src=raw[:, M:M + N], engine=nisa.engine.vector)
    stats = [stat0, stat1]

    ps = []
    for t in range(N_PS):
        ps.append(nl.ndarray([M, N], dtype=nl.float32, buffer=nl.psum))

    with nl.no_reorder():
        for i in nl.static_range(NUM_MM):
            nisa.nc_matmul(dst=ps[i % N_PS], stationary=stats[i % 2], moving=mov, accumulate=False)

    acc = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.sbuf)
    tmp = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    for t in range(N_PS):
        nisa.tensor_copy(dst=tmp, src=ps[t], engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)
    nl.store(out_tensor, value=acc)
    return out_tensor


KERNELS = {"mm_bf16_fixed": mm_bf16_fixed_kernel, "ls_bf16_alt": ls_bf16_alt_kernel}


def get_kernel(name, num_mm, m=128):
    global NUM_MM, M_DIM
    NUM_MM = num_mm
    M_DIM = m
    return KERNELS[name]
