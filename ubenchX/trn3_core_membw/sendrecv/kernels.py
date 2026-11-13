# pyright: reportAttributeAccessIssue=false
"""Raw core-to-core exchange inside an LNC=2 pair (NKI 0.7): nisa.sendrecv and nisa.sendrecv_compute.

Both physical cores of the logical core run the kernel (rank = nl.program_id(0)). Each step exchanges a
[P, F] float32 SBUF tile with the peer, and the next step sends what this step received, so the NSTEP
steps form a dependent chain: one hop in each direction per step. Device time vs NSTEP; the slope is the
cost per step (the TPU port's method: slope of device time against round trips).

  xchg_dma      nisa.sendrecv, dma_engine=dma (the default; CoreBarrier synchronisation)
  xchg_gpsimd   nisa.sendrecv, dma_engine=gpsimd_dma ("low-latency SB-to-SB swaps"; <= 1 KiB/partition)
  xchg_reduce   nisa.sendrecv_compute, reduce_op=add (exchange + add in the DMA engine)

in_tensor is [2, P, F]: core k starts from in_tensor[k]. out is [2, P, F]: core k's final tile.
Expected results: xchg_* after N steps: core k holds in[k] if N even else in[1-k]; xchg_reduce: both
cores hold (in[0] + in[1]) * 2^(N-1).
"""

import nki
import nki.isa as nisa
import nki.language as nl

NSTEP = 1


@nki.jit
def xchg_dma(in_tensor):
    _, P, F = in_tensor.shape
    out = nl.ndarray([2, P, F], dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    pid = nl.program_id(0)
    peer = 1 - pid
    a = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.sbuf)
    b = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a, src=in_tensor[pid, :, :])
    for i in nl.static_range(NSTEP):
        if i % 2 == 0:
            nisa.sendrecv(src=a, dst=b, send_to_rank=peer, recv_from_rank=peer, pipe_id=0)
        else:
            nisa.sendrecv(src=b, dst=a, send_to_rank=peer, recv_from_rank=peer, pipe_id=0)
    if NSTEP % 2 == 0:
        nisa.dma_copy(dst=out[pid, :, :], src=a)
    else:
        nisa.dma_copy(dst=out[pid, :, :], src=b)
    return out


@nki.jit
def xchg_gpsimd(in_tensor):
    _, P, F = in_tensor.shape
    out = nl.ndarray([2, P, F], dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    pid = nl.program_id(0)
    peer = 1 - pid
    a = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.sbuf)
    b = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a, src=in_tensor[pid, :, :])
    for i in nl.static_range(NSTEP):
        if i % 2 == 0:
            nisa.sendrecv(src=a, dst=b, send_to_rank=peer, recv_from_rank=peer, pipe_id=0,
                          dma_engine=nisa.dma_engine.gpsimd_dma)
        else:
            nisa.sendrecv(src=b, dst=a, send_to_rank=peer, recv_from_rank=peer, pipe_id=0,
                          dma_engine=nisa.dma_engine.gpsimd_dma)
    if NSTEP % 2 == 0:
        nisa.dma_copy(dst=out[pid, :, :], src=a)
    else:
        nisa.dma_copy(dst=out[pid, :, :], src=b)
    return out


@nki.jit
def xchg_reduce(in_tensor):
    _, P, F = in_tensor.shape
    out = nl.ndarray([2, P, F], dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    pid = nl.program_id(0)
    peer = 1 - pid
    a = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.sbuf)
    b = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a, src=in_tensor[pid, :, :])
    for i in nl.static_range(NSTEP):
        if i % 2 == 0:
            nisa.sendrecv_compute(src=a, dst=b, send_to_rank=peer, recv_from_rank=peer, reduce_op=nl.add)
        else:
            nisa.sendrecv_compute(src=b, dst=a, send_to_rank=peer, recv_from_rank=peer, reduce_op=nl.add)
    if NSTEP % 2 == 0:
        nisa.dma_copy(dst=out[pid, :, :], src=a)
    else:
        nisa.dma_copy(dst=out[pid, :, :], src=b)
    return out


KERNELS = {"xchg_dma": xchg_dma, "xchg_gpsimd": xchg_gpsimd, "xchg_reduce": xchg_reduce}


def expected(name, x, nstep):
    """Reference output for input x [2, P, F] (numpy)."""
    import numpy as np
    if name == "xchg_reduce":
        s = (x[0].astype(np.float64) + x[1]) * 2.0 ** (nstep - 1)
        return np.stack([s, s]).astype(np.float32)
    return x if nstep % 2 == 0 else x[::-1].copy()
