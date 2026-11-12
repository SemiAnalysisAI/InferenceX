# pyright: reportAttributeAccessIssue=false
"""Trainium3 communication between the two physical NeuronCores of one LNC=2 logical core (NKI 0.7).

At LNC=2 a kernel runs on both cores of an HBM stack; nl.program_id(0) is the rank (0 or 1), peer =
1 - rank. Every kernel loads a [128, F] tile (the kernel input, rank r uses in_tensor[r]) into SBUF and
issues NUM_OPS operations back to back in a no_reorder block; each moves the same bytes per rank:

  sr_dma      nisa.sendrecv, standard DMA: send src to the peer, receive the peer's src into dst (a
              2-rank all-gather: src is kept).
  sr_gpsimd   the same with dma_engine=gpsimd_dma (GpSimd's internal DMA, SBUF to SBUF; <= 1 KiB per
              partition for 32-bit, 512 B for 16-bit types).
  sr_add      sendrecv then nisa.tensor_tensor(add) on VectorE: the unfused 2-rank all-reduce nkilib uses.
  src_add     nisa.sendrecv_compute(reduce_op=add): the fused 2-rank all-reduce in the DMA engine.
  shm_barrier each rank dma_copy's src into its half of a shared_hbm tensor, nisa.core_barrier, then
              dma_copy's the peer's half into SBUF (2-rank all-gather through shared HBM). Two shared
              buffers alternate between operations.
  local_copy  the same SBUF -> HBM -> SBUF round trip into the rank's own private_hbm buffer, with no
              barrier and no peer: the cost of the copies alone.

After the loop each rank stores its last result to out[rank] so no operation is dead. NUM_OPS and DTYPE
are module globals set by get_kernel.
"""

import nki
import nki.isa as nisa
import nki.language as nl

NUM_OPS = 64
DTYPE = nl.bfloat16


def _load(in_tensor):
    rank = nl.program_id(0)
    P, F = in_tensor.shape[1], in_tensor.shape[2]
    src = nl.ndarray([P, F], dtype=DTYPE, buffer=nl.sbuf)
    nisa.dma_copy(dst=src, src=in_tensor[rank])
    out = nl.ndarray([2, P, F], dtype=DTYPE, buffer=nl.shared_hbm)
    return rank, 1 - rank, src, out, P, F


def _sendrecv(in_tensor, engine):
    rank, peer, src, out, P, F = _load(in_tensor)
    dst = nl.ndarray([P, F], dtype=DTYPE, buffer=nl.sbuf)
    with nl.no_reorder():
        for i in nl.static_range(NUM_OPS):
            nisa.sendrecv(src=src, dst=dst, send_to_rank=peer, recv_from_rank=peer, pipe_id=0, dma_engine=engine)
    nisa.dma_copy(dst=out[rank], src=dst)
    return out


@nki.jit
def sr_dma_kernel(in_tensor):
    return _sendrecv(in_tensor, nisa.dma_engine.dma)


@nki.jit
def sr_gpsimd_kernel(in_tensor):
    return _sendrecv(in_tensor, nisa.dma_engine.gpsimd_dma)


@nki.jit
def sr_add_kernel(in_tensor):
    rank, peer, src, out, P, F = _load(in_tensor)
    recv = nl.ndarray([P, F], dtype=DTYPE, buffer=nl.sbuf)
    acc = nl.ndarray([P, F], dtype=DTYPE, buffer=nl.sbuf)
    with nl.no_reorder():
        for i in nl.static_range(NUM_OPS):
            nisa.sendrecv(src=src, dst=recv, send_to_rank=peer, recv_from_rank=peer, pipe_id=0)
            nisa.tensor_tensor(dst=acc, data1=src, data2=recv, op=nl.add)
    nisa.dma_copy(dst=out[rank], src=acc)
    return out


@nki.jit
def src_add_kernel(in_tensor):
    rank, peer, src, out, P, F = _load(in_tensor)
    acc = nl.ndarray([P, F], dtype=DTYPE, buffer=nl.sbuf)
    with nl.no_reorder():
        for i in nl.static_range(NUM_OPS):
            nisa.sendrecv_compute(src=src, dst=acc, send_to_rank=peer, recv_from_rank=peer, reduce_op=nl.add)
    nisa.dma_copy(dst=out[rank], src=acc)
    return out


@nki.jit
def shm_barrier_kernel(in_tensor):
    rank, peer, src, out, P, F = _load(in_tensor)
    shm = [nl.ndarray([2, P, F], dtype=DTYPE, buffer=nl.shared_hbm),
           nl.ndarray([2, P, F], dtype=DTYPE, buffer=nl.shared_hbm)]
    recv = nl.ndarray([P, F], dtype=DTYPE, buffer=nl.sbuf)
    with nl.no_reorder():
        for i in nl.static_range(NUM_OPS):
            sh = shm[i % 2]
            nisa.dma_copy(dst=sh[rank], src=src)
            nisa.core_barrier(data=sh, cores=(0, 1))
            nisa.dma_copy(dst=recv, src=sh[peer])
    nisa.dma_copy(dst=out[rank], src=recv)
    return out


@nki.jit
def local_copy_kernel(in_tensor):
    rank, peer, src, out, P, F = _load(in_tensor)
    priv = [nl.ndarray([P, F], dtype=DTYPE, buffer=nl.private_hbm),
            nl.ndarray([P, F], dtype=DTYPE, buffer=nl.private_hbm)]
    recv = nl.ndarray([P, F], dtype=DTYPE, buffer=nl.sbuf)
    with nl.no_reorder():
        for i in nl.static_range(NUM_OPS):
            nisa.dma_copy(dst=priv[i % 2], src=src)
            nisa.dma_copy(dst=recv, src=priv[i % 2])
    nisa.dma_copy(dst=out[rank], src=recv)
    return out


KERNELS = {"sr_dma": sr_dma_kernel, "sr_gpsimd": sr_gpsimd_kernel, "sr_add": sr_add_kernel,
           "src_add": src_add_kernel, "shm_barrier": shm_barrier_kernel, "local_copy": local_copy_kernel}
DTYPES = {"bf16": nl.bfloat16, "fp32": nl.float32}


def get_kernel(name, num_ops, dtype="bf16"):
    global NUM_OPS, DTYPE
    NUM_OPS = num_ops
    DTYPE = DTYPES[dtype]
    return KERNELS[name]
