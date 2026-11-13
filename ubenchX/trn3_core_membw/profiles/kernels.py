# pyright: reportAttributeAccessIssue=false
"""Trainium3 HBM pointer chase, the analogue of the GPU/Ascend core-memory-profile kernel (NKI 0.7).

GPU page: one warp per SM chases a random linked list (`ld.global.cg [p]`), timing each hop with
clock64. Here one NeuronCore chases a random cyclic linked list stored in HBM as int32 slot indices:

    idx <- chain[idx]   (H times)

Each hop is one indirect DMA (scalar_offset = the 1x1 SBUF tile written by the previous hop), so every
hop depends on the one before it and the per-hop time is one HBM round trip through the DMA path.
Per-hop timestamps come from the profiler (the DMA packet table), not from an on-device clock.

chain is [N, W] int32; slot i occupies one row of W int32 (W*4 bytes = the slot stride). Only column 0
is read. The final idx is stored to the output so the chain is live.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode as dge_mode_enum

HOPS = 512


@nki.jit
def chase_swdge(chain):
    N, W = chain.shape
    out = nl.ndarray([1, 1], dtype=chain.dtype, buffer=nl.shared_hbm)
    idx = nl.ndarray([1, 1], dtype=chain.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx, src=chain[0:1, 0:1], dge_mode=dge_mode_enum.none)
    for h in nl.static_range(HOPS):
        nisa.dma_copy(dst=idx, src=chain.ap(pattern=[[W, 1], [1, 1]], offset=0, scalar_offset=idx,
                                            indirect_dim=0),
                      dge_mode=dge_mode_enum.swdge)
    nisa.dma_copy(dst=out, src=idx, dge_mode=dge_mode_enum.none)
    return out


@nki.jit
def chase_hwdge(chain):
    N, W = chain.shape
    out = nl.ndarray([1, 1], dtype=chain.dtype, buffer=nl.shared_hbm)
    idx = nl.ndarray([1, 1], dtype=chain.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx, src=chain[0:1, 0:1], dge_mode=dge_mode_enum.none)
    for h in nl.static_range(HOPS):
        nisa.dma_copy(dst=idx, src=chain.ap(pattern=[[W, 1], [1, 1]], offset=0, scalar_offset=idx,
                                            indirect_dim=0),
                      dge_mode=dge_mode_enum.hwdge)
    nisa.dma_copy(dst=out, src=idx, dge_mode=dge_mode_enum.none)
    return out


@nki.jit
def chase_register(chain):
    """Engine-sequencer chase: register_load of chain[reg] into reg."""
    N, W = chain.shape
    out = nl.ndarray([1, 1], dtype=chain.dtype, buffer=nl.shared_hbm)
    reg = nisa.register_alloc(0)
    for h in nl.static_range(HOPS):
        nisa.register_load(reg, chain.ap(pattern=[[W, 1], [1, 1]], offset=0, scalar_offset=reg,
                                         indirect_dim=0))
    res = nl.ndarray([1, 1], dtype=chain.dtype, buffer=nl.sbuf)
    nisa.register_store(res, reg)
    nisa.dma_copy(dst=out, src=res, dge_mode=dge_mode_enum.none)
    return out


KERNELS = {"chase_swdge": chase_swdge, "chase_hwdge": chase_hwdge, "chase_register": chase_register}
