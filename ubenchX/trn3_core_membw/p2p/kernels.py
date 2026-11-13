# pyright: reportAttributeAccessIssue=false
"""Point-to-point core-to-core candidates on Trainium3 (NKI 0.7). See p2p/probe.py.

Every kernel is a dependent chain of NSTEP exchanges between paired ranks: step i sends what step i-1
received. After N steps a rank holds its own tile if N is even, its peer's if N is odd.

  sr_lnc1     nisa.sendrecv at LNC=1, one NeuronCore per rank, peer given as the constant PEER
  sr_lncn     nisa.sendrecv at LNC=N (N = in_tensor.shape[0]), peer = program_id ^ MASK
  cpi_sbuf    nki.collectives.collective_permute_implicit on SBUF tiles, 2-rank rings (PAIRS)
  cp_sbuf     nki.collectives.collective_permute on SBUF tiles (documented HBM-only; confirm)
  cp_hbm      nki.collectives.collective_permute on private HBM, source_target_pairs both ways
  ag_sbuf     2-rank nki.collectives.all_gather on SBUF (c2c/kernels.xchg_ag): out [P, 2F]
  ag_hbm      the same on private HBM along dim 0 (c2c/kernels.xchg_ag_hbm): out [2P, F]
              all_gather chains forward the higher rank's half, so after N >= 2 steps both ranks hold
              [x_hi | x_hi] (N = 1: [x_lo | x_hi]).
"""

import nki
import nki.collectives as cc
import nki.isa as nisa
import nki.language as nl

NSTEP = 1
PEER = 1
MASK = 1
PAIRS = [[0, 1], [2, 3], [4, 5], [6, 7]]
GPSIMD = False
ST_PAIRS = [(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4), (6, 7), (7, 6)]  # set by probe.py with PAIRS


@nki.jit
def sr_lnc1(in_tensor):
    P, F = in_tensor.shape
    out = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    a = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.sbuf)
    b = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a, src=in_tensor)
    for i in nl.static_range(NSTEP):
        s, d = (a, b) if i % 2 == 0 else (b, a)
        nisa.sendrecv(src=s, dst=d, send_to_rank=PEER, recv_from_rank=PEER, pipe_id=0)
    nisa.dma_copy(dst=out, src=a if NSTEP % 2 == 0 else b)
    return out


@nki.jit
def sr_lncn(in_tensor):
    N, P, F = in_tensor.shape
    out = nl.ndarray([N, P, F], dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    pid = nl.program_id(0)
    peer = pid ^ MASK
    a = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.sbuf)
    b = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a, src=in_tensor[pid, :, :])
    for i in nl.static_range(NSTEP):
        s, d = (a, b) if i % 2 == 0 else (b, a)
        if GPSIMD:
            nisa.sendrecv(src=s, dst=d, send_to_rank=peer, recv_from_rank=peer, pipe_id=0,
                          dma_engine=nisa.dma_engine.gpsimd_dma)
        else:
            nisa.sendrecv(src=s, dst=d, send_to_rank=peer, recv_from_rank=peer, pipe_id=0)
    nisa.dma_copy(dst=out[pid, :, :], src=a if NSTEP % 2 == 0 else b)
    return out


@nki.jit
def cpi_sbuf(in_tensor):
    P, F = in_tensor.shape
    out = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    a = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.sbuf)
    b = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a, src=in_tensor)
    rg = cc.ReplicaGroup(PAIRS)
    for i in nl.static_range(NSTEP):
        s, d = (a, b) if i % 2 == 0 else (b, a)
        cc.collective_permute_implicit(srcs_by_channel=[[s]], dsts_by_channel=[[d]], replica_group=rg)
    nisa.dma_copy(dst=out, src=a if NSTEP % 2 == 0 else b)
    return out


@nki.jit
def cp_sbuf(in_tensor):
    P, F = in_tensor.shape
    out = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    a = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.sbuf)
    b = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a, src=in_tensor)
    for i in nl.static_range(NSTEP):
        s, d = (a, b) if i % 2 == 0 else (b, a)
        cc.collective_permute(srcs=[s], dsts=[d], source_target_pairs=ST_PAIRS)
    nisa.dma_copy(dst=out, src=a if NSTEP % 2 == 0 else b)
    return out


@nki.jit
def cp_hbm(in_tensor):
    P, F = in_tensor.shape
    out = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    a = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.private_hbm)
    b = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.private_hbm)
    t = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=in_tensor)
    nisa.dma_copy(dst=a, src=t)
    for i in nl.static_range(NSTEP):
        s, d = (a, b) if i % 2 == 0 else (b, a)
        cc.collective_permute(srcs=[s], dsts=[d], source_target_pairs=ST_PAIRS)
    nisa.dma_copy(dst=t, src=a if NSTEP % 2 == 0 else b)
    nisa.dma_copy(dst=out, src=t)
    return out


@nki.jit
def ag_sbuf(in_tensor):
    P, F = in_tensor.shape
    out = nl.ndarray([P, 2 * F], dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    a = nl.ndarray([P, 2 * F], dtype=in_tensor.dtype, buffer=nl.sbuf)
    b = nl.ndarray([P, 2 * F], dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a[:, 0:F], src=in_tensor)
    rg = cc.ReplicaGroup(PAIRS)
    for i in nl.static_range(NSTEP):
        if i % 2 == 0:
            cc.all_gather(srcs=[a[:, 0:F]], dsts=[b], replica_group=rg, collective_dim=1)
        else:
            cc.all_gather(srcs=[b[:, F:2 * F]], dsts=[a], replica_group=rg, collective_dim=1)
    nisa.dma_copy(dst=out, src=a if NSTEP % 2 == 0 else b)
    return out


@nki.jit
def ag_hbm(in_tensor):
    P, F = in_tensor.shape
    out = nl.ndarray([2 * P, F], dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    a = nl.ndarray([2 * P, F], dtype=in_tensor.dtype, buffer=nl.private_hbm)
    b = nl.ndarray([2 * P, F], dtype=in_tensor.dtype, buffer=nl.private_hbm)
    t = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=in_tensor)
    nisa.dma_copy(dst=a[0:P, :], src=t)
    rg = cc.ReplicaGroup(PAIRS)
    for i in nl.static_range(NSTEP):
        if i % 2 == 0:
            cc.all_gather(srcs=[a[0:P, :]], dsts=[b], replica_group=rg, collective_dim=0)
        else:
            cc.all_gather(srcs=[b[P:2 * P, :]], dsts=[a], replica_group=rg, collective_dim=0)
    src = a if NSTEP % 2 == 0 else b
    for h in nl.static_range(2):
        nisa.dma_copy(dst=t, src=src[h * P:(h + 1) * P, :])
        nisa.dma_copy(dst=out[h * P:(h + 1) * P, :], src=t)
    return out


KERNELS = {k: v for k, v in globals().items() if k in ("sr_lnc1", "sr_lncn", "cpi_sbuf", "cp_sbuf", "cp_hbm", "ag_sbuf", "ag_hbm")}
