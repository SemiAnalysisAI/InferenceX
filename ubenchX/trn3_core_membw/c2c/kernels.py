# pyright: reportAttributeAccessIssue=false
"""SBUF -> SBUF core-to-core exchange between any two NeuronCores of a chip (NKI 0.7, LNC=1).

nisa.sendrecv only reaches the other core of an LNC=2 pair, and nki.collectives.collective_permute and
all_to_all are HBM-only. all_gather accepts SBUF tensors, so a 2-rank all_gather on SBUF tiles is the
exchange: each core sends its [P, F] tile to the peer's SBUF and receives the peer's tile into its own.

All 8 cores of a chip run the kernel at once (ranks 0-7); REPLICA is a perfect matching (pairs i, i^M),
so one launch times 4 disjoint pairs. Each step gathers from what the previous step received, so the
NSTEP steps form a dependent chain; the slope of device time against NSTEP is the cost per exchange.

in_tensor [P, F] (same on every rank). Expected after N steps: [in | in] on every rank.
"""

import nki
import nki.collectives as cc
import nki.isa as nisa
import nki.language as nl

NSTEP = 1
REPLICA = [[0, 1], [2, 3], [4, 5], [6, 7]]


@nki.jit
def xchg_ag(in_tensor):
    P, F = in_tensor.shape
    out = nl.ndarray([P, 2 * F], dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    a = nl.ndarray([P, 2 * F], dtype=in_tensor.dtype, buffer=nl.sbuf)
    b = nl.ndarray([P, 2 * F], dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a[:, 0:F], src=in_tensor)
    rg = cc.ReplicaGroup(REPLICA)
    for i in nl.static_range(NSTEP):
        if i % 2 == 0:
            cc.all_gather(srcs=[a[:, 0:F]], dsts=[b], replica_group=rg, collective_dim=1)
        else:
            cc.all_gather(srcs=[b[:, F:2 * F]], dsts=[a], replica_group=rg, collective_dim=1)
    if NSTEP % 2 == 0:
        nisa.dma_copy(dst=out, src=a)
    else:
        nisa.dma_copy(dst=out, src=b)
    return out


KERNELS = {"xchg_ag": xchg_ag}


@nki.jit
def xchg_ag_hbm(in_tensor):
    """Same chain on HBM tensors (all_gather along dim 0): for pairs the SBUF path does not route."""
    P, F = in_tensor.shape
    out = nl.ndarray([2 * P, F], dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    a = nl.ndarray([2 * P, F], dtype=in_tensor.dtype, buffer=nl.private_hbm)
    b = nl.ndarray([2 * P, F], dtype=in_tensor.dtype, buffer=nl.private_hbm)
    t = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=in_tensor)
    nisa.dma_copy(dst=a[0:P, :], src=t)
    rg = cc.ReplicaGroup(REPLICA)
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


KERNELS["xchg_ag_hbm"] = xchg_ag_hbm

SEQ = [[[0, 4], [1, 5], [2, 6], [3, 7]], [[0, 2], [1, 3], [4, 6], [5, 7]]]  # replica group per step, cycled


@nki.jit
def relay_hbm(in_tensor):
    """xchg_ag_hbm with the replica group cycling through SEQ: [mask 4, mask 2] carries rank r's data to r^6
    in two hops (0 -> 4 -> 6). NSTEP counts collectives (a multiple of len(SEQ))."""
    P, F = in_tensor.shape
    out = nl.ndarray([2 * P, F], dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    a = nl.ndarray([2 * P, F], dtype=in_tensor.dtype, buffer=nl.private_hbm)
    b = nl.ndarray([2 * P, F], dtype=in_tensor.dtype, buffer=nl.private_hbm)
    t = nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=in_tensor)
    nisa.dma_copy(dst=a[0:P, :], src=t)
    for i in nl.static_range(NSTEP):
        rg = cc.ReplicaGroup(SEQ[i % len(SEQ)])
        if i % 2 == 0:
            cc.all_gather(srcs=[a[0:P, :]], dsts=[b], replica_group=rg, collective_dim=0)
        else:
            cc.all_gather(srcs=[b[P:2 * P, :]], dsts=[a], replica_group=rg, collective_dim=0)
    src = a if NSTEP % 2 == 0 else b
    for h in nl.static_range(2):
        nisa.dma_copy(dst=t, src=src[h * P:(h + 1) * P, :])
        nisa.dma_copy(dst=out[h * P:(h + 1) * P, :], src=t)
    return out


KERNELS["relay_hbm"] = relay_hbm
