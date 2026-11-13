# pyright: reportAttributeAccessIssue=false
"""Trainium3 HBM copy kernels for the mem-bw page (NKI 0.7).

The GPU page times b.copy_(a): one read and one write of every byte. On Trainium a copy is an explicit
DMA program, so the compiler cannot elide it (on TPU, XLA could, which is why that port used y = x + 1).

The input is [C, P, F]: C chunks of one [P, F] DMA each. layout() picks the shape for a byte count.

  copy_h2h_<dge>   out[c] = in[c] for every chunk, HBM -> HBM, one dma_copy per chunk.
  copy_stage_<dge> HBM -> SBUF -> HBM through a ring of SBUF slots (the path a compute kernel takes).
                   Only valid when one chunk fits an SBUF slot (P = 128, F*4 <= SLOT_BYTES).

dge: static (descriptors written by the runtime before execution), hwdge (DGE hardware block at run time).
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode as dge_mode_enum

DTYPE_BYTES = 4  # float32
SLOT_BYTES = 32768  # per-partition bytes per SBUF ring slot (F <= 8192 fp32)
RING_SLOTS = 4
MAX_CHUNK_BYTES = 4 * 2**20


def layout(nbytes, max_chunk=MAX_CHUNK_BYTES):
    """(C, P, F) for a float32 array of nbytes. P = 128 when it divides, else 1; C is the fewest equal
    chunks of at most max_chunk bytes each."""
    if nbytes % DTYPE_BYTES:
        raise ValueError(f"{nbytes} B is not a whole number of float32")
    e = nbytes // DTYPE_BYTES
    p = 128 if e % 128 == 0 else 1
    ftot = e // p
    for c in range(1, ftot + 1):
        if ftot % c == 0 and (ftot // c) * p * DTYPE_BYTES <= max_chunk:
            return c, p, ftot // c
    raise ValueError(nbytes)


@nki.jit
def copy_h2h_static(in_tensor):
    C, P, F = in_tensor.shape
    out = nl.ndarray([C, P, F], dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    for c in nl.static_range(C):
        nisa.dma_copy(dst=out[c, :, :], src=in_tensor[c, :, :], dge_mode=dge_mode_enum.none)
    return out


@nki.jit
def copy_stage_static(in_tensor):
    C, P, F = in_tensor.shape
    out = nl.ndarray([C, P, F], dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    n = min(RING_SLOTS, C)
    slots = []
    for s in range(n):
        slots.append(nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.sbuf, address=(0, s * SLOT_BYTES)))
    for c in nl.static_range(C):
        nisa.dma_copy(dst=slots[c % n], src=in_tensor[c, :, :], dge_mode=dge_mode_enum.none)
        nisa.dma_copy(dst=out[c, :, :], src=slots[c % n], dge_mode=dge_mode_enum.none)
    return out


@nki.jit
def copy_h2h_hwdge(in_tensor):
    C, P, F = in_tensor.shape
    out = nl.ndarray([C, P, F], dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    for c in nl.static_range(C):
        nisa.dma_copy(dst=out[c, :, :], src=in_tensor[c, :, :], dge_mode=dge_mode_enum.hwdge)
    return out


@nki.jit
def copy_stage_hwdge(in_tensor):
    C, P, F = in_tensor.shape
    out = nl.ndarray([C, P, F], dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    n = min(RING_SLOTS, C)
    slots = []
    for s in range(n):
        slots.append(nl.ndarray([P, F], dtype=in_tensor.dtype, buffer=nl.sbuf, address=(0, s * SLOT_BYTES)))
    for c in nl.static_range(C):
        nisa.dma_copy(dst=slots[c % n], src=in_tensor[c, :, :], dge_mode=dge_mode_enum.hwdge)
        nisa.dma_copy(dst=out[c, :, :], src=slots[c % n], dge_mode=dge_mode_enum.hwdge)
    return out


KERNELS = {
    "copy_h2h_static": copy_h2h_static,
    "copy_h2h_hwdge": copy_h2h_hwdge,
    "copy_stage_static": copy_stage_static,
    "copy_stage_hwdge": copy_stage_hwdge,
}


@nki.jit
def copy_h2h_static_spmd(in_tensor):
    """LNC=2: in_tensor is [2*C, P, F]; physical core k (nl.program_id(0)) copies chunks k*C .. k*C+C-1."""
    C2, P, F = in_tensor.shape
    C = C2 // 2
    out = nl.ndarray([C2, P, F], dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    pid = nl.program_id(0)
    for c in nl.static_range(C):
        nisa.dma_copy(dst=out[pid * C + c, :, :], src=in_tensor[pid * C + c, :, :], dge_mode=dge_mode_enum.none)
    return out


KERNELS["copy_h2h_static_spmd"] = copy_h2h_static_spmd
