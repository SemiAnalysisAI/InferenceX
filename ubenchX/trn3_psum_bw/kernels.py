# pyright: reportAttributeAccessIssue=false
"""Trainium3 PSUM -> SBUF copy bandwidth kernels (NKI 0.7), one NeuronCore at LNC=1.

Each kernel is a single burst of fp32 -> fp32 nisa.tensor_copy instructions on one engine. Copy i
reads one [P, F] PSUM tile (a ring of 8 tiles, one per 2 KiB PSUM bank, when F fits in a bank;
otherwise one tile spanning several banks) and writes it to one slot of a ring of SBUF tiles.
After the burst, a reduce over the ring slots (SBUF only, so it is not counted) keeps every copy
live, and the sum is stored to HBM.

  copy_vector   tensor_copy on the Vector Engine
  copy_scalar   tensor_copy on the Scalar Engine
  dual_alternate  NUM_COPIES copies on each of VectorE and ScalarE, alternating in program order
                (even i on VectorE, odd i on ScalarE), the usual way kernels split PSUM evacuation
  (GpSimd cannot access PSUM.)

dual_alternate leaves PSUM placement to the allocator, like a normal kernel. Its PSUM ring holds as
many tiles as fit in PSUM (up to 8), so the two engines read different tiles; each engine writes
its own SBUF ring, so nothing makes one engine wait for the other.

The two kernels differ only in the tensor_copy engine. The kernel input is a [P, F] fp32 tile that
only carries the shape; the copy count is the NUM_COPIES global set by get_kernel.
"""

import nki
import nki.isa as nisa
import nki.language as nl

PSUM_BANK_BYTES = 2048  # per partition; PSUM is 8 banks
SBUF_RING_BYTES = 192000  # per-partition SBUF bytes used for the destination ring
DTYPE_BYTES = 4
NUM_COPIES = 1024  # set by get_kernel before compiling; NKI kernels see module globals, not closures


@nki.jit
def copy_vector_kernel(shape_tensor):
    P = shape_tensor.shape[0]
    F = shape_tensor.shape[1]
    out_tensor = nl.ndarray([P, F], dtype=nl.float32, buffer=nl.shared_hbm)

    n_ps = min(8, NUM_COPIES) if F * DTYPE_BYTES <= PSUM_BANK_BYTES else 1
    ps = []
    for t in range(n_ps):
        ps.append(nl.ndarray([P, F], dtype=nl.float32, buffer=nl.psum))
    for t in range(n_ps):
        nisa.memset(dst=ps[t], value=1.0 + t)

    n_slots = min(SBUF_RING_BYTES // (F * DTYPE_BYTES), NUM_COPIES)
    slots = []
    for t in range(n_slots):
        slots.append(nl.ndarray([P, F], dtype=nl.float32, buffer=nl.sbuf, address=(0, t * F * DTYPE_BYTES)))

    with nl.no_reorder():
        for i in nl.static_range(NUM_COPIES):
            nisa.tensor_copy(dst=slots[i % n_slots], src=ps[i % n_ps], engine=nisa.engine.vector)

    # Read every slot afterwards so the compiler cannot drop copies into slots nothing reads.
    acc = nl.ndarray([P, F], dtype=nl.float32, buffer=nl.sbuf, address=(0, n_slots * F * DTYPE_BYTES))
    nisa.memset(dst=acc, value=0.0)
    for t in range(n_slots):
        nisa.tensor_tensor(dst=acc, data1=acc, data2=slots[t], op=nl.add)
    nl.store(out_tensor, value=acc)
    return out_tensor


@nki.jit
def copy_scalar_kernel(shape_tensor):
    P = shape_tensor.shape[0]
    F = shape_tensor.shape[1]
    out_tensor = nl.ndarray([P, F], dtype=nl.float32, buffer=nl.shared_hbm)

    n_ps = min(8, NUM_COPIES) if F * DTYPE_BYTES <= PSUM_BANK_BYTES else 1
    ps = []
    for t in range(n_ps):
        ps.append(nl.ndarray([P, F], dtype=nl.float32, buffer=nl.psum))
    for t in range(n_ps):
        nisa.memset(dst=ps[t], value=1.0 + t)

    n_slots = min(SBUF_RING_BYTES // (F * DTYPE_BYTES), NUM_COPIES)
    slots = []
    for t in range(n_slots):
        slots.append(nl.ndarray([P, F], dtype=nl.float32, buffer=nl.sbuf, address=(0, t * F * DTYPE_BYTES)))

    with nl.no_reorder():
        for i in nl.static_range(NUM_COPIES):
            nisa.tensor_copy(dst=slots[i % n_slots], src=ps[i % n_ps], engine=nisa.engine.scalar)

    # Read every slot afterwards so the compiler cannot drop copies into slots nothing reads.
    acc = nl.ndarray([P, F], dtype=nl.float32, buffer=nl.sbuf, address=(0, n_slots * F * DTYPE_BYTES))
    nisa.memset(dst=acc, value=0.0)
    for t in range(n_slots):
        nisa.tensor_tensor(dst=acc, data1=acc, data2=slots[t], op=nl.add)
    nl.store(out_tensor, value=acc)
    return out_tensor


@nki.jit
def dual_alternate_kernel(shape_tensor):
    P = shape_tensor.shape[0]
    F = shape_tensor.shape[1]
    out_tensor = nl.ndarray([P, F], dtype=nl.float32, buffer=nl.shared_hbm)

    # An even ring size gives VectorE the even tiles and ScalarE the odd ones.
    n_ps = max(1, min(8, PSUM_BANK_BYTES * 8 // (F * DTYPE_BYTES)))
    ps = []
    for t in range(n_ps):
        ps.append(nl.ndarray([P, F], dtype=nl.float32, buffer=nl.psum))
    for t in range(n_ps):
        nisa.memset(dst=ps[t], value=1.0 + t)

    # One SBUF ring per engine (VectorE in the first half, ScalarE in the second), so no slot is
    # written by both engines.
    n_slots = min(SBUF_RING_BYTES // 2 // (F * DTYPE_BYTES), NUM_COPIES)
    vslots = []
    sslots = []
    for t in range(n_slots):
        vslots.append(nl.ndarray([P, F], dtype=nl.float32, buffer=nl.sbuf, address=(0, t * F * DTYPE_BYTES)))
    for t in range(n_slots):
        sslots.append(nl.ndarray([P, F], dtype=nl.float32, buffer=nl.sbuf,
                                 address=(0, (n_slots + t) * F * DTYPE_BYTES)))

    with nl.no_reorder():
        for i in nl.static_range(NUM_COPIES):
            nisa.tensor_copy(dst=vslots[i % n_slots], src=ps[(2 * i) % n_ps], engine=nisa.engine.vector)
            nisa.tensor_copy(dst=sslots[i % n_slots], src=ps[(2 * i + 1) % n_ps], engine=nisa.engine.scalar)

    acc = nl.ndarray([P, F], dtype=nl.float32, buffer=nl.sbuf, address=(0, 2 * n_slots * F * DTYPE_BYTES))
    nisa.memset(dst=acc, value=0.0)
    for t in range(n_slots):
        nisa.tensor_tensor(dst=acc, data1=acc, data2=vslots[t], op=nl.add)
    for t in range(n_slots):
        nisa.tensor_tensor(dst=acc, data1=acc, data2=sslots[t], op=nl.add)
    nl.store(out_tensor, value=acc)
    return out_tensor


KERNELS = {"copy_vector": copy_vector_kernel, "copy_scalar": copy_scalar_kernel,
           "dual_alternate": dual_alternate_kernel}
ENGINES = {"dual_alternate": 2}  # engines issuing NUM_COPIES copies each; 1 for the others


def get_kernel(name, num_copies):
    global NUM_COPIES
    NUM_COPIES = num_copies
    return KERNELS[name]
