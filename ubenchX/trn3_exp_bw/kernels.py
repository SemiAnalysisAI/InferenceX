# pyright: reportAttributeAccessIssue=false
"""Trainium3 exp / activation throughput kernels (NKI 0.7), one NeuronCore at LNC=1.

Each kernel is a single burst of NUM_OPS instructions on one engine. Op i reads one [P, F] SBUF
source tile (a ring of up to 8 holding 0.1 .. 0.8) and writes one slot of a ring of SBUF tiles, in
the kernel's source and destination dtypes. After the burst, a reduce over the destination slots
(on the Vector Engine, not counted) keeps every result live, and the sum is stored to HBM.

Softmax exp, fp32 -> bf16, with the per-row max subtracted and the row sum chained across the burst
(reset_reduce on the first op, reduce on the rest, the fp32 [P, 1] sum read out once on the last):
  exp_vector_softmax   nisa.exponential(max_value=row_max)                (Vector Engine)
  exp_scalar_softmax   nisa.activation(op=nl.exp, bias=-row_max, reduce_op=nl.add)  (Scalar Engine)

The same softmax exp with a 16-bit source, bf16 -> bf16; the max and the row sum stay fp32, the
only dtype nisa.exponential takes for them:
  exp_vector_softmax_bf16   nisa.exponential(max_value=row_max)           (Vector Engine)

Activation functions on the Scalar Engine, plain (no scale, bias or reduction):
  act_silu                                     fp32 -> bf16
  act_sigmoid, act_rsqrt, act_softplus,
  act_sqrt, act_tanh                           fp32 -> fp32

The kernel input is a [P, F] fp32 tile that only carries the shape; NUM_OPS, SRC_DTYPE, SRC_BYTES,
DST_DTYPE, DST_BYTES and ACT_OP are module globals set by get_kernel (NKI kernels see module globals,
not closures).
"""

import nki
import nki.isa as nisa
import nki.language as nl

SRC_SBUF_BYTES = 65536  # per-partition SBUF bytes for the source ring
DST_SBUF_BYTES = 112000  # per-partition SBUF bytes for the destination ring
ROW_MAX = 0.8  # >= every source value, so exp(x - max) <= 1
NUM_OPS = 1024
SRC_DTYPE = nl.float32
SRC_BYTES = 4
DST_DTYPE = nl.bfloat16
DST_BYTES = 2
ACT_OP = nl.sigmoid


@nki.jit
def exp_vector_softmax_kernel(shape_tensor):
    P = shape_tensor.shape[0]
    F = shape_tensor.shape[1]
    out_tensor = nl.ndarray([P, F], dtype=nl.float32, buffer=nl.shared_hbm)

    n_src = max(1, min(8, SRC_SBUF_BYTES // (F * SRC_BYTES), NUM_OPS))
    src = []
    for t in range(n_src):
        src.append(nl.ndarray([P, F], dtype=SRC_DTYPE, buffer=nl.sbuf))
    for t in range(n_src):
        nisa.memset(dst=src[t], value=0.1 + 0.1 * t)

    n_slots = min(DST_SBUF_BYTES // (F * DST_BYTES), NUM_OPS)
    dst = []
    for t in range(n_slots):
        dst.append(nl.ndarray([P, F], dtype=DST_DTYPE, buffer=nl.sbuf))
    row_max = nl.ndarray([P, 1], dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=row_max, value=ROW_MAX)
    row_sum = nl.ndarray([P, 1], dtype=nl.float32, buffer=nl.sbuf)

    with nl.no_reorder():
        nisa.exponential(dst=dst[0], src=src[0], max_value=row_max, reduce_cmd=nisa.reduce_cmd.reset_reduce)
        for i in nl.static_range(1, NUM_OPS - 1):
            nisa.exponential(dst=dst[i % n_slots], src=src[i % n_src], max_value=row_max,
                             reduce_cmd=nisa.reduce_cmd.reduce)
        nisa.exponential(dst=dst[(NUM_OPS - 1) % n_slots], src=src[(NUM_OPS - 1) % n_src], max_value=row_max,
                         reduce_cmd=nisa.reduce_cmd.reduce, reduce_res=row_sum)

    # Read every slot afterwards so the compiler cannot drop ops into slots nothing reads.
    acc = nl.ndarray([P, F], dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    for t in range(n_slots):
        nisa.tensor_tensor(dst=acc, data1=acc, data2=dst[t], op=nl.add)
    nisa.tensor_scalar(dst=acc, data=acc, op0=nl.add, operand0=row_sum)
    nl.store(out_tensor, value=acc)
    return out_tensor


@nki.jit
def exp_scalar_softmax_kernel(shape_tensor):
    P = shape_tensor.shape[0]
    F = shape_tensor.shape[1]
    out_tensor = nl.ndarray([P, F], dtype=nl.float32, buffer=nl.shared_hbm)

    n_src = max(1, min(8, SRC_SBUF_BYTES // (F * 4), NUM_OPS))
    src = []
    for t in range(n_src):
        src.append(nl.ndarray([P, F], dtype=nl.float32, buffer=nl.sbuf))
    for t in range(n_src):
        nisa.memset(dst=src[t], value=0.1 + 0.1 * t)

    n_slots = min(DST_SBUF_BYTES // (F * DST_BYTES), NUM_OPS)
    dst = []
    for t in range(n_slots):
        dst.append(nl.ndarray([P, F], dtype=DST_DTYPE, buffer=nl.sbuf))
    neg_max = nl.ndarray([P, 1], dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=neg_max, value=-ROW_MAX)
    row_sum = nl.ndarray([P, 1], dtype=nl.float32, buffer=nl.sbuf)

    with nl.no_reorder():
        nisa.activation(dst=dst[0], op=nl.exp, data=src[0], bias=neg_max, scale=1.0, reduce_op=nl.add,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
        for i in nl.static_range(1, NUM_OPS - 1):
            nisa.activation(dst=dst[i % n_slots], op=nl.exp, data=src[i % n_src], bias=neg_max, scale=1.0,
                            reduce_op=nl.add, reduce_cmd=nisa.reduce_cmd.reduce)
        nisa.activation(dst=dst[(NUM_OPS - 1) % n_slots], op=nl.exp, data=src[(NUM_OPS - 1) % n_src], bias=neg_max,
                        scale=1.0, reduce_op=nl.add, reduce_cmd=nisa.reduce_cmd.reduce, reduce_res=row_sum)

    # Read every slot afterwards so the compiler cannot drop ops into slots nothing reads.
    acc = nl.ndarray([P, F], dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    for t in range(n_slots):
        nisa.tensor_tensor(dst=acc, data1=acc, data2=dst[t], op=nl.add)
    nisa.tensor_scalar(dst=acc, data=acc, op0=nl.add, operand0=row_sum)
    nl.store(out_tensor, value=acc)
    return out_tensor


@nki.jit
def act_kernel(shape_tensor):
    P = shape_tensor.shape[0]
    F = shape_tensor.shape[1]
    out_tensor = nl.ndarray([P, F], dtype=nl.float32, buffer=nl.shared_hbm)

    n_src = max(1, min(8, SRC_SBUF_BYTES // (F * 4), NUM_OPS))
    src = []
    for t in range(n_src):
        src.append(nl.ndarray([P, F], dtype=nl.float32, buffer=nl.sbuf))
    for t in range(n_src):
        nisa.memset(dst=src[t], value=0.1 + 0.1 * t)

    n_slots = min(DST_SBUF_BYTES // (F * DST_BYTES), NUM_OPS)
    dst = []
    for t in range(n_slots):
        dst.append(nl.ndarray([P, F], dtype=DST_DTYPE, buffer=nl.sbuf))

    with nl.no_reorder():
        for i in nl.static_range(NUM_OPS):
            nisa.activation(dst=dst[i % n_slots], op=ACT_OP, data=src[i % n_src], engine=nisa.engine.scalar)

    # Read every slot afterwards so the compiler cannot drop ops into slots nothing reads.
    acc = nl.ndarray([P, F], dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    for t in range(n_slots):
        nisa.tensor_tensor(dst=acc, data1=acc, data2=dst[t], op=nl.add)
    nl.store(out_tensor, value=acc)
    return out_tensor


DTYPES = {"fp32": (nl.float32, 4), "bf16": (nl.bfloat16, 2)}  # name -> (dtype, bytes)
# name -> (kernel, source dtype, destination dtype, activation op or None)
KERNELS = {
    "exp_vector_softmax": (exp_vector_softmax_kernel, "fp32", "bf16", None),
    "exp_vector_softmax_bf16": (exp_vector_softmax_kernel, "bf16", "bf16", None),
    "exp_scalar_softmax": (exp_scalar_softmax_kernel, "fp32", "bf16", None),
    "act_silu": (act_kernel, "fp32", "bf16", nl.silu),
    "act_sigmoid": (act_kernel, "fp32", "fp32", nl.sigmoid),
    "act_rsqrt": (act_kernel, "fp32", "fp32", nl.rsqrt),
    "act_softplus": (act_kernel, "fp32", "fp32", nl.softplus),
    "act_sqrt": (act_kernel, "fp32", "fp32", nl.sqrt),
    "act_tanh": (act_kernel, "fp32", "fp32", nl.tanh),
}
# every other kernel runs on the Scalar Engine
ENGINE = {"exp_vector_softmax": "vector", "exp_vector_softmax_bf16": "vector"}


def get_kernel(name, num_ops):
    global NUM_OPS, SRC_DTYPE, SRC_BYTES, DST_DTYPE, DST_BYTES, ACT_OP
    kernel, src, dst, op = KERNELS[name]
    SRC_DTYPE, SRC_BYTES = DTYPES[src]
    DST_DTYPE, DST_BYTES = DTYPES[dst]
    NUM_OPS = num_ops
    if op is not None:
        ACT_OP = op
    return kernel
