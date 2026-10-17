"""NKI backend for Trainium.

Compute ops are @nki.jit kernels (real NKI):
  - gemm           -> LNC=2-sharded matmul (bundled NKI, in this file)
  - attention_mha  -> nkilib.core.attention.attention_cte

Collectives route through torch_xla's xm.* primitives:
  - allreduce, allgather, reduce_scatter, alltoall  -> xm.*

Why not nki.collectives? In the currently shipped Neuron SDK (2.29.0 /
neuronx-cc 2.24 / nki 0.3.0) on Trainium3, the NKI-emitted collective NEFFs
trigger a runtime assertion in libnrt (`enc.cc:3663:
Assertion alg == ENC_ALG_MESH failed`) — independent of LNC mode, replica
group shape, or torchrun init. Compiler internal errors (`NCC_ILLC059`,
`NCC_INLA001`) also block the canonical SBUF-staged pattern under LNC=2.
The error messages themselves say "Please open a support ticket". torch_xla
collectives use a different (HLO-level) lowering that *does* work, so we use
that for the operatorx collective ops until the NKI path is fixed in a
future SDK release.

The `nl.spmd_dim` / `nl.nc()` SPMD-launch APIs that the AWS docs reference
are NOT yet shipped (NKI 0.3.0 only has `kernel[N]` with N ∈ {1, 2}). We use
the bracket form + in-kernel `nl.program_id` for multi-NC sharding.
"""
from __future__ import annotations

import math

import torch
import torch_neuronx  # noqa: F401  registers the neuron device
import torch_xla.core.xla_model as xm

import nki
import nki.isa as nisa
import nki.language as nl


from operatorx.core import BackendImpl, Op, UnsupportedOpError, lookup_versions

# Import once at module load so the @nki.jit dispatchers are ready.
from nkilib.core.attention.attention_cte import attention_cte



def versions() -> dict[str, str]:
    return lookup_versions("nki", "nkilib", "neuronx-cc", "torch_neuronx", "torch_xla")


_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    # NKI accepts both float8_e4m3 and float8_e5m2 for nc_matmul, but PyTorch
    # only ships e4m3fn (finite-NaN variant) which doesn't map cleanly:
    # _torch_to_neuron_dtype either rejects it ("float8_e4m3fn is not
    # supported in nki") or requires UNSAFE_FP8FNCAST which then creates a
    # mixed legacy/OCP type at the compiler level (NCC_EOCP001). float8_e5m2
    # maps to nl.float8_e5m2 directly with no casting layer, so use that as
    # our "fp8" dtype on Trainium. Numerically different from e4m3 (more
    # exponent, less mantissa) but supported end-to-end.
    "fp8":  torch.float8_e5m2,
}


def _resolve(dtype: str) -> torch.dtype:
    if dtype not in _DTYPES or _DTYPES[dtype] is None:
        raise UnsupportedOpError(f"nkilib doesn't support dtype={dtype!r}")
    return _DTYPES[dtype]


def _device():
    return xm.xla_device()


# ---------- gemm: LNC=2 matmul via standalone NKI ----------
#
# FIXME: revisit for LNC=2 gemm entrypoint
#
# Shape constraint: M%2048, K%1024, N%2048 (so NUM_BLOCK_N = N/1024 is even).


_GEMM_TILES_IN_BLOCK_M = 16
_GEMM_TILES_IN_BLOCK_N = 2
_GEMM_TILES_IN_BLOCK_K = 8
_GEMM_TILES_IN_BLOCK_K_FP8 = 4   # double_row consumes 256 K-elts/call vs 128 → half the K-subtiles for same 1024 BLOCK_K
_GEMM_BLOCK_M = 128 * _GEMM_TILES_IN_BLOCK_M   # 2048
_GEMM_BLOCK_N = 512 * _GEMM_TILES_IN_BLOCK_N   # 1024
_GEMM_BLOCK_K = 128 * _GEMM_TILES_IN_BLOCK_K   # 1024


@nki.jit
def _matmul_lnc2(lhsT, rhs):
    """N-block-sharded matmul. With kernel[2], num_programs(0)=2 and each
    physical NC handles NUM_BLOCK_N/2 N-blocks. With kernel[1] (or no
    bracket), it falls back to one program handling all N-blocks."""
    K, M = lhsT.shape
    _, N = rhs.shape
    TILE_M = nl.tile_size.gemm_stationary_fmax
    TILE_K = nl.tile_size.pmax
    TILE_N = nl.tile_size.gemm_moving_fmax
    BLOCK_M = TILE_M * _GEMM_TILES_IN_BLOCK_M
    BLOCK_N = TILE_N * _GEMM_TILES_IN_BLOCK_N
    BLOCK_K = TILE_K * _GEMM_TILES_IN_BLOCK_K
    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N
    NUM_BLOCK_K = K // BLOCK_K

    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    n_progs = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    BLOCKS_PER_PID = NUM_BLOCK_N // n_progs
    N_START = pid * BLOCKS_PER_PID

    for ln in nl.affine_range(BLOCKS_PER_PID):
        n = N_START + ln

        result_tmps = []
        for m_idx in range(NUM_BLOCK_M):
            block_m = []
            for bm_idx in range(_GEMM_TILES_IN_BLOCK_M):
                block_n = []
                for bn_idx in range(_GEMM_TILES_IN_BLOCK_N):
                    t = nl.ndarray(shape=(TILE_M, TILE_N), dtype=lhsT.dtype, buffer=nl.sbuf)
                    nisa.memset(dst=t, value=0.0)
                    block_n.append(t)
                block_m.append(block_n)
            result_tmps.append(block_m)

        for k in nl.sequential_range(NUM_BLOCK_K):
            rhs_tiles = []
            for bk_r in range(_GEMM_TILES_IN_BLOCK_K):
                rt = nl.ndarray(shape=(TILE_K, BLOCK_N), dtype=rhs.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=rt[0:TILE_K, 0:BLOCK_N],
                              src=rhs[(_GEMM_TILES_IN_BLOCK_K * k + bk_r) * TILE_K:(_GEMM_TILES_IN_BLOCK_K * k + bk_r + 1) * TILE_K,
                                      BLOCK_N * n:BLOCK_N * (n + 1)])
                rhs_tiles.append(rt)
            for m in nl.affine_range(NUM_BLOCK_M):
                lhsT_tiles = []
                for bk_l in nl.affine_range(_GEMM_TILES_IN_BLOCK_K):
                    lt = nl.ndarray(shape=(TILE_K, BLOCK_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=lt[0:TILE_K, 0:BLOCK_M],
                                  src=lhsT[(_GEMM_TILES_IN_BLOCK_K * k + bk_l) * TILE_K:(_GEMM_TILES_IN_BLOCK_K * k + bk_l + 1) * TILE_K,
                                           BLOCK_M * m:BLOCK_M * (m + 1)])
                    lhsT_tiles.append(lt)
                for bn in nl.affine_range(_GEMM_TILES_IN_BLOCK_N):
                    for bm in nl.affine_range(_GEMM_TILES_IN_BLOCK_M):
                        rt2 = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                        for bk in nl.affine_range(_GEMM_TILES_IN_BLOCK_K):
                            nisa.nc_matmul(dst=rt2,
                                           stationary=lhsT_tiles[bk][0:TILE_K, bm * TILE_M:(bm + 1) * TILE_M],
                                           moving=rhs_tiles[bk][0:TILE_K, bn * TILE_N:(bn + 1) * TILE_N])
                        nisa.tensor_tensor(dst=result_tmps[m][bm][bn],
                                           data1=result_tmps[m][bm][bn], data2=rt2, op=nl.add)

        for m in nl.affine_range(NUM_BLOCK_M):
            for bm in nl.affine_range(_GEMM_TILES_IN_BLOCK_M):
                packed = nl.ndarray(shape=(TILE_M, BLOCK_N), dtype=nl.float32, buffer=nl.sbuf)
                for bn in nl.affine_range(_GEMM_TILES_IN_BLOCK_N):
                    nisa.tensor_copy(dst=packed[0:TILE_M, bn * TILE_N:(bn + 1) * TILE_N],
                                     src=result_tmps[m][bm][bn][0:TILE_M, 0:TILE_N])
                nisa.dma_copy(dst=result[(_GEMM_TILES_IN_BLOCK_M * m + bm) * TILE_M:(_GEMM_TILES_IN_BLOCK_M * m + bm + 1) * TILE_M,
                                         BLOCK_N * n:BLOCK_N * (n + 1)],
                              src=packed[0:TILE_M, 0:BLOCK_N])
    return result


@nki.jit
def _matmul_lnc2_fp8(lhsT, rhs, out_dtype=nl.bfloat16):
    """Same N-block sharding as _matmul_lnc2, but stationary+moving tiles are
    laid out (partition=128, 2, free) and fed to nc_matmul with
    perf_mode="double_row" — each nc_matmul consumes 256 K-elements (2x the
    bf16 path), accumulator in fp32, output cast to out_dtype on writeback."""
    K, M = lhsT.shape
    _, N = rhs.shape
    TILE_M = nl.tile_size.gemm_stationary_fmax     # 128
    TILE_K = nl.tile_size.pmax                     # 128 (partition dim)
    TILE_N = nl.tile_size.gemm_moving_fmax         # 512
    DOUBLE = 2
    K_PER_CALL = TILE_K * DOUBLE                   # 256 K-elements per nc_matmul
    BLOCK_M = TILE_M * _GEMM_TILES_IN_BLOCK_M      # 2048
    BLOCK_N = TILE_N * _GEMM_TILES_IN_BLOCK_N      # 1024
    BLOCK_K = K_PER_CALL * _GEMM_TILES_IN_BLOCK_K_FP8  # 256 * 4 = 1024
    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N
    NUM_BLOCK_K = K // BLOCK_K

    result = nl.ndarray((M, N), dtype=out_dtype, buffer=nl.shared_hbm)
    n_progs = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    BLOCKS_PER_PID = NUM_BLOCK_N // n_progs
    N_START = pid * BLOCKS_PER_PID

    for ln in nl.affine_range(BLOCKS_PER_PID):
        n = N_START + ln

        result_tmps = []
        for m_idx in range(NUM_BLOCK_M):
            block_m = []
            for bm_idx in range(_GEMM_TILES_IN_BLOCK_M):
                block_n = []
                for bn_idx in range(_GEMM_TILES_IN_BLOCK_N):
                    t = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.memset(dst=t, value=0.0)
                    block_n.append(t)
                block_m.append(block_n)
            result_tmps.append(block_m)

        for k in nl.sequential_range(NUM_BLOCK_K):
            rhs_tiles = []
            for bk_r in range(_GEMM_TILES_IN_BLOCK_K_FP8):
                rt = nl.ndarray(shape=(TILE_K, DOUBLE, BLOCK_N), dtype=rhs.dtype, buffer=nl.sbuf)
                k_start = k * BLOCK_K + bk_r * K_PER_CALL
                nisa.dma_copy(dst=rt[0:TILE_K, 0, 0:BLOCK_N],
                              src=rhs[k_start:k_start + TILE_K,
                                      BLOCK_N * n:BLOCK_N * (n + 1)])
                nisa.dma_copy(dst=rt[0:TILE_K, 1, 0:BLOCK_N],
                              src=rhs[k_start + TILE_K:k_start + K_PER_CALL,
                                      BLOCK_N * n:BLOCK_N * (n + 1)])
                rhs_tiles.append(rt)
            for m in nl.affine_range(NUM_BLOCK_M):
                lhsT_tiles = []
                for bk_l in nl.affine_range(_GEMM_TILES_IN_BLOCK_K_FP8):
                    lt = nl.ndarray(shape=(TILE_K, DOUBLE, BLOCK_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                    k_start = k * BLOCK_K + bk_l * K_PER_CALL
                    nisa.dma_copy(dst=lt[0:TILE_K, 0, 0:BLOCK_M],
                                  src=lhsT[k_start:k_start + TILE_K,
                                           BLOCK_M * m:BLOCK_M * (m + 1)])
                    nisa.dma_copy(dst=lt[0:TILE_K, 1, 0:BLOCK_M],
                                  src=lhsT[k_start + TILE_K:k_start + K_PER_CALL,
                                           BLOCK_M * m:BLOCK_M * (m + 1)])
                    lhsT_tiles.append(lt)
                for bn in nl.affine_range(_GEMM_TILES_IN_BLOCK_N):
                    for bm in nl.affine_range(_GEMM_TILES_IN_BLOCK_M):
                        rt2 = nl.ndarray(shape=(TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)
                        for bk in nl.affine_range(_GEMM_TILES_IN_BLOCK_K_FP8):
                            nisa.nc_matmul(
                                dst=rt2,
                                stationary=lhsT_tiles[bk][0:TILE_K, 0:DOUBLE,
                                                          bm * TILE_M:(bm + 1) * TILE_M],
                                moving=rhs_tiles[bk][0:TILE_K, 0:DOUBLE,
                                                     bn * TILE_N:(bn + 1) * TILE_N],
                                perf_mode="double_row",
                            )
                        nisa.tensor_tensor(dst=result_tmps[m][bm][bn],
                                           data1=result_tmps[m][bm][bn], data2=rt2, op=nl.add)

        for m in nl.affine_range(NUM_BLOCK_M):
            for bm in nl.affine_range(_GEMM_TILES_IN_BLOCK_M):
                packed = nl.ndarray(shape=(TILE_M, BLOCK_N), dtype=out_dtype, buffer=nl.sbuf)
                for bn in nl.affine_range(_GEMM_TILES_IN_BLOCK_N):
                    nisa.tensor_copy(dst=packed[0:TILE_M, bn * TILE_N:(bn + 1) * TILE_N],
                                     src=result_tmps[m][bm][bn][0:TILE_M, 0:TILE_N])
                nisa.dma_copy(dst=result[(_GEMM_TILES_IN_BLOCK_M * m + bm) * TILE_M:(_GEMM_TILES_IN_BLOCK_M * m + bm + 1) * TILE_M,
                                         BLOCK_N * n:BLOCK_N * (n + 1)],
                              src=packed[0:TILE_M, 0:BLOCK_N])
    return result


# Map operatorx dtype strings -> nki.language dtypes for the kernel's
# out_dtype kwarg (compile-time constant).
_NL_DTYPES = {
    "bf16": nl.bfloat16,
    "fp16": nl.float16,
    "fp32": nl.float32,
}


def _ceil_to(x: int, multiple: int) -> int:
    """Round x up to the next multiple of `multiple`. Used to pad gemm dims
    so any shape can run through the LNC=2 block-tiled kernel — the kernel's
    extra rows/cols see random padding and waste a small slice of compute,
    but the testlist no longer has to be filtered to aligned-only shapes."""
    return ((x + multiple - 1) // multiple) * multiple


def _prepare_gemm(op: Op) -> dict:
    a = op.args
    if a["dtype_b"] != a["dtype_a"]:
        raise UnsupportedOpError("nkilib gemm requires dtype_a == dtype_b")
    if a.get("bias") or a.get("activation"):
        raise UnsupportedOpError("nkilib gemm has no bias / activation fusion")
    M_req, N_req, K_req = a["m"], a["n"], a["k"]
    # Pad up to the next LNC=2-tiled block multiple. The kernel runs on the
    # padded shape; latency reflects that work. Dashboard FLOPs/bytes still
    # use the original (M_req, N_req, K_req) so %SOL is reported against the
    # logical op, not the padded one — that understates SOL when the pad
    # ratio is large but matches what a real model run with this op would
    # actually see latency-wise.
    M = _ceil_to(M_req, _GEMM_BLOCK_M)
    K = _ceil_to(K_req, _GEMM_BLOCK_K)
    N = _ceil_to(N_req, _GEMM_BLOCK_N * 2)
    dt_in = _resolve(a["dtype_a"])
    is_fp8 = (a["dtype_a"] == "fp8")
    out_dt_str = a.get("dtype_out", "bf16")
    if out_dt_str not in _NL_DTYPES:
        raise UnsupportedOpError(f"nkilib gemm dtype_out={out_dt_str!r} not in {list(_NL_DTYPES)}")
    out_nl_dtype = _NL_DTYPES[out_dt_str]
    # FP8 has no native randn — generate bf16 noise and cast.
    if is_fp8:
        lhsT = torch.randn(K, M, dtype=torch.bfloat16, device="cpu").to(dt_in).to(_device())
        rhs  = torch.randn(K, N, dtype=torch.bfloat16, device="cpu").to(dt_in).to(_device())
    else:
        lhsT = torch.randn(K, M, dtype=dt_in, device=_device())
        rhs  = torch.randn(K, N, dtype=dt_in, device=_device())
    return {"lhsT": lhsT, "rhs": rhs, "is_fp8": is_fp8, "out_dtype": out_nl_dtype}


def _kernel_gemm(ctx: dict) -> None:
    # kernel[2] sets LNC=2 mode: 2 SPMD programs across the 2 NC-v4 of the
    # LNC=2 logical NC. The kernel uses nl.program_id(0) to differentiate.
    # OPERATORX_NKILIB_LNC=1 forces single-NC for A/B benchmarking.
    import os as _os
    _lnc = int(_os.environ.get("OPERATORX_NKILIB_LNC", "2"))
    if ctx.get("is_fp8"):
        ctx["out"] = _matmul_lnc2_fp8[_lnc](ctx["lhsT"], ctx["rhs"], out_dtype=ctx["out_dtype"])
    else:
        ctx["out"] = _matmul_lnc2[_lnc](ctx["lhsT"], ctx["rhs"])
    xm.mark_step()


def _prepare_attention_mha(op: Op) -> dict:
    """attention_cte kernel layout (defaults: tp_q=True, tp_k=False):
      q: (batch_size * num_heads_q,  seqlen_q,  head_dim)
      k: (batch_size * num_heads_kv, head_dim,  seqlen_kv)   <- seq is LAST
      v: (batch_size * num_heads_kv, seqlen_kv, head_dim)
    GQA is expressed by the bs/bs_kv ratio (q's first dim vs k's first dim).
    """
    a = op.args
    if a.get("kv_layout", "contig") != "contig":
        raise UnsupportedOpError("nkilib attention_cte: only contiguous KV")
    for k in ("dtype_q", "dtype_k", "dtype_v"):
        if a[k] == "fp8":
            raise UnsupportedOpError(
                f"nkilib attention_cte fp8 KV not wired here ({k}={a[k]})"
            )
    if not (a["dtype_q"] == a["dtype_k"] == a["dtype_v"]):
        raise UnsupportedOpError(
            f"nkilib attention_cte needs uniform Q/K/V dtype "
            f"(got {a['dtype_q']}/{a['dtype_k']}/{a['dtype_v']})"
        )

    B, S_q, S_kv = a["batch_size"], a["seq_len_q"], a["seq_len_kv"]
    H, H_kv, D = a["num_heads"], a["num_heads_kv"], a["head_dim"]
    dt = _resolve(a["dtype_q"])
    causal = a.get("causal", True)
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(B * H,    S_q,  D,    dtype=dt, device=_device())
    k = torch.randn(B * H_kv, D,    S_kv, dtype=dt, device=_device())
    v = torch.randn(B * H_kv, S_kv, D,    dtype=dt, device=_device())
    return {"q": q, "k": k, "v": v, "scale": scale, "causal": causal}


def _kernel_attention_mha(ctx: dict) -> None:
    ctx["out"] = attention_cte(
        ctx["q"], ctx["k"], ctx["v"],
        scale=ctx["scale"],
        causal_mask=ctx["causal"],
    )
    xm.mark_step()


# ---------- Collectives via nki.collectives ----------
#
# Each @nki.jit wrapper takes the local tensor + world_size (compile-time
# constant), builds a ReplicaGroup spanning every rank, and dispatches to the
# corresponding nki.collectives primitive. Rank assignment is provided by the
# Neuron PJRT plugin at runtime from the torchrun-set RANK / WORLD_SIZE /
# MASTER_ADDR env vars.


# Collectives route through torch_xla's xm.* (see file docstring for why).


def _ensure_dist() -> None:
    """Initialize torch.distributed with the xla backend if not yet."""
    import torch.distributed as dist
    if dist.is_initialized():
        return
    import torch_xla.distributed.xla_backend  # noqa: F401  registers backend
    dist.init_process_group("xla")


# NKI collectives require tensors with access patterns of 2D–5D, so the 1D
# byte counts from the op spec are reshaped to (N, 1) for the kernel call.


def _prepare_allreduce(op: Op) -> dict:
    _ensure_dist()
    a = op.args
    dt = _resolve(a["dtype"])
    t = torch.randn(a["num_elements"], dtype=dt, device=_device())
    return {"t": t}


def _kernel_allreduce(ctx: dict) -> None:
    xm.all_reduce(xm.REDUCE_SUM, [ctx["t"]])
    # allreduce is in-place on ctx["t"]; expose it as ctx["out"] so the
    # runner's _sync(ctx) readback can force a real device-side wait.
    ctx["out"] = ctx["t"]
    xm.mark_step()


def _prepare_allgather(op: Op) -> dict:
    _ensure_dist()
    a = op.args
    dt = _resolve(a["dtype"])
    t = torch.randn(a["num_elements_per_rank"], dtype=dt, device=_device())
    return {"t": t}


def _kernel_allgather(ctx: dict) -> None:
    ctx["out"] = xm.all_gather(ctx["t"], dim=0)
    xm.mark_step()


def _prepare_reduce_scatter(op: Op) -> dict:
    _ensure_dist()
    a = op.args
    ws = int(a["world_size"])
    if a["num_elements"] % ws != 0:
        raise UnsupportedOpError(
            f"reduce_scatter requires num_elements divisible by world_size "
            f"({a['num_elements']} % {ws} != 0)"
        )
    dt = _resolve(a["dtype"])
    t = torch.randn(a["num_elements"], dtype=dt, device=_device())
    return {"t": t, "ws": ws}


def _kernel_reduce_scatter(ctx: dict) -> None:
    ctx["out"] = xm.reduce_scatter(
        xm.REDUCE_SUM, ctx["t"], scale=1.0, scatter_dim=0, shard_count=ctx["ws"],
    )
    xm.mark_step()


def _prepare_alltoall(op: Op) -> dict:
    # On Trainium3 with neuronx-cc 2.24 (the only available build in Neuron
    # 2.29.0), both lowerings of all_to_all fail:
    #   - xm.all_to_all with default groups -> `NCC_IVRF100` on the resulting
    #     `mhlo.all-to-all` HLO with `replica_groups={}`.
    #   - xm.all_to_all with explicit groups -> compiler reports "CustomCallOp
    #     unsupported target: mhlo.all_to_all" (no lowering at all).
    #   - nki.collectives.all_to_all -> runtime asserts ENC_ALG_MESH (broken).
    # No user-side workaround is possible until the compiler supports it. We
    # surface this as Unsupported so the sweep reports a clear reason instead
    # of an opaque ValueError mid-compile.
    raise UnsupportedOpError(
        "alltoall: neuronx-cc 2.24 (Neuron SDK 2.29) has no working lowering "
        "for mhlo.all_to_all on Trainium3. Both xm.all_to_all paths (empty + "
        "explicit replica_groups) and nki.collectives.all_to_all fail. Needs "
        "a future SDK release. Tracking: aws-neuron/aws-neuron-sdk."
    )


def _kernel_alltoall(ctx: dict) -> None:  # pragma: no cover  unreachable
    raise UnsupportedOpError("alltoall not reachable")


IMPLS = [
    BackendImpl(op_type="gemm", prepare=_prepare_gemm, kernel=_kernel_gemm),
    BackendImpl(op_type="attention_mha", prepare=_prepare_attention_mha, kernel=_kernel_attention_mha),
    BackendImpl(op_type="allreduce", prepare=_prepare_allreduce, kernel=_kernel_allreduce),
    BackendImpl(op_type="allgather", prepare=_prepare_allgather, kernel=_kernel_allgather),
    BackendImpl(op_type="reduce_scatter", prepare=_prepare_reduce_scatter, kernel=_kernel_reduce_scatter),
    BackendImpl(op_type="alltoall", prepare=_prepare_alltoall, kernel=_kernel_alltoall),
]
