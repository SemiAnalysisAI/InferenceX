# pyright: reportAttributeAccessIssue=false
"""Trainium3 quad MX (MXFP8) matmul throughput at smaller K, untiled vs row-tiled (NKI 0.7, LNC=1).

nc_matmul_mx contracts 4 packed fp8 values per partition, so a Kp-partition matmul has K = 4 Kp.
Kp = 128 (K = 512) uses the whole 128-row PE array and is measured in ../trn3_quad_mx_matmul. Here
Kp = 32 (K = 128) or 64 (K = 256), every matmul loading a new stationary (alternating stationary):

  ls_mx_alt_untiled  matmuls on partitions 0..Kp-1 with no tiling arguments: one small matmul at a time,
                     PE rows Kp..127 idle. Matmul i uses stationary i % 2.

  ls_mx_alt_tiled    row tiling: the 128 PE rows split into T = 128 / Kp row tiles (tile_size (Kp, 128)).
                     Matmul i goes to tile t = i % T (tile_position (t Kp, 0)), with its stationary, moving
                     tile and scales at SBUF partitions t Kp .. t Kp + Kp - 1, and within its tile it
                     alternates between 2 stationaries ((i // T) % 2). Each tile has its own moving data.

Every matmul writes [M, N] fp32 to PSUM tile i % P_T (accumulate=False), P_T = 8 up to N = 512. Operands
are quantized from one [128, 4 (M + N + W)] fp32 input: two 128-partition stationaries, one moving tile and
(with a warm-up) a 512-column warm-up moving tile, all via nisa.quantize_mx on 128 partitions; a matmul's
operands are partition slices of those (scales sliced the same way, which keeps each 32-partition
quadrant's scales with its data). Warm-up (WARMUP_MM > 0): fixed-stationary full-array (128-partition,
untiled) matmuls with N = 512 right before the burst, in the same no_reorder block, to release TensorE's
activity throttle; ../trn3_matmul/parse.py drops them (meta.json warmup_mm). NUM_MM, M_DIM, WARMUP_MM and
KP are module globals set by get_kernel.
"""

import nki
import nki.isa as nisa
import nki.language as nl

NUM_MM = 1024
WARMUP_MM = 0
WARMUP_N = 512
M_DIM = 128
KP = 32  # partitions per matmul (K = 4 KP); set by get_kernel
PSUM_BANKS = 8
Q = 4
MX = nl.float8_e4m3fn_x4
SCALE = nl.uint8


def psum_tiles(N):
    """PSUM tiles to rotate through: each [M, N] fp32 tile takes ceil(4 N / 2 KiB) banks of the 8."""
    return max(1, PSUM_BANKS // -(-4 * N // 2048))


def _quantize(raw, start, cols):
    P = raw.shape[0]
    src = nl.ndarray([P, Q * cols], dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=src, src=raw[:, Q * start:Q * (start + cols)], engine=nisa.engine.vector)
    data = nl.ndarray([P, cols], dtype=MX, buffer=nl.sbuf)
    scale = nl.ndarray([P, cols], dtype=SCALE, buffer=nl.sbuf)
    nisa.quantize_mx(dst=data, src=src, dst_scale=scale)
    return data, scale


def _burst(in_tensor, tiled):
    P = in_tensor.shape[0]
    M = M_DIM
    W = WARMUP_N if WARMUP_MM else 0
    N = in_tensor.shape[1] // Q - M - W
    T = P // KP if tiled else 1
    out_tensor = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.shared_hbm)

    raw = nl.ndarray([P, Q * (M + N + W)], dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=raw, src=in_tensor)
    stats = [_quantize(raw, 0, M), _quantize(raw, 0, M)]
    mov, mov_s = _quantize(raw, M, N)
    if W:
        wmov, wmov_s = _quantize(raw, M + N, W)

    width = max(N, W)
    n_ps = psum_tiles(width)
    ps = []
    for t in range(n_ps):
        ps.append(nl.ndarray([M, width], dtype=nl.float32, buffer=nl.psum))

    with nl.no_reorder():
        for i in nl.static_range(WARMUP_MM):
            nisa.nc_matmul_mx(dst=ps[i % n_ps][:, 0:W], stationary=stats[0][0], moving=wmov,
                              stationary_scale=stats[0][1], moving_scale=wmov_s, accumulate=False)
        for i in nl.static_range(NUM_MM):
            t = i % T
            st, st_s = stats[(i // T) % 2]
            r0 = t * KP
            if tiled:
                nisa.nc_matmul_mx(dst=ps[i % n_ps][:, 0:N], stationary=st[r0:r0 + KP, :], moving=mov[r0:r0 + KP, :],
                                  stationary_scale=st_s[r0:r0 + KP, :], moving_scale=mov_s[r0:r0 + KP, :],
                                  accumulate=False, tile_position=(r0, 0), tile_size=(KP, 128))
            else:
                nisa.nc_matmul_mx(dst=ps[i % n_ps][:, 0:N], stationary=st[0:KP, :], moving=mov[0:KP, :],
                                  stationary_scale=st_s[0:KP, :], moving_scale=mov_s[0:KP, :], accumulate=False)

    acc = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.sbuf)
    tmp = nl.ndarray([M, N], dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    for t in range(n_ps):
        nisa.tensor_copy(dst=tmp, src=ps[t][:, 0:N], engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=tmp, op=nl.add)
    nl.store(out_tensor, value=acc)
    return out_tensor


@nki.jit
def ls_mx_alt_untiled_kernel(in_tensor):
    return _burst(in_tensor, False)


@nki.jit
def ls_mx_alt_tiled_kernel(in_tensor):
    return _burst(in_tensor, True)


KERNELS = {"ls_mx_alt_untiled": ls_mx_alt_untiled_kernel, "ls_mx_alt_tiled": ls_mx_alt_tiled_kernel}


def get_kernel(name, num_mm, m=128, warmup=0, kp=32):
    global NUM_MM, M_DIM, WARMUP_MM, KP
    NUM_MM = num_mm
    M_DIM = m
    WARMUP_MM = warmup
    KP = kp
    return KERNELS[name]
