from __future__ import annotations

import flashinfer
import torch

from operatorx.core import BackendImpl, Op, UnsupportedOpError, lookup_versions


def versions() -> dict[str, str]:
    return lookup_versions("flashinfer", "torch")

_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

# nvfp4 vs mxfp4 differ only in block size: 16 vs 32. mm_fp4 selects via flags.
_FP4_FAMILIES = {"nvfp4": (16, True), "mxfp4": (32, False)}


def _prepare_gemm(op: Op) -> dict:
    """flashinfer GEMMs by dtype:
      - bf16/bf16 -> flashinfer.mm_bf16 (native bias supported)
      - fp8/fp8   -> flashinfer.mm_fp8 (no bias param)
      - nvfp4/nvfp4 -> flashinfer.mm_fp4 (no bias param)
      - mxfp4/mxfp4 -> flashinfer.mm_fp4 (no bias param)

    Mixed-precision (e.g. bf16 act × mxfp4/int4 weight) and activation fusion
    are NOT exposed by any flashinfer API at this layer — those land in
    UnsupportedOpError so the dashboard reflects real backend coverage.
    Quantization happens in prepare (outside the timed window).
    """
    a = op.args
    if a.get("activation") is not None:
        raise UnsupportedOpError(
            f"flashinfer gemm: no native activation fusion in mm_bf16/mm_fp8/mm_fp4; "
            f"got activation={a['activation']!r}"
        )
    if a["dtype_a"] != a["dtype_b"]:
        raise UnsupportedOpError(
            f"flashinfer gemm: no native mixed-precision GEMM (a={a['dtype_a']} b={a['dtype_b']})"
        )
    if a["dtype_out"] not in _DTYPES:
        raise UnsupportedOpError(
            f"flashinfer gemm out_dtype must be bf16/fp16; got {a['dtype_out']!r}"
        )
    out_dt = _DTYPES[a["dtype_out"]]
    m, n, k = a["m"], a["n"], a["k"]
    has_bias = bool(a.get("bias"))
    family = a["dtype_a"]

    if family == "bf16":
        A = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
        bias = torch.randn(n, dtype=out_dt, device="cuda") if has_bias else None
        return {"kind": "bf16", "A": A, "B": B, "bias": bias, "out_dt": out_dt}

    if family == "fp8":
        # flashinfer.mm_fp8 is trtllm_low_latency which expects a block-scaled
        # 3D weight layout — not a plain MxK @ KxN matmul. flashinfer's only
        # plain-FP8 path is gemm_fp8_nt_groupwise which also wants block scales.
        # Plain FP8 GEMM is covered by torch._scaled_mm / deepgemm / sglang.
        raise UnsupportedOpError(
            "flashinfer has no plain FP8 GEMM (mm_fp8 needs block-scaled 3D B); "
            "use the torch / deepgemm / sglang backends for FP8 GEMM"
        )

    if family not in _FP4_FAMILIES:
        raise UnsupportedOpError(
            f"flashinfer gemm: unsupported dtype family {family!r} "
            f"(supported: bf16, fp8, nvfp4, mxfp4)"
        )
    if has_bias:
        raise UnsupportedOpError(
            f"flashinfer mm_fp4 does not expose a bias parameter; got bias=True for {family!r}"
        )
    block_size, use_nvfp4 = _FP4_FAMILIES[family]
    # Pad K and N up to a multiple of block_size; the kernel requires it for
    # the block-scaled FP4 layout.
    k = ((k + block_size - 1) // block_size) * block_size
    n = ((n + block_size - 1) // block_size) * block_size
    a_hi = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    b_hi = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    g_a = (448 * 6) / a_hi.float().abs().nan_to_num().max()
    g_b = (448 * 6) / b_hi.float().abs().nan_to_num().max()
    alpha = (1.0 / (g_a * g_b)).reshape(1).contiguous()
    if use_nvfp4:
        a_q, a_sf = flashinfer.nvfp4_quantize(a_hi, g_a, sf_vec_size=block_size, do_shuffle=False)
        b_q, b_sf = flashinfer.nvfp4_quantize(b_hi, g_b, sf_vec_size=block_size, do_shuffle=False)
        backend = "cute-dsl"
    else:
        if not hasattr(flashinfer, "mxfp4_quantize"):
            raise UnsupportedOpError("flashinfer has no mxfp4_quantize in this build")
        a_q, a_sf = flashinfer.mxfp4_quantize(a_hi)
        b_q, b_sf = flashinfer.mxfp4_quantize(b_hi)
        backend = "cudnn"
        with flashinfer.autotune(tune_mode=True):
            flashinfer.mm_fp4(
                a_q, b_q.T, a_sf, b_sf.T, alpha=alpha,
                block_size=block_size, use_nvfp4=use_nvfp4,
                out_dtype=out_dt, backend=backend,
            )
    return {
        "kind": "fp4",
        "a": a_q, "b": b_q.T, "a_sf": a_sf, "b_sf": b_sf.T,
        "alpha": alpha, "block_size": block_size, "use_nvfp4": use_nvfp4,
        "out_dt": out_dt, "backend": backend,
    }


def _kernel_gemm(ctx: dict) -> None:
    kind = ctx["kind"]
    if kind == "bf16":
        ctx["out"] = flashinfer.mm_bf16(
            ctx["A"], ctx["B"], bias=ctx["bias"], out_dtype=ctx["out_dt"],
        )
    else:  # fp4
        ctx["out"] = flashinfer.mm_fp4(
            ctx["a"], ctx["b"], ctx["a_sf"], ctx["b_sf"], alpha=ctx["alpha"],
            block_size=ctx["block_size"], use_nvfp4=ctx["use_nvfp4"],
            out_dtype=ctx["out_dt"], backend=ctx["backend"],
        )


def _routing_logits(nt: int, ne: int, distribution: str, device: str = "cuda",
                    dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Generate routing logits matching the requested expert distribution.

    The kernel's downstream topk picks the dominant experts; biasing the logits
    here is equivalent to biasing the router's output.
    """
    base = torch.randn(nt, ne, dtype=dtype, device=device)
    if distribution == "uniform":
        return base
    if distribution == "zipf":
        # Boost the first few experts. log(1/(rank+1)) for s=1.
        rank = torch.arange(1, ne + 1, dtype=torch.float32, device=device)
        bias = -torch.log(rank).to(dtype)
        return base + bias  # broadcast over tokens
    if distribution == "single_hot":
        base[:, 0] += 10.0
        return base
    raise UnsupportedOpError(
        f"unknown expert_distribution={distribution!r} "
        f"(expected 'uniform' | 'zipf' | 'single_hot')"
    )


def _prepare_moe_gemm(op: Op) -> dict:
    """Per-rank routed-expert GEMM via flashinfer's fp4 MoE kernels.

    Routes by weight dtype:
      - mxfp4 weights + bf16 acts -> trtllm_fp4_block_scale_moe (bf16 acts work here)
      - nvfp4 weights + bf16/nvfp4 acts -> cute_dsl_fused_moe_nvfp4
      - fp8 weights + fp8 acts -> trtllm_fp8_block_scale_moe (DeepSeek block-FP8)
    """
    a = op.args
    # trtllm fp4/fp8 block-scale kernels and cute_dsl nvfp4 all use 128-wide
    # K-tiles; flashinfer's tactic table has no entry when hidden % 128 != 0
    # (e.g. gptoss h=2880 → NO_VALID_CONFIG on mxfp4, illegal address on nvfp4).
    if a["hidden"] % 128 != 0:
        raise UnsupportedOpError(
            f"flashinfer moe_gemm kernels require hidden % 128 == 0; got hidden={a['hidden']}"
        )
    if a["num_tokens"] == 0:
        raise UnsupportedOpError("flashinfer moe_gemm requires num_tokens > 0")
    if a["dtype_weight"] == "fp8" and a["dtype_act"] == "fp8":
        return _prepare_moe_gemm_fp8(op)
    if a["dtype_weight"] not in ("nvfp4", "mxfp4"):
        raise UnsupportedOpError(
            f"flashinfer moe_gemm only implements nvfp4/mxfp4/fp8 weights; "
            f"got dtype_weight={a['dtype_weight']!r}"
        )
    # cute_dsl path takes pre-quantized nvfp4 acts and we generate them in
    # prepare, so dtype_act in {bf16, nvfp4} both route there. trtllm path
    # (mxfp4 weights) only accepts bf16 acts.
    if a["dtype_weight"] == "mxfp4" and a["dtype_act"] != "bf16":
        raise UnsupportedOpError(
            f"flashinfer mxfp4 path requires bf16 acts; got dtype_act={a['dtype_act']!r}"
        )
    if a["dtype_act"] not in ("bf16", "nvfp4"):
        raise UnsupportedOpError(
            f"flashinfer moe_gemm only supports bf16/nvfp4 acts; got dtype_act={a['dtype_act']!r}"
        )
    nt, h = a["num_tokens"], a["hidden"]
    ne, k = a["num_experts"], a["top_k"]
    ep = a.get("expert_parallel_size", 1)
    routed_tp = a.get("routed_tensor_parallel_size", 1)
    shared_tp = a.get("shared_tensor_parallel_size", 1)
    n_shared = a.get("n_shared_experts", 0)
    # Single-rank kernel: each rank's slice is (intermediate / routed_tp)
    # along intermediate, (num_experts / ep) along routed experts, plus
    # n_shared always-on shared experts appended as extra groups.
    if a["intermediate"] % max(1, routed_tp) != 0:
        raise UnsupportedOpError(
            f"intermediate={a['intermediate']} not divisible by routed_tp={routed_tp}"
        )
    im = a["intermediate"] // max(1, routed_tp)
    local_routed = ne // max(1, ep)
    # Shared experts run as a separate dense FP4 MLP after the routed kernel
    # (matches real production: trtllm routed kernel + dense shared MLP +
    # accumulate). Fusing via routing biases interacts badly with trtllm's
    # softmax-then-topk and suppresses routed compute.
    local_total = local_routed
    use_nvfp4 = (a["dtype_weight"] == "nvfp4")
    block_size = 16 if use_nvfp4 else 32

    for name, dim in (("hidden", h), ("intermediate", im)):
        if dim % block_size != 0:
            raise UnsupportedOpError(
                f"flashinfer fp4 MoE needs {name}={dim} divisible by block_size={block_size}"
            )
    if local_routed < 1:
        raise UnsupportedOpError(f"local_routed={local_routed} < 1 (num_experts={ne} ep={ep})")

    g_w = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    if use_nvfp4:
        def quant(x):
            return flashinfer.nvfp4_quantize(x, g_w, sf_vec_size=block_size, do_shuffle=False)
    else:
        def quant(x):
            return flashinfer.mxfp4_quantize(x)
    # gate_up: (local_total, 2*im, h); down: (local_total, h, im).
    w1_hi = torch.randn(local_total * 2 * im, h, dtype=torch.bfloat16, device="cuda")
    w2_hi = torch.randn(local_total * h, im, dtype=torch.bfloat16, device="cuda")
    w1_q, w1_sf = quant(w1_hi)
    w2_q, w2_sf = quant(w2_hi)
    # trtllm_fp4_block_scale_moe wants scales as float8_e4m3fn. Both mxfp4 and
    # nvfp4 quantizers return uint8 — reinterpret-view to fp8 (same bytes).
    if not use_nvfp4:
        w1_sf = w1_sf.view(torch.float8_e4m3fn)
        w2_sf = w2_sf.view(torch.float8_e4m3fn)
    # The swizzled 128x4 scale layout pads BOTH dimensions: rows to multiples
    # of 128 and columns to multiples of 4. Reshape to 3D using the actual
    # column count returned by the quantizer (which may be > h//block_size or
    # im//block_size after padding).
    w1_q = w1_q.view(local_total, 2 * im, w1_q.size(-1))
    w1_sf = w1_sf.view(local_total, 2 * im, w1_sf.size(-1))
    w2_q = w2_q.view(local_total, h, w2_q.size(-1))
    w2_sf = w2_sf.view(local_total, h, w2_sf.size(-1))

    distribution = a.get("expert_distribution", "uniform")

    # Shared experts: per-expert FP4 dense MLP weights sized for the shared TP
    # slice (intermediate / shared_tp). Run as plain mm_fp4 calls after the
    # routed kernel and accumulate. Per-rank: n_shared two-GEMM MLPs.
    shared_pack: dict | None = None
    if n_shared > 0:
        if a["intermediate"] % max(1, shared_tp) != 0:
            raise UnsupportedOpError(
                f"intermediate={a['intermediate']} not divisible by shared_tp={shared_tp}"
            )
        im_s = a["intermediate"] // max(1, shared_tp)
        if im_s % block_size != 0:
            raise UnsupportedOpError(
                f"shared MLP intermediate={im_s} not divisible by block_size={block_size}"
            )
        shared_experts = []
        for _ in range(n_shared):
            sw1_hi = torch.randn(2 * im_s, h, dtype=torch.bfloat16, device="cuda")
            sw2_hi = torch.randn(h, im_s, dtype=torch.bfloat16, device="cuda")
            sw1_q, sw1_sf = quant(sw1_hi)
            sw2_q, sw2_sf = quant(sw2_hi)
            if not use_nvfp4:
                sw1_sf = sw1_sf.view(torch.float8_e4m3fn)
                sw2_sf = sw2_sf.view(torch.float8_e4m3fn)
            shared_experts.append({"w1": sw1_q, "w1_sf": sw1_sf, "w2": sw2_q, "w2_sf": sw2_sf})
        shared_pack = {
            "experts": shared_experts,
            "use_nvfp4": use_nvfp4,
            "block_size": block_size,
            "alpha": torch.tensor(1.0, dtype=torch.float32, device="cuda"),
        }

    if not use_nvfp4:
        hidden_states = torch.randn(nt, h, dtype=torch.bfloat16, device="cuda")
        routing_logits = _routing_logits(nt, ne, distribution)
        return {
            "path": "trtllm",
            "routing_logits": routing_logits, "hidden_states": hidden_states,
            "gemm1_weights": w1_q, "gemm1_weights_scale": w1_sf,
            "gemm2_weights": w2_q, "gemm2_weights_scale": w2_sf,
            "num_experts": ne, "top_k": k, "intermediate_size": im,
            "local_expert_offset": 0, "local_num_experts": local_total,
            "shared": shared_pack,
            "num_tokens_for_tune": nt,
        }

    x_hi = torch.randn(nt, h, dtype=torch.bfloat16, device="cuda")
    # nvfp4_quantize returns scale factors in the swizzled 128x4 layout (rows
    # padded to multiples of 128). The kernel expects that native layout — do
    # not .view() it back to the dense (nt, k//block_size) shape.
    x_q, x_sf = flashinfer.nvfp4_quantize(x_hi, g_w, sf_vec_size=block_size, do_shuffle=False)
    router_logits = _routing_logits(nt, ne, distribution)
    token_final_scales, token_selected_experts = torch.topk(router_logits.float(), k, dim=-1)
    token_selected_experts = token_selected_experts.to(torch.int32)
    token_final_scales = torch.softmax(token_final_scales, dim=-1).to(torch.bfloat16)
    w1_alpha = torch.ones(local_total, dtype=torch.float32, device="cuda")
    w2_alpha = torch.ones(local_total, dtype=torch.float32, device="cuda")
    fc2_input_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    return {
        "path": "cute_dsl",
        "x": x_q, "x_sf": x_sf,
        "x_hi": x_hi,
        "token_selected_experts": token_selected_experts,
        "token_final_scales": token_final_scales,
        "w1_weight": w1_q, "w1_weight_sf": w1_sf, "w1_alpha": w1_alpha,
        "w2_weight": w2_q, "w2_weight_sf": w2_sf, "w2_alpha": w2_alpha,
        "fc2_input_scale": fc2_input_scale,
        "num_experts": ne, "top_k": k,
        "num_local_experts": local_total, "local_expert_offset": 0,
        "shared": shared_pack,
    }


def _prepare_moe_gemm_fp8(op: Op) -> dict:
    """fp8/fp8 path via DeepSeek block-FP8 (128x128 weight blocks, per-token-128 act scale).

    Both the routed kernel and the shared MLP are dispatched through a single
    `flashinfer.trtllm_fp8_block_scale_moe` call. For the shared expert we pad
    `num_experts` to 4 (the kernel routing checks `num_experts % 4 == 0`),
    route all tokens to expert 0 with top_k=1, and ignore the other 3 dummy
    slots — that gives us the fused gate_up + silu + down kernel in a single
    launch instead of chaining gemm_fp8_nt_groupwise → silu → quant → groupwise.

    Returns `_call` (routed) and `_call_shared` (shared, when n_shared>0) closures
    that capture all prep temporaries. Without the pinning, intermediate bf16
    tensors (x_bf16 + per-iteration randn fed to per_block_cast_to_fp8) get freed
    when this function returns; the kernel then allocates its workspace into
    those freed blocks at addresses that fail the kernel's launch-param check
    (cudaLaunchKernelEx → "operation not supported on global/shared address
    space" at trtllm_fused_moe_dev_kernel.cu:376).
    """
    from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8
    from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8
    a = op.args
    nt, h = a["num_tokens"], a["hidden"]
    intermediate = a["intermediate"]
    ne, k = a["num_experts"], a["top_k"]
    ep = a.get("expert_parallel_size", 1)
    routed_tp = a.get("routed_tensor_parallel_size", 1)
    shared_tp = a.get("shared_tensor_parallel_size", 1)
    n_shared = a.get("n_shared_experts", 0)
    distribution = a.get("expert_distribution", "uniform")

    BLOCK = 128
    if intermediate % max(1, routed_tp) != 0:
        raise UnsupportedOpError(f"intermediate={intermediate} not divisible by routed_tp={routed_tp}")
    im = intermediate // max(1, routed_tp)
    local_routed = ne // max(1, ep)
    for name, dim in (("hidden", h), ("intermediate/routed_tp", im)):
        if dim % BLOCK != 0:
            raise UnsupportedOpError(f"fp8 path requires {name}={dim} divisible by {BLOCK}")
    if local_routed < 1:
        raise UnsupportedOpError(f"local_routed={local_routed} < 1 (num_experts={ne} ep={ep})")

    pinned: list = []  # keep every intermediate bf16 tensor alive across kernel calls
    x_bf16 = torch.randn(nt, h, dtype=torch.bfloat16, device="cuda")
    pinned.append(x_bf16)
    # per_token_group_quant_fp8 returns (M, K//128); trtllm_fp8_block_scale_moe
    # expects (K//128, M) — transpose + .contiguous() to materialize the layout.
    x_fp8, x_scale_mk = per_token_group_quant_fp8(x_bf16, BLOCK)
    x_scale = x_scale_mk.T.contiguous()

    w1_q = torch.empty(local_routed, 2 * im, h, dtype=torch.float8_e4m3fn, device="cuda")
    w1_sf = torch.empty(local_routed, 2 * im // BLOCK, h // BLOCK, dtype=torch.float32, device="cuda")
    w2_q = torch.empty(local_routed, h, im, dtype=torch.float8_e4m3fn, device="cuda")
    w2_sf = torch.empty(local_routed, h // BLOCK, im // BLOCK, dtype=torch.float32, device="cuda")
    for i in range(local_routed):
        tmp1 = torch.randn(2 * im, h, dtype=torch.bfloat16, device="cuda")
        pinned.append(tmp1)
        w1_q[i], w1_sf[i] = per_block_cast_to_fp8(tmp1)
        tmp2 = torch.randn(h, im, dtype=torch.bfloat16, device="cuda")
        pinned.append(tmp2)
        w2_q[i], w2_sf[i] = per_block_cast_to_fp8(tmp2)

    routing_logits = _routing_logits(nt, ne, distribution)
    nt_for_tune = 1
    while nt_for_tune < max(1, nt):
        nt_for_tune <<= 1

    # Shared MLP via fused trtllm_fp8_block_scale_moe with num_experts padded to
    # 4 (kernel asserts num_experts % 4 == 0), top_k=1, all tokens routed to
    # expert 0. The other 3 expert slots are dummy weights that never get touched.
    call_shared = None
    if n_shared > 0:
        if intermediate % max(1, shared_tp) != 0:
            raise UnsupportedOpError(f"intermediate={intermediate} not divisible by shared_tp={shared_tp}")
        im_s = intermediate // max(1, shared_tp)
        if im_s % BLOCK != 0:
            raise UnsupportedOpError(f"fp8 shared MLP requires intermediate/shared_tp={im_s} divisible by {BLOCK}")
        pad_ne = max(4, ((n_shared + 3) // 4) * 4)
        sw1 = torch.empty(pad_ne, 2 * im_s, h, dtype=torch.float8_e4m3fn, device="cuda")
        sw1_sf = torch.ones(pad_ne, 2 * im_s // BLOCK, h // BLOCK, dtype=torch.float32, device="cuda")
        sw2 = torch.empty(pad_ne, h, im_s, dtype=torch.float8_e4m3fn, device="cuda")
        sw2_sf = torch.ones(pad_ne, h // BLOCK, im_s // BLOCK, dtype=torch.float32, device="cuda")
        # Routing: large logit on expert 0 forces every token through it
        shared_logits = torch.full((nt, pad_ne), -100.0, dtype=torch.bfloat16, device="cuda")
        shared_logits[:, 0] = 100.0

        def _call_shared(_pin=pinned):
            return flashinfer.trtllm_fp8_block_scale_moe(
                routing_logits=shared_logits, routing_bias=None,
                hidden_states=x_fp8, hidden_states_scale=x_scale,
                gemm1_weights=sw1, gemm1_weights_scale=sw1_sf,
                gemm2_weights=sw2, gemm2_weights_scale=sw2_sf,
                num_experts=pad_ne, top_k=1, n_group=None, topk_group=None,
                intermediate_size=im_s,
                local_expert_offset=0, local_num_experts=pad_ne,
                routed_scaling_factor=None,
                routing_method_type=1,
                tune_max_num_tokens=nt_for_tune,
            )
        call_shared = _call_shared

    def _call_routed():
        # `pinned` must be referenced here so the closure cell keeps it alive
        # (and with it, x_bf16 + per-iteration randn temps). Without this
        # reference the kernel reuses freed allocator blocks and fails.
        _ = pinned
        return flashinfer.trtllm_fp8_block_scale_moe(
            routing_logits=routing_logits, routing_bias=None,
            hidden_states=x_fp8, hidden_states_scale=x_scale,
            gemm1_weights=w1_q, gemm1_weights_scale=w1_sf,
            gemm2_weights=w2_q, gemm2_weights_scale=w2_sf,
            num_experts=ne, top_k=k, n_group=None, topk_group=None,
            intermediate_size=im,
            local_expert_offset=0, local_num_experts=local_routed,
            routed_scaling_factor=None,
            # routing_method_type=1 (Renormalize) picks a robust tactic across
            # all (local_num_experts, num_tokens) we benchmark. The default
            # rmt=0 (Default/TopK-Softmax) crashes for ne_local=48 at nt=1
            # (cudaLaunchKernelEx launch reject) and for ne_local∈{256,512}
            # at nt=256 (trtllm_batched_gemm_runner runtime error). We're
            # benchmarking the routed-GEMM cost, not the routing softmax
            # itself, so the choice of routing post-process is incidental.
            routing_method_type=1,
            tune_max_num_tokens=nt_for_tune,
        )

    return {
        "path": "trtllm_fp8",
        "_call": _call_routed,
        "_call_shared": call_shared,
    }


def _shared_mlp_fp4(x_bf16: torch.Tensor, shared_pack: dict) -> torch.Tensor:
    """Run the shared expert(s) as plain FP4 GEMMs and return the bf16 sum."""
    from flashinfer.gemm import mm_fp4
    block_size = shared_pack["block_size"]
    use_nvfp4 = shared_pack["use_nvfp4"]
    alpha = shared_pack["alpha"]
    g_w = torch.tensor([1.0], dtype=torch.float32, device=x_bf16.device)

    if use_nvfp4:
        x_q, x_sf = flashinfer.nvfp4_quantize(x_bf16, g_w, sf_vec_size=block_size, do_shuffle=False)
    else:
        x_q, x_sf = flashinfer.mxfp4_quantize(x_bf16)
        x_sf = x_sf.view(torch.float8_e4m3fn)

    out = None
    for expert in shared_pack["experts"]:
        gate_up = mm_fp4(
            x_q, expert["w1"].T, x_sf, expert["w1_sf"].T, alpha,
            out_dtype=torch.bfloat16, block_size=block_size, use_nvfp4=use_nvfp4,
        )
        gate, up_proj = gate_up.chunk(2, dim=-1)
        mid = torch.nn.functional.silu(gate) * up_proj
        if use_nvfp4:
            mid_q, mid_sf = flashinfer.nvfp4_quantize(mid, g_w, sf_vec_size=block_size, do_shuffle=False)
        else:
            mid_q, mid_sf = flashinfer.mxfp4_quantize(mid)
            mid_sf = mid_sf.view(torch.float8_e4m3fn)
        down = mm_fp4(
            mid_q, expert["w2"].T, mid_sf, expert["w2_sf"].T, alpha,
            out_dtype=torch.bfloat16, block_size=block_size, use_nvfp4=use_nvfp4,
        )
        out = down if out is None else out + down
    return out


def _kernel_moe_gemm(ctx: dict) -> None:
    path = ctx["path"]
    # Match sglang's pattern: lock the autotuner's tuning window to this
    # shape's M via tune_max_num_tokens=next_power_of_2(num_tokens). Without
    # this, the kernel keeps the default tune_max_num_tokens=8192 and the
    # autotuner cache is keyed for a different M range than the runtime tactic
    # selector picks, producing CUDA illegal memory access for shapes outside
    # the default tuning window.
    nt_for_tune = 1
    while nt_for_tune < max(1, ctx.get("num_tokens_for_tune", 1)):
        nt_for_tune <<= 1
    if path == "trtllm":
        out = flashinfer.trtllm_fp4_block_scale_moe(
            routing_logits=ctx["routing_logits"], routing_bias=None,
            hidden_states=ctx["hidden_states"], hidden_states_scale=None,
            gemm1_weights=ctx["gemm1_weights"], gemm1_weights_scale=ctx["gemm1_weights_scale"],
            gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None,
            gemm2_weights=ctx["gemm2_weights"], gemm2_weights_scale=ctx["gemm2_weights_scale"],
            gemm2_bias=None,
            output1_scale_scalar=None, output1_scale_gate_scalar=None, output2_scale_scalar=None,
            num_experts=ctx["num_experts"], top_k=ctx["top_k"],
            n_group=None, topk_group=None,
            intermediate_size=ctx["intermediate_size"],
            local_expert_offset=ctx["local_expert_offset"], local_num_experts=ctx["local_num_experts"],
            routed_scaling_factor=None,
            tune_max_num_tokens=nt_for_tune,
        )
    elif path == "trtllm_fp8":
        # Closure call: keeps prep-time temporaries alive across the kernel
        # launch so the workspace allocator doesn't reuse freed blocks.
        out = ctx["_call"]()
    else:  # cute_dsl
        out = flashinfer.cute_dsl_fused_moe_nvfp4(
            x=ctx["x"], x_sf=ctx["x_sf"],
            token_selected_experts=ctx["token_selected_experts"],
            token_final_scales=ctx["token_final_scales"],
            w1_weight=ctx["w1_weight"], w1_weight_sf=ctx["w1_weight_sf"], w1_alpha=ctx["w1_alpha"],
            fc2_input_scale=ctx["fc2_input_scale"],
            w2_weight=ctx["w2_weight"], w2_weight_sf=ctx["w2_weight_sf"], w2_alpha=ctx["w2_alpha"],
            num_experts=ctx["num_experts"], top_k=ctx["top_k"],
            num_local_experts=ctx["num_local_experts"], local_expert_offset=ctx["local_expert_offset"],
        )
    out = out[0] if isinstance(out, (list, tuple)) else out
    if path == "trtllm_fp8":
        cs = ctx.get("_call_shared")
        if cs is not None:
            shared_out = cs()
            shared_out = shared_out[0] if isinstance(shared_out, (list, tuple)) else shared_out
            out = out + shared_out
    elif ctx.get("shared") is not None:
        x_for_shared = ctx["hidden_states"] if path == "trtllm" else ctx["x_hi"]
        out = out + _shared_mlp_fp4(x_for_shared, ctx["shared"])
    ctx["out"] = out


def _prepare_attention_mha(op: Op) -> dict:
    a = op.args
    if a["batch_size"] != 1:
        raise NotImplementedError("flashinfer attention_mha currently supports batch_size=1")
    if a.get("kv_layout", "contig") != "contig":
        raise NotImplementedError("flashinfer attention_mha here only handles contiguous KV")

    S_q, S_kv = a["seq_len_q"], a["seq_len_kv"]
    H, H_kv, D = a["num_heads"], a["num_heads_kv"], a["head_dim"]
    if a["dtype_q"] not in _DTYPES:
        raise UnsupportedOpError(f"flashinfer doesn't support dtype_q={a['dtype_q']!r}")
    dt = _DTYPES[a["dtype_q"]]
    causal = a.get("causal", True)

    q = torch.randn(S_q, H, D, dtype=dt, device="cuda")
    k = torch.randn(S_kv, H_kv, D, dtype=dt, device="cuda")
    v = torch.randn(S_kv, H_kv, D, dtype=dt, device="cuda")

    if S_q == 1:
        def fn(q_=q, k_=k, v_=v):
            return flashinfer.single_decode_with_kv_cache(q_[0], k_, v_)
    else:
        def fn(q_=q, k_=k, v_=v, causal_=causal):
            return flashinfer.single_prefill_with_kv_cache(q_, k_, v_, causal=causal_)

    return {"fn": fn}


def _kernel_attention_mha(ctx: dict) -> None:
    ctx["out"] = ctx["fn"]()


# ---------- attention_mla ----------
#
# flashinfer ships two distinct MLA kernels (per the project's tracing issue
# https://github.com/flashinfer-ai/flashinfer/issues/792):
#
#   1. Decode  -> BatchMLAPagedAttentionWrapper (matrix-absorbed, paged KV)
#      q_nope lives in kv_lora_rank space, ckv/kpe are the compressed cache.
#      Loads O(S_kv * (R + D_pe)) bytes — dominates over S_kv.
#
#   2. Prefill -> BatchPrefillWithRaggedKVCacheWrapper (no absorption, ragged)
#      Plain self-attention with head_dim_qk = nope+rope = 192,
#      head_dim_vo = D_v = 128. Loads O(S_kv * H * D) bytes but the inner
#      attention dim stays small (192 vs the absorbed 512), so prefill is
#      faster here than through the absorbed wrapper.
#
# We dispatch on S_q == 1: pure decode goes via path 1, anything else uses
# path 2. (Chunked prefill / incremental prefill that mixes both kernels is
# out of scope — neither shape in our testlist hits that regime.)
#
# Either path: workspace 128 MiB; only wrapper.run() is timed. Up-projection
# gemms (c_kv -> K_nope/V) are scored separately via the gemm op.

_MLA_WORKSPACE_BYTES = 128 * 1024 * 1024  # 128 MiB, recommended by flashinfer docs


def _prepare_mla_decode(a: dict, dt: torch.dtype) -> dict:
    B = a["batch_size"]
    S_q = a["seq_len_q"]
    S_kv = a["seq_len_kv"]
    H = a["num_heads"]
    R = a["kv_lora_rank"]
    D_pe = a["head_dim_qk_rope"]
    D_nope = a["head_dim_qk_nope"]
    causal = a.get("causal", True)
    page_size = 1

    q_indptr = torch.arange(0, B + 1, dtype=torch.int32, device="cuda") * S_q
    kv_indptr = torch.arange(0, B + 1, dtype=torch.int32, device="cuda") * S_kv
    kv_indices = torch.arange(0, B * S_kv, dtype=torch.int32, device="cuda")
    kv_lens = torch.full((B,), S_kv, dtype=torch.int32, device="cuda")

    q_nope = torch.randn(B * S_q, H, R, dtype=dt, device="cuda")
    q_pe = torch.randn(B * S_q, H, D_pe, dtype=dt, device="cuda")
    ckv = torch.randn(B * S_kv, page_size, R, dtype=dt, device="cuda")
    kpe = torch.randn(B * S_kv, page_size, D_pe, dtype=dt, device="cuda")

    workspace = torch.empty(_MLA_WORKSPACE_BYTES, dtype=torch.int8, device="cuda")
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="auto")

    # Per flashinfer docs: "use head dimension before matrix absorption" for sm_scale.
    sm_scale = 1.0 / ((D_nope + D_pe) ** 0.5)
    wrapper.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        H, R, D_pe, page_size,
        causal, sm_scale,
        dt, dt,
    )

    def _run():
        return wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)

    return {"fn": _run}


def _prepare_mla_prefill(a: dict, dt: torch.dtype) -> dict:
    B = a["batch_size"]
    S_q = a["seq_len_q"]
    S_kv = a["seq_len_kv"]
    H = a["num_heads"]
    qk_dim = a["head_dim_qk_nope"] + a["head_dim_qk_rope"]
    D_v = a["head_dim_v"]
    causal = a.get("causal", True)

    qo_indptr = torch.arange(0, B + 1, dtype=torch.int32, device="cuda") * S_q
    kv_indptr = torch.arange(0, B + 1, dtype=torch.int32, device="cuda") * S_kv

    # Ragged layout NHD: [nnz, num_heads, head_dim]. MLA has num_kv_heads == num_heads.
    q = torch.randn(B * S_q, H, qk_dim, dtype=dt, device="cuda")
    k = torch.randn(B * S_kv, H, qk_dim, dtype=dt, device="cuda")
    v = torch.randn(B * S_kv, H, D_v, dtype=dt, device="cuda")

    workspace = torch.empty(_MLA_WORKSPACE_BYTES, dtype=torch.int8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")

    sm_scale = 1.0 / (qk_dim ** 0.5)
    wrapper.plan(
        qo_indptr, kv_indptr,
        num_qo_heads=H, num_kv_heads=H,
        head_dim_qk=qk_dim, head_dim_vo=D_v,
        causal=causal, sm_scale=sm_scale,
        q_data_type=dt, kv_data_type=dt,
    )

    def _run():
        return wrapper.run(q, k, v)

    return {"fn": _run}


def _prepare_attention_mla(op: Op) -> dict:
    a = op.args
    if a["dtype_q"] not in _DTYPES:
        raise UnsupportedOpError(
            f"flashinfer MLA dtype_q={a['dtype_q']!r} not supported (need bf16/fp16)"
        )
    if a["dtype_kv"] not in _DTYPES:
        raise UnsupportedOpError(
            f"flashinfer MLA dtype_kv={a['dtype_kv']!r} not supported (need bf16/fp16); "
            "neither MLA paged wrapper nor ragged prefill wrapper has a native fp8 path"
        )
    if a["dtype_q"] != a["dtype_kv"]:
        raise UnsupportedOpError(
            f"flashinfer MLA needs dtype_q == dtype_kv "
            f"(got {a['dtype_q']} vs {a['dtype_kv']})"
        )

    dt = _DTYPES[a["dtype_q"]]
    # Pure decode (S_q == 1) goes through the matrix-absorbed paged MLA wrapper.
    # Anything with S_q > 1 (self-attention prefill) goes through the ragged
    # prefill wrapper with head_dim_qk = nope+rope, head_dim_vo = D_v.
    if a["seq_len_q"] == 1:
        return _prepare_mla_decode(a, dt)
    return _prepare_mla_prefill(a, dt)


def _kernel_attention_mla(ctx: dict) -> None:
    ctx["out"] = ctx["fn"]()


IMPLS = [
    BackendImpl(op_type="gemm", prepare=_prepare_gemm, kernel=_kernel_gemm),
    BackendImpl(op_type="moe_gemm", prepare=_prepare_moe_gemm, kernel=_kernel_moe_gemm),
    BackendImpl(op_type="attention_mha", prepare=_prepare_attention_mha, kernel=_kernel_attention_mha),
    BackendImpl(op_type="attention_mla", prepare=_prepare_attention_mla, kernel=_kernel_attention_mla),
]
