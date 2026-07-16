from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from operatorx.core import BackendImpl, Op, lookup_versions


def versions() -> dict[str, str]:
    return lookup_versions("sglang", "torch")

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
_MOE_CACHE: dict[tuple, object] = {}


def _ensure_dist() -> None:
    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        dist.init_process_group("nccl")


_SGLANG_READY = False
_MP_STATE: dict[str, int | None] = {"tp": None, "ep": None}


def _ensure_sglang(tp: int, ep: int) -> None:
    # sglang's parallel-group state is process-global and can't be safely
    # re-initialized while cached MoE modules still reference the old groups.
    # First (tp, ep) wins for this process; mismatched later shapes are rejected.
    global _SGLANG_READY
    if _SGLANG_READY:
        if _MP_STATE["tp"] != tp or _MP_STATE["ep"] != ep:
            from operatorx.core import UnsupportedOpError
            raise UnsupportedOpError(
                f"sglang already initialised with tp={_MP_STATE['tp']} ep={_MP_STATE['ep']}; "
                f"this op needs tp={tp} ep={ep}"
            )
        return
    _ensure_dist()

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    import sglang.srt.layers.moe.utils as moe_utils
    import sglang.srt.layers.dp_attention as dp_attn
    import sglang.srt.server_args as sa_module

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not model_parallel_is_initialized():
        init_distributed_environment(
            backend="nccl",
            world_size=world_size, rank=rank, local_rank=local_rank,
            distributed_init_method="env://",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=ep,
        )

    class _FakeServerArgs:
        disable_shared_experts_fusion = True
        ep_num_redundant_experts = 0
        enable_eplb = False
        enable_deterministic_inference = False
        enable_symm_mem = False
        kt_weight_path = None
        kt_num_gpu_experts = None
        kt_cpuinfer = None
        kt_threadpool_count = None
        dp_size = 1
        chunked_prefill_size = 8192

        def __getattr__(self, name):
            if name.startswith("enable_") or name.startswith("disable_"):
                return False
            if name.endswith("_size"):
                return 1
            return None

    sa_module._global_server_args = _FakeServerArgs()
    moe_utils.MOE_A2A_BACKEND = moe_utils.MoeA2ABackend.NONE
    dp_attn._DP_ATTENTION_ENABLED = False

    _MP_STATE["tp"] = tp
    _MP_STATE["ep"] = ep
    _SGLANG_READY = True


def _prepare_gemm(op: Op) -> dict:
    from operatorx.core import UnsupportedOpError
    a = op.args
    if a["dtype_a"] != "fp8" or a["dtype_b"] != "fp8":
        raise UnsupportedOpError(f"sglang requires fp8 inputs (got {a['dtype_a']}/{a['dtype_b']})")
    if a.get("bias"):
        raise UnsupportedOpError("sglang w8a8_block_fp8_matmul does not expose a bias parameter")
    if a.get("activation") is not None:
        raise UnsupportedOpError(
            f"sglang w8a8_block_fp8_matmul has no fused activation; got activation={a['activation']!r}"
        )
    if a["m"] <= 0 or a["n"] <= 0 or a["k"] <= 0:
        raise UnsupportedOpError(f"degenerate gemm shape m={a['m']} n={a['n']} k={a['k']}")
    from sglang.srt.layers.quantization.fp8_kernel import (
        per_token_group_quant_fp8,
        w8a8_block_fp8_matmul,
    )
    from sglang.srt.layers.quantization.fp8_utils import per_block_cast_to_fp8

    # Pad K up to a multiple of 128 (block size); the kernel's grid wants this.
    K_pad = ((a["k"] + 127) // 128) * 128
    M, N, K = a["m"], a["n"], K_pad
    A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    block_size = [128, 128]
    A_fp8, A_scales = per_token_group_quant_fp8(A_bf16, block_size[1])
    B_fp8, B_scales = per_block_cast_to_fp8(B_bf16)
    return {
        "matmul_fn": w8a8_block_fp8_matmul,
        "a": A_fp8, "b": B_fp8,
        "a_scales": A_scales, "b_scales": B_scales,
        "block_size": block_size,
    }


def _kernel_gemm(ctx: dict) -> None:
    ctx["out"] = ctx["matmul_fn"](
        ctx["a"], ctx["b"], ctx["a_scales"], ctx["b_scales"],
        ctx["block_size"], torch.bfloat16,
    )


def _build_moe(
    num_experts: int, hidden: int, intermediate: int, top_k: int,
    dtype: torch.dtype, weight_dtype: str,
    expert_distribution: str = "uniform",
    n_shared_experts: int = 1,
):
    """Mirror master's bench_deepseek_moe_standalone.create_moe_layer.

    ``weight_dtype`` selects the expert weight precision:
      - 'fp8'   -> Fp8Config (DeepSeek block-FP8)
      - 'nvfp4' -> ModelOptFp4Config (NVFP4). Routes through MOE_RUNNER_BACKEND
                  (scoped to FLASHINFER_CUTEDSL during construction).
      - 'bf16'/'fp16' -> unquantized

    ``intermediate`` is the *global* per-expert intermediate dim. sglang's
    FusedMoE and DeepseekV2MLP shard it internally based on the TP/EP groups
    set up by ``initialize_model_parallel`` — don't pre-divide here.
    """
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

    if weight_dtype == "fp8":
        quant_config_dict = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    elif weight_dtype == "nvfp4":
        quant_config_dict = {"quant_method": "modelopt_fp4"}
    else:
        quant_config_dict = None

    config = SimpleNamespace(
        n_routed_experts=num_experts,
        num_experts_per_tok=top_k,
        n_group=8,
        topk_group=4,
        topk_method="noaux_tc",
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        hidden_size=hidden,
        moe_intermediate_size=intermediate,
        hidden_act="silu",
        n_shared_experts=n_shared_experts,
        first_k_dense_replace=3,
        num_hidden_layers=61,
        quantization_config=quant_config_dict,
    )

    if weight_dtype == "fp8":
        from sglang.srt.layers.quantization.fp8 import Fp8Config
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[128, 128],
        )
    elif weight_dtype == "nvfp4":
        from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
        # group_size=16 (NVFP4 block size). The router gate and shared-experts
        # are bf16 Linear layers; exclude them so apply() doesn't look for
        # checkpoint-only `input_scale_inv` attrs we don't generate.
        quant_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True, group_size=16,
            exclude_modules=["gate", "shared_experts", "lm_head", "embed"],
            packed_modules_mapping={},
        )
    else:
        quant_config = None

    torch.set_default_dtype(dtype)
    try:
        moe = DeepseekV2MoE(
            config=config, layer_id=3, quant_config=quant_config,
            prefix="model.layers.3.mlp", alt_stream=None, is_nextn=False,
        )
    finally:
        torch.set_default_dtype(torch.float32)

    moe = moe.to(device="cuda")
    with torch.no_grad():
        for name, param in moe.named_parameters():
            if param.dtype == torch.uint8:
                # nvfp4-packed weight tensors; any byte is a valid fp4 pair.
                param.random_(0, 256)
            elif param.dtype == torch.float8_e4m3fn:
                # block scales — positive fp8 values around 1.0 via bf16.
                param.copy_(torch.rand_like(param, dtype=dtype).add_(0.5).to(param.dtype))
            elif param.dtype.is_floating_point:
                param.normal_(0, 0.02)
            if "correction_bias" in name:
                param.abs_()
            if "scale" in name and param.dtype == torch.float32:
                param.fill_(1.0)

        # Bias the router's correction_bias to shape the expert distribution
        # observed by the kernel. We patch correction_bias (added directly to
        # the routing scores) rather than the gate's weight matrix so the bias
        # is independent of the input activations.
        if expert_distribution != "uniform":
            for name, param in moe.named_parameters():
                if "correction_bias" in name and param.shape[-1] == num_experts:
                    if expert_distribution == "zipf":
                        rank = torch.arange(1, num_experts + 1,
                                            dtype=torch.float32, device=param.device)
                        bias = -torch.log(rank).to(param.dtype)
                    elif expert_distribution == "single_hot":
                        bias = torch.zeros(num_experts, dtype=param.dtype, device=param.device)
                        bias[0] = 10.0
                    else:
                        raise ValueError(f"unknown expert_distribution={expert_distribution!r}")
                    param.copy_(bias)

    # NVFP4 needs `process_weights_after_loading` to consolidate per-expert
    # scales into the `*_scale_quant` tensors the kernel reads. The relevant
    # `enable_flashinfer_*_moe` flags are @property methods that read the
    # global MOE_RUNNER_BACKEND at call time, so the override must already be
    # active when this is called — _prepare_moe_forward sets it before invoking.
    if weight_dtype == "nvfp4":
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        for sub in moe.modules():
            if isinstance(sub, FusedMoE) and hasattr(sub, "quant_method"):
                sub.quant_method.process_weights_after_loading(sub)
    return moe


def _prepare_moe_forward(op: Op) -> dict:
    from operatorx.core import UnsupportedOpError
    a = op.args
    if a["dtype_weight"] not in ("fp8", "nvfp4", "bf16", "fp16"):
        raise UnsupportedOpError(
            f"sglang moe_forward expects fp8/nvfp4/bf16/fp16 weights (got dtype_weight={a['dtype_weight']!r})"
        )
    if a["dtype_act"] not in _DTYPES:
        raise UnsupportedOpError(f"sglang moe_forward doesn't support dtype_act={a['dtype_act']!r}")
    ne, h, im = a["num_experts"], a["hidden"], a["intermediate"]
    k, ws = a["top_k"], a["world_size"]
    ep = a.get("expert_parallel_size", 1)
    tp = a.get("routed_tensor_parallel_size", 1)
    s_tp = a.get("shared_tensor_parallel_size", 1)
    if ep * tp != ws:
        raise UnsupportedOpError(
            f"expert_parallel_size * routed_tensor_parallel_size must equal world_size "
            f"(got {ep} * {tp} != {ws})"
        )
    if s_tp not in (1, tp):
        raise UnsupportedOpError(
            f"shared_tensor_parallel_size must be 1 or routed_tensor_parallel_size "
            f"(got {s_tp}, routed_tp={tp})"
        )
    # SGLANG_SHARED_EXPERT_TP1=1 forces DeepseekV2MoE to build the shared expert
    # with tp_size=1 (replicated within the TP group). Must be set before
    # construction since it's read at __init__ time.
    if s_tp == 1 and tp > 1:
        os.environ["SGLANG_SHARED_EXPERT_TP1"] = "1"
    else:
        os.environ.pop("SGLANG_SHARED_EXPERT_TP1", None)
    _ensure_sglang(tp=tp, ep=ep)
    # NVFP4: pin MOE_RUNNER_BACKEND to a flashinfer variant for the lifetime of
    # this op. `enable_flashinfer_*_moe` are @property methods that re-read the
    # global at every call, so we need this set during build, weight-process,
    # and the timed kernel iterations. AUTO has no nvfp4 dispatch branch.
    if a["dtype_weight"] == "nvfp4":
        import sglang.srt.layers.moe.utils as _moe_utils
        runner_name = os.environ.get("OPERATORX_SGLANG_MOE_RUNNER", "FLASHINFER_CUTLASS")
        _moe_utils.MOE_RUNNER_BACKEND = _moe_utils.MoeRunnerBackend[runner_name]
    # sglang's fp8 MoE config (weight_block_size=[128,128]) requires the per-rank
    # expert count to be a multiple of block_n=128, else weight loading aborts.
    if a["dtype_weight"] == "fp8":
        experts_per_rank = ne // max(1, ep)
        if experts_per_rank % 128 != 0:
            raise UnsupportedOpError(
                f"sglang fp8 MoE requires num_experts/ep to be a multiple of 128 "
                f"(weight block_n); got num_experts={ne} ep={ep} -> {experts_per_rank}"
            )
    dt = _DTYPES[a["dtype_act"]]
    nt = a["num_tokens"]

    distribution = a.get("expert_distribution", "uniform")
    n_shared = a.get("n_shared_experts", 0)
    cache_key = (ne, h, im, k, ws, tp, ep, s_tp, dt, a["dtype_weight"], distribution, n_shared)
    if cache_key not in _MOE_CACHE:
        _MOE_CACHE[cache_key] = _build_moe(
            ne, h, im, k, dt, a["dtype_weight"],
            expert_distribution=distribution, n_shared_experts=n_shared,
        )
    moe = _MOE_CACHE[cache_key]

    x = torch.randn(nt, h, dtype=dt, device="cuda")
    return {"moe": moe, "x": x}


def _kernel_moe_forward(ctx: dict) -> None:
    with torch.no_grad():
        ctx["out"] = ctx["moe"].forward_normal(ctx["x"])


IMPLS = [
    BackendImpl(op_type="gemm", prepare=_prepare_gemm, kernel=_kernel_gemm),
    BackendImpl(op_type="moe_forward", prepare=_prepare_moe_forward, kernel=_kernel_moe_forward),
]
