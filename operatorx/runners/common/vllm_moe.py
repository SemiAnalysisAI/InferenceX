"""MoE layer through vLLM's own MoE pipeline, so vLLM picks router, expert and
shared-expert kernels and the stream layout.

Each op builds the pieces a vLLM MoE block wires together - a GateLinear
router, the routed experts from FusedMoEFactory under the quantization config
the expert descriptors imply, and a shared-expert MLP under its own - loads
synthetic checkpoint-format weights, runs process_weights_after_loading and
times the block on bf16 hidden states (as a CUDA graph where vLLM would
capture one). The quant methods and expert kernels vLLM chose are reported.
"""
from __future__ import annotations

import torch

from operatorx.core import BackendImpl, Op, UnsupportedOpError
from operatorx.runners.common import vllm_linear
from operatorx.runners.common.vllm_linear import _ENV_KEYS, _fill, _is_fault, _launcher, versions

__all__ = ["IMPLS", "versions"]

_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
_WORKSPACE = False
_LAYER = 0


def _context():
    global _WORKSPACE
    vllm_linear._vllm_context()
    if not _WORKSPACE:
        from vllm.v1.worker.workspace import init_workspace_manager
        init_workspace_manager(torch.device("cuda", torch.cuda.current_device()))
        _WORKSPACE = True


def _quant(x: dict, w: dict, where: str):
    """(activation, weight) descriptors -> vLLM quantization config, or None for bf16."""
    if x.get("input", "bf16") != "bf16":
        raise UnsupportedOpError(f"{where}: vLLM MoE layers take a bf16 input")
    sch = vllm_linear._scheme(x, w)
    if sch is None:
        return None
    if not sch:
        raise UnsupportedOpError(f"{where}: no vLLM quantization config for x={x} w={w}")
    from vllm.model_executor.layers.quantization import get_quantization_config
    method, cfg = sch
    return get_quantization_config(method).from_config(cfg)


def _act_kwargs(act: dict) -> dict:
    kind = act["kind"]
    if not act.get("gated", True):
        raise UnsupportedOpError("only gated expert activations are wired")
    if kind == "silu":
        return {"activation": "silu", "swiglu_limit": act.get("limit")}
    if kind == "swigluoai":
        return {"activation": "swigluoai", "swiglu_limit": act.get("limit", 7.0),
                "swiglu_alpha": act.get("alpha", 1.702), "swiglu_beta": act.get("beta")}
    if kind == "situ":
        return {"activation": "situ", "activation_situ_beta": act.get("alpha", 1.0),
                "activation_situ_linear_beta": act.get("beta")}
    raise UnsupportedOpError(f"activation {kind!r} is not wired for vLLM MoE")


def _act_fn(act: dict):
    from vllm.model_executor.layers import activation as A
    kind = act["kind"]
    if kind == "silu":
        return A.SiluAndMulWithClamp(act["limit"]) if act.get("limit") else A.SiluAndMul()
    if kind == "swigluoai":
        return A.SwigluOAIAndMul(alpha=act.get("alpha", 1.702), limit=act.get("limit", 7.0))
    if kind == "situ":
        return A.SituAndMul(beta=act.get("alpha", 1.0), linear_beta=act.get("beta"))
    raise UnsupportedOpError(f"shared-expert activation {kind!r} is not wired")


class _SharedMLP(torch.nn.Module):
    def __init__(self, hidden: int, inter: int, act: dict, qc, prefix: str):
        super().__init__()
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear, RowParallelLinear
        self.gate_up_proj = MergedColumnParallelLinear(hidden, [inter] * 2, bias=False, quant_config=qc,
                                                       disable_tp=True, prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(inter, hidden, bias=False, quant_config=qc, reduce_results=False,
                                           disable_tp=True, prefix=f"{prefix}.down_proj")
        self.act_fn = _act_fn(act)

    def forward(self, x):
        h, _ = self.gate_up_proj(x)
        h, _ = self.down_proj(self.act_fn(h))
        return h


class _MoeBlock(torch.nn.Module):
    """Router + routed experts (+ shared experts), called as the model's block would."""

    def __init__(self, a: dict, prefix: str):
        super().__init__()
        from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
        from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
        from vllm.model_executor.layers.fused_moe.utils import resolve_layer_fused_shared_expert
        ex, rt, sh, act = a["experts"], a["router"], a.get("shared"), a["activation"]
        for key, what in (("latent", "latent experts"), ("zero", "zero experts")):
            if ex.get(key):
                raise UnsupportedOpError(f"{what} are not wired for vLLM MoE yet")
        if rt["select"]["kind"] == "hash":
            raise UnsupportedOpError("hash routing is not wired for vLLM MoE yet")
        if rt.get("weight_on_input"):
            raise UnsupportedOpError("router weight on the expert input is not wired")
        q = ex["quant"]
        if q["w13"] != q["w2"] or q["a2"] != q["x"]:
            raise UnsupportedOpError("vLLM MoE takes one scheme for w13/w2 and for x/a2")
        qc = _quant(q["x"], q["w13"], "experts")
        vllm_linear._set_quant_fp8_op(qc)
        H = a["hidden"]
        logits = _DTYPES[rt["gate"].get("logits", rt["gate"]["dtype"])]
        self.gate = GateLinear(H, ex["num"], params_dtype=_DTYPES[rt["gate"]["dtype"]],
                               out_dtype=None if logits == torch.bfloat16 else logits, prefix=f"{prefix}.gate")
        self.gate.e_score_correction_bias = (
            torch.nn.Parameter(torch.zeros(ex["num"], dtype=torch.float32)) if rt.get("bias") else None)
        shared, shared_gate, fused_shared = None, None, False
        if sh is not None:
            sq = sh["quant"]
            if sq["w13"] != sq["w2"]:
                raise UnsupportedOpError("shared experts take one weight scheme for w13/w2")
            sqc = _quant(sq["x"], sq["w13"], "shared")
            if sq == {"x": q["x"], "w13": q["w13"], "w2": q["w2"]} and sh.get("gate") is None:
                fused_shared = resolve_layer_fused_shared_expert(qc, prefix)
            if not fused_shared:
                shared = _SharedMLP(H, sh["inter"] * sh["count"], act, sqc, f"{prefix}.shared_experts")
            if sh.get("gate") == "sigmoid":
                from vllm.model_executor.layers.linear import ReplicatedLinear
                shared_gate = ReplicatedLinear(H, 1, bias=False, quant_config=None, disable_tp=True,
                                               prefix=f"{prefix}.shared_expert_gate")
        sel = rt["select"]
        grouped = sel["kind"] == "grouped_topk"
        self.experts = FusedMoEFactory(
            num_experts=ex["num"], top_k=ex["top_k"], hidden_size=H, intermediate_size=ex["inter"],
            renormalize=bool(rt.get("renormalize")), quant_config=qc,
            use_grouped_topk=grouped or rt.get("bias", False),
            num_expert_group=sel["groups"] if grouped else 1,
            topk_group=sel["topk_groups"] if grouped else 1,
            prefix=f"{prefix}.experts", scoring_func=rt["scoring"],
            routed_scaling_factor=rt.get("scale") or 1.0,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            has_bias=bool(ex.get("bias")), reduce_results=False,
            n_shared_experts=sh["count"] if fused_shared else None, fuse_shared_experts=fused_shared,
            router_logits_dtype=self.gate.out_dtype, gate=self.gate,
            shared_experts=shared, shared_expert_gate=shared_gate, **_act_kwargs(act))
        self.shared = shared
        self.fused_shared = fused_shared

    def forward(self, x):
        from vllm.config import get_current_vllm_config
        from vllm.forward_context import set_forward_context
        with set_forward_context(None, get_current_vllm_config(), num_tokens=x.shape[0]):
            return self.experts(hidden_states=x, router_logits=x)


def _describe(block) -> dict:
    """Quant method and kernel objects vLLM attached to every quantized submodule."""
    out = {}
    for name, m in block.named_modules():
        qm = getattr(m, "quant_method", None)
        if qm is None or not hasattr(qm, "process_weights_after_loading"):
            continue
        entry = {"method": type(qm).__name__}
        for owner in (qm, getattr(qm, "scheme", None), getattr(m, "scheme", None)):
            for k, v in (vars(owner).items() if owner is not None else ()):
                t = type(v).__name__
                if any(t.endswith(s) for s in ("Kernel", "Experts", "PrepareAndFinalize", "Impl")):
                    entry[k] = t
                impl = getattr(v, "impl", None)  # FusedMoEKernel: the experts and dispatch it runs
                for part in ("fused_experts", "prepare_finalize"):
                    if getattr(impl, part, None) is not None:
                        entry[f"{k}.{part}"] = type(getattr(impl, part)).__name__
        out[name or "."] = entry
    return out


def _prepare_moe(op: Op) -> dict:
    global _LAYER
    a = op.args
    if a.get("out", "bf16") != "bf16":
        raise UnsupportedOpError("vLLM MoE layers return bf16")
    routing = a.get("routing") or {"distribution": "natural", "seed": 0}
    if routing["distribution"] != "natural":
        raise UnsupportedOpError("forced expert-load distributions are not wired for vLLM MoE yet")
    _context()
    import vllm.envs as envs
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    torch.manual_seed(routing["seed"])
    _LAYER += 1  # vLLM registers layers by name; each op builds a fresh one
    prefix = f"model.layers.{_LAYER}.mlp"
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        block = _MoeBlock(a, prefix).cuda()
        for m in block.modules():
            _fill(m)
        if block.gate.e_score_correction_bias is not None:
            block.gate.e_score_correction_bias.data.zero_()
        for m in block.modules():
            qm = getattr(m, "quant_method", None)
            if isinstance(qm, QuantizeMethodBase):
                qm.process_weights_after_loading(m)
    except (NotImplementedError, AssertionError, ValueError, RuntimeError, TypeError, KeyError) as e:
        if _is_fault(e):
            raise
        raise UnsupportedOpError(f"vLLM rejected this MoE layer: {type(e).__name__}: {e}"[:400]) from e
    finally:
        torch.set_default_dtype(prev)
    kernels = _describe(block)
    if any("Emulation" in str(v) for e in kernels.values() for v in e.values()):
        raise UnsupportedOpError(f"vLLM has only an emulation kernel for this MoE layer here: {kernels}")
    x = torch.randn(a["tokens"], a["hidden"], device="cuda", dtype=torch.bfloat16)
    ctx = {"layer": block, "x": x,
           "meta": {"vllm_modules": kernels, "fused_shared_experts": block.fused_shared,
                    "vllm_env": {k: getattr(envs, k) for k in (*_ENV_KEYS, *_MOE_ENV_KEYS) if hasattr(envs, k)}}}
    try:
        _kernel_moe(ctx)
        torch.cuda.synchronize()
    except (NotImplementedError, AssertionError, RuntimeError, ValueError, TypeError) as e:
        if _is_fault(e):
            raise
        raise UnsupportedOpError(f"vLLM MoE kernel failed: {type(e).__name__}: {e}"[:400]) from e
    return ctx


# Env that steers vLLM's MoE kernel and stream choice; recorded with every result.
_MOE_ENV_KEYS = ("VLLM_ROCM_USE_AITER_MOE", "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS",
                 "VLLM_USE_FLASHINFER_MOE_FP8", "VLLM_USE_FLASHINFER_MOE_FP4", "VLLM_FLASHINFER_MOE_BACKEND",
                 "VLLM_DISABLE_SHARED_EXPERTS_STREAM", "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD")


def _kernel_moe(ctx: dict) -> None:
    ctx["out"] = ctx["layer"](ctx["x"])


IMPLS = [BackendImpl(op_type="moe_layer", prepare=_prepare_moe, kernel=_kernel_moe, launcher=_launcher)]
