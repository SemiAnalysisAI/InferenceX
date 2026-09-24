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
from operatorx.runners.common.vllm import linear as vllm_linear
from operatorx.runners.common.vllm.linear import (_ENV_KEYS, _fill, _is_fault, _launcher, device, sync,
                                                  versions)

__all__ = ["IMPLS", "versions"]

_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
_MAX_TOKENS = 2048  # covers the largest CUDA-graph capture size
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
    return vllm_linear.quant_config(*sch)


def _act_kwargs(act: dict) -> dict:
    kind = act["kind"]
    if not act.get("gated", True):
        raise UnsupportedOpError("only gated expert activations are wired")
    if kind == "silu":
        return {"activation": "silu", "swiglu_limit": act.get("limit")}
    if kind == "swigluoai":
        return {"activation": "swigluoai" if act.get("interleaved") else "swigluoai_uninterleave",
                "swiglu_limit": act.get("limit", 7.0), "swiglu_alpha": act.get("alpha", 1.702),
                "swiglu_beta": act.get("beta")}
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
        if act.get("interleaved"):
            return A.SwigluOAIAndMul(alpha=act.get("alpha", 1.702), limit=act.get("limit", 7.0))
        return A.SiluAndMulWithClamp(act.get("limit", 7.0), alpha=act.get("alpha", 1.702),
                                     beta=act.get("beta") or 0.0)
    if kind == "situ":
        return A.SituAndMul(beta=act.get("alpha", 1.0), linear_beta=act.get("beta"))
    raise UnsupportedOpError(f"shared-expert activation {kind!r} is not wired")


class _SharedMLP(torch.nn.Module):
    def __init__(self, hidden: int, inter: int, act: dict, qc, prefix: str, expert_gate=None):
        super().__init__()
        self.expert_gate = expert_gate  # Qwen: sigmoid(gate(x)) scales the shared output
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear, RowParallelLinear
        self.gate_up_proj = MergedColumnParallelLinear(hidden, [inter] * 2, bias=False, quant_config=qc,
                                                       disable_tp=True, prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(inter, hidden, bias=False, quant_config=qc, reduce_results=False,
                                           disable_tp=True, prefix=f"{prefix}.down_proj")
        self.act_fn = _act_fn(act)

    def forward(self, x):
        h, _ = self.gate_up_proj(x)
        h, _ = self.down_proj(self.act_fn(h))
        if self.expert_gate is not None:
            h = torch.sigmoid(self.expert_gate(x)[0]) * h
        return h


# DeepSeek-V4 routed experts: MXFP4 weights under the checkpoint's fp8 (ue8m0) config,
# which vLLM routes to its MXFP4 MoE method (expert_dtype "fp4").
_DSV4_FP4_EXPERTS = ("deepseek_v4_fp8", {"quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3",
                                         "scale_fmt": "ue8m0", "weight_block_size": [128, 128]})


def _expert_quant(x: dict, w: dict):
    if (x.get("dtype") == "e4m3" and x.get("scale", {}).get("dtype") == "ue8m0" and w["dtype"] == "e2m1"
            and w.get("scale") == {"dtype": "ue8m0", "static": True, "group": [1, 32]} and "scale2" not in w):
        return vllm_linear.quant_config(*_DSV4_FP4_EXPERTS)
    return _quant(x, w, "experts")


def _scores(logits: torch.Tensor, scoring: str) -> torch.Tensor:
    logits = logits.float()
    if scoring == "softmax":
        return torch.softmax(logits, dim=-1)
    if scoring == "sigmoid":
        return torch.sigmoid(logits)
    return torch.sqrt(torch.nn.functional.softplus(logits))  # sqrtsoftplus


def _forced_ids(dist, tokens: int, experts: int, top_k: int, seed: int) -> torch.Tensor:
    """[tokens, top_k] expert ids with the requested expert-load distribution."""
    g = torch.Generator().manual_seed(seed)
    if dist == "balanced":  # round-robin: every expert gets tokens*top_k/experts slots
        base = torch.arange(tokens * top_k) % experts
        ids = base.view(tokens, top_k)
        return ids[:, torch.randperm(top_k, generator=g)]
    if dist == "single_hot":
        rest = torch.stack([torch.randperm(experts - 1, generator=g)[: top_k - 1] + 1 for _ in range(tokens)])
        return torch.cat([torch.zeros(tokens, 1, dtype=torch.long), rest], dim=1)
    p = 1.0 / torch.arange(1, experts + 1, dtype=torch.float64) ** dist["s"]  # zipf over expert rank
    ranked = torch.randperm(experts, generator=g)
    return ranked[torch.multinomial(p.expand(tokens, -1), top_k, replacement=False, generator=g)]


def _kimi_latent():
    """Kimi-K3's latent-MoE runner and output transform (platform-specific modules)."""
    from vllm.platforms import current_platform
    if current_platform.is_rocm():
        from vllm.models.kimi_k3.amd.latent_moe_runner import ROCmLatentMoERunner as LatentMoERunner
        from vllm.models.kimi_k3.amd.linear import KimiRoutedOutputTransform
        return LatentMoERunner, KimiRoutedOutputTransform, 256
    from vllm.models.kimi_k3.nvidia.latent_moe_runner import LatentMoERunner
    from vllm.models.kimi_k3.nvidia.model import (_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD,
                                                  KimiRoutedOutputTransform)
    return LatentMoERunner, KimiRoutedOutputTransform, _ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD


class _ForcedRouting:
    """vLLM custom_routing_function: fixed expert choice, the router's weights for it."""

    def __init__(self, owner: torch.nn.Module, scoring: str):
        self.owner, self.scoring = owner, scoring

    def __call__(self, hidden_states, gating_output, topk, renormalize):
        ids = self.owner.forced_ids[: gating_output.shape[0]]
        w = _scores(gating_output, self.scoring).gather(1, ids.long())
        if renormalize:
            w = w / w.sum(dim=-1, keepdim=True)
        return w, ids


class _MoeBlock(torch.nn.Module):
    """Router + routed experts (+ shared experts), wired as vLLM's model blocks wire them."""

    def __init__(self, a: dict, prefix: str):
        super().__init__()
        from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
        from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
        from vllm.model_executor.layers.fused_moe.utils import resolve_layer_fused_shared_expert
        ex, rt, sh, act = a["experts"], a["router"], a.get("shared"), a["activation"]
        if ex.get("zero"):
            raise UnsupportedOpError("zero experts are not wired for vLLM MoE yet")
        if rt.get("weight_on_input"):
            raise UnsupportedOpError("router weight on the expert input is not wired")
        q = ex["quant"]
        if q["w13"] != q["w2"] or q["a2"] != q["x"]:
            raise UnsupportedOpError("vLLM MoE takes one scheme for w13/w2 and for x/a2")
        qc = _expert_quant(q["x"], q["w13"])
        vllm_linear._set_quant_fp8_op(qc)
        H, E, K = a["hidden"], ex["num"], ex["top_k"]
        L = ex.get("latent") or H
        sel = rt["select"]
        routing = a.get("routing") or {"distribution": "natural", "seed": 0}
        logits = _DTYPES[rt["gate"].get("logits", rt["gate"]["dtype"])]
        self.gate = GateLinear(H, E, params_dtype=_DTYPES[rt["gate"]["dtype"]],
                               out_dtype=None if logits == torch.bfloat16 else logits, prefix=f"{prefix}.gate")
        self.gate.e_score_correction_bias = (
            torch.nn.Parameter(torch.zeros(E, dtype=torch.float32)) if rt.get("bias") else None)
        self.gate.tid2eid = None
        self.register_buffer("input_ids", None)
        if sel["kind"] == "hash":  # DeepSeek-V4 hash layers: token id -> experts table
            g = torch.Generator().manual_seed(routing["seed"])
            self.gate.tid2eid = torch.nn.Parameter(
                torch.randint(0, E, (sel["vocab"], K), generator=g, dtype=torch.int32), requires_grad=False)
            self.register_buffer("input_ids", torch.randint(0, sel["vocab"], (_MAX_TOKENS,), generator=g))
        custom = None
        if routing["distribution"] != "natural":
            if sel["kind"] == "hash":
                raise UnsupportedOpError("a forced expert-load distribution conflicts with hash routing")
            ids = _forced_ids(routing["distribution"], _MAX_TOKENS, E, K, routing["seed"])
            self.register_buffer("forced_ids", ids.to(torch.int32))
            custom = _ForcedRouting(self, rt["scoring"])
        shared, shared_gate, fused_shared = None, None, False
        if sh is not None:
            sq = sh["quant"]
            if sq["w13"] != sq["w2"]:
                raise UnsupportedOpError("shared experts take one weight scheme for w13/w2")
            sqc = _quant(sq["x"], sq["w13"], "shared")
            if sq == {"x": q["x"], "w13": q["w13"], "w2": q["w2"]} and not ex.get("latent"):
                fused_shared = resolve_layer_fused_shared_expert(qc, prefix)
            expert_gate = None
            if sh.get("gate") == "sigmoid":
                from vllm.model_executor.layers.linear import ReplicatedLinear
                expert_gate = ReplicatedLinear(H, 1, bias=False, quant_config=None, disable_tp=True,
                                               prefix=f"{prefix}.shared_expert_gate")
            if fused_shared:  # as Qwen's block: the runner takes the gate only with fused shared experts
                shared_gate = expert_gate
            else:
                shared = _SharedMLP(H, sh["inter"] * sh["count"], act, sqc, f"{prefix}.shared_experts",
                                    expert_gate=expert_gate)
        latent = {}
        self.down_proj = None
        if ex.get("latent"):  # Kimi-K3: routed experts run at width L between bf16 projections
            from vllm.model_executor.layers.layernorm import RMSNorm
            from vllm.model_executor.layers.linear import ReplicatedLinear
            LatentMoERunner, KimiRoutedOutputTransform, self._stream_tokens = _kimi_latent()
            self.down_proj = ReplicatedLinear(H, L, bias=False, quant_config=None,
                                              prefix=f"{prefix}.routed_expert_down_proj")
            norm = RMSNorm(L) if ex.get("latent_norm") else None
            up = ReplicatedLinear(L, H, bias=False, quant_config=None, prefix=f"{prefix}.routed_expert_up_proj")
            self.up_transform = KimiRoutedOutputTransform(norm, up)
            latent = {"routed_output_transform": self.up_transform, "runner_cls": LatentMoERunner}
        # a forced expert choice replaces selection, so vLLM's grouped/bias selection is off
        grouped = sel["kind"] == "grouped_topk" and custom is None
        bias = None if custom is not None else self.gate.e_score_correction_bias
        self.experts_quant_config = qc
        self.experts = FusedMoEFactory(
            num_experts=E, top_k=K, hidden_size=L, intermediate_size=ex["inter"],
            renormalize=bool(rt.get("renormalize")), quant_config=qc, use_grouped_topk=grouped,
            num_expert_group=sel["groups"] if grouped else None,
            topk_group=sel["topk_groups"] if grouped else None,
            prefix=f"{prefix}.experts", scoring_func=rt["scoring"],
            routed_scaling_factor=rt.get("scale") or 1.0,
            e_score_correction_bias=bias,
            hash_indices_table=self.gate.tid2eid, custom_routing_function=custom,
            has_bias=bool(ex.get("bias")), reduce_results=False,
            n_shared_experts=sh["count"] if fused_shared else None, fuse_shared_experts=fused_shared,
            router_logits_dtype=self.gate.out_dtype, gate=None if ex.get("latent") else self.gate,
            shared_experts=shared, shared_expert_gate=shared_gate, **latent, **_act_kwargs(act))
        self.shared = shared
        self.fused_shared = fused_shared
        if ex.get("latent") and device().type == "cuda":
            from vllm.utils.torch_utils import aux_stream
            self._aux = aux_stream()
            self._events = (torch.cuda.Event(), torch.cuda.Event())

    def forward(self, x):
        from vllm.config import get_current_vllm_config
        from vllm.forward_context import set_forward_context
        T = x.shape[0]
        with set_forward_context(None, get_current_vllm_config(), num_tokens=T):
            if self.down_proj is None or not hasattr(self, "_aux"):
                if self.down_proj is not None:  # latent projection without an aux stream
                    lat, _ = self.down_proj(x)
                    return self.experts(hidden_states=lat, router_logits=self.gate(x)[0],
                                        shared_experts_input=x)
                ids = None if self.input_ids is None else self.input_ids[:T]
                return self.experts(hidden_states=x, router_logits=x, input_ids=ids)
            # as Kimi-K3's block: router and latent down projection on two streams at decode sizes
            from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
            logits, (lat, _) = maybe_execute_in_parallel(
                lambda: self.gate(x)[0], lambda: self.down_proj(x), self._events[0], self._events[1],
                self._aux if T <= self._stream_tokens else None)
            return self.experts(hidden_states=lat, router_logits=logits, shared_experts_input=x)


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


def _release_previous() -> None:
    """vLLM registers every MoE layer by name in the forward context; drop the previous
    op's so its weights and workspaces are freed before the next layer is built."""
    import gc

    from vllm.config import get_current_vllm_config
    ctx = get_current_vllm_config().compilation_config.static_forward_context
    for name in [k for k in ctx if k.startswith("model.layers.")]:
        del ctx[name]
    gc.collect()
    torch.cuda.empty_cache()


def _missing_op(e: BaseException) -> bool:
    """A vLLM custom op this build does not ship (e.g. Marlin repack on ROCm)."""
    return isinstance(e, AttributeError) and "_OpNamespace" in str(e)


def _prepare_moe(op: Op) -> dict:
    global _LAYER
    a = op.args
    if a.get("out", "bf16") != "bf16":
        raise UnsupportedOpError("vLLM MoE layers return bf16")
    if a["tokens"] > _MAX_TOKENS:
        raise UnsupportedOpError(f"tokens > {_MAX_TOKENS} is not wired")
    routing = a.get("routing") or {"distribution": "natural", "seed": 0}
    _context()
    _release_previous()
    import vllm.envs as envs
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    torch.manual_seed(routing["seed"])
    _LAYER += 1  # vLLM registers layers by name; each op builds a fresh one
    prefix = f"model.layers.{_LAYER}.mlp"
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        block = _MoeBlock(a, prefix).to(device())
        for m in block.modules():
            _fill(m)
        if block.gate.e_score_correction_bias is not None:
            block.gate.e_score_correction_bias.data.zero_()
        if block.gate.tid2eid is not None:  # the filler knows nothing of expert ids
            block.gate.tid2eid.data.random_(0, a["experts"]["num"])
        methods = []
        for m in block.modules():
            qm = getattr(m, "quant_method", None)
            if isinstance(qm, QuantizeMethodBase):
                methods.append(type(qm).__name__)
                qm.process_weights_after_loading(m)
        # A quantized block whose every method came out unquantized computed bf16 and
        # would be recorded under the quantized row it is not. A bf16 shared expert
        # beside quantized routed ones is normal, so only an all-unquantized block counts.
        if block.experts_quant_config is not None and methods and all("Unquantized" in n for n in methods):
            raise UnsupportedOpError(
                f"{type(block.experts_quant_config).__name__} has no quantized MoE method here; "
                f"the layer fell back to {sorted(set(methods))}")
    except (NotImplementedError, AssertionError, ValueError, RuntimeError, TypeError, KeyError,
            AttributeError) as e:
        if _is_fault(e) or (isinstance(e, AttributeError) and not _missing_op(e)):
            raise
        raise UnsupportedOpError(f"vLLM rejected this MoE layer: {type(e).__name__}: {e}"[:400]) from e
    finally:
        torch.set_default_dtype(prev)
    kernels = _describe(block)
    if any("Emulation" in str(v) for e in kernels.values() for v in e.values()):
        raise UnsupportedOpError(f"vLLM has only an emulation kernel for this MoE layer here: {kernels}")
    x = torch.randn(a["tokens"], a["hidden"], device=device(), dtype=torch.bfloat16)
    ctx = {"layer": block, "x": x,
           "meta": {"vllm_modules": kernels, "fused_shared_experts": block.fused_shared,
                    "vllm_env": {k: getattr(envs, k) for k in (*_ENV_KEYS, *_MOE_ENV_KEYS) if hasattr(envs, k)}}}
    try:
        _kernel_moe(ctx)
        sync()
    except (NotImplementedError, AssertionError, RuntimeError, ValueError, TypeError, AttributeError) as e:
        if _is_fault(e) or (isinstance(e, AttributeError) and not _missing_op(e)):
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
