from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from operatorx.core.op import OpSpec
from operatorx.core.op_registry import register
from operatorx.ops.gemm import ELEMENT_DTYPES, check_operand

SCORING = {"softmax", "sigmoid", "sqrtsoftplus"}
ACTIVATIONS = {"silu", "gelu", "gelu_tanh", "swigluoai", "situ", "swiglustep"}
GATE_DTYPES = {"bf16", "fp32"}
DISTRIBUTIONS = {"natural", "balanced"}


def _keys(d: Any, where: str, required: set[str], optional: set[str] = frozenset()) -> None:
    if not isinstance(d, dict):
        raise ValueError(f"{where} must be a dict, got {d!r}")
    missing, extra = required - set(d), set(d) - required - set(optional)
    if missing or extra:
        raise ValueError(f"{where}: missing {sorted(missing)}, unknown {sorted(extra)}")


def _pos_int(v: Any, where: str) -> None:
    if not isinstance(v, int) or isinstance(v, bool) or v < 1:
        raise ValueError(f"{where} must be a positive int, got {v!r}")


def _number(v: Any, where: str) -> None:
    if v is not None and (not isinstance(v, (int, float)) or isinstance(v, bool)):
        raise ValueError(f"{where} must be a number or null, got {v!r}")


def _check_quant(q: Any, where: str, operands: set[str]) -> None:
    _keys(q, where, set(operands))
    for name in operands:
        check_operand(q[name], f"{where}.{name}")


def _check_experts(e: Any) -> None:
    _keys(e, "experts", {"num", "top_k", "inter", "quant"}, {"bias", "latent", "zero"})
    for k in ("num", "top_k", "inter"):
        _pos_int(e[k], f"experts.{k}")
    if e["top_k"] > e["num"]:
        raise ValueError("experts.top_k exceeds experts.num")
    if e.get("latent") is not None:
        _pos_int(e["latent"], "experts.latent")
    if not isinstance(e.get("zero", 0), int) or e.get("zero", 0) < 0:
        raise ValueError("experts.zero must be a non-negative int")
    if not isinstance(e.get("bias", False), bool):
        raise ValueError("experts.bias must be a bool")
    _check_quant(e["quant"], "experts.quant", {"x", "w13", "w2", "a2"})


def _check_router(r: Any, num_experts: int) -> None:
    _keys(r, "router", {"gate", "scoring", "select"}, {"bias", "renormalize", "scale", "weight_on_input"})
    _keys(r["gate"], "router.gate", {"dtype"}, {"logits"})
    for k in ("dtype", "logits"):
        if r["gate"].get(k, "bf16") not in GATE_DTYPES:
            raise ValueError(f"router.gate.{k} must be one of {sorted(GATE_DTYPES)}")
    if r["scoring"] not in SCORING:
        raise ValueError(f"router.scoring must be one of {sorted(SCORING)}")
    s = r["select"]
    kind = s.get("kind") if isinstance(s, dict) else None
    if kind == "topk":
        _keys(s, "router.select", {"kind"})
    elif kind == "grouped_topk":
        _keys(s, "router.select", {"kind", "groups", "topk_groups"})
        _pos_int(s["groups"], "router.select.groups")
        _pos_int(s["topk_groups"], "router.select.topk_groups")
        if num_experts % s["groups"] or s["topk_groups"] > s["groups"]:
            raise ValueError("router.select: groups must divide experts.num and topk_groups <= groups")
    elif kind == "hash":
        _keys(s, "router.select", {"kind", "vocab"})
        _pos_int(s["vocab"], "router.select.vocab")
    else:
        raise ValueError(f"router.select.kind must be topk, grouped_topk or hash, got {s!r}")
    for k in ("bias", "renormalize", "weight_on_input"):
        if not isinstance(r.get(k, False), bool):
            raise ValueError(f"router.{k} must be a bool")
    _number(r.get("scale"), "router.scale")


def _check_activation(a: Any) -> None:
    _keys(a, "activation", {"kind"}, {"gated", "limit", "alpha", "beta"})
    if a["kind"] not in ACTIVATIONS:
        raise ValueError(f"activation.kind must be one of {sorted(ACTIVATIONS)}")
    if not isinstance(a.get("gated", True), bool):
        raise ValueError("activation.gated must be a bool")
    for k in ("limit", "alpha", "beta"):
        _number(a.get(k), f"activation.{k}")


def _check_shared(s: Any) -> None:
    _keys(s, "shared", {"count", "inter", "quant"}, {"gate"})
    _pos_int(s["count"], "shared.count")
    _pos_int(s["inter"], "shared.inter")
    if s.get("gate") not in (None, "sigmoid"):
        raise ValueError("shared.gate must be null or 'sigmoid'")
    _check_quant(s["quant"], "shared.quant", {"x", "w13", "w2"})


def _check_routing(r: Any) -> None:
    _keys(r, "routing", {"distribution", "seed"})
    d = r["distribution"]
    if isinstance(d, dict):
        _keys(d, "routing.distribution", {"kind", "s"})
        if d["kind"] != "zipf":
            raise ValueError("routing.distribution dict must be {'kind': 'zipf', 's': ...}")
        _number(d["s"], "routing.distribution.s")
    elif d not in DISTRIBUTIONS:
        raise ValueError(f"routing.distribution must be one of {sorted(DISTRIBUTIONS)} or a zipf dict")
    if not isinstance(r["seed"], int) or isinstance(r["seed"], bool):
        raise ValueError("routing.seed must be an int")


@dataclass(frozen=True)
class MoeLayerArgs:
    """y[T,H] = shared(x) + sum over the selected experts e of w_e * expert_e(x).

    The op starts at the router GEMM on normed hidden states x [T, H] and ends at
    the combined output; residual add and norms are outside it. Operand
    descriptors are the gemm op's ({"dtype", "scale"?, "scale2"?, "symmetric"?}).

    experts: {"num": E, "top_k": K, "inter": I, "quant": {x, w13, w2, a2},
              "bias"?: bool, "latent"?: L, "zero"?: n}
      quant.x is the experts' input as they consume it, w13 the fused gate/up
      weight, w2 the down weight, a2 the intermediate activation. latent: experts
      run at width L with H->L / L->H projections. zero: identity experts.
    router: {"gate": {"dtype", "logits"?}, "scoring": softmax|sigmoid|sqrtsoftplus,
             "select": {"kind": "topk"} | {"kind": "grouped_topk", "groups", "topk_groups"}
                       | {"kind": "hash", "vocab"},
             "bias"?: score-correction bias, "renormalize"?, "scale"?: routed scaling factor,
             "weight_on_input"?: router weight applied to the expert input}
      gate.dtype is the router weight's dtype, gate.logits the dtype of the logits it
      produces (default: gate.dtype).
    activation: {"kind", "gated"?: default true, "limit"?, "alpha"?, "beta"?}
      swigluoai: alpha scales the gate sigmoid, limit clamps. situ: alpha and beta
      soft-cap the gate and up halves (alpha*tanh(g/alpha)*sigmoid(g) * beta*tanh(u/beta)).
    shared: null | {"count", "inter", "quant": {x, w13, w2}, "gate"?: null|"sigmoid"}
    routing: {"distribution": "natural" | "balanced" | {"kind": "zipf", "s"}, "seed"}
      natural: routing is whatever the router computes on seeded random inputs;
      balanced / zipf: expert load is forced to that distribution.
    """
    tokens: int
    hidden: int
    experts: dict
    router: dict
    activation: dict
    shared: dict | None = None
    routing: dict | None = None
    out: str = "bf16"

    def __post_init__(self):
        _pos_int(self.tokens, "tokens")
        _pos_int(self.hidden, "hidden")
        _check_experts(self.experts)
        _check_router(self.router, self.experts["num"])
        _check_activation(self.activation)
        if self.shared is not None:
            _check_shared(self.shared)
        _check_routing(self.routing if self.routing is not None else {"distribution": "natural", "seed": 0})
        if self.out not in ELEMENT_DTYPES:
            raise ValueError(f"out {self.out!r} not in {sorted(ELEMENT_DTYPES)}")


MOE_LAYER = OpSpec(
    type="moe_layer",
    arg_schema=MoeLayerArgs,
    description="y = shared(x) + sum_{e in topk(route(x))} w_e * expert_e(x)",
)

register(MOE_LAYER)
