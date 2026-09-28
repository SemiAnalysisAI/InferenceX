"""Attention modules: one op type per module, each timed whole.

Every op starts at the post-input_layernorm hidden states x [T, hidden] (plus
positions) and ends at the module output [T, hidden] before the residual add. It
includes the module's projections, norms, RoPE, cache/state writes, selection
(indexer, compressor) and the attention itself; the tensor-parallel all-reduce,
residual, hyper-connections and MTP layers are outside it. The KV cache (or linear
attention state) holds each request's ctx tokens of random, format-valid data before
the timed call.

Shared args:
  proj: {"<checkpoint module>": {"a": operand, "b": operand}}. Operand descriptors are
    the gemm op's; a projection left out is bf16 x bf16.
  batch: {"groups": [{"count", "q", "ctx"}], "pages"?: "contiguous" | "shuffled", "seed"?}
    count requests with q new tokens each and ctx tokens already cached; ctx is an int
    or {"dist": "uniform", "min", "max"} drawn per request from seed. Whether a request
    runs the backend's prefill or decode path is the backend's choice, recorded with
    the result.
  selection (sparse modules): which tokens the selection step picks. natural: whatever
    the module's indexer computes on the random cache; uniform / recent / clustered
    force the indices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from operatorx.core.op import OpSpec
from operatorx.core.op_registry import register
from operatorx.ops.gemm import check_operand

SELECTIONS = {"natural", "uniform", "recent", "clustered"}
PAGES = {"contiguous", "shuffled"}
STATE_DTYPES = {"fp32", "bf16"}


def _pos_int(v: Any, where: str, zero: bool = False) -> None:
    if not isinstance(v, int) or isinstance(v, bool) or v < (0 if zero else 1):
        raise ValueError(f"{where} must be a {'non-negative' if zero else 'positive'} int, got {v!r}")


def _number(v: Any, where: str) -> None:
    if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
        raise ValueError(f"{where} must be a positive number, got {v!r}")


def _choice(v: Any, where: str, allowed: set) -> None:
    if v not in allowed:
        raise ValueError(f"{where} must be one of {sorted(allowed, key=str)}, got {v!r}")


def _bool(v: Any, where: str) -> None:
    if not isinstance(v, bool):
        raise ValueError(f"{where} must be a bool, got {v!r}")


def _keys(d: Any, where: str, required: set, optional: set = frozenset()) -> None:
    if not isinstance(d, dict):
        raise ValueError(f"{where} must be a dict, got {d!r}")
    missing, extra = required - set(d), set(d) - required - set(optional)
    if missing or extra:
        raise ValueError(f"{where}: missing {sorted(missing)}, unknown {sorted(extra)}")


def _check_proj(p: Any, names: tuple[str, ...]) -> None:
    _keys(p, "proj", set(), set(names))
    for name, pair in p.items():
        _keys(pair, f"proj.{name}", {"a", "b"})
        check_operand(pair["a"], f"proj.{name}.a")
        check_operand(pair["b"], f"proj.{name}.b")


def _check_batch(b: Any) -> None:
    _keys(b, "batch", {"groups"}, {"pages", "seed"})
    groups = b["groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("batch.groups must be a non-empty list")
    for i, g in enumerate(groups):
        where = f"batch.groups[{i}]"
        _keys(g, where, {"count", "q", "ctx"})
        _pos_int(g["count"], f"{where}.count")
        _pos_int(g["q"], f"{where}.q")
        ctx = g["ctx"]
        if isinstance(ctx, dict):
            _keys(ctx, f"{where}.ctx", {"dist", "min", "max"})
            _choice(ctx["dist"], f"{where}.ctx.dist", {"uniform"})
            _pos_int(ctx["min"], f"{where}.ctx.min", zero=True)
            _pos_int(ctx["max"], f"{where}.ctx.max", zero=True)
            if ctx["min"] > ctx["max"]:
                raise ValueError(f"{where}.ctx.min exceeds ctx.max")
        else:
            _pos_int(ctx, f"{where}.ctx", zero=True)
    _choice(b.get("pages", "contiguous"), "batch.pages", PAGES)
    if not isinstance(b.get("seed", 0), int) or isinstance(b.get("seed", 0), bool):
        raise ValueError("batch.seed must be an int")


def _check_rope_scaling(s: Any) -> None:
    if s is None:
        return
    _keys(s, "rope_scaling", {"type", "factor", "original_max"}, {"beta_fast", "beta_slow", "mscale", "mscale_all_dim"})
    _choice(s["type"], "rope_scaling.type", {"yarn"})
    _number(s["factor"], "rope_scaling.factor")
    _pos_int(s["original_max"], "rope_scaling.original_max")
    for k in ("beta_fast", "beta_slow", "mscale", "mscale_all_dim"):
        if k in s:
            _number(s[k], f"rope_scaling.{k}")


def _check_mrope(m: Any, rope_dim: int) -> None:
    if m is None:
        return
    if not (isinstance(m, list) and len(m) == 3 and all(isinstance(x, int) and x >= 0 for x in m)):
        raise ValueError(f"mrope_section must be three non-negative ints, got {m!r}")
    if 2 * sum(m) != rope_dim:
        raise ValueError(f"mrope_section {m} must cover rope_dim/2 = {rope_dim // 2}")


MLA_PROJ = ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "o_proj", "g_proj")


@dataclass(frozen=True)
class MlaArgs:
    """DeepSeek multi-head latent attention (DeepSeek-R1, Kimi-K3's MLA layers).

    Timed: q_a_proj + kv_a_proj_with_mqa (+ g_proj when gate) -> q_a / kv_a RMSNorm ->
    q_b_proj -> RoPE on q_pe and k_pe (when rope) -> latent cache write (kv_c, k_pe) ->
    prefill: kv_b_proj up-projection of new and cached tokens, MHA, merge of context
    chunks; decode: q_nope x W_UK, MQA over the latent cache, x W_UV -> sigmoid output
    gate (when gate) -> o_proj.

    rope false: NoPE; the rope_dim part is still cached and used, never rotated.
    kv_cache_dtype: vLLM's --kv-cache-dtype (auto = the model dtype).
    """
    hidden: int
    heads: int
    q_lora_rank: int
    kv_lora_rank: int
    nope: int
    rope_dim: int
    v: int
    batch: dict
    rope: bool = True
    rope_theta: float | None = None
    rope_scaling: dict | None = None
    gate: bool = False
    kv_cache_dtype: str = "auto"
    proj: dict | None = None

    def __post_init__(self):
        for k in ("hidden", "heads", "q_lora_rank", "kv_lora_rank", "nope", "rope_dim", "v"):
            _pos_int(getattr(self, k), k)
        _bool(self.rope, "rope")
        _bool(self.gate, "gate")
        if self.rope:
            _number(self.rope_theta, "rope_theta")
            _check_rope_scaling(self.rope_scaling)
        elif self.rope_theta is not None or self.rope_scaling is not None:
            raise ValueError("rope_theta / rope_scaling need rope")
        _choice(self.kv_cache_dtype, "kv_cache_dtype", {"auto", "bf16", "fp8"})
        _check_proj(self.proj or {}, MLA_PROJ if self.gate else MLA_PROJ[:-1])
        _check_batch(self.batch)


MLA_DSA_PROJ = ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "o_proj",
                "indexer.wq_b", "indexer.wk", "indexer.weights_proj")


@dataclass(frozen=True)
class MlaDsaArgs:
    """MLA with DeepSeek sparse attention (GLM-5.1 / 5.2 / 5.3).

    Timed: q_a_proj + kv_a_proj_with_mqa (+ indexer.wk / weights_proj when indexer is
    own) -> q_a / kv_a RMSNorm, RoPE, latent cache write (and index-K cache write when
    own) -> q_b_proj -> q_nope x W_UK -> indexer (own: indexer.wq_b, index q RoPE and
    quant, logits over the index-K cache, top-k; reuse: the top-k of an earlier layer,
    pre-filled) -> MQA over the top-k tokens (dense MHA prefill where the backend picks
    it for short sequences) -> x W_UV -> o_proj.

    kv_cache_dtype: vLLM's --kv-cache-dtype; fp8_ds_mla is the fp8 latent with per-128
    scales and a bf16 rope part.
    """
    hidden: int
    heads: int
    q_lora_rank: int
    kv_lora_rank: int
    nope: int
    rope_dim: int
    v: int
    rope_theta: float
    topk: int
    index_heads: int
    index_dim: int
    batch: dict
    indexer: str = "own"
    rope_scaling: dict | None = None
    kv_cache_dtype: str = "auto"
    selection: str = "natural"
    proj: dict | None = None

    def __post_init__(self):
        for k in ("hidden", "heads", "q_lora_rank", "kv_lora_rank", "nope", "rope_dim", "v", "topk",
                  "index_heads", "index_dim"):
            _pos_int(getattr(self, k), k)
        _number(self.rope_theta, "rope_theta")
        _check_rope_scaling(self.rope_scaling)
        _choice(self.indexer, "indexer", {"own", "reuse"})
        _choice(self.kv_cache_dtype, "kv_cache_dtype", {"auto", "bf16", "fp8", "fp8_ds_mla"})
        _choice(self.selection, "selection", SELECTIONS)
        _check_proj(self.proj or {}, MLA_DSA_PROJ if self.indexer == "own" else MLA_DSA_PROJ[:5])
        _check_batch(self.batch)


DSV4_PROJ = ("wq_a", "wkv", "wq_b", "wo_a", "wo_b", "compressor.wkv", "compressor.wgate",
             "indexer.wq_b", "indexer.wk", "indexer.weights_proj",
             "indexer.compressor.wkv", "indexer.compressor.wgate")
DSV4_RATIOS = {0, 1, 2, 4, 128}


@dataclass(frozen=True)
class Dsv4AttnArgs:
    """DeepSeek-V4 attention (DeepSeek-V4-Pro, DeepSeek-V4.1-Flash): a sliding window of
    raw tokens plus compressed tokens, one latent used as both K and V, learned sinks.

    Timed: wq_a + wkv -> q_norm / kv_norm -> wq_b -> compressor (compress_ratio > 0 and
    source: compressor.wkv (+ wgate), pooling of compress_ratio tokens, RMSNorm, RoPE,
    compressed cache write, compressor state update) -> indexer (own: indexer.wq_b and
    weights_proj, index keys (ratio 4: its own compressor; ratio 1 / 2: indexer.wk over
    the source latent), logits, top-k; shared: own queries over a pre-filled index-K
    cache; reuse: a pre-filled top-k) -> per-head q RMSNorm, RoPE, quant and window
    cache write -> sparse MQA over window + compressed tokens with attn_sink -> inverse
    RoPE -> wo_a (o_groups grouped) -> wo_b.

    compress_ratio: 4 / 128 V4-Pro (4 with an indexer, 128 attends every compressed
    token); 0 (window only), 1 / 2 V4.1. The compressor's overlap, positional embedding and
    gate follow from the ratio as in the checkpoints. source false: a layer that reads
    a pre-filled compressed cache written by another layer and has no compressor.
    rope_theta applies to window-only layers, compress_rope_theta (+ rope_scaling) to
    compressed ones. index_cache_dtype: the indexer K cache (vLLM indexer_kv_dtype).
    """
    hidden: int
    heads: int
    head_dim: int
    rope_dim: int
    q_lora_rank: int
    window: int
    o_groups: int
    o_lora_rank: int
    compress_ratio: int
    batch: dict
    source: bool = True
    indexer: str | None = None
    topk: int | None = None
    index_heads: int | None = None
    index_dim: int | None = None
    rope_theta: float = 10000.0
    compress_rope_theta: float | None = None
    rope_scaling: dict | None = None
    kv_cache_dtype: str = "auto"
    index_cache_dtype: str | None = None
    selection: str = "natural"
    proj: dict | None = None

    def __post_init__(self):
        for k in ("hidden", "heads", "head_dim", "rope_dim", "q_lora_rank", "window", "o_groups", "o_lora_rank"):
            _pos_int(getattr(self, k), k)
        if self.heads % self.o_groups:
            raise ValueError("o_groups must divide heads")
        _choice(self.compress_ratio, "compress_ratio", DSV4_RATIOS)
        _bool(self.source, "source")
        _choice(self.indexer, "indexer", {None, "own", "shared", "reuse"})
        if self.compress_ratio in (0, 128) and self.indexer is not None:
            raise ValueError(f"compress_ratio {self.compress_ratio} has no indexer")
        if self.compress_ratio == 4 and self.indexer != "own":
            raise ValueError("compress_ratio 4 runs its own indexer")
        if self.compress_ratio == 0 and not self.source:
            raise ValueError("a window-only layer has no compressed cache to share")
        index = ("topk", "index_heads", "index_dim")
        if self.indexer is None:
            if any(getattr(self, k) is not None for k in index) or self.index_cache_dtype is not None:
                raise ValueError("topk / index_heads / index_dim / index_cache_dtype need an indexer")
        else:
            for k in index:
                _pos_int(getattr(self, k), k)
            _choice(self.index_cache_dtype or "fp8", "index_cache_dtype", {"bf16", "fp8", "mxfp4", "nvfp4"})
        _number(self.rope_theta, "rope_theta")
        if self.compress_ratio:
            _number(self.compress_rope_theta, "compress_rope_theta")
            _check_rope_scaling(self.rope_scaling)
        elif self.compress_rope_theta is not None or self.rope_scaling is not None:
            raise ValueError("compress_rope_theta / rope_scaling need compress_ratio > 0")
        _choice(self.kv_cache_dtype, "kv_cache_dtype", {"auto", "bf16", "fp8", "fp8_ds_mla"})
        _choice(self.selection, "selection", SELECTIONS)
        _check_proj(self.proj or {}, DSV4_PROJ)
        _check_batch(self.batch)


GQA_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True)
class GqaArgs:
    """Dense grouped-query attention (MiniMax-M3 layers 0-2, Qwen3.5 full attention).

    Timed: q_proj (with the gate packed in when gate), k_proj, v_proj -> per-head
    q / k RMSNorm (when qk_norm) -> RoPE on the first rope_dim dims (M-RoPE when
    mrope_section) -> KV cache write -> causal attention -> sigmoid output gate (when
    gate) -> o_proj.
    """
    hidden: int
    q_heads: int
    kv_heads: int
    head_dim: int
    rope_dim: int
    rope_theta: float
    batch: dict
    mrope_section: list | None = None
    qk_norm: bool = False
    gate: bool = False
    kv_cache_dtype: str = "auto"
    proj: dict | None = None

    def __post_init__(self):
        for k in ("hidden", "q_heads", "kv_heads", "head_dim", "rope_dim"):
            _pos_int(getattr(self, k), k)
        if self.q_heads % self.kv_heads:
            raise ValueError("kv_heads must divide q_heads")
        if self.rope_dim > self.head_dim:
            raise ValueError("rope_dim exceeds head_dim")
        _number(self.rope_theta, "rope_theta")
        _check_mrope(self.mrope_section, self.rope_dim)
        _bool(self.qk_norm, "qk_norm")
        _bool(self.gate, "gate")
        _choice(self.kv_cache_dtype, "kv_cache_dtype", {"auto", "bf16", "fp8"})
        _check_proj(self.proj or {}, GQA_PROJ)
        _check_batch(self.batch)


QSA_PROJ = (*GQA_PROJ, "indexer.index_qk_proj")


@dataclass(frozen=True)
class QsaArgs:
    """Qwen3.8-Flash-Next gated sparse attention.

    Timed: q_proj (with the gate), k_proj, v_proj -> q / k RMSNorm, RoPE -> KV cache
    write -> indexer (indexer.index_qk_proj to index_heads queries and one key,
    RMSNorm, RoPE; keys pooled compress tokens at a time into the compressed index-K
    cache; score = sum over index heads of ReLU(q . k); top budget/compress groups,
    expanded to budget tokens) -> attention over the selected tokens with the sigmoid
    output gate fused -> o_proj. The KV cache is bf16.
    """
    hidden: int
    q_heads: int
    kv_heads: int
    head_dim: int
    rope_dim: int
    rope_theta: float
    index_heads: int
    index_dim: int
    compress: int
    budget: int
    batch: dict
    mrope_section: list | None = None
    selection: str = "natural"
    proj: dict | None = None

    def __post_init__(self):
        for k in ("hidden", "q_heads", "kv_heads", "head_dim", "rope_dim", "index_heads", "index_dim",
                  "compress", "budget"):
            _pos_int(getattr(self, k), k)
        if self.q_heads % self.kv_heads:
            raise ValueError("kv_heads must divide q_heads")
        if self.budget % self.compress:
            raise ValueError("compress must divide budget")
        _number(self.rope_theta, "rope_theta")
        _check_mrope(self.mrope_section, self.rope_dim)
        _choice(self.selection, "selection", SELECTIONS)
        _check_proj(self.proj or {}, QSA_PROJ)
        _check_batch(self.batch)


GDN_PROJ = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")


@dataclass(frozen=True)
class GdnArgs:
    """Gated DeltaNet linear attention (Qwen3.5 / Qwen3.8 linear_attn layers).

    Timed: in_proj_qkv, in_proj_z, in_proj_a, in_proj_b -> causal depthwise conv1d
    (conv_kernel) with conv-state update -> decay g and beta gates -> q / k L2 norm ->
    delta rule (chunked for prefill, recurrent for decode) reading and writing the
    recurrent state -> RMSNorm gated by z (norm_act) -> out_proj. ctx > 0 means the
    request starts from a pre-filled state.
    """
    hidden: int
    qk_heads: int
    v_heads: int
    head_dim: int
    conv_kernel: int
    batch: dict
    norm_act: str = "silu"
    state_dtype: str = "fp32"
    proj: dict | None = None

    def __post_init__(self):
        for k in ("hidden", "qk_heads", "v_heads", "head_dim", "conv_kernel"):
            _pos_int(getattr(self, k), k)
        if self.v_heads % self.qk_heads:
            raise ValueError("qk_heads must divide v_heads")
        _choice(self.norm_act, "norm_act", {"silu", "sigmoid"})
        _choice(self.state_dtype, "state_dtype", STATE_DTYPES)
        _check_proj(self.proj or {}, GDN_PROJ)
        _check_batch(self.batch)


KDA_PROJ = ("q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "f_b_proj", "b_proj", "o_proj")


@dataclass(frozen=True)
class KdaArgs:
    """Kimi Delta Attention (Kimi-K3 linear layers).

    Timed: q_proj, k_proj, v_proj, g_proj, f_a_proj, b_proj -> f_b_proj (the
    head_dim-rank per-channel decay gate) -> causal depthwise conv1d on q / k / v with
    conv-state update -> decay and beta gates -> gated delta rule with q / k L2 norm,
    reading and writing the recurrent state -> RMSNorm gated by g -> o_proj.
    """
    hidden: int
    heads: int
    head_dim: int
    conv_kernel: int
    batch: dict
    state_dtype: str = "fp32"
    proj: dict | None = None

    def __post_init__(self):
        for k in ("hidden", "heads", "head_dim", "conv_kernel"):
            _pos_int(getattr(self, k), k)
        _choice(self.state_dtype, "state_dtype", STATE_DTYPES)
        _check_proj(self.proj or {}, KDA_PROJ)
        _check_batch(self.batch)


for _spec in (
    OpSpec(type="mla", arg_schema=MlaArgs, description="DeepSeek MLA module"),
    OpSpec(type="mla_dsa", arg_schema=MlaDsaArgs, description="MLA with DeepSeek sparse attention"),
    OpSpec(type="dsv4_attn", arg_schema=Dsv4AttnArgs, description="DeepSeek-V4 window + compressed attention"),
    OpSpec(type="gqa", arg_schema=GqaArgs, description="dense grouped-query attention module"),
    OpSpec(type="qsa", arg_schema=QsaArgs, description="Qwen3.8 gated sparse attention"),
    OpSpec(type="gdn", arg_schema=GdnArgs, description="Gated DeltaNet linear attention"),
    OpSpec(type="kda", arg_schema=KdaArgs, description="Kimi Delta Attention"),
):
    register(_spec)
