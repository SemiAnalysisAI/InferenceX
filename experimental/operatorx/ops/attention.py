from __future__ import annotations

from dataclasses import dataclass

from operatorx.core.op import OpSpec
from operatorx.core.op_registry import register


@dataclass(frozen=True)
class AttentionMhaArgs:
    batch_size: int
    seq_len_q: int
    seq_len_kv: int
    num_heads: int
    num_heads_kv: int
    head_dim: int
    dtype_q: str = "bf16"
    dtype_k: str = "bf16"
    dtype_v: str = "bf16"
    dtype_o: str = "bf16"
    causal: bool = True
    sliding_window: int | None = None
    kv_layout: str = "contig"   # "contig" | "paged"
    block_size: int | None = None


@dataclass(frozen=True)
class AttentionMlaArgs:
    batch_size: int
    seq_len_q: int
    seq_len_kv: int
    num_heads: int
    head_dim_qk_nope: int
    head_dim_qk_rope: int
    head_dim_v: int
    kv_lora_rank: int
    dtype_q: str = "bf16"
    dtype_kv: str = "bf16"          # compressed cache + rope share one dtype in practice
    dtype_o: str = "bf16"
    rope: bool = True
    causal: bool = True


ATTENTION_MHA = OpSpec(
    type="attention_mha",
    arg_schema=AttentionMhaArgs,
    description="Vanilla multi-head attention",
)

ATTENTION_MLA = OpSpec(
    type="attention_mla",
    arg_schema=AttentionMlaArgs,
    description="Multi-head latent attention with compressed KV cache.",
)

register(ATTENTION_MHA)
register(ATTENTION_MLA)
