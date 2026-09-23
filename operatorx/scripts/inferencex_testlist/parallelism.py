"""TP / EP / DP-attn sharding helpers.

The InferenceX matrix exposes `tp`, `ep`, `dp_attn`. We translate those into:
  - `attn_tp`: how attention weight matrices are sharded across heads (TP).
  - `attn_dp`: data-parallel factor for the attention block; when `dp_attn`
    is true, attention is replicated per rank and each rank processes a
    `1/attn_tp` slice of the batch instead of sharding heads.
  - `moe_tp`: TP shard count for MoE expert weights (only when EP doesn't
    cover the full world).
  - `moe_ep`: expert-parallel rank count.
  - `world_size`: max of the parallelism dimensions, used for collectives.

This mirrors the SGLang / vLLM convention where `ep > 1` shards experts
across ranks while attention stays TP-replicated (or DP-replicated under
`dp_attn=true`).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Parallelism:
    tp: int
    ep: int
    dp_attn: bool

    @property
    def world_size(self) -> int:
        # In InferenceX configs `tp` is the total tensor-parallel rank count
        # and `ep` <= tp partitions experts across a subset (or all) of those
        # ranks. World size for collectives equals tp.
        return self.tp

    @property
    def attn_tp(self) -> int:
        """Attention head-sharding factor."""
        return 1 if self.dp_attn else self.tp

    @property
    def attn_dp(self) -> int:
        """Attention data-parallel factor (batch is split this many ways)."""
        return self.tp if self.dp_attn else 1

    @property
    def moe_ep(self) -> int:
        return max(1, self.ep)

    @property
    def moe_tp(self) -> int:
        # Remaining TP after EP carves out its dimension; SGLang convention.
        return max(1, self.tp // self.moe_ep)
