from __future__ import annotations

from dataclasses import dataclass

from operatorx.core.op import OpSpec
from operatorx.core.op_registry import register


@dataclass(frozen=True)
class AllreduceArgs:
    num_elements: int
    dtype: str
    world_size: int


@dataclass(frozen=True)
class AllgatherArgs:
    num_elements_per_rank: int
    dtype: str
    world_size: int


@dataclass(frozen=True)
class ReduceScatterArgs:
    num_elements: int       # total elements; output per rank is num_elements // world_size
    dtype: str
    world_size: int


@dataclass(frozen=True)
class AlltoallArgs:
    num_elements_per_rank: int   # total elements per rank, split into world_size chunks
    dtype: str
    world_size: int


@dataclass(frozen=True)
class DispatchArgs:
    num_tokens: int
    num_experts: int
    top_k: int
    hidden: int
    dtype: str
    world_size: int


@dataclass(frozen=True)
class CombineArgs:
    num_tokens: int
    num_experts: int
    top_k: int
    hidden: int
    dtype: str
    world_size: int


ALLREDUCE = OpSpec(
    type="allreduce", arg_schema=AllreduceArgs,
    description="AllReduce",
)
ALLGATHER = OpSpec(
    type="allgather", arg_schema=AllgatherArgs,
    description="AllGather",
)
REDUCE_SCATTER = OpSpec(
    type="reduce_scatter", arg_schema=ReduceScatterArgs,
    description="ReduceScatter",
)
ALLTOALL = OpSpec(
    type="alltoall", arg_schema=AlltoallArgs,
    description="AlltoAll",
)
DISPATCH = OpSpec(
    type="dispatch", arg_schema=DispatchArgs,
    description="MoE Dispatch",
)
COMBINE = OpSpec(
    type="combine", arg_schema=CombineArgs,
    description="MoE Combine",
)

register(ALLREDUCE)
register(ALLGATHER)
register(REDUCE_SCATTER)
register(ALLTOALL)
register(DISPATCH)
register(COMBINE)
