# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness coverage for fused route reduction, all-reduce, and RMSNorm."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.jit.utils.chip_info import get_gfx
from torch.multiprocessing import spawn

_TP_SIZE = 8
_TOKENS = 7
_TOPK = 16
_HIDDEN_SIZE = 3584
_EPS = 1e-5
_DTYPE = torch.bfloat16


def _make_routes(rank: int, phase: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(
        _TOKENS * _TOPK * _HIDDEN_SIZE,
        dtype=torch.float32,
        device=device,
    ).reshape(_TOKENS, _TOPK, _HIDDEN_SIZE)
    values = (values.remainder(251) - 125).mul_(0.0005)
    values.add_(rank * 0.002 + phase * 0.003)
    return values.to(_DTYPE)


def _make_weight(device: torch.device) -> torch.Tensor:
    values = torch.arange(_HIDDEN_SIZE, dtype=torch.float32, device=device)
    return (1.0 + values.remainder(37).mul_(0.0005)).to(_DTYPE)


def _ordinary_route_reduce_ar_rms(
    routes: torch.Tensor,
    weight: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    local_sum = torch.zeros(
        (_TOKENS, _HIDDEN_SIZE),
        dtype=torch.float32,
        device=routes.device,
    )
    for slot in range(_TOPK):
        local_sum.add_(routes[:, slot].float())
    reduced = local_sum.to(_DTYPE)
    dist.all_reduce(reduced, group=group)
    return F.rms_norm(reduced, (_HIDDEN_SIZE,), weight=weight, eps=_EPS)


def _contract_reference(
    phase: int,
    weight: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    reduced = torch.zeros(
        (_TOKENS, _HIDDEN_SIZE),
        dtype=torch.float32,
        device=device,
    )
    for rank in range(_TP_SIZE):
        routes = _make_routes(rank, phase, device)
        local_sum = torch.zeros_like(reduced)
        for slot in range(_TOPK):
            local_sum.add_(routes[:, slot].float())
        reduced.add_(local_sum.to(_DTYPE).float())
    reduced = reduced.to(_DTYPE)
    return F.rms_norm(reduced, (_HIDDEN_SIZE,), weight=weight, eps=_EPS)


def _assert_matches_reference(
    actual: torch.Tensor,
    routes: torch.Tensor,
    weight: torch.Tensor,
    group: dist.ProcessGroup,
    phase: int,
) -> None:
    contract = _contract_reference(phase, weight, routes.device)
    ordinary = _ordinary_route_reduce_ar_rms(routes, weight, group)
    torch.testing.assert_close(actual, contract, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual, ordinary, atol=1e-2, rtol=1e-2)


def _run_rank(
    rank: int,
    world_size: int,
    distributed_init_method: str,
    with_graph: bool,
) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(world_size, 1)

    try:
        tp_group = get_tp_group()
        group = tp_group.device_group
        ca_comm = tp_group.device_communicator.ca_comm
        assert ca_comm is not None and not ca_comm.disabled

        routes = _make_routes(rank, phase=0, device=device)
        weight = _make_weight(device)
        assert ca_comm.should_custom_ar(routes)

        dist.barrier(group=group)
        if with_graph:
            graph = torch.cuda.CUDAGraph()
            with (
                graph_capture() as graph_context,
                torch.cuda.graph(graph, stream=graph_context.stream),
            ):
                output = ca_comm.custom_fused_route_reduce_ar_rms(
                    routes,
                    weight,
                    _EPS,
                )
            assert output is not None
            assert output.shape == (_TOKENS, _HIDDEN_SIZE)
            assert output.is_contiguous()

            output.zero_()
            graph.replay()
            torch.cuda.synchronize()
            first_output = output.clone()
            _assert_matches_reference(first_output, routes, weight, group, phase=0)

            routes.copy_(_make_routes(rank, phase=1, device=device))
            dist.barrier(group=group)
            output.zero_()
            graph.replay()
            torch.cuda.synchronize()
            second_output = output.clone()
            _assert_matches_reference(second_output, routes, weight, group, phase=1)
            assert not torch.equal(first_output, second_output)
        else:
            output = ca_comm.custom_fused_route_reduce_ar_rms(
                routes,
                weight,
                _EPS,
            )
            assert output is not None
            assert output.shape == (_TOKENS, _HIDDEN_SIZE)
            assert output.is_contiguous()
            _assert_matches_reference(output, routes, weight, group, phase=0)
    finally:
        if dist.is_initialized():
            destroy_model_parallel()
            destroy_distributed_environment()
        torch.cuda.empty_cache()


@pytest.mark.parametrize("with_graph", [False, True], ids=["eager", "graph"])
def test_fused_route_reduce_allreduce_rmsnorm_tp8(with_graph: bool) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < _TP_SIZE:
        pytest.skip("requires eight gfx950 GPUs")
    if get_gfx() != "gfx950":
        pytest.skip("requires eight gfx950 GPUs")
    distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
    spawn(
        _run_rank,
        args=(_TP_SIZE, distributed_init_method, with_graph),
        nprocs=_TP_SIZE,
        join=True,
    )
