# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the InferenceX project

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)
from vllm.platforms import current_platform


def run_shared(shared: SharedExperts, value: torch.Tensor) -> torch.Tensor:
    shared.maybe_sync_shared_experts_stream(value)
    shared(value, SharedExpertsOrder.MULTI_STREAM_OVERLAPPED)
    return shared.output


def main() -> None:
    result_dir = Path(os.environ["K3_SHARED_STREAM_TEST_RESULT_DIR"])
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    result_dir.mkdir(parents=True, exist_ok=True)

    if torch.version.hip is None or not current_platform.is_rocm():
        raise RuntimeError("ROCm shared-expert stream test requires a ROCm runtime")

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.manual_seed(20260902 + rank)

    width = 256
    expansion = 384
    layer = torch.nn.Sequential(
        torch.nn.Linear(
            width,
            expansion,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        ),
        torch.nn.SiLU(),
        torch.nn.Linear(
            expansion,
            width,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        ),
    )
    parallel_config = SimpleNamespace(
        enable_eplb=False,
        use_fi_nvl_two_sided_kernels=False,
    )
    shared = SharedExperts(
        layer,
        moe_config=SimpleNamespace(moe_parallel_config=parallel_config),
        enable_dbo=False,
        mk_can_overlap_shared_experts=lambda: False,
    )

    probe = torch.randn(7, width, device=device, dtype=torch.bfloat16)
    order = shared._determine_shared_experts_order(probe)
    if order != SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
        raise AssertionError(f"M=7 selected unexpected shared-expert order: {order}")
    if shared._stream is None:
        raise AssertionError("ROCm auxiliary stream was not created")

    threshold = envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
    if threshold < probe.shape[0]:
        raise AssertionError(
            f"shared-expert stream threshold is below M=7: {threshold}"
        )
    large_probe = torch.empty(threshold + 1, width, device=device, dtype=torch.bfloat16)
    large_order = shared._determine_shared_experts_order(large_probe)
    if large_order != SharedExpertsOrder.NO_OVERLAP:
        raise AssertionError(
            f"M={threshold + 1} selected unexpected shared-expert order: {large_order}"
        )

    eager_expected = layer(probe)
    eager_actual = run_shared(shared, probe)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(eager_actual, eager_expected, rtol=2e-2, atol=2e-2)

    static_input = torch.randn(7, width, device=device, dtype=torch.bfloat16)
    warmup_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            run_shared(shared, static_input)
    warmup_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = run_shared(shared, static_input)

    changed_input = torch.randn(7, width, device=device, dtype=torch.bfloat16)
    changed_expected = layer(changed_input)
    static_input.copy_(changed_input)
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(graph_output, changed_expected, rtol=2e-2, atol=2e-2)

    max_abs_error = float(
        (graph_output.float() - changed_expected.float()).abs().max().item()
    )
    result = {
        "device": str(device),
        "hip_version": torch.version.hip,
        "max_abs_error": max_abs_error,
        "rank": rank,
        "selected_order": order.name,
        "status": "passed",
        "world_size": world_size,
    }
    output = result_dir / f"rank_{rank}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
