# SPDX-License-Identifier: Apache-2.0
"""Benchmark a portable ROCm Tier-0 Kimi-K3 latent-MoE tail topology.

The benchmark compares the current Tier-2 tail against a reduce-scatter,
sharded projection, and all-gather decomposition.  The overlap arm uses a
second AITER communicator so the routed all-reduce and shared reduce-scatter
do not alias the same registered IPC workspace.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiter
import torch
import torch.distributed as dist
import torch.nn.functional as F
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import get_tp_group
from vllm.distributed.device_communicators.aiter_custom_all_reduce import (
    AiterCustomAllreduce,
)
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)

NUM_TOKENS = 7
HIDDEN_SIZE = 7168
LATENT_SIZE = 3584
TP_SIZE = 8
SHARD_SIZE = HIDDEN_SIZE // TP_SIZE
RMS_EPS = 1.0e-6


@dataclass
class Case:
    routed: torch.Tensor
    shared: torch.Tensor
    shared_workspace: torch.Tensor
    rms_weight: torch.Tensor
    up_weight_shard: torch.Tensor
    rs_workspace: torch.Tensor
    start_event: torch.cuda.Event
    done_event: torch.cuda.Event


@dataclass
class CapturedPath:
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    initial_output: torch.Tensor
    reset: Callable[[], None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotations", type=int, default=48)
    parser.add_argument("--warmup-replays", type=int, default=10)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(samples: Sequence[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    upper_weight = position - lower
    return ordered[lower] * (1.0 - upper_weight) + ordered[upper] * upper_weight


def summarize(samples_us: Sequence[float]) -> dict[str, Any]:
    mean_us = statistics.mean(samples_us)
    return {
        "median_us": statistics.median(samples_us),
        "p10_us": percentile(samples_us, 0.1),
        "p90_us": percentile(samples_us, 0.9),
        "mean_us": mean_us,
        "cv_pct": statistics.pstdev(samples_us) / mean_us * 100.0,
        "samples_us": list(samples_us),
    }


def make_cases(
    rotations: int,
    rank: int,
    device: torch.device,
    seed: int,
) -> list[Case]:
    cases = []
    for index in range(rotations):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + 1009 * index + 17 * rank)
        routed = torch.randn(
            (NUM_TOKENS, LATENT_SIZE),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).mul_(0.01)
        shared = torch.randn(
            (NUM_TOKENS, HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        rms_weight = 1 + 0.1 * torch.randn(
            LATENT_SIZE,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        up_weight_shard = (
            torch.randn(
                (SHARD_SIZE, LATENT_SIZE),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            / LATENT_SIZE**0.5
        )
        cases.append(
            Case(
                routed=routed,
                shared=shared,
                shared_workspace=shared.clone(),
                rms_weight=rms_weight,
                up_weight_shard=up_weight_shard,
                rs_workspace=torch.empty(
                    (NUM_TOKENS, SHARD_SIZE),
                    dtype=torch.bfloat16,
                    device=device,
                ),
                start_event=torch.cuda.Event(),
                done_event=torch.cuda.Event(),
            )
        )
    return cases


def custom_all_reduce(ca: Any, value: torch.Tensor) -> torch.Tensor:
    output = ca.custom_all_reduce(value)
    if output is None:
        raise RuntimeError(
            f"AITER custom all-reduce rejected shape {tuple(value.shape)}"
        )
    return output


def custom_reduce_scatter(
    ca: Any,
    value: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    if not ca.should_custom_rs(value, -1):
        raise RuntimeError(
            f"AITER custom reduce-scatter rejected shape {tuple(value.shape)}"
        )
    ca.custom_reduce_scatter(value, output, dim=-1)
    return output


def custom_all_gather(ca: Any, value: torch.Tensor) -> torch.Tensor:
    if not ca.should_custom_ag(value):
        raise RuntimeError(
            f"AITER custom all-gather rejected shape {tuple(value.shape)}"
        )
    output = ca.custom_all_gather(value, dim=-1)
    if output is None:
        raise RuntimeError(
            f"AITER custom all-gather rejected shape {tuple(value.shape)}"
        )
    return output


def make_tier2_launch(
    case: Case, main_ca: Any, rank: int
) -> Callable[[], torch.Tensor]:
    shard_start = rank * SHARD_SIZE

    def launch() -> torch.Tensor:
        latent = custom_all_reduce(main_ca, case.routed)
        latent = aiter.rmsnorm2d_fwd(latent, case.rms_weight, RMS_EPS)
        hidden_shard = case.shared_workspace.narrow(-1, shard_start, SHARD_SIZE)
        hidden_shard.addmm_(latent, case.up_weight_shard.t())
        return custom_all_reduce(main_ca, case.shared_workspace)

    return launch


def make_tier0_sequential_launch(
    case: Case,
    main_ca: Any,
    shared_ca: Any,
) -> Callable[[], torch.Tensor]:
    def launch() -> torch.Tensor:
        latent = custom_all_reduce(main_ca, case.routed)
        latent = aiter.rmsnorm2d_fwd(latent, case.rms_weight, RMS_EPS)
        shared_shard = custom_reduce_scatter(
            shared_ca,
            case.shared,
            case.rs_workspace,
        )
        local_output = torch.addmm(
            shared_shard,
            latent,
            case.up_weight_shard.t(),
        )
        return custom_all_gather(shared_ca, local_output)

    return launch


def make_tier0_overlap_launch(
    case: Case,
    main_ca: Any,
    shared_ca: Any,
    aux_stream: torch.cuda.Stream,
) -> Callable[[], torch.Tensor]:
    def launch() -> torch.Tensor:
        case.start_event.record()
        with torch.cuda.stream(aux_stream):
            case.start_event.wait()
            custom_reduce_scatter(shared_ca, case.shared, case.rs_workspace)
            case.done_event.record()

        latent = custom_all_reduce(main_ca, case.routed)
        latent = aiter.rmsnorm2d_fwd(latent, case.rms_weight, RMS_EPS)
        local_output = torch.mm(latent, case.up_weight_shard.t())
        case.done_event.wait()
        local_output.add_(case.rs_workspace)
        return custom_all_gather(shared_ca, local_output)

    return launch


def reset_tier2(cases: Sequence[Case]) -> None:
    for case in cases:
        case.shared_workspace.copy_(case.shared)


def capture_path(
    launches: Sequence[Callable[[], torch.Tensor]],
    reset: Callable[[], None],
    capture_comms: Sequence[Any],
    cpu_group: dist.ProcessGroup,
) -> CapturedPath:
    with ExitStack() as stack:
        for communicator in capture_comms:
            stack.enter_context(communicator.capture())

        for _ in range(3):
            reset()
            for launch in launches:
                output = launch()
        torch.cuda.synchronize()

        dist.barrier(group=cpu_group)
        reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for launch in launches:
                output = launch()
        torch.cuda.synchronize()

    initial_output = output.clone()
    torch.cuda.synchronize()
    return CapturedPath(
        graph=graph,
        output=output,
        initial_output=initial_output,
        reset=reset,
    )


def expected_output(
    case: Case,
    device_group: dist.ProcessGroup,
    world_size: int,
) -> torch.Tensor:
    routed = case.routed.clone()
    dist.all_reduce(routed, group=device_group)
    normalized = F.rms_norm(
        routed.float(),
        (LATENT_SIZE,),
        case.rms_weight.float(),
        RMS_EPS,
    )

    full_weight = torch.empty(
        (world_size * SHARD_SIZE, LATENT_SIZE),
        dtype=case.up_weight_shard.dtype,
        device=case.up_weight_shard.device,
    )
    dist.all_gather_into_tensor(
        full_weight,
        case.up_weight_shard,
        group=device_group,
    )
    shared = case.shared.clone()
    dist.all_reduce(shared, group=device_group)
    return (
        F.linear(normalized, full_weight.float())
        .add_(shared.float())
        .to(torch.bfloat16)
    )


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = (actual.float() - expected.float()).abs()
    max_abs = float(delta.max().item())
    denominator = expected.float().abs().clamp_min(1.0e-5)
    max_rel = float((delta / denominator).max().item())
    relative_l2 = float(
        torch.linalg.vector_norm(delta)
        .div(torch.linalg.vector_norm(expected.float()).clamp_min(1.0e-12))
        .item()
    )
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "mean_abs": float(delta.mean().item()),
        "relative_l2": relative_l2,
    }


def validate_changed_input_replay(
    paths: dict[str, CapturedPath],
    cases: Sequence[Case],
    device_group: dist.ProcessGroup,
    cpu_group: dist.ProcessGroup,
    world_size: int,
) -> dict[str, dict[str, float]]:
    case = cases[-1]
    case.routed.add_(0.03125)
    case.shared.mul_(0.9375).add_(0.015625)
    case.up_weight_shard.add_(0.0009765625)
    case.rms_weight.mul_(0.9921875)
    expected = expected_output(case, device_group, world_size)

    errors: dict[str, dict[str, float]] = {}
    replayed_outputs: dict[str, torch.Tensor] = {}
    failures = []
    for name, path in paths.items():
        path.reset()
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
        path.graph.replay()
        torch.cuda.synchronize()
        actual = path.output.clone()
        replayed_outputs[name] = actual
        fp32_error = error_metrics(actual, expected)
        changed_input_delta = error_metrics(actual, path.initial_output)
        if fp32_error["max_abs"] > 1.0 or fp32_error["relative_l2"] > 0.02:
            failures.append(
                f"{name} differs materially from the FP32 reference: {fp32_error}"
            )
        if changed_input_delta["relative_l2"] < 0.001:
            failures.append(
                f"{name} did not respond to changed graph inputs: {changed_input_delta}"
            )
        error_tensor = torch.tensor(
            [
                fp32_error["max_abs"],
                fp32_error["max_rel"],
                fp32_error["mean_abs"],
                fp32_error["relative_l2"],
                changed_input_delta["relative_l2"],
            ],
            dtype=torch.float64,
            device=case.routed.device,
        )
        dist.all_reduce(error_tensor, op=dist.ReduceOp.MAX, group=device_group)
        errors[name] = {
            "max_abs": float(error_tensor[0].item()),
            "max_rel": float(error_tensor[1].item()),
            "mean_abs": float(error_tensor[2].item()),
            "relative_l2": float(error_tensor[3].item()),
            "changed_vs_capture_relative_l2": float(error_tensor[4].item()),
        }

    baseline = replayed_outputs["tier2_baseline"]
    for name in ("tier0_sequential", "tier0_overlap"):
        topology_error = error_metrics(replayed_outputs[name], baseline)
        errors[name].update(
            {f"vs_tier2_{key}": value for key, value in topology_error.items()}
        )
        if topology_error["max_abs"] > 1.0 or topology_error["relative_l2"] > 0.02:
            failures.append(f"{name} differs materially from Tier-2: {topology_error}")

    if failures:
        raise AssertionError(
            "changed-input graph validation failed:\n"
            + "\n".join(failures)
            + "\nmetrics="
            + json.dumps(errors, sort_keys=True)
        )
    return errors


def benchmark_paths(
    paths: dict[str, CapturedPath],
    warmup_replays: int,
    samples: int,
    operations_per_replay: int,
    device_group: dist.ProcessGroup,
    cpu_group: dist.ProcessGroup,
) -> tuple[dict[str, dict[str, Any]], list[list[str]]]:
    for path in paths.values():
        for _ in range(warmup_replays):
            path.reset()
            path.graph.replay()
        torch.cuda.synchronize()

    sample_map: dict[str, list[float]] = {name: [] for name in paths}
    orders = []
    names = tuple(paths)
    for trial in range(samples):
        shift = trial % len(names)
        order = names[shift:] + names[:shift]
        if (trial // len(names)) % 2:
            order = tuple(reversed(order))
        orders.append(list(order))

        for name in order:
            path = paths[name]
            path.reset()
            torch.cuda.synchronize()
            dist.barrier(group=cpu_group)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            path.graph.replay()
            end.record()
            end.synchronize()
            elapsed = torch.tensor(
                start.elapsed_time(end) * 1000.0 / operations_per_replay,
                dtype=torch.float64,
                device=torch.cuda.current_device(),
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=device_group)
            sample_map[name].append(float(elapsed.item()))

    return {name: summarize(values) for name, values in sample_map.items()}, orders


def run(args: argparse.Namespace) -> None:
    if not {"RANK", "WORLD_SIZE", "LOCAL_RANK"} <= os.environ.keys():
        raise RuntimeError("launch this benchmark with torchrun")
    if args.rotations < 2 or args.warmup_replays < 0 or args.samples <= 0:
        raise ValueError("invalid benchmark iteration counts")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != TP_SIZE:
        raise RuntimeError(f"expected TP={TP_SIZE}, got world_size={world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    tp_group = get_tp_group()
    device_group = tp_group.device_group
    cpu_group = dist.new_group(backend="gloo")

    properties = torch.cuda.get_device_properties(device)
    arch = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if (
        not torch.version.hip
        or arch != "gfx950"
        or properties.multi_processor_count != 256
    ):
        raise RuntimeError(
            "expected a 256-CU gfx950 ROCm device, got "
            f"hip={torch.version.hip!r}, arch={arch!r}, "
            f"cu={properties.multi_processor_count}"
        )

    main_wrapper = getattr(tp_group.device_communicator, "aiter_ar_comm", None)
    if main_wrapper is None or main_wrapper.disabled:
        raise RuntimeError("vLLM did not initialize the AITER custom all-reduce")
    main_ca = main_wrapper.aiter_ca
    shared_wrapper = AiterCustomAllreduce(tp_group.cpu_group, device)
    if shared_wrapper.disabled:
        raise RuntimeError("second AITER communicator is disabled")
    shared_ca = shared_wrapper.aiter_ca

    try:
        cases = make_cases(args.rotations, rank, device, args.seed)
        probe = cases[0]
        if not main_ca.should_custom_ar(probe.routed):
            raise RuntimeError("routed M=7 tensor does not select AITER custom AR")
        if not main_ca.should_custom_ar(probe.shared):
            raise RuntimeError("shared M=7 tensor does not select AITER custom AR")
        if not shared_ca.should_custom_rs(probe.shared, -1):
            raise RuntimeError("shared M=7 tensor does not select AITER custom RS")
        if not shared_ca.should_custom_ag(probe.rs_workspace):
            raise RuntimeError("local M=7 shard does not select AITER custom AG")

        aux_stream = torch.cuda.Stream()
        launches = {
            "tier2_baseline": [
                make_tier2_launch(case, main_ca, rank) for case in cases
            ],
            "tier0_sequential": [
                make_tier0_sequential_launch(case, main_ca, shared_ca) for case in cases
            ],
            "tier0_overlap": [
                make_tier0_overlap_launch(case, main_ca, shared_ca, aux_stream)
                for case in cases
            ],
        }

        def no_reset() -> None:
            return None

        paths = {
            "tier2_baseline": capture_path(
                launches["tier2_baseline"],
                lambda: reset_tier2(cases),
                (main_ca,),
                cpu_group,
            ),
            "tier0_sequential": capture_path(
                launches["tier0_sequential"],
                no_reset,
                (main_ca, shared_ca),
                cpu_group,
            ),
            "tier0_overlap": capture_path(
                launches["tier0_overlap"],
                no_reset,
                (main_ca, shared_ca),
                cpu_group,
            ),
        }

        changed_input_errors = validate_changed_input_replay(
            paths,
            cases,
            device_group,
            cpu_group,
            world_size,
        )
        timings, timing_orders = benchmark_paths(
            paths,
            args.warmup_replays,
            args.samples,
            args.rotations,
            device_group,
            cpu_group,
        )
        baseline_us = timings["tier2_baseline"]["median_us"]
        report = {
            "shape": "Kimi-K3 M=7 BF16 latent-MoE distributed tail",
            "runtime": {
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "device": properties.name,
                "arch": arch,
                "cu_num": properties.multi_processor_count,
                "world_size": world_size,
            },
            "route_support": {
                "routed_custom_all_reduce": True,
                "shared_custom_all_reduce": True,
                "shared_custom_reduce_scatter_last_dim": True,
                "local_custom_all_gather_last_dim": True,
                "fully_connected": bool(main_ca.fully_connected),
                "dual_communicator_overlap": True,
            },
            "rotations": args.rotations,
            "rotating_sharded_weight_bytes_per_rank": (
                args.rotations * SHARD_SIZE * LATENT_SIZE * 2
            ),
            "warmup_replays": args.warmup_replays,
            "samples": args.samples,
            "changed_input_graph_replay": changed_input_errors,
            "timings": timings,
            "baseline_over_path_speedup": {
                name: baseline_us / result["median_us"]
                for name, result in timings.items()
            },
            "timing_orders": timing_orders,
        }
        if rank == 0:
            rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered, encoding="utf-8")
            print(rendered, end="")
    finally:
        shared_wrapper.close()
        dist.barrier(group=cpu_group)


def main() -> None:
    args = parse_args()
    with set_current_vllm_config(VllmConfig()):
        run(args)


if __name__ == "__main__":
    main()
