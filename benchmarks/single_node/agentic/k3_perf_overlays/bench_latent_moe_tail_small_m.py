# SPDX-License-Identifier: MIT
"""Benchmark Kimi-K3 BF16 projection and fused-tail graph decode paths."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
from aiter.ops.flydsl.latent_moe_tail import latent_moe_tail

LATENT = 3584
HIDDEN = 7168
EPSILON = 1.0e-6
ROTATIONS = 8

Case = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
Operation = Callable[[Case], torch.Tensor]
CapturedGraph = tuple[torch.cuda.CUDAGraph, list[torch.Tensor]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, nargs="+", default=[1, 2, 7, 14])
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--operations-per-graph", type=int, default=24)
    parser.add_argument("--warmup-operations", type=int, default=2400)
    parser.add_argument("--replays-per-trial", type=int, default=20)
    parser.add_argument("--trials", type=int, default=21)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_cases(num_tokens: int, seed: int) -> list[Case]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = []
    for _ in range(ROTATIONS):
        routed = torch.randn(
            (num_tokens, LATENT),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        shared = torch.randn(
            (num_tokens, HIDDEN),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        rms_weight = torch.randn(
            (LATENT,), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        up_weight = torch.randn(
            (HIDDEN, LATENT),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ).mul_(LATENT**-0.5)
        result.append((routed, shared, rms_weight, up_weight))
    return result


def load_projection_paths() -> tuple[Operation, Operation]:
    import aiter
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    def torch_mm(case: Case) -> torch.Tensor:
        routed, shared, rms_weight, up_weight = case
        normalized = aiter.rmsnorm2d_fwd(routed, rms_weight, EPSILON)
        return torch.mm(normalized, up_weight.T).add_(shared)

    def control(case: Case) -> torch.Tensor:
        routed, shared, rms_weight, up_weight = case
        normalized = aiter.rmsnorm2d_fwd(routed, rms_weight, EPSILON)
        return rocm_unquantized_gemm_impl(normalized, up_weight).add_(shared)

    return torch_mm, control


def candidate(case: Case) -> torch.Tensor:
    routed, shared, rms_weight, up_weight = case
    return latent_moe_tail(routed, shared, rms_weight, up_weight, EPSILON)


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual.float() - expected.float()
    scale = expected.float().square().mean().sqrt().clamp_min(1.0e-12)
    return {
        "relative_rmse": (difference.square().mean().sqrt() / scale).item(),
        "max_abs_error": difference.abs().max().item(),
    }


def assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    try:
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.015625)
    except AssertionError as error:
        raise AssertionError(f"{label} failed correctness: {error}") from error


def capture(operation: Operation, cases: list[Case], operations: int) -> CapturedGraph:
    for case in cases:
        operation(case)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    outputs = []
    with torch.cuda.graph(graph):
        for index in range(operations):
            outputs.append(operation(cases[index % len(cases)]))
    return graph, outputs


def validate_changed_input_replay(
    graphs: dict[str, CapturedGraph],
    cases: list[Case],
    control: Operation,
    num_tokens: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    changed_cases = make_cases(num_tokens, seed)
    for destinations, sources in zip(cases, changed_cases, strict=True):
        for destination, source in zip(destinations, sources, strict=True):
            destination.copy_(source)
    torch.cuda.synchronize()

    expected = [control(case) for case in cases]
    torch.cuda.synchronize()
    results = {}
    for name, (graph, outputs) in graphs.items():
        graph.replay()
        torch.cuda.synchronize()
        metrics = []
        for case_index in range(len(cases)):
            output_index = max(
                index
                for index in range(len(outputs))
                if index % len(cases) == case_index
            )
            actual = outputs[output_index]
            assert_close(
                actual,
                expected[case_index],
                f"M={num_tokens} {name} changed-input graph replay case {case_index}",
            )
            metrics.append(error_metrics(actual, expected[case_index]))
        results[name] = {
            "max_relative_rmse": max(value["relative_rmse"] for value in metrics),
            "max_abs_error": max(value["max_abs_error"] for value in metrics),
        }
    return results


def elapsed_us(graph: torch.cuda.CUDAGraph, replays: int, operations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / (replays * operations)


def benchmark_shape(args: argparse.Namespace, num_tokens: int) -> dict:
    cases = make_cases(num_tokens, args.seed + num_tokens)
    torch_mm, control = load_projection_paths()
    expected = torch_mm(cases[0])
    control_actual = control(cases[0])
    candidate_actual = candidate(cases[0])
    torch.cuda.synchronize()
    assert_close(control_actual, expected, f"M={num_tokens} control eager")
    assert_close(candidate_actual, expected, f"M={num_tokens} candidate eager")
    eager_error = error_metrics(candidate_actual, control_actual)

    graphs = {
        "torch_mm": capture(torch_mm, cases, args.operations_per_graph),
        "control": capture(control, cases, args.operations_per_graph),
        "candidate": capture(candidate, cases, args.operations_per_graph),
    }
    replay_errors = validate_changed_input_replay(
        graphs,
        cases,
        torch_mm,
        num_tokens,
        args.seed + 1000 + num_tokens,
    )

    warmups = math.ceil(args.warmup_operations / args.operations_per_graph)
    names = tuple(graphs)
    for index in range(warmups * len(graphs)):
        graphs[names[index % len(names)]][0].replay()
    torch.cuda.synchronize()

    samples = {name: [] for name in graphs}
    timing_orders = (
        ("torch_mm", "control", "candidate"),
        ("control", "candidate", "torch_mm"),
        ("candidate", "torch_mm", "control"),
    )
    for trial in range(args.trials):
        order = timing_orders[trial % len(timing_orders)]
        for name in order:
            samples[name].append(
                elapsed_us(
                    graphs[name][0],
                    args.replays_per_trial,
                    args.operations_per_graph,
                )
            )
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "num_tokens": num_tokens,
        "eager_error": eager_error,
        "eager_errors": {
            "control_vs_torch_mm": error_metrics(control_actual, expected),
            "candidate_vs_torch_mm": error_metrics(candidate_actual, expected),
        },
        "changed_input_graph_replay": replay_errors,
        "p50_us": medians,
        "candidate_minus_control_us": medians["candidate"] - medians["control"],
        "speedup": medians["control"] / medians["candidate"],
        "torch_mm_minus_control_us": medians["torch_mm"] - medians["control"],
        "torch_mm_over_control_speedup": medians["torch_mm"] / medians["control"],
        "samples_us": samples,
    }


def main() -> None:
    args = parse_args()
    if (
        min(
            *args.num_tokens,
            args.operations_per_graph,
            args.warmup_operations,
            args.replays_per_trial,
            args.trials,
        )
        <= 0
    ):
        raise ValueError("token and benchmark counts must be positive")
    if max(args.num_tokens) > 14:
        raise ValueError("num_tokens must not exceed 14")
    if args.operations_per_graph < ROTATIONS:
        raise ValueError(
            f"operations-per-graph must cover all {ROTATIONS} rotating weights"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a ROCm GPU")
    properties = torch.cuda.get_device_properties(0)
    arch = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if not torch.version.hip or arch != "gfx950":
        raise RuntimeError(f"this benchmark requires ROCm gfx950, got {arch!r}")

    rotating_weight_bytes = ROTATIONS * HIDDEN * LATENT * 2
    result = {
        "shape": "Kimi-K3 latent-MoE BF16 local tail",
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": properties.name,
            "arch": arch,
        },
        "seed": args.seed,
        "rotations": ROTATIONS,
        "rotating_weight_bytes": rotating_weight_bytes,
        "cache_valid_rotation": rotating_weight_bytes > 256 * 1024 * 1024,
        "operations_per_graph": args.operations_per_graph,
        "warmup_operations": args.warmup_operations,
        "replays_per_trial": args.replays_per_trial,
        "trials": args.trials,
        "all_changed_input_graph_replays_passed": True,
        "results": [benchmark_shape(args, value) for value in args.num_tokens],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
