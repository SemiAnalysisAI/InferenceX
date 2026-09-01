# SPDX-License-Identifier: MIT
"""Compare Kimi-K3 M=7 latent-tail token tilings under HIP graph replay."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import aiter
import torch
from aiter.ops.flydsl.latent_moe_tail import (
    latent_moe_projection_add,
    latent_moe_tail,
)
from bench_latent_moe_tail_small_m import (
    EPSILON,
    HIDDEN,
    LATENT,
    ROTATIONS,
    Case,
    Operation,
    assert_close,
    capture,
    elapsed_us,
    error_metrics,
    load_projection_paths,
    make_cases,
    validate_changed_input_replay,
)

NUM_TOKENS = 7
TILINGS = {
    "one_token_r14": (1, 14),
    "token_tile_2_r7": (2, 7),
    "token_tile_4_r4": (4, 4),
    "token_tile_7_r2": (7, 2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--operations-per-graph", type=int, default=24)
    parser.add_argument("--warmup-operations", type=int, default=2400)
    parser.add_argument("--replays-per-trial", type=int, default=20)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_candidate(tokens_per_block: int, rows_per_block: int) -> Operation:
    def operation(case: Case) -> torch.Tensor:
        routed, shared, rms_weight, up_weight = case
        return latent_moe_tail(
            routed,
            shared,
            rms_weight,
            up_weight,
            EPSILON,
            tokens_per_block=tokens_per_block,
            rows_per_block=rows_per_block,
        )

    return operation


def make_pre_normalized_candidate(
    tokens_per_block: int, rows_per_block: int
) -> Operation:
    def operation(case: Case) -> torch.Tensor:
        routed, shared, rms_weight, up_weight = case
        normalized = aiter.rmsnorm2d_fwd(routed, rms_weight, EPSILON)
        return latent_moe_projection_add(
            normalized,
            shared,
            up_weight,
            tokens_per_block=tokens_per_block,
            rows_per_block=rows_per_block,
        )

    return operation


def timing_order(names: tuple[str, ...], trial: int) -> tuple[str, ...]:
    shift = trial % len(names)
    order = names[shift:] + names[:shift]
    if (trial // len(names)) % 2:
        order = tuple(reversed(order))
    return order


def benchmark(args: argparse.Namespace) -> dict:
    cases = make_cases(NUM_TOKENS, args.seed)
    torch_mm, control = load_projection_paths()
    operations: dict[str, Operation] = {
        "torch_mm": torch_mm,
        "control": control,
        **{
            name: make_candidate(tokens_per_block, rows_per_block)
            for name, (tokens_per_block, rows_per_block) in TILINGS.items()
        },
        **{
            f"pre_norm_{name}": make_pre_normalized_candidate(
                tokens_per_block, rows_per_block
            )
            for name, (tokens_per_block, rows_per_block) in TILINGS.items()
        },
    }

    expected = torch_mm(cases[0])
    eager_errors = {}
    for name, operation in operations.items():
        actual = operation(cases[0])
        torch.cuda.synchronize()
        assert_close(actual, expected, f"M=7 {name} eager")
        eager_errors[f"{name}_vs_torch_mm"] = error_metrics(actual, expected)

    graphs = {
        name: capture(operation, cases, args.operations_per_graph)
        for name, operation in operations.items()
    }
    replay_errors = validate_changed_input_replay(
        graphs,
        cases,
        torch_mm,
        NUM_TOKENS,
        args.seed + 1000,
    )

    warmups = math.ceil(args.warmup_operations / args.operations_per_graph)
    names = tuple(graphs)
    for index in range(warmups * len(graphs)):
        graphs[names[index % len(names)]][0].replay()
    torch.cuda.synchronize()

    samples = {name: [] for name in graphs}
    orders = []
    for trial in range(args.trials):
        order = timing_order(names, trial)
        orders.append(order)
        for name in order:
            samples[name].append(
                elapsed_us(
                    graphs[name][0],
                    args.replays_per_trial,
                    args.operations_per_graph,
                )
            )

    medians = {name: statistics.median(values) for name, values in samples.items()}
    control_us = medians["control"]
    one_token_us = medians["one_token_r14"]
    return {
        "num_tokens": NUM_TOKENS,
        "tilings": {
            candidate_name: {
                "normalization": normalization,
                "tokens_per_block": tokens_per_block,
                "rows_per_block": rows_per_block,
                "token_groups": math.ceil(NUM_TOKENS / tokens_per_block),
                "workgroups": math.ceil(HIDDEN / rows_per_block)
                * math.ceil(NUM_TOKENS / tokens_per_block),
                "weight_row_load_reduction_vs_one_token": NUM_TOKENS
                / math.ceil(NUM_TOKENS / tokens_per_block),
            }
            for name, (tokens_per_block, rows_per_block) in TILINGS.items()
            for candidate_name, normalization in (
                (name, "in_kernel"),
                (f"pre_norm_{name}", "separate_aiter_rmsnorm"),
            )
        },
        "eager_errors": eager_errors,
        "changed_input_graph_replay": replay_errors,
        "p50_us": medians,
        "control_over_path_speedup": {
            name: control_us / value for name, value in medians.items()
        },
        "one_token_over_path_speedup": {
            name: one_token_us / value for name, value in medians.items()
        },
        "samples_us": samples,
        "timing_orders": orders,
    }


def main() -> None:
    args = parse_args()
    if (
        min(
            args.operations_per_graph,
            args.warmup_operations,
            args.replays_per_trial,
            args.trials,
        )
        <= 0
    ):
        raise ValueError("benchmark counts must be positive")
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
        "shape": "Kimi-K3 M=7 latent-MoE BF16 local tail token tiling",
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": properties.name,
            "arch": arch,
            "cu_num": properties.multi_processor_count,
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
        "result": benchmark(args),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
