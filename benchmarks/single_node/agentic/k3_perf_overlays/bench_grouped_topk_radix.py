# SPDX-License-Identifier: MIT
"""Benchmark Kimi-K3's FP32 grouped-top-k router under HIP graph replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

import aiter
import torch
from aiter.jit.core import get_user_jit_dir

NUM_EXPERTS = 896
TOPK = 16
NUM_EXPERT_GROUPS = 1
TOPK_GROUPS = 1
ROUTED_SCALING_FACTOR = 2.827
ROUTER_CALLS_PER_STEP = 92
DTYPE = torch.float32


@dataclass
class RouterCase:
    logits: torch.Tensor
    bias: torch.Tensor
    weights: torch.Tensor
    ids: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("stock", "radix"), required=True)
    parser.add_argument("--aiter-commit", required=True)
    parser.add_argument(
        "--num-tokens", type=int, nargs="+", default=[1, 2, 4, 7, 14]
    )
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--warmup-graphs", type=int, default=100)
    parser.add_argument("--replays-per-trial", type=int, default=20)
    parser.add_argument("--trials", type=int, default=31)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_cases(num_tokens: int, seed: int) -> list[RouterCase]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    cases = []
    for layer in range(ROUTER_CALLS_PER_STEP):
        logits = torch.randn(
            (num_tokens, NUM_EXPERTS),
            dtype=DTYPE,
            device="cuda",
            generator=generator,
        )
        bias = torch.randn(
            NUM_EXPERTS,
            dtype=DTYPE,
            device="cuda",
            generator=generator,
        ).mul_(0.1)
        # Keep every row away from an accidental boundary tie while rotating
        # the selected experts from layer to layer.
        first = (layer * TOPK) % (NUM_EXPERTS - TOPK)
        bias[first : first + TOPK].add_(8.0)
        cases.append(
            RouterCase(
                logits=logits,
                bias=bias,
                weights=torch.full(
                    (num_tokens, TOPK),
                    float("nan"),
                    dtype=torch.float32,
                    device="cuda",
                ),
                ids=torch.full(
                    (num_tokens, TOPK),
                    -777,
                    dtype=torch.int32,
                    device="cuda",
                ),
            )
        )
    return cases


def run_router(case: RouterCase) -> None:
    aiter.biased_grouped_topk(
        case.logits,
        case.bias,
        case.weights,
        case.ids,
        NUM_EXPERT_GROUPS,
        TOPK_GROUPS,
        True,
        ROUTED_SCALING_FACTOR,
    )


def assert_case_matches_reference(case: RouterCase, label: str) -> None:
    reference_weights, reference_ids = aiter.biased_grouped_topk_torch(
        torch.nan_to_num(case.logits, nan=-float("inf")),
        torch.nan_to_num(case.bias, nan=-float("inf")),
        TOPK,
        True,
        NUM_EXPERT_GROUPS,
        TOPK_GROUPS,
    )
    reference_weights.mul_(ROUTED_SCALING_FACTOR)
    actual_ids, actual_order = case.ids.sort(dim=-1)
    expected_ids, expected_order = reference_ids.sort(dim=-1)
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    try:
        torch.testing.assert_close(
            case.weights.gather(1, actual_order),
            reference_weights.gather(1, expected_order),
            rtol=2.0e-3,
            atol=2.0e-3,
        )
    except AssertionError as error:
        raise AssertionError(f"{label}: {error}") from error
    if not bool(torch.isfinite(case.weights).all()):
        raise AssertionError(f"{label}: non-finite router weights")
    if not bool((case.ids >= 0).all() and (case.ids < NUM_EXPERTS).all()):
        raise AssertionError(f"{label}: out-of-range expert id")
    sorted_ids = case.ids.sort(dim=-1).values
    if not bool((sorted_ids[:, 1:] != sorted_ids[:, :-1]).all()):
        raise AssertionError(f"{label}: duplicate expert id")


def dispatch_canary(implementation: str) -> list[int]:
    case = RouterCase(
        logits=torch.zeros((1, NUM_EXPERTS), dtype=DTYPE, device="cuda"),
        bias=torch.arange(NUM_EXPERTS, dtype=DTYPE, device="cuda"),
        weights=torch.empty((1, TOPK), dtype=torch.float32, device="cuda"),
        ids=torch.empty((1, TOPK), dtype=torch.int32, device="cuda"),
    )
    run_router(case)
    torch.cuda.synchronize()
    observed = case.ids[0].cpu().tolist()
    expected = (
        [
            880,
            884,
            888,
            892,
            881,
            885,
            889,
            893,
            882,
            886,
            890,
            894,
            883,
            887,
            891,
            895,
        ]
        if implementation == "radix"
        else list(range(895, 879, -1))
    )
    if observed != expected:
        raise AssertionError(
            f"{implementation} dispatch canary mismatch: "
            f"expected {expected}, observed {observed}"
        )
    return observed


def capture(cases: list[RouterCase]) -> torch.cuda.CUDAGraph:
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for case in cases:
            run_router(case)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for case in cases:
            run_router(case)
    return graph


def validate_changed_input_replay(
    graph: torch.cuda.CUDAGraph,
    cases: list[RouterCase],
    num_tokens: int,
    seed: int,
) -> int:
    previous_ids = [case.ids.clone() for case in cases]
    changed = make_cases(num_tokens, seed)
    for destination, source in zip(cases, changed, strict=True):
        destination.logits.copy_(source.logits)
        destination.bias.copy_(source.bias)
    graph.replay()
    torch.cuda.synchronize()

    changed_rows = 0
    for index, (case, old_ids) in enumerate(zip(cases, previous_ids, strict=True)):
        assert_case_matches_reference(case, f"M={num_tokens} changed layer={index}")
        changed_rows += int((case.ids != old_ids).any(dim=-1).sum().item())
    if changed_rows == 0:
        raise AssertionError(f"M={num_tokens}: graph replay ignored changed inputs")
    return changed_rows


def elapsed_us(
    graph: torch.cuda.CUDAGraph,
    replays: int,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / (replays * ROUTER_CALLS_PER_STEP)


def benchmark_shape(args: argparse.Namespace, num_tokens: int) -> dict:
    cases = make_cases(num_tokens, args.seed + num_tokens)
    for index, case in enumerate(cases):
        run_router(case)
        torch.cuda.synchronize()
        assert_case_matches_reference(case, f"M={num_tokens} eager layer={index}")

    graph = capture(cases)
    changed_rows = validate_changed_input_replay(
        graph,
        cases,
        num_tokens,
        args.seed + 1000 + num_tokens,
    )
    for _ in range(args.warmup_graphs):
        graph.replay()
    torch.cuda.synchronize()

    samples = [elapsed_us(graph, args.replays_per_trial) for _ in range(args.trials)]
    return {
        "num_tokens": num_tokens,
        "router_calls_per_graph": ROUTER_CALLS_PER_STEP,
        "changed_input_rows": changed_rows,
        "changed_input_graph_replay_passed": True,
        "p50_us_per_call": statistics.median(samples),
        "samples_us_per_call": samples,
    }


def main() -> None:
    args = parse_args()
    if (
        min(
            *args.num_tokens,
            args.warmup_graphs,
            args.replays_per_trial,
            args.trials,
        )
        <= 0
    ):
        raise ValueError("benchmark counts and token sizes must be positive")
    if sorted(args.num_tokens) != [1, 2, 4, 7, 14]:
        raise ValueError("the diagnostic must cover exactly M=1,2,4,7,14")
    if len(set(args.num_tokens)) != len(args.num_tokens):
        raise ValueError("num_tokens must not contain duplicates")
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a ROCm GPU")

    properties = torch.cuda.get_device_properties(0)
    arch = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if not torch.version.hip or arch != "gfx950":
        raise RuntimeError(
            f"this benchmark requires ROCm gfx950, got hip={torch.version.hip!r} "
            f"arch={arch!r}"
        )
    if properties.multi_processor_count != 256:
        raise RuntimeError(
            f"this benchmark requires a 256-CU gfx950, got "
            f"{properties.multi_processor_count} CUs"
        )

    canary_ids = dispatch_canary(args.implementation)
    results = [benchmark_shape(args, value) for value in args.num_tokens]
    module_path = Path(get_user_jit_dir()) / "module_moe_asm.so"
    if not module_path.is_file():
        raise RuntimeError(f"compiled module is missing: {module_path}")

    payload = {
        "implementation": args.implementation,
        "aiter_commit": args.aiter_commit,
        "aiter_file": str(Path(aiter.__file__).resolve()),
        "aiter_meta_dir": os.environ.get("AITER_META_DIR"),
        "aiter_jit_dir": str(Path(get_user_jit_dir()).resolve()),
        "module_moe_asm": str(module_path.resolve()),
        "module_moe_asm_sha256": sha256(module_path),
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": properties.name,
            "arch": arch,
            "cu_num": properties.multi_processor_count,
        },
        "contract": {
            "dtype": str(DTYPE),
            "num_experts": NUM_EXPERTS,
            "topk": TOPK,
            "num_expert_groups": NUM_EXPERT_GROUPS,
            "topk_groups": TOPK_GROUPS,
            "need_renorm": True,
            "routed_scaling_factor": ROUTED_SCALING_FACTOR,
            "router_calls_per_decode_step": ROUTER_CALLS_PER_STEP,
        },
        "seed": args.seed,
        "warmup_graphs": args.warmup_graphs,
        "replays_per_trial": args.replays_per_trial,
        "trials": args.trials,
        "dispatch_canary_ids": canary_ids,
        "dispatch_canary_passed": True,
        "all_eager_correctness_passed": True,
        "all_changed_input_graph_replays_passed": True,
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
