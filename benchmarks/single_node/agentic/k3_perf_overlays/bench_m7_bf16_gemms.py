# SPDX-License-Identifier: Apache-2.0
"""Graph-replay A/B/A benchmark for exact small-M Kimi-K3 BF16 GEMMs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument("--rotations", type=int, default=4)
    parser.add_argument("--warmup-replays", type=int, default=50)
    parser.add_argument("--replays-per-trial", type=int, default=25)
    parser.add_argument("--trials", type=int, default=21)
    parser.add_argument("--min-speedup-percent", type=float, default=3.0)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--worker-label")
    return parser.parse_args()


def load_shapes(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no GEMM shapes found in {path}")
    return rows


def shape_key(row: dict[str, object]) -> tuple[int, int, int]:
    return int(row["M"]), int(row["N"]), int(row["K"])


def json_scalar(value: object) -> object:
    if hasattr(value, "item"):
        value = value.item()  # type: ignore[union-attr]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def run_worker(args: argparse.Namespace) -> None:
    import torch
    from aiter.tuned_gemm import gemm_a16w16, get_GEMM_A16W16_config
    from torch.nn import functional

    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("this benchmark requires a ROCm GPU")
    properties = torch.cuda.get_device_properties(0)
    arch = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if arch != "gfx950":
        raise RuntimeError(f"this benchmark requires gfx950, got {arch!r}")

    torch.manual_seed(20260901)
    results: list[dict[str, object]] = []
    for row in load_shapes(args.input_csv):
        m, n, k = shape_key(row)
        cases: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(args.rotations):
            inp = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
            weight.mul_(k**-0.5)
            cases.append((inp, weight))

        selected = get_GEMM_A16W16_config(
            M=m,
            N=n,
            K=k,
            bias=False,
            dtype=str(torch.bfloat16),
            otype=str(torch.bfloat16),
            scaleAB=False,
            bpreshuffle=False,
        )

        eager_outputs = [gemm_a16w16(inp, weight) for inp, weight in cases]
        torch.cuda.synchronize()
        for (inp, weight), actual in zip(cases, eager_outputs, strict=True):
            expected = functional.linear(inp.float(), weight.float()).bfloat16()
            torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)

        captured_outputs: list[torch.Tensor] = []
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for inp, weight in cases:
                captured_outputs.append(gemm_a16w16(inp, weight))

        changed_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for inp, weight in cases:
            new_inp = torch.randn_like(inp)
            new_weight = torch.randn_like(weight).mul_(k**-0.5)
            inp.copy_(new_inp)
            weight.copy_(new_weight)
            changed_inputs.append((new_inp, new_weight))
        graph.replay()
        torch.cuda.synchronize()
        for (inp, weight), actual in zip(changed_inputs, captured_outputs, strict=True):
            expected = functional.linear(inp.float(), weight.float()).bfloat16()
            torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)

        for _ in range(args.warmup_replays):
            graph.replay()
        torch.cuda.synchronize()

        samples_us: list[float] = []
        for _ in range(args.trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.replays_per_trial):
                graph.replay()
            end.record()
            end.synchronize()
            samples_us.append(
                start.elapsed_time(end)
                * 1000.0
                / (args.replays_per_trial * args.rotations)
            )

        results.append(
            {
                "M": m,
                "N": n,
                "K": k,
                "selected_config": {
                    key: json_scalar(value) for key, value in selected.items()
                },
                "p50_us": statistics.median(samples_us),
                "samples_us": samples_us,
                "graph_changed_input_replay": "passed",
            }
        )
        del cases, eager_outputs, captured_outputs, changed_inputs, graph
        torch.cuda.empty_cache()

    payload = {
        "label": args.worker_label,
        "config_env": os.environ.get("AITER_CONFIG_GEMM_BF16"),
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": properties.name,
            "arch": arch,
            "cu_num": properties.multi_processor_count,
        },
        "rotations": args.rotations,
        "warmup_replays": args.warmup_replays,
        "replays_per_trial": args.replays_per_trial,
        "trials": args.trials,
        "results": results,
    }
    if args.worker_output is None:
        raise ValueError("--worker-output is required in worker mode")
    args.worker_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_subprocess(
    args: argparse.Namespace,
    label: str,
    output: Path,
    config: Path | None,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--input-csv",
        str(args.input_csv.resolve()),
        "--candidate-csv",
        str(args.candidate_csv.resolve()),
        "--output",
        str(args.output.resolve()),
        "--selected-csv",
        str(args.selected_csv.resolve()),
        "--rotations",
        str(args.rotations),
        "--warmup-replays",
        str(args.warmup_replays),
        "--replays-per-trial",
        str(args.replays_per_trial),
        "--trials",
        str(args.trials),
        "--min-speedup-percent",
        str(args.min_speedup_percent),
        "--worker-output",
        str(output),
        "--worker-label",
        label,
    ]
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = "0"
    env["ROCR_VISIBLE_DEVICES"] = "0"
    if config is None:
        env.pop("AITER_CONFIG_GEMM_BF16", None)
    else:
        env["AITER_CONFIG_GEMM_BF16"] = str(config.resolve())
    subprocess.run(command, check=True, env=env)
    return json.loads(output.read_text(encoding="utf-8"))


def write_selected_csv(
    candidate_csv: Path,
    selected_csv: Path,
    selected_keys: set[tuple[int, int, int]],
) -> None:
    with candidate_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"candidate CSV has no header: {candidate_csv}")
        rows = [row for row in reader if shape_key(row) in selected_keys]
    with selected_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_controller(args: argparse.Namespace) -> None:
    for name, value in (
        ("rotations", args.rotations),
        ("warmup_replays", args.warmup_replays),
        ("replays_per_trial", args.replays_per_trial),
        ("trials", args.trials),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.min_speedup_percent < 0:
        raise ValueError("min_speedup_percent must be non-negative")

    input_rows = load_shapes(args.input_csv)
    matrix_ms = {shape_key(row)[0] for row in input_rows}
    if len(matrix_ms) != 1:
        raise ValueError(
            f"input CSV must contain one exact M value, got {sorted(matrix_ms)}"
        )
    matrix_m = next(iter(matrix_ms))

    with tempfile.TemporaryDirectory(prefix=f"k3-m{matrix_m}-gemm-graph-") as temp_dir:
        temp = Path(temp_dir)
        default_a = run_subprocess(args, "default-a", temp / "default-a.json", None)
        candidate = run_subprocess(
            args, "candidate", temp / "candidate.json", args.candidate_csv
        )
        default_b = run_subprocess(args, "default-b", temp / "default-b.json", None)

    indexed = {
        payload["label"]: {
            shape_key(row): row
            for row in payload["results"]  # type: ignore[index]
        }
        for payload in (default_a, candidate, default_b)
    }
    comparisons: list[dict[str, object]] = []
    selected_keys: set[tuple[int, int, int]] = set()
    for source_row in input_rows:
        key = shape_key(source_row)
        first_default = indexed["default-a"][key]
        second_default = indexed["default-b"][key]
        candidate_row = indexed["candidate"][key]
        default_us = statistics.median(
            [float(first_default["p50_us"]), float(second_default["p50_us"])]
        )
        candidate_us = float(candidate_row["p50_us"])
        speedup = default_us / candidate_us
        speedup_percent = (speedup - 1.0) * 100.0
        candidate_config = candidate_row["selected_config"]
        candidate_is_tuned = candidate_config.get("libtype") != "torch"  # type: ignore[union-attr]
        promote = candidate_is_tuned and speedup_percent >= args.min_speedup_percent
        if promote:
            selected_keys.add(key)
        comparisons.append(
            {
                "M": key[0],
                "N": key[1],
                "K": key[2],
                "default_a_us": first_default["p50_us"],
                "default_b_us": second_default["p50_us"],
                "default_median_us": default_us,
                "candidate_us": candidate_us,
                "speedup": speedup,
                "speedup_percent": speedup_percent,
                "candidate_config": candidate_config,
                "promote": promote,
            }
        )

    write_selected_csv(args.candidate_csv, args.selected_csv, selected_keys)
    result = {
        "method": "HIP graph replay with rotating weights; default/candidate/default",
        "matrix_m": matrix_m,
        "minimum_speedup_percent": args.min_speedup_percent,
        "all_changed_input_graph_replays_passed": True,
        "selected_shape_count": len(selected_keys),
        "comparisons": comparisons,
        "workers": [default_a, candidate, default_b],
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.worker_output is not None:
        run_worker(args)
    else:
        run_controller(args)


if __name__ == "__main__":
    main()
