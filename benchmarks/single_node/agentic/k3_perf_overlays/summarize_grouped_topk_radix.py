# SPDX-License-Identifier: MIT
"""Adjudicate alternating stock/radix Kimi-K3 router timing rounds."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

SHAPES = (1, 2, 4, 7, 14)
PRIMARY_SHAPE = 7
ROUTER_CALLS_PER_STEP = 92


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", type=Path, nargs=2, required=True)
    parser.add_argument("--candidate", type=Path, nargs=2, required=True)
    parser.add_argument("--stock-commit", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def by_m(payload: dict) -> dict[int, dict]:
    rows = {int(row["num_tokens"]): row for row in payload["results"]}
    if set(rows) != set(SHAPES):
        raise ValueError(f"expected M={SHAPES}, got {sorted(rows)}")
    return rows


def validate_round(
    payload: dict,
    implementation: str,
    expected_commit: str,
) -> None:
    runtime = payload.get("runtime", {})
    if runtime.get("arch") != "gfx950" or int(runtime.get("cu_num", -1)) != 256:
        raise ValueError(f"invalid runtime identity: {runtime}")
    contract = payload.get("contract", {})
    if contract.get("dtype") != "torch.float32":
        raise ValueError("router diagnostic did not use the vLLM FP32 gate contract")
    if contract.get("router_calls_per_decode_step") != ROUTER_CALLS_PER_STEP:
        raise ValueError("router-call projection count is not 92")
    if payload.get("implementation") != implementation:
        raise ValueError(
            f"expected {implementation} round, got {payload.get('implementation')}"
        )
    if payload.get("aiter_commit") != expected_commit:
        raise ValueError(
            f"expected AITER {expected_commit}, got {payload.get('aiter_commit')}"
        )
    for field in (
        "dispatch_canary_passed",
        "all_eager_correctness_passed",
        "all_changed_input_graph_replays_passed",
    ):
        if not payload.get(field):
            raise ValueError(f"{implementation} round failed {field}")

    trials = int(payload["trials"])
    for num_tokens, row in by_m(payload).items():
        samples = row.get("samples_us_per_call", [])
        if len(samples) != trials:
            raise ValueError(
                f"{implementation} M={num_tokens}: expected {trials} samples, "
                f"got {len(samples)}"
            )
        if not row.get("changed_input_graph_replay_passed"):
            raise ValueError(
                f"{implementation} M={num_tokens}: changed-input replay failed"
            )
        if int(row.get("changed_input_rows", 0)) <= 0:
            raise ValueError(
                f"{implementation} M={num_tokens}: inputs did not change outputs"
            )


def main() -> None:
    args = parse_args()
    stock = [load(path) for path in args.stock]
    candidate = [load(path) for path in args.candidate]
    for payload in stock:
        validate_round(payload, "stock", args.stock_commit)
    for payload in candidate:
        validate_round(payload, "radix", args.candidate_commit)

    runtime_identities = {
        (
            payload["runtime"]["device"],
            payload["runtime"]["arch"],
            int(payload["runtime"]["cu_num"]),
            payload["runtime"]["torch"],
            payload["runtime"]["hip"],
        )
        for payload in stock + candidate
    }
    if len(runtime_identities) != 1:
        raise ValueError(f"timing rounds used different runtimes: {runtime_identities}")
    if len({payload["module_moe_asm_sha256"] for payload in stock}) != 1:
        raise ValueError("stock rounds loaded different module_moe_asm binaries")
    if len({payload["module_moe_asm_sha256"] for payload in candidate}) != 1:
        raise ValueError("candidate rounds loaded different module_moe_asm binaries")
    if stock[0]["module_moe_asm_sha256"] == candidate[0]["module_moe_asm_sha256"]:
        raise ValueError("stock and candidate module_moe_asm hashes are identical")

    rows = []
    for num_tokens in SHAPES:
        stock_rows = [by_m(payload)[num_tokens] for payload in stock]
        candidate_rows = [by_m(payload)[num_tokens] for payload in candidate]
        stock_round_medians = [float(row["p50_us_per_call"]) for row in stock_rows]
        candidate_round_medians = [
            float(row["p50_us_per_call"]) for row in candidate_rows
        ]
        stock_samples = [
            float(value)
            for row in stock_rows
            for value in row["samples_us_per_call"]
        ]
        candidate_samples = [
            float(value)
            for row in candidate_rows
            for value in row["samples_us_per_call"]
        ]
        stock_us = statistics.median(stock_samples)
        candidate_us = statistics.median(candidate_samples)
        saving_us = stock_us - candidate_us
        stock_drift = abs(stock_round_medians[1] - stock_round_medians[0]) / stock_us
        candidate_drift = (
            abs(candidate_round_medians[1] - candidate_round_medians[0])
            / candidate_us
        )
        rows.append(
            {
                "num_tokens": num_tokens,
                "stock_p50_us_per_call": stock_us,
                "candidate_p50_us_per_call": candidate_us,
                "saving_us_per_call": saving_us,
                "speedup": stock_us / candidate_us,
                "projected_saving_ms_per_decode_step": saving_us
                * ROUTER_CALLS_PER_STEP
                / 1000.0,
                "stock_round_p50_us": stock_round_medians,
                "candidate_round_p50_us": candidate_round_medians,
                "stock_round_drift_fraction": stock_drift,
                "candidate_round_drift_fraction": candidate_drift,
                "stock_samples_us_per_call": stock_samples,
                "candidate_samples_us_per_call": candidate_samples,
            }
        )

    primary = next(row for row in rows if row["num_tokens"] == PRIMARY_SHAPE)
    all_shape_non_regression = all(
        row["candidate_p50_us_per_call"]
        <= row["stock_p50_us_per_call"] * 1.03
        for row in rows
    )
    all_round_drift_bounded = all(
        row["stock_round_drift_fraction"] <= 0.02
        and row["candidate_round_drift_fraction"] <= 0.02
        for row in rows
    )
    promoted = (
        primary["speedup"] >= 1.03
        and primary["projected_saving_ms_per_decode_step"] >= 0.5
        and all_shape_non_regression
        and all_round_drift_bounded
    )
    summary = {
        "primary_shape": PRIMARY_SHAPE,
        "router_calls_per_decode_step": ROUTER_CALLS_PER_STEP,
        "stock_commit": args.stock_commit,
        "candidate_commit": args.candidate_commit,
        "promotion_thresholds": {
            "minimum_primary_shape_speedup": 1.03,
            "minimum_projected_saving_ms_per_decode_step": 0.5,
            "maximum_other_shape_regression_fraction": 0.03,
            "maximum_round_drift_fraction": 0.02,
        },
        "all_shape_non_regression": all_shape_non_regression,
        "all_round_drift_bounded": all_round_drift_bounded,
        "promoted_to_natural_agentx": promoted,
        "stock_module_moe_asm_sha256": stock[0]["module_moe_asm_sha256"],
        "candidate_module_moe_asm_sha256": candidate[0][
            "module_moe_asm_sha256"
        ],
        "results": rows,
    }
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with args.output_tsv.open("w", encoding="utf-8") as handle:
        handle.write(
            "M\tstock_us\tcandidate_us\tsaving_us\tspeedup"
            "\tprojected_ms_per_step\tstock_drift\tcandidate_drift\n"
        )
        for row in rows:
            handle.write(
                f"{row['num_tokens']}\t{row['stock_p50_us_per_call']:.6f}\t"
                f"{row['candidate_p50_us_per_call']:.6f}\t"
                f"{row['saving_us_per_call']:.6f}\t{row['speedup']:.6f}\t"
                f"{row['projected_saving_ms_per_decode_step']:.6f}\t"
                f"{row['stock_round_drift_fraction']:.6f}\t"
                f"{row['candidate_round_drift_fraction']:.6f}\n"
            )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
