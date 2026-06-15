#!/usr/bin/env python3
"""Collect fixed-global-batch TRT offline results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from io_utils import read_json, write_json


RENDERER_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
RENDERER_MODEL_PREFIX = "dsv4"
RENDERER_FRAMEWORK = "trt"
RENDERER_PRECISION = "fp4"


def parse_expected(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def discover_results(root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    discovered: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(root.rglob("offline_result_*.json")):
        result = read_json(path)
        benchmark = result.get("benchmark") or {}
        global_batch = benchmark.get(
            "global_batch_size",
            benchmark.get("concurrency"),
        )
        if global_batch is None:
            continue
        experiment_id = str(
            benchmark.get("experiment_id") or f"gbs{int(global_batch)}"
        )
        if experiment_id in discovered:
            raise RuntimeError(
                f"Duplicate result for {experiment_id}: "
                f"{discovered[experiment_id][0]} and {path}"
            )
        discovered[experiment_id] = (path, result)
    return discovered


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _seconds(aggregate: dict[str, Any], key: str) -> float | None:
    value = aggregate.get(key)
    return float(value) / 1000.0 if value is not None else None


def renderer_row(result: dict[str, Any]) -> dict[str, Any] | None:
    if result.get("status") != "success":
        return None
    benchmark = result.get("benchmark") or {}
    aggregate = result.get("aggregate") or {}
    global_batch = benchmark.get(
        "global_batch_size",
        benchmark.get("concurrency"),
    )
    output_tput = aggregate.get("output_tput_per_gpu")
    if global_batch is None or output_tput is None:
        return None

    config = result.get("config") or {}
    active_gpu_count = int(
        benchmark.get("active_gpu_count")
        or config.get("active_gpu_count")
        or 8
    )
    tensor_parallel_size = int(
        config.get("tensor_parallel_size") or active_gpu_count
    )
    expert_parallel_size = int(
        config.get("moe_expert_parallel_size") or active_gpu_count
    )
    replica_count = int(benchmark.get("replica_count") or 1)
    is_multinode = bool(
        benchmark.get(
            "is_multinode",
            int(benchmark.get("physical_nodes") or 1) > 1,
        )
    )
    tokens_per_step = float(aggregate["observed_tokens_per_step"])
    equivalent_percentiles = {
        "median_tpot": (
            float(aggregate["median_decode_round_tpot_ms"])
            / tokens_per_step
            / 1000.0
        ),
        "p90_tpot": (
            float(aggregate["p90_decode_round_tpot_ms"])
            / tokens_per_step
            / 1000.0
        ),
        "p99_tpot": (
            float(aggregate["p99_decode_round_tpot_ms"])
            / tokens_per_step
            / 1000.0
        ),
    }
    equivalent_tpot_s = (
        float(aggregate["equivalent_output_tpot_ms"]) / 1000.0
    )
    row: dict[str, Any] = {
        "hw": str(
            benchmark.get("renderer_hw")
            or benchmark.get("hardware_profile")
            or "b300"
        ),
        "model": RENDERER_MODEL,
        "infmax_model_prefix": RENDERER_MODEL_PREFIX,
        "framework": RENDERER_FRAMEWORK,
        "precision": RENDERER_PRECISION,
        "isl": int(benchmark.get("input_tokens") or 8192),
        "osl": int(benchmark.get("generated_output_tokens") or 1025),
        "conc": int(global_batch),
        "image": (result.get("provenance") or {}).get("image"),
        "disagg": False,
        "is_multinode": is_multinode,
        "spec_decoding": "mtp",
        "tput_per_gpu": float(output_tput),
        "output_tput_per_gpu": float(output_tput),
        "mean_tpot": equivalent_tpot_s,
        "mean_intvty": 1.0 / equivalent_tpot_s,
        "prefill_tp": tensor_parallel_size,
        "prefill_ep": expert_parallel_size,
        "prefill_dp_attention": True,
        "prefill_num_workers": replica_count if replica_count > 1 else 0,
        "decode_tp": tensor_parallel_size,
        "decode_ep": expert_parallel_size,
        "decode_dp_attention": True,
        "decode_num_workers": replica_count if replica_count > 1 else 0,
        "num_prefill_gpu": active_gpu_count,
        "num_decode_gpu": active_gpu_count,
        # Custom fields remain useful in downloaded flat rows. The standard
        # renderer ignores fields it does not understand.
        "global_batch_size": int(global_batch),
        "benchmark_profile": benchmark.get("benchmark_profile"),
        "local_batch_size": int(aggregate["local_batch_size"]),
        "engine_max_batch_size": benchmark.get(
            "engine_max_batch_size"
        ),
        "decode_round_tpot_ms": float(
            aggregate["decode_round_tpot_ms"]
        ),
        "decode_step_tput_per_gpu": float(
            aggregate["decode_step_tput_per_gpu"]
        ),
        "observed_tokens_per_step": tokens_per_step,
        "measured_decode_rounds": int(
            aggregate["measured_decode_rounds"]
        ),
        "timing_source": aggregate.get("timing_source"),
        "replica_count": replica_count,
        **equivalent_percentiles,
    }
    pr_reference = result.get("pr_reference") or {}
    for key in (
        "reference_concurrency",
        "reference_active_global_batch",
        "reference_prefill_gpu_count",
        "reference_decode_gpu_count",
        "reference_total_gpu_count",
        "reference_output_tput_per_decode_gpu",
        "reference_output_tput_per_total_gpu",
        "measured_output_tput_per_reference_total_gpu",
        "offline_to_reference_decode_gpu_ratio",
        "offline_to_reference_total_gpu_ratio",
        "reference_recipe_url",
    ):
        if key in pr_reference:
            row[f"pr_{key}"] = pr_reference[key]
    for renderer_key, aggregate_key in (
        ("mean_ttft", "mean_ttft_ms"),
        ("median_ttft", "median_ttft_ms"),
        ("p90_ttft", "p90_ttft_ms"),
        ("p99_ttft", "p99_ttft_ms"),
        ("mean_e2el", "mean_e2e_ms"),
        ("median_e2el", "median_e2e_ms"),
        ("p90_e2el", "p90_e2e_ms"),
        ("p99_e2el", "p99_e2e_ms"),
    ):
        value = _seconds(aggregate, aggregate_key)
        if value is not None:
            row[renderer_key] = value
    return row


def renderer_rows(
    discovered: dict[str, tuple[Path, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for _, result in sorted(
        discovered.values(),
        key=lambda item: str(item[0]),
    ):
        row = renderer_row(result)
        if row is not None:
            rows.append(row)
    return rows


def result_row(
    experiment_id: str,
    source: Path | None,
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    if result is None:
        return {
            "experiment_id": experiment_id,
            "global_batch_size": None,
            "status": "missing",
            "source": None,
        }
    benchmark = result.get("benchmark") or {}
    aggregate = result.get("aggregate") or {}
    huawei = result.get("huawei") or {}
    pr_reference = result.get("pr_reference") or {}
    hardware_profile = str(
        benchmark.get("hardware_profile")
        or huawei.get("hardware_key")
        or "b300"
    )
    step_ratio = huawei.get(
        "hardware_to_huawei_decode_step_ratio"
    )
    if step_ratio is None:
        step_ratio = huawei.get(
            f"{hardware_profile}_to_huawei_decode_step_ratio"
        )
    if step_ratio is None:
        step_ratio = huawei.get("b300_to_huawei_decode_step_ratio")
    output_ratio = huawei.get("hardware_to_huawei_output_ratio")
    if output_ratio is None:
        output_ratio = huawei.get(
            f"{hardware_profile}_to_huawei_output_ratio"
        )
    if output_ratio is None:
        output_ratio = huawei.get("b300_to_huawei_output_ratio")
    huawei_step_tput = huawei.get("decode_step_tput_per_chip")
    measured_tokens_per_step = aggregate.get(
        "observed_tokens_per_step"
    )
    huawei_same_yield_output_tput = huawei.get(
        "huawei_output_tput_per_chip_at_measured_tokens_per_step"
    )
    if (
        huawei_same_yield_output_tput is None
        and huawei_step_tput is not None
        and measured_tokens_per_step is not None
    ):
        huawei_same_yield_output_tput = (
            float(huawei_step_tput)
            * float(measured_tokens_per_step)
        )
    same_yield_ratio = huawei.get(
        "hardware_to_huawei_same_yield_output_ratio"
    )
    if (
        same_yield_ratio is None
        and aggregate.get("output_tput_per_gpu") is not None
        and huawei_same_yield_output_tput
    ):
        same_yield_ratio = (
            float(aggregate["output_tput_per_gpu"])
            / float(huawei_same_yield_output_tput)
        )
    return {
        "experiment_id": experiment_id,
        "hardware": benchmark.get("hardware"),
        "hardware_profile": hardware_profile,
        "benchmark_profile": benchmark.get("benchmark_profile"),
        "effective_parallelism": benchmark.get("effective_parallelism"),
        "physical_nodes": benchmark.get("physical_nodes"),
        "global_batch_size": benchmark.get(
            "global_batch_size",
            benchmark.get("concurrency"),
        ),
        "local_batch_size": benchmark.get("local_batch_size"),
        "engine_max_batch_size": benchmark.get(
            "engine_max_batch_size"
        ),
        "active_gpu_count": benchmark.get("active_gpu_count"),
        "status": result.get("status", "unknown"),
        "decode_round_tpot_ms": aggregate.get("decode_round_tpot_ms"),
        "median_decode_round_tpot_ms": aggregate.get(
            "median_decode_round_tpot_ms"
        ),
        "p90_decode_round_tpot_ms": aggregate.get(
            "p90_decode_round_tpot_ms"
        ),
        "p99_decode_round_tpot_ms": aggregate.get(
            "p99_decode_round_tpot_ms"
        ),
        "decode_step_tput_per_gpu": aggregate.get(
            "decode_step_tput_per_gpu"
        ),
        "output_tput_per_gpu": aggregate.get("output_tput_per_gpu"),
        "wall_output_tput_per_gpu": aggregate.get(
            "wall_output_tput_per_gpu"
        ),
        "observed_tokens_per_step": aggregate.get(
            "observed_tokens_per_step"
        ),
        "effective_acceptance_rate": aggregate.get(
            "effective_acceptance_rate"
        ),
        "token_yield_source": aggregate.get("token_yield_source"),
        "timing_source": aggregate.get("timing_source"),
        "mean_ttft_ms": aggregate.get("mean_ttft_ms"),
        "measured_decode_rounds": aggregate.get(
            "measured_decode_rounds"
        ),
        "retained_rounds": (aggregate.get("filter") or {}).get(
            "retained_rounds"
        ),
        "outlier_rounds": (aggregate.get("filter") or {}).get(
            "outlier_rounds"
        ),
        "skipped_rounds": (aggregate.get("filter") or {}).get(
            "rounds_skipped"
        ),
        "huawei_decode_round_tpot_ms": huawei.get(
            "decode_round_tpot_ms"
        ),
        "huawei_reference_global_batch_size": huawei.get(
            "reference_global_batch_size",
            huawei.get("global_batch_size"),
        ),
        "huawei_decode_step_tput_per_chip": huawei.get(
            "decode_step_tput_per_chip"
        ),
        "huawei_output_tput_per_chip_at_measured_tokens_per_step": (
            huawei_same_yield_output_tput
        ),
        "hardware_to_huawei_decode_step_ratio": step_ratio,
        "hardware_to_huawei_same_yield_output_ratio": (
            same_yield_ratio
        ),
        "hardware_to_huawei_output_ratio": output_ratio,
        "pr_reference_concurrency": pr_reference.get(
            "reference_concurrency"
        ),
        "pr_reference_active_global_batch": pr_reference.get(
            "reference_active_global_batch"
        ),
        "pr_reference_output_tput_per_decode_gpu": pr_reference.get(
            "reference_output_tput_per_decode_gpu"
        ),
        "pr_reference_output_tput_per_total_gpu": pr_reference.get(
            "reference_output_tput_per_total_gpu"
        ),
        "pr_measured_output_tput_per_reference_total_gpu": (
            pr_reference.get(
                "measured_output_tput_per_reference_total_gpu"
            )
        ),
        "offline_to_pr_decode_gpu_ratio": pr_reference.get(
            "offline_to_reference_decode_gpu_ratio"
        ),
        "offline_to_pr_total_gpu_ratio": pr_reference.get(
            "offline_to_reference_total_gpu_ratio"
        ),
        "failure_kind": result.get("failure_kind"),
        "error": result.get("error"),
        "source": str(source) if source else None,
    }


def markdown(rows: list[dict[str, Any]]) -> str:
    hardware = next(
        (
            str(row["hardware"])
            for row in rows
            if row.get("hardware")
        ),
        "TRT hardware",
    )
    hardware_profile = next(
        (
            str(row["hardware_profile"]).upper()
            for row in rows
            if row.get("hardware_profile")
        ),
        "TRT",
    )
    gpu_count = next(
        (
            int(row["active_gpu_count"])
            for row in rows
            if row.get("active_gpu_count") is not None
        ),
        None,
    )
    topology = next(
        (
            str(row["effective_parallelism"])
            for row in rows
            if row.get("effective_parallelism")
        ),
        None,
    )
    is_pr_max = any(
        str(row.get("benchmark_profile") or "").startswith("pr-")
        for row in rows
    )
    is_rack = any(
        str(row.get("benchmark_profile") or "").startswith("rack-")
        for row in rows
    )
    if is_rack:
        lines = [
            "# DeepSeek-V4 GB300 NVL72 TRT Fixed-GBS Rack Benchmark",
            "",
            "| Rack GBS | Local/GPU | GPUs | Status | Slowest-replica step ms | Decode steps/s/GPU | Tok/step | Output tok/s/GPU | Wall tok/s/GPU | Huawei GBS | Huawei step ms | Huawei steps/s/chip | Huawei @ GB300 yield tok/s/chip | GB300/Huawei same-yield |",
            "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                "| {gbs} | {local} | {gpus} | {status} | {tpot} | "
                "{step_tput} | {tokens_per_step} | {output_tput} | "
                "{wall} | {huawei_gbs} | {huawei_tpot} | "
                "{huawei_step_tput} | {huawei_same_yield_output} | "
                "{same_yield_ratio} |".format(
                    gbs=row.get("global_batch_size") or "-",
                    local=row.get("local_batch_size") or "-",
                    gpus=row.get("active_gpu_count") or "-",
                    status=row["status"],
                    tpot=_fmt(row.get("decode_round_tpot_ms"), 3),
                    step_tput=_fmt(
                        row.get("decode_step_tput_per_gpu"),
                        2,
                    ),
                    tokens_per_step=_fmt(
                        row.get("observed_tokens_per_step"),
                        3,
                    ),
                    output_tput=_fmt(
                        row.get("output_tput_per_gpu"),
                        2,
                    ),
                    wall=_fmt(
                        row.get("wall_output_tput_per_gpu"),
                        2,
                    ),
                    huawei_gbs=(
                        row.get("huawei_reference_global_batch_size")
                        or "-"
                    ),
                    huawei_tpot=_fmt(
                        row.get("huawei_decode_round_tpot_ms"),
                        2,
                    ),
                    huawei_step_tput=_fmt(
                        row.get("huawei_decode_step_tput_per_chip"),
                        2,
                    ),
                    huawei_same_yield_output=_fmt(
                        row.get(
                            "huawei_output_tput_per_chip_at_measured_tokens_per_step"
                        ),
                        2,
                    ),
                    same_yield_ratio=_fmt(
                        row.get(
                            "hardware_to_huawei_same_yield_output_ratio"
                        ),
                        3,
                    ),
                )
            )
        lines.extend(
            [
                "",
                "## Metric meanings",
                "",
                "- One rack row is nine synchronized TP8/EP8 MTP1 TensorRT engines on 18 four-GPU nodes. All 72 GPUs must report one NVLink Fabric UUID and clique.",
                "- `Rack GBS` is fixed and divided equally across the nine engines. `Local/GPU` is `Rack GBS / 72`; the Huawei-comparable rows 72/288/576 preserve Huawei local batches 1/4/8.",
                "- `Slowest-replica step` takes the maximum same-index rank-0 TRT `host_step_time` across the nine engines for each of 256 logical decode rounds, then discards eight startup rounds and upper-IQR outliers.",
                "- `Decode steps/s/GPU` is `Rack GBS / step time / 72`. `Tok/step` is measured MTP acceptance; output throughput multiplies the two.",
                "- `Huawei @ GB300 yield` multiplies Huawei's published raw decode-step rate by the measured GB300 `Tok/step`. Its GB300/Huawei ratio is therefore exactly the raw decode-step ratio, without crediting either stack for a different MTP depth or acceptance rate.",
                "- Huawei uses MTP3 while the copied attempt-14 TP8 recipe uses MTP1. The table therefore keeps raw decode-step and acceptance-adjusted output rates separate.",
                "- Rack GBS 30960/36864 are saturation points copied from nine times the attempt-14 TP8 active/capacity batches; they have no Huawei local-batch row.",
                "",
            ]
        )
    elif is_pr_max:
        lines = [
            "# DeepSeek-V4 GB300 TRT PR-Config Offline Decode Saturation",
            "",
            "| Profile | GBS | Local/rank | Decode GPUs | Status | Host-step ms | Decode steps/s/GPU | Tok/step | Output tok/s/decode-GPU | PR output tok/s/decode-GPU | Offline/PR decode | PR-fleet-normalized tok/s/GPU |",
            "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                "| {profile} | {gbs} | {local} | {gpus} | {status} | "
                "{tpot} | {step_tput} | {tokens_per_step} | "
                "{output_tput} | {pr_output} | {ratio} | "
                "{fleet_output} |".format(
                    profile=row.get("benchmark_profile") or "-",
                    gbs=row.get("global_batch_size") or "-",
                    local=row.get("local_batch_size") or "-",
                    gpus=row.get("active_gpu_count") or "-",
                    status=row["status"],
                    tpot=_fmt(row.get("decode_round_tpot_ms"), 3),
                    step_tput=_fmt(
                        row.get("decode_step_tput_per_gpu"),
                        2,
                    ),
                    tokens_per_step=_fmt(
                        row.get("observed_tokens_per_step"),
                        3,
                    ),
                    output_tput=_fmt(
                        row.get("output_tput_per_gpu"),
                        2,
                    ),
                    pr_output=_fmt(
                        row.get(
                            "pr_reference_output_tput_per_decode_gpu"
                        ),
                        2,
                    ),
                    ratio=_fmt(
                        row.get("offline_to_pr_decode_gpu_ratio"),
                        3,
                    ),
                    fleet_output=_fmt(
                        row.get(
                            "pr_measured_output_tput_per_reference_total_gpu"
                        ),
                        2,
                    ),
                )
            )
        lines.extend(
            [
                "",
                "## Metric meanings",
                "",
                "- `GBS` is a fixed decode population. Each copied recipe runs its attempt-14 active estimate and engine-capacity endpoint: TP32=192/256, TP16=400/512, TP8=3440/4096. It is not HTTP concurrency.",
                "- `Host-step ms` is rank-0 TRT `host_step_time` for 256 consecutive full-batch decode iterations under overlap scheduling, after discarding eight startup rounds and upper-IQR outliers.",
                "- `Output tok/s/decode-GPU` multiplies full-batch decode-step rate by the measured MTP token yield.",
                "- `PR output tok/s/decode-GPU` is attempt 14's serving result for the copied recipe. `Offline/PR decode` compares the same decode-GPU denominator.",
                "- `PR-fleet-normalized` divides offline total output throughput by the PR's decode plus prefill GPU count. It is a comparison normalization, not the offline run's actual allocation.",
                "- The offline adaptation performs 8K prefill on the decode GPUs with `max_num_tokens=32768`; the PR worker receives transferred KV and uses its smaller decode-only token cap.",
                "",
            ]
        )
    else:
        lines = [
            f"# DeepSeek-V4 {hardware} TRT Fixed-GBS Offline Benchmark",
            "",
            f"| GBS | Local/rank | GPUs | Status | Decode round TPOT ms | Decode steps/s/GPU | Tok/step | Output tok/s/GPU | Wall tok/s/GPU | Retained rounds | Huawei TPOT ms | Huawei steps/s/chip | {hardware_profile}/Huawei step |",
            "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            retained = row.get("retained_rounds")
            measured = row.get("measured_decode_rounds")
            skipped = int(row.get("skipped_rounds") or 0)
            retained_label = (
                f"{retained}/{int(measured) - skipped}"
                if retained is not None and measured is not None
                else "-"
            )
            lines.append(
                "| {gbs} | {local} | {gpus} | {status} | {tpot} | "
                "{step_tput} | {tokens_per_step} | {output_tput} | {wall} | "
                "{retained} | {huawei_tpot} | {huawei_tput} | {ratio} |".format(
                    gbs=row.get("global_batch_size") or "-",
                    local=row.get("local_batch_size") or "-",
                    gpus=row.get("active_gpu_count") or "-",
                    status=row["status"],
                    tpot=_fmt(row.get("decode_round_tpot_ms"), 3),
                    step_tput=_fmt(
                        row.get("decode_step_tput_per_gpu"),
                        2,
                    ),
                    tokens_per_step=_fmt(
                        row.get("observed_tokens_per_step"),
                        3,
                    ),
                    output_tput=_fmt(row.get("output_tput_per_gpu"), 2),
                    wall=_fmt(row.get("wall_output_tput_per_gpu"), 2),
                    retained=retained_label,
                    huawei_tpot=_fmt(
                        row.get("huawei_decode_round_tpot_ms"),
                        2,
                    ),
                    huawei_tput=_fmt(
                        row.get("huawei_decode_step_tput_per_chip"),
                        2,
                    ),
                    ratio=_fmt(
                        row.get(
                            "hardware_to_huawei_decode_step_ratio"
                        ),
                        3,
                    ),
                )
            )
        lines.extend(
            [
                "",
                "## Metric meanings",
                "",
                (
                    "- `GBS` is the one authoritative global batch submitted "
                    f"to TRT. `Local/rank` is exactly `GBS / {gpu_count}`"
                    + (f" for {topology}." if topology else ".")
                    if gpu_count is not None
                    else (
                        "- `GBS` is the one authoritative global batch "
                        "submitted to TRT."
                    )
                ),
                "- `Decode round TPOT` is the mean `iterLatencyMS` for 256 consecutive full-local-batch decode iterations after skipping the first and removing only upper-IQR outliers, matching Huawei's `process_infer_time` path.",
                (
                    "- `Decode steps/s/GPU` is "
                    f"`GBS / decode_round_TPOT / {gpu_count}`. This is the "
                    "direct comparison with Huawei's published table."
                    if gpu_count is not None
                    else (
                        "- `Decode steps/s/GPU` divides the full-batch "
                        "decode-step rate by active GPUs."
                    )
                ),
                "- `Tok/step` is MTP output yield and is reported separately. `Output tok/s/GPU` multiplies decode-step throughput by that yield.",
                "- `Wall tok/s/GPU` covers the entire `LLM.generate()` call and remains diagnostic; it includes the deliberately longer output cap used to guarantee at least 256 decode rounds.",
                (
                    "- The global batch and timing method match Huawei. "
                    f"The hardware does not: this run uses {gpu_count} "
                    f"{hardware} GPUs, while Huawei publishes 16 950DT chips."
                    if gpu_count is not None
                    else (
                        "- The global batch and timing method match Huawei; "
                        "the hardware does not."
                    )
                ),
                "",
            ]
        )
    failures = [row for row in rows if row["status"] != "success"]
    if failures:
        lines.extend(["## Missing Or Failed Rows", ""])
        for row in failures:
            detail = row.get("error") or ""
            failure_kind = row.get("failure_kind")
            if failure_kind:
                detail = f"{failure_kind}: {detail}".rstrip()
            lines.append(
                f"- `{row['experiment_id']}`: "
                f"{row['status']} {detail}".rstrip()
            )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--renderer-json-out", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = parse_expected(args.expected)
    discovered = discover_results(args.results_dir)
    rows = [
        result_row(
            experiment_id,
            discovered.get(experiment_id, (None, None))[0],
            discovered.get(experiment_id, (None, None))[1],
        )
        for experiment_id in expected
    ]
    aggregate = {
        "schema_version": 2,
        "expected_experiments": expected,
        "rows": rows,
        "results": {
            experiment_id: result
            for experiment_id, (_, result) in sorted(discovered.items())
        },
    }
    write_json(args.json_out, aggregate)
    if args.renderer_json_out is not None:
        write_json(args.renderer_json_out, renderer_rows(discovered))
    summary = markdown(rows)
    args.markdown_out.write_text(summary, encoding="utf-8")
    print(summary)
    if args.strict and any(row["status"] != "success" for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
