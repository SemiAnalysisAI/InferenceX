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
        "hw": "b300",
        "model": RENDERER_MODEL,
        "infmax_model_prefix": RENDERER_MODEL_PREFIX,
        "framework": RENDERER_FRAMEWORK,
        "precision": RENDERER_PRECISION,
        "isl": int(benchmark.get("input_tokens") or 8192),
        "osl": int(benchmark.get("generated_output_tokens") or 1025),
        "conc": int(global_batch),
        "image": (result.get("provenance") or {}).get("image"),
        "disagg": False,
        "is_multinode": False,
        "spec_decoding": "mtp",
        "tput_per_gpu": float(output_tput),
        "output_tput_per_gpu": float(output_tput),
        "mean_tpot": equivalent_tpot_s,
        "mean_intvty": 1.0 / equivalent_tpot_s,
        "prefill_tp": 8,
        "prefill_ep": 8,
        "prefill_dp_attention": True,
        "prefill_num_workers": 0,
        "decode_tp": 8,
        "decode_ep": 8,
        "decode_dp_attention": True,
        "decode_num_workers": 0,
        "num_prefill_gpu": 8,
        "num_decode_gpu": 8,
        # Custom fields remain useful in downloaded flat rows. The standard
        # renderer ignores fields it does not understand.
        "global_batch_size": int(global_batch),
        "local_batch_size": int(aggregate["local_batch_size"]),
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
        **equivalent_percentiles,
    }
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
    return {
        "experiment_id": experiment_id,
        "global_batch_size": benchmark.get(
            "global_batch_size",
            benchmark.get("concurrency"),
        ),
        "local_batch_size": benchmark.get("local_batch_size"),
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
        "huawei_decode_round_tpot_ms": huawei.get(
            "decode_round_tpot_ms"
        ),
        "huawei_decode_step_tput_per_chip": huawei.get(
            "decode_step_tput_per_chip"
        ),
        "b300_to_huawei_decode_step_ratio": huawei.get(
            "b300_to_huawei_decode_step_ratio"
        ),
        "b300_to_huawei_output_ratio": huawei.get(
            "b300_to_huawei_output_ratio"
        ),
        "failure_kind": result.get("failure_kind"),
        "error": result.get("error"),
        "source": str(source) if source else None,
    }


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# DeepSeek-V4 B300 TRT Fixed-GBS Offline Benchmark",
        "",
        "| GBS | Local/rank | GPUs | Status | Decode round TPOT ms | Decode steps/s/GPU | Tok/step | Output tok/s/GPU | Wall tok/s/GPU | Retained rounds | Huawei TPOT ms | Huawei steps/s/chip | B300/Huawei step |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        retained = row.get("retained_rounds")
        measured = row.get("measured_decode_rounds")
        retained_label = (
            f"{retained}/{int(measured) - 1}"
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
                    row.get("b300_to_huawei_decode_step_ratio"),
                    3,
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Metric meanings",
            "",
            "- `GBS` is the one authoritative global batch submitted to TRT. `Local/rank` is exactly `GBS / 8` for DEP8.",
            "- `Decode round TPOT` is the mean `iterLatencyMS` for 256 consecutive full-local-batch decode iterations after skipping the first and removing only upper-IQR outliers, matching Huawei's `process_infer_time` path.",
            "- `Decode steps/s/GPU` is `GBS / decode_round_TPOT / 8`. This is the direct comparison with Huawei's published table.",
            "- `Tok/step` is MTP output yield and is reported separately. `Output tok/s/GPU` multiplies decode-step throughput by that yield.",
            "- `Wall tok/s/GPU` covers the entire `LLM.generate()` call and remains diagnostic; it includes the deliberately longer output cap used to guarantee at least 256 decode rounds.",
            "- The global batch and timing method match Huawei. The topology does not: this run uses eight B300 GPUs, while Huawei publishes sixteen 950DT chips.",
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
