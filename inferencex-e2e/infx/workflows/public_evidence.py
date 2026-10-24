"""Ingested InferenceX evidence for the sign-off verifier (Checks 2 and 18)."""

from __future__ import annotations

import argparse
import json
from typing import Any

from infx.klaud import api

SERIES = ("model", "precision", "framework", "hardware", "spec_method")
TOPOLOGY = (
    "disagg",
    "is_multinode",
    "prefill_tp",
    "prefill_ep",
    "prefill_dp_attention",
    "prefill_num_workers",
    "num_prefill_gpu",
    "decode_tp",
    "decode_ep",
    "decode_dp_attention",
    "decode_num_workers",
    "num_decode_gpu",
)
EVAL_METRICS = ("em_strict", "em_strict_se", "n_eff")
POINT_FIELDS = ("benchmark_type", "isl", "osl", "image", "recipe_fingerprint")
POINT_METRICS = ("tput_per_gpu", "mean_intvty", "median_e2el", "p90_e2el")


def _summary(row: dict[str, Any], fields: tuple[str, ...], metrics: tuple[str, ...]) -> dict:
    values = row.get("metrics") or {}
    return {
        **{field: row.get(field) for field in (*fields, "date", "conc", *TOPOLOGY)},
        **{metric: values.get(metric) for metric in metrics},
        "run_url": row.get("run_url"),
    }


def _newest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (str(row["date"]), str(row["run_url"])), reverse=True)


def eval_cohorts(rows: list[Any], target: dict[str, str], limit: int) -> dict[str, Any]:
    """Split ingested evaluations into the exact cohort and same model/precision/task peers."""
    exact, wider = [], []
    for row in rows:
        if not isinstance(row, dict) or any(
            row.get(field) != target[field] for field in ("model", "precision", "task")
        ):
            continue
        same = all(row.get(field) == target[field] for field in target)
        (exact if same else wider).append(_summary(row, (*SERIES, "task"), EVAL_METRICS))
    exact, wider = _newest(exact), _newest(wider)
    return {
        "target": target,
        "exact": exact[:limit],
        "exact_total": len(exact),
        "same_model_precision_task": wider[:limit],
        "same_model_precision_task_total": len(wider),
    }


def series_points(rows: list[Any], target: dict[str, str], limit: int) -> dict[str, Any]:
    """Return the ingested benchmark points of one model/precision/framework/hardware/spec series."""
    points = _newest(
        [
            _summary(row, (*SERIES, *POINT_FIELDS), POINT_METRICS)
            for row in rows
            if isinstance(row, dict) and all(row.get(field) == target[field] for field in target)
        ]
    )
    return {"target": target, "points": points[:limit], "points_total": len(points)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evals = commands.add_parser("evals", help="peer evaluations for Check 2")
    points = commands.add_parser("points", help="ingested benchmark points for Check 18")
    points.add_argument(
        "--model-name", required=True, help="public model name, e.g. DeepSeek-V4.1-Flash"
    )
    for command in (evals, points):
        for field in SERIES:
            command.add_argument("--" + field.replace("_", "-"), required=True)
        command.add_argument("--limit", type=int, default=25)
    evals.add_argument("--task", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "evals":
            feed = api.fetch("evaluations")
        else:
            feed = api.fetch("benchmarks", model=args.model_name)
    except api.ReadError as error:
        parser.exit(1, f"{args.command} feed unavailable: {error}\n")
    if not isinstance(feed.payload, list):
        parser.exit(1, f"{args.command} feed is not a list\n")
    target = {field: getattr(args, field) for field in SERIES}
    report: dict[str, Any] = {"source": feed.url, "retrieved_at": feed.retrieved_at}
    if args.command == "evals":
        report |= eval_cohorts(feed.payload, target | {"task": args.task}, args.limit)
    else:
        report |= series_points(feed.payload, target, args.limit)
    print(json.dumps(report, separators=(",", ":")))


if __name__ == "__main__":
    main()
