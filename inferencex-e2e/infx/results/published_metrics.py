"""What the published-curve snapshot derives from a dashboard row: its curve and line, Pareto
frontier membership, physical GPUs, tok/$ and engine source.

These reproduce the dashboard's computations from the row's published fields, so they can
differ from what the dashboard shows if its formulas change.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from typing import Any

ENGINES = {
    "sglang": "sgl-project/sglang",
    "vllm": "vllm-project/vllm",
    "trt": "NVIDIA/TensorRT-LLM",
}
IMAGE_COMMIT = re.compile(r"-([0-9a-f]{7,40})(?:@|$)")
PRICE_KEY = "h"
DERIVED = ("tput_per_physical_gpu", "tok_per_usd")
# The frontier axes and the direction that is better: the metrics the snapshot reads or derives
BETTER = {
    **{f"{percentile}_intvty": "max" for percentile in ("median", "p75", "p90")},
    **{f"{p}_{latency}": "min" for p in ("median", "p75", "p90") for latency in ("ttft", "e2el")},
    **dict.fromkeys(("tput_per_gpu", "input_tput_per_gpu", "output_tput_per_gpu", *DERIVED), "max"),
}
DEFAULT_FRONTIER = ("p90_intvty", "tput_per_gpu")
# The dashboard's x axes and the direction that is better
X_AXES = {"intvty": "max", "e2el": "min", "ttft": "min", "e2e_norm_intvty": "max"}


def curve_of(row: Mapping[str, Any]) -> dict[str, Any]:
    """The curve a row is drawn on: the API replaces a curve's points together."""
    fields = ("hardware", "framework", "precision", "benchmark_type", "isl", "osl")
    return {field: row.get(field) for field in fields}


def line_of(row: Mapping[str, Any]) -> list[Any]:
    """The line whose frontier the dashboard marks a row on: its curve on the curve's date, and
    for a fixed sequence its spec method too. Pooled models' curves share a line on one date."""
    spec = None if row.get("benchmark_type") == "agentic_traces" else row.get("spec_method")
    return [*curve_of(row).values(), spec, row.get("curve_date") or row.get("date")]


def interactivity(row: Mapping[str, Any], percentile: str = "p90") -> float | None:
    """Interactivity at ``percentile`` as the dashboard plots it: for AgentX, 1 / ITL,
    full-response where the row has full-response ITLs; for a fixed sequence, the published
    ``<percentile>_intvty``."""
    metrics = row.get("metrics") or {}
    if row.get("benchmark_type") != "agentic_traces":
        published = metrics.get(f"{percentile}_intvty")
        return published if isinstance(published, int | float) and published > 0 else None
    full = any(k.endswith("_full_response_itl") and k != "std_full_response_itl" for k in metrics)
    itl = metrics.get(f"{percentile}_full_response_itl")
    if not (isinstance(itl, int | float) and itl > 0):
        itl = None if full else metrics.get(f"{percentile}_itl")
    return 1 / itl if isinstance(itl, int | float) and itl > 0 else None


def metric(row: Mapping[str, Any], name: str) -> float | None:
    """``row``'s value of the frontier axis ``name``, or None: interactivity as the dashboard
    plots it, a field the snapshot derives, else the row's published metric."""
    if name.endswith("_intvty"):
        return interactivity(row, name.removesuffix("_intvty"))
    value = row.get(name) if name in DERIVED else (row.get("metrics") or {}).get(name)
    return value if isinstance(value, int | float) else None


def x_axes(row: Mapping[str, Any]) -> dict[str, float | None]:
    """``row``'s value on each dashboard x axis at the dashboard's default statistic, P90 for
    AgentX and median for a fixed sequence. E2E normalized interactivity is AgentX only. A
    persisted row needs the derived-agentic-metrics value merged into its metrics first."""
    agentic = row.get("benchmark_type") == "agentic_traces"
    statistic = "p90" if agentic else "median"
    metrics = row.get("metrics") or {}

    def positive(name: str) -> float | None:
        value = metrics.get(f"{statistic}_{name}")
        return value if isinstance(value, int | float) and value > 0 else None

    return {
        "intvty": interactivity(row, statistic),
        "e2el": positive("e2el"),
        "ttft": positive("ttft"),
        "e2e_norm_intvty": positive("e2e_norm_intvty") if agentic else None,
    }


def frontier(rows: list[dict], axes: tuple[str, ...] = DEFAULT_FRONTIER) -> list[dict]:
    """The rows of one line on its Pareto frontier of the two ``axes``, each toward its
    ``BETTER`` direction; by default P90 interactivity and tok/s/GPU, as the dashboard's
    interactivity chart draws it (paretoFrontUpperLeft)."""
    points = []
    for row in rows:
        values = [metric(row, name) for name in axes]
        if None not in values:
            signs = [1 if BETTER[name] == "max" else -1 for name in axes]
            points.append((signs[0] * values[0], signs[1] * values[1], row))
    front: list[tuple[float, float, dict]] = []
    for x, y, row in sorted(points, key=lambda point: (point[0], -point[1])):
        if front and x == front[-1][0]:
            if y > front[-1][1]:
                front[-1] = (x, y, row)
            continue
        while front and y >= front[-1][1]:
            front.pop()
        front.append((x, y, row))
    return [row for _, _, row in front]


def plotted(row: Mapping[str, Any], axes: tuple[str, ...] = DEFAULT_FRONTIER) -> bool:
    """Whether ``row`` has both ``axes``, by default those of the dashboard's interactivity chart."""
    return all(metric(row, name) is not None for name in axes)


def frontier_axes(text: str) -> tuple[str, ...]:
    """``X,Y`` as two frontier axes; raise ArgumentTypeError naming the valid ones."""
    axes = tuple(name.strip() for name in text.split(","))
    if len(axes) != 2 or not set(axes) <= BETTER.keys():
        raise argparse.ArgumentTypeError(f"{text!r} is not X,Y of {', '.join(BETTER)}")
    return axes


def physical_gpus(row: Mapping[str, Any]) -> int | None:
    """The GPUs a row's deployment holds; the dashboard reports an aggregated row's TP x EP."""
    prefill, decode = row.get("num_prefill_gpu"), row.get("num_decode_gpu")
    if row.get("disagg"):
        return prefill + decode if isinstance(prefill, int) and isinstance(decode, int) else None
    tp, ep = row.get("prefill_tp"), row.get("prefill_ep")
    if isinstance(tp, int) and isinstance(ep, int) and tp > 1 and ep > 1 and prefill == tp * ep:
        return max(tp, ep)
    return prefill if isinstance(prefill, int) else None


def engine(framework: str) -> str:
    return str(framework).removeprefix("dynamo-")


def engine_source(row: Mapping[str, Any]) -> dict[str, str | None]:
    """The engine repository a row ran and the commit its image tag names; a date is no commit."""
    tag = IMAGE_COMMIT.search(str(row.get("image") or ""))
    commit = tag[1] if tag and not re.fullmatch(r"20\d{6}", tag[1]) else None
    return {"repository": ENGINES.get(engine(row.get("framework", ""))), "commit": commit}


def published(row: dict, prices: Mapping[str, Any]) -> dict[str, Any]:
    """A dashboard row with its curve, physical GPUs, tok/$ and engine."""
    metrics = row.get("metrics") or {}
    gpus, total = physical_gpus(row), metrics.get("total_tput_tps")
    price = (prices.get(row.get("hardware")) or {}).get(PRICE_KEY)
    tput = metrics.get("tput_per_gpu")
    return {
        **row,
        "curve": curve_of(row),
        "physical_gpus": gpus,
        "tput_per_physical_gpu": total / gpus if gpus and isinstance(total, int | float) else None,
        "tok_per_usd": tput * 3600 / price if price and isinstance(tput, int | float) else None,
        "price_key": PRICE_KEY if price else None,
        "engine": engine_source(row),
    }
