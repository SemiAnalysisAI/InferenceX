"""Plan and summarize the AgentX speculative acceptance gate (milestone M1 of #1651).

Before AgentX throughput, each deployment topology x drafting configuration measures its real
SPEED-Bench acceptance length (AL) once against the golden curve. The gate is report-only.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from infx.clusters import CLUSTER_LABEL_PREFIX, load_clusters
from infx.golden_al_distribution import GOLDEN_DIR, curve_name, golden_length, load_curve
from infx.launch.drivers.srt.recipe import recipe_mirror_path
from infx.srt_slurm.single_node import select_recipe
from infx.srt_slurm.synthetic_acceptance import (
    ENGINES,
    generation_role,
    selected_recipes,
    spec_parameters,
)

# Points that differ only in these share a gate (KV offloading restores the same cache).
POINT_FIELDS = frozenset((
    "conc", "exp-name", "recipe-fingerprint", "priority", "queue-token", "run-eval", "eval-only",
    "eval-conc", "eval-framework", "eval-suite", "kv-offloading", "kv-offload-backend",
    "total-cpu-dram-gb",
))  # fmt: skip
SPEC_FIELDS = (
    "method", "num_speculative_tokens", "model", "draft_sample_method", "rejection_sample_method"
)  # fmt: skip
EXEMPT_FRAMEWORKS = {
    "dynamo-trt": "exempt: the Dynamo TRT-LLM recipes disable TRT-LLM metrics",
    "tilert": "exempt: TileRT rejects streaming /v1/completions and serves no metrics",
}
LEADER_ONLY_METRICS = (
    "exempt: srtctl gives the gate only the leader's metrics of this multi-node Dynamo vLLM "
    "data-parallel worker, so AL would cover part of the prompts"
)


def _concurrency(row: Mapping[str, Any]) -> int:
    return min(row["conc"]) if isinstance(row["conc"], list) else row["conc"]


def point_environment(row: Mapping[str, Any]) -> dict[str, str]:
    """The launch environment benchmark-tmpl.yml derives from a single-node AgentX row."""
    tp, pp, pcp = int(row["tp"]), int(row["pp"]), int(row["pcp-size"])
    kv_offloading, dram = row.get("kv-offloading"), row.get("total-cpu-dram-gb")
    return {
        "FRAMEWORK": row["framework"], "MODEL": row["model"], "IMAGE": row["image"],
        "PRECISION": row["precision"], "TP": str(tp), "PP_SIZE": str(pp),
        "DCP_SIZE": str(row["dcp-size"]), "PCP_SIZE": str(pcp), "EP_SIZE": str(row["ep"]),
        "DP_ATTENTION": "true" if row["dp-attn"] else "false", "GPU_COUNT": str(tp * pp * pcp),
        "CONC": str(row["conc"]), "SPEC_DECODING": row["spec-decoding"], "IS_AGENTIC": "1",
        "KV_OFFLOADING": "" if kv_offloading is None else str(kv_offloading),
        "TOTAL_CPU_DRAM_GB": "" if dram is None else str(dram),
    }  # fmt: skip


def point_recipe(row: Mapping[str, Any], workspace: Path) -> dict[str, Any]:
    """The resolved recipe variant a point launches, as the srt driver selects it."""
    if row.get("prefill") is None:
        _, recipe = select_recipe(f"{workspace}/{row['srt-recipe']}", point_environment(row))
        return recipe
    configs = {
        setting.split("=", 1)[1]
        for role in ("prefill", "decode")
        for setting in (row.get(role) or {}).get("additional-settings") or []
        if setting.startswith("CONFIG_FILE=")
    }
    if len(configs) != 1:
        raise ValueError(f"multi-node point names CONFIG_FILE {sorted(configs)}, expected one")
    [config] = configs
    path = recipe_mirror_path(workspace, config)
    if not path.is_file():
        raise ValueError(f"recipe {config} is not in the workspace mirror")
    variants = selected_recipes(yaml.safe_load(path.read_text()), config.partition(":")[2] or None)
    if len(variants) != 1:
        raise ValueError(f"{config} selects {len(variants)} recipe variants, expected one")
    return variants[0][1]


def deployment_key(row: Mapping[str, Any]) -> str:
    """Identify a deployment: the point without concurrency-level fields or recipe files."""
    deployment = {key: value for key, value in row.items() if key not in POINT_FIELDS}
    for role in ("prefill", "decode"):
        if isinstance(worker := deployment.get(role), Mapping):
            settings = worker.get("additional-settings") or []
            kept = [setting for setting in settings if not setting.startswith("CONFIG_FILE=")]
            deployment[role] = {**worker, "additional-settings": kept}
    return json.dumps(deployment, sort_keys=True, default=str)


def leader_only_metrics(role: Mapping[str, Any], recipe: Mapping[str, Any], runner: str) -> bool:
    """Whether srtctl runs a Dynamo vLLM worker's data-parallel ranks as one process per node,
    as it does when they span nodes but a tensor x pipeline replica fits on one."""
    args = role.get("args") or {}
    dp, tp, pp = (int(args.get(f"{k}-parallel-size", 1)) for k in ("data", "tensor", "pipeline"))
    if dp == 1:
        return False
    gpus_per_node = (recipe.get("resources") or {}).get("gpus_per_node")
    if gpus_per_node is None:  # srtctl then uses the cluster's, from runners.yaml
        gpus_per_node = load_clusters()[runner.removeprefix(CLUSTER_LABEL_PREFIX)].gpus_per_node
    return tp * pp <= gpus_per_node < tp * pp * dp


def _gate(
    row: Mapping[str, Any], workspace: Path, thinking_mode: str, golden_dir: Path
) -> dict[str, Any] | str | None:
    """The point's gate, the reason it gets none, or None when it does not draft."""
    if row["framework"] in EXEMPT_FRAMEWORKS:
        return EXEMPT_FRAMEWORKS[row["framework"]]
    recipe = point_recipe(row, workspace)
    engine = ENGINES[row["framework"]]
    role = generation_role(recipe, engine)
    spec = spec_parameters(role, engine)
    if not spec:
        return None
    if row["framework"] == "dynamo-vllm" and leader_only_metrics(role, recipe, row["runner"]):
        return LEADER_ONLY_METRICS
    curve = curve_name(row["model-prefix"], spec)
    golden = golden_length(row["model-prefix"], spec, thinking_mode, golden_dir)
    collection = load_curve(curve, golden_dir).collection
    if collection is None:
        return f"golden curve {curve} records no collection settings to replay"
    key = json.dumps([deployment_key(row), [spec.get(field) for field in SPEC_FIELDS]])
    return {
        "id": hashlib.sha256(key.encode()).hexdigest()[:12],
        "curve": curve,
        "draft-tokens": spec["num_speculative_tokens"],
        "golden-acceptance-length": golden,
        "concurrency": collection["concurrency"],
    }


def plan(
    space: Mapping[str, Any], workspace: Path, thinking_mode: str, *, golden_dir: Path = GOLDEN_DIR
) -> dict[str, Any]:
    """Gate a search space's AgentX speculative points; only an ImportError stops the plan."""
    groups: dict[str, tuple[dict[str, Any], str, list[dict[str, Any]]]] = {}
    unresolved = []
    for kind in ("single_node", "multi_node"):
        for row in (space.get(kind) or {}).get("agentic", []):
            if row.get("spec-decoding", "none") == "none":
                continue
            try:
                gate = _gate(row, workspace, thinking_mode, golden_dir)
            except ImportError:
                raise
            except Exception as error:  # noqa: BLE001 - one point's error must not stop the plan
                gate = f"planning error: {type(error).__name__}: {error}"
            if isinstance(gate, str):
                unresolved.append({"point": row["exp-name"], "conc": row["conc"], "reason": gate})
            elif gate is not None:
                groups.setdefault(gate["id"], (gate, kind, []))[2].append(row)
    gates: dict[str, list[dict[str, Any]]] = {"single_node": [], "multi_node": []}
    for gate, kind, rows in groups.values():
        rows.sort(key=_concurrency)
        # The smallest point at or above the golden collection concurrency, else the largest.
        chosen = next((row for row in rows if _concurrency(row) >= gate["concurrency"]), rows[-1])
        gates[kind].append({**chosen, "al-gate": {**gate, "covers": [row["conc"] for row in rows]}})
    return {**gates, "unresolved": unresolved}


def _number(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def summarize(planned: Mapping[str, Any], results: list[Mapping[str, Any]]) -> str:
    """Markdown of the gate verdicts, planned gates without a result, and unresolved points."""
    lines = [
        "## Speculative acceptance gate (report-only)",
        "",
        "| Point | Curve | Draft tokens | Golden AL | Measured AL | Ratio | Status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| `{r.get('result_filename')}` | {r.get('curve')} | {r.get('draft_tokens')} | "
        f"{_number(r.get('golden_acceptance_length'))} | {_number(r.get('acceptance_length'))} | "
        f"{_number(r.get('ratio'), 3)} | {r.get('status')} |"
        for r in sorted(results, key=lambda r: str(r.get("result_filename")))
    ]
    gates = sum(len(planned.get(kind, [])) for kind in ("single_node", "multi_node"))
    if gates > len(results):
        lines += ["", f"{gates - len(results)} planned gate(s) have no result; see their logs."]
    by_reason: dict[str, list[str]] = {}
    for point in planned.get("unresolved", []):
        by_reason.setdefault(point["reason"], []).append(
            f"`{point['point']}` (conc {point['conc']})"
        )
    if by_reason:
        lines += ["", f"### Unresolved points ({len(planned['unresolved'])}, no gate)", ""]
    for reason, points in by_reason.items():
        lines.append(f"- {reason}: {len(points)} point(s), e.g. {', '.join(points[:3])}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan", help="gate points of a run-sweep search space")
    plan_parser.add_argument("search_space", type=Path)
    plan_parser.add_argument("--workspace", type=Path, required=True)
    plan_parser.add_argument("--thinking-mode", required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    summary_parser = commands.add_parser("summarize", help="tabulate downloaded gate results")
    summary_parser.add_argument("results", type=Path)
    summary_parser.add_argument("--plan", type=Path)
    args = parser.parse_args(argv)
    if args.command == "plan":
        gates = plan(json.loads(args.search_space.read_text()), args.workspace, args.thinking_mode)
        args.output.write_text(json.dumps(gates, indent=2) + "\n")
        return 0
    # The collect job also runs when no plan was uploaded.
    planned = json.loads(args.plan.read_text()) if args.plan and args.plan.is_file() else {}
    results = [json.loads(path.read_text()) for path in args.results.rglob("al_gate_*.json")]
    sys.stdout.write(summarize(planned, [result for result in results if "status" in result]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
