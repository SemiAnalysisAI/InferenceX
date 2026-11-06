"""Plan and summarize the report-only SPEED-Bench acceptance gate of AgentX speculative evals.

An agentic eval job whose generation role drafts replays the golden curve's SPEED-Bench
collection before its eval, against the same server: srt-slurm runs the gate as the job's
benchmark stage, which alone receives the workers' metrics endpoints, and the eval as its
post-benchmark eval. The gate never fails the job.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from infx.golden_al_distribution import GOLDEN_DIR, curve_name, load_curve
from infx.srt_slurm.synthetic_acceptance import (
    ENGINES,
    generation_role,
    spec_parameters,
    verifier,
)

COMMAND = "bash /infmax-workspace/benchmarks/srt_al_gate.sh"
OUTPUT_DIR = "/logs"
# srt-slurm runs the benchmark stage, then the eval, only without EVAL_ONLY.
SRT_ENVIRONMENT = {"EVAL_ONLY": "false", "RUN_EVAL": "true"}
EXEMPT_FRAMEWORKS = {
    "tilert": "TileRT rejects streaming /v1/completions and serves no metrics",
}
LEADER_ONLY_METRICS = (
    "srtctl gives the gate only the leader's metrics of this multi-node Dynamo vLLM "
    "data-parallel worker, so AL would cover part of the prompts"
)


def requested(environment: Mapping[str, str]) -> bool:
    """Whether the launcher found the prompts for this agentic speculative eval."""
    return environment.get("EVAL_AL_GATE") == "true"


def leader_only_metrics(
    role: Mapping[str, Any], recipe: Mapping[str, Any], environment: Mapping[str, str]
) -> bool:
    """Whether srtctl runs a Dynamo vLLM worker's data-parallel ranks as one process per node,
    as it does when they span nodes but a tensor x pipeline replica fits on one."""
    args = role.get("args") or {}
    dp, tp, pp = (int(args.get(f"{k}-parallel-size", 1)) for k in ("data", "tensor", "pipeline"))
    if dp == 1:
        return False
    gpus_per_node = (recipe.get("resources") or {}).get("gpus_per_node")
    if gpus_per_node is None:  # srtctl then uses the cluster's, from runners.yaml
        from infx.clusters import CLUSTER_LABEL_PREFIX, load_clusters

        cluster = environment["RUNNER_TYPE"].removeprefix(CLUSTER_LABEL_PREFIX)
        gpus_per_node = load_clusters()[cluster].gpus_per_node
    return tp * pp <= gpus_per_node < tp * pp * dp


def plan(
    recipe: Mapping[str, Any],
    framework: str,
    environment: Mapping[str, str],
    *,
    golden_dir: Path = GOLDEN_DIR,
) -> dict[str, Any] | None:
    """The gate a resolved eval recipe runs, a ``skipped`` record, or None when it does not draft.

    A runnable gate has ``curve``, ``draft_tokens`` and ``golden_acceptance_length``. The curve
    is the one synthetic acceptance selects for the generation role and its verifier.
    """
    if framework in EXEMPT_FRAMEWORKS:
        return skipped(EXEMPT_FRAMEWORKS[framework], environment)
    engine = ENGINES.get(framework)
    if engine is None:
        return skipped(f"{framework} has no speculative settings to read", environment)
    role = generation_role(recipe)
    spec = spec_parameters(role, engine)
    if not spec:
        return None
    if framework == "dynamo-vllm" and leader_only_metrics(role, recipe, environment):
        return skipped(LEADER_ONLY_METRICS, environment)
    try:
        verify = verifier(role, engine).value
        name = curve_name(environment["MODEL_PREFIX"], spec, golden_dir, verify)
        curve = load_curve(name, golden_dir)
        tokens = spec.get("num_speculative_tokens")
        golden = curve.acceptance(environment["THINKING_MODE"], tokens)
    except (KeyError, TypeError, ValueError) as error:
        return skipped(f"no golden AL: {error}", environment)
    if curve.collection is None:
        return skipped(f"golden curve {curve.name} records no collection to replay", environment)
    return {"curve": curve.name, "draft_tokens": tokens, "golden_acceptance_length": golden}


def skipped(reason: str, environment: Mapping[str, str]) -> dict[str, Any]:
    """The result document of a gate that did not run."""
    return {
        "result_filename": environment.get("RESULT_FILENAME"),
        "framework": environment.get("FRAMEWORK"),
        "eval_framework": environment.get("EVAL_FRAMEWORK"),
        "eval_suite": environment.get("EVAL_SUITE"),
        "status": "skipped",
        "error": reason,
        "report_only": True,
    }


def overrides(gate: Mapping[str, Any], environment: Mapping[str, str]) -> list[str]:
    """``srtctl apply`` arguments that run ``gate`` as the benchmark stage, without power.

    ``benchmark.env`` outranks the job environment. The container resolves the relative
    ``AL_GATE_DATASET`` in ``/infmax-workspace``.
    """
    env = {
        "AL_GATE": "true",
        "AL_GATE_CURVE": gate["curve"],
        "AL_GATE_DRAFT_TOKENS": str(gate["draft_tokens"]),
        "AL_GATE_DATASET": environment["AL_GATE_DATASET"],
        "AL_GATE_OUTPUT_DIR": OUTPUT_DIR,
        "THINKING_MODE": environment["THINKING_MODE"],
        **{
            name: environment[name]
            for name in ("EVAL_FRAMEWORK", "EVAL_SUITE")
            if environment.get(name)
        },
    }
    arguments = [
        "--set", 'benchmark.type="custom"',
        "--set", f"benchmark.command={json.dumps(COMMAND)}",
    ]  # fmt: skip
    for name, value in env.items():
        arguments += ["--set", f"benchmark.env.{name}={json.dumps(value)}"]
    return [*arguments, "--unset", "telemetry"]


def _number(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def summarize(results: list[Mapping[str, Any]]) -> str:
    """Markdown of the gate verdicts, or an empty string without any."""
    if not results:
        return ""
    lines = [
        "## Speculative acceptance gate (report-only)",
        "",
        (
            "Measured with real verification on the eval's server; "
            "`pass` is within 5% of the golden AL."
        ),
        "",
        "| Point | Eval | Curve | Draft tokens | Golden AL | Measured AL | Ratio | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(
        results, key=lambda r: (str(r.get("result_filename")), str(r.get("eval_framework")))
    ):
        evaluation = " ".join(filter(None, (r.get("eval_framework"), r.get("eval_suite")))) or "-"
        status = r.get("status")
        if r.get("error"):
            status = f"{status}: {r['error']}"
        lines.append(
            f"| `{r.get('result_filename')}` | {evaluation} | {r.get('curve') or '-'} | "
            f"{r.get('draft_tokens') or '-'} | {_number(r.get('golden_acceptance_length'))} | "
            f"{_number(r.get('acceptance_length'))} | {_number(r.get('ratio'), 3)} | {status} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summarize", help="tabulate downloaded al_gate_*.json results")
    summary.add_argument("results", type=Path)
    args = parser.parse_args(argv)
    results = []
    for path in sorted(args.results.rglob("al_gate_*.json")):
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            print(f"WARNING: unreadable {path}: {error}", file=sys.stderr)
            continue
        if isinstance(document, dict) and "status" in document:
            results.append(document)
    sys.stdout.write(summarize(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
