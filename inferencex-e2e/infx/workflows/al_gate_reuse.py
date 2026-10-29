"""Skip acceptance gates an earlier run-sweep run on the same PR branch already measured."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

from infx import github
from infx.workflows import sweep_runs

KINDS = ("single_node", "multi_node")
LOOKBACK_RUNS = 30
GATE_INPUTS = (
    "benchmarks/srt_al_gate.sh",
    "infx/bench/al_gate.py",
    "infx/bench/speedbench.py",
    "infx/bench/agentic/requirements.txt",
    "infx/bench_serving/encoding_dsv4.py",
    "infx/results/agentic/spec_decode.py",
    "infx/srt_slurm/synthetic_acceptance.py",
    "infx/launch/drivers/srt/submit.py",
)


def gate_key(gate: dict[str, Any], workspace: Path, srt_slurm_commit: str) -> str:
    """Hash the gate's deployment, recipe, golden curve, srt-slurm commit and gate code."""
    identity = [gate["al-gate"]["id"], gate["recipe-fingerprint"], srt_slurm_commit]
    digest = hashlib.sha256(json.dumps(identity).encode())
    curve = Path("infx/golden_al_distribution") / f"{gate['al-gate']['curve']}.yaml"
    for name in (curve, *GATE_INPUTS):
        digest.update((workspace / name).read_bytes())
    return digest.hexdigest()[:16]


def measured(repo: str, branch: str, keys: set[str], token: str) -> dict[str, dict[str, Any]]:
    """The newest non-error result of each key among the branch's recent completed runs."""
    found: dict[str, dict[str, Any]] = {}
    runs = sweep_runs.completed_pr_runs(repo, "run-sweep.yml", branch, token)
    for run in sorted(runs, key=lambda run: run["created_at"], reverse=True)[:LOOKBACK_RUNS]:
        if found.keys() == keys:
            break
        artifacts = github.paginate(
            repo, f"/actions/runs/{run['id']}/artifacts", token, "artifacts"
        )
        for artifact in artifacts:
            key = artifact["name"].removeprefix("algate_")
            if key not in keys or key in found or artifact.get("expired"):
                continue
            archive = github.download(repo, f"/actions/artifacts/{artifact['id']}/zip", token)
            with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
                for name in fnmatch.filter(bundle.namelist(), "al_gate_*.json"):
                    result = json.loads(bundle.read(name))
                    if result.get("status") != "error":
                        found[key] = result
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--srt-slurm-commit", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    gates = [gate for kind in KINDS for gate in plan[kind]]
    for gate in gates:
        gate["al-gate"]["key"] = gate_key(gate, args.workspace, args.srt_slurm_commit)
    keys = {gate["al-gate"]["key"] for gate in gates}
    try:
        found = measured(args.repo, args.branch, keys, os.environ["GH_TOKEN"]) if keys else {}
    except (github.APIError, github.ListingError) as error:
        print(f"::warning title=Acceptance gate reuse::{error}", file=sys.stderr)
        found = {}
    for key, result in found.items():
        path = args.results / key / f"al_gate_{result['result_filename']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    for kind in KINDS:
        plan[kind] = [gate for gate in plan[kind] if gate["al-gate"]["key"] not in found]
    args.output.write_text(json.dumps(plan) + "\n")
    print(f"Reused {len(found)} of {len(keys)} acceptance gate result(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
