"""Skip acceptance gates an earlier run-sweep run of the same PR already measured."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

from infx import github
from infx.workflows import sweep_runs

KINDS = ("single_node", "multi_node")
LOOKUP_ERRORS = (
    github.APIError,
    github.ListingError,
    subprocess.TimeoutExpired,
    zipfile.BadZipFile,
    ValueError,
)


def gate_key(gate: dict[str, Any], code_version: str) -> str:
    """Hash the gate's deployment id, recipe fingerprint and the version of all code it runs."""
    identity = [gate["al-gate"]["id"], gate["recipe-fingerprint"], code_version]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]


def measured(
    repo: str, pr_number: int, run_id: int, keys: set[str], token: str
) -> dict[str, dict[str, Any]]:
    """The newest non-error result of each key among this PR's earlier runs."""
    shas = sweep_runs.pr_commit_shas(repo, pr_number, token)
    found: dict[str, dict[str, Any]] = {}
    for key in keys:
        params = {"name": f"algate_{key}"}
        artifacts = github.paginate(repo, "/actions/artifacts", token, "artifacts", params)
        ours = [
            artifact
            for artifact in artifacts
            if not artifact.get("expired")
            and (run := artifact.get("workflow_run") or {}).get("head_sha") in shas
            and run.get("id") != run_id
        ]
        for artifact in sorted(ours, key=lambda artifact: artifact["created_at"], reverse=True):
            if result := _result(repo, artifact, token):
                found[key] = result
                break
    return found


def _result(repo: str, artifact: dict[str, Any], token: str) -> dict[str, Any] | None:
    archive = github.download(repo, f"/actions/artifacts/{artifact['id']}/zip", token)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for name in fnmatch.filter(bundle.namelist(), "al_gate_*.json"):
            result = json.loads(bundle.read(name))
            if result.get("status") != "error" and "result_filename" in result:
                return result
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    gates = [gate for kind in KINDS for gate in plan[kind]]
    for gate in gates:
        gate["al-gate"]["key"] = gate_key(gate, args.code_version)
    keys = {gate["al-gate"]["key"] for gate in gates}
    try:
        token = os.environ["GH_TOKEN"]
        found = measured(args.repo, args.pr_number, args.run_id, keys, token) if keys else {}
    except LOOKUP_ERRORS as error:
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
