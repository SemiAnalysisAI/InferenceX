#!/usr/bin/env python3
"""Retain original sources so paired fidelity is auditable without GPU reruns."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import ci
import export_ci
from evaluator.mvp_compare import compare_runs
from evaluator.mvp_report import write_report


def selected_run(root: Path, source: dict) -> Path:
    expected = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        ci.need(match is not None, "Malformed source checksum entry")
        digest, name = match.groups()
        ci.need(name not in expected, "Duplicate source checksum entry")
        expected[name] = digest
    ci.need(expected and ci.inventory(root) == expected, "Source artifact checksum mismatch")
    record = ci.read(root / "ci.json")
    ci.need(str(record.get("run_id")) == str(source["databaseId"])
            and str(record.get("run_attempt")) == str(source["runAttempt"])
            and record.get("source_sha") == source["headSha"], "Source CI identity mismatch")
    manifest = ci.read(root / "manifest.json")
    ci.need(str(manifest.get("run_id")) == str(source["databaseId"])
            and str(manifest.get("run_attempt")) == str(source["runAttempt"])
            and manifest.get("git_commit") == source["headSha"]
            and manifest.get("mode") == "serving-smoke", "Source manifest identity mismatch")
    matrix = ci.read(root / "serving-smoke.json")
    ci.need(matrix.get("bundle_type") == "h3_serving_smoke_matrix"
            and matrix.get("schema_version") == "1.0.0", "Expected the existing serving matrix contract")
    cells = [cell for cell in matrix["cells"] if cell.get("concurrency") == 1]
    ci.need(len(cells) == 1, "Source requires exactly one C1 cell")
    cell = cells[0]
    relative = cell.get("run", {}).get("path")
    ci.need(isinstance(relative, str) and relative in expected, "C1 run is missing from the sealed artifact")
    path = (root / relative).resolve()
    ci.need(path.is_relative_to(root.resolve()) and path.name == "run.json"
            and expected[relative] == cell["run"]["sha256"], "C1 run identity mismatch")
    run = ci.read(path)
    ci.need(run.get("configuration", {}).get("serving", {}).get("concurrency") == 1,
            "Selected run is not C1")
    return path.parent


def publish(run_ids: list[str], output: Path) -> None:
    ci.need(len(run_ids) == 2 and run_ids == export_ci.source_ids(",".join(run_ids)),
            "Exactly two distinct source runs are required")
    sha = os.environ.get("GITHUB_SHA", "")
    ci.need(re.fullmatch(r"[0-9a-f]{40}", sha)
            and ci.command(["git", "rev-parse", "HEAD"]).strip() == sha, "Exact exporter checkout required")
    run_id, attempt = os.environ.get("GITHUB_RUN_ID", ""), os.environ.get("GITHUB_RUN_ATTEMPT", "")
    ci.need(run_id.isdigit() and attempt.isdigit()
            and os.environ.get("GITHUB_REPOSITORY") == export_ci.REPOSITORY, "GitHub producer identity required")
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="h3-fidelity-sources-") as scratch:
        compare_sources(run_ids, output, Path(scratch), sha, run_id, attempt)


def compare_sources(run_ids: list[str], output: Path, scratch: Path, sha: str, run_id: str, attempt: str) -> None:
    sources, directories = [], []
    for source in run_ids:
        metadata, artifact = export_ci.verified_execution(source)
        target = scratch / ("source-" + source)
        subprocess.run(["gh", "run", "download", source, "--repo", export_ci.REPOSITORY,
                        "--name", artifact["name"], "--dir", str(target)], check=True, timeout=300)
        selected = selected_run(target, metadata)
        directories.append(selected)
        snapshot = output / "sources" / source
        snapshot.mkdir(parents=True)
        for name in ("ci.json", "manifest.json", "serving-smoke.json"):
            shutil.copyfile(target / name, snapshot / name)
        shutil.copyfile(target / "SHA256SUMS", snapshot / "original-SHA256SUMS")
        shutil.copyfile(selected / "run.json", snapshot / "c1-run.json")
        sources.append({"ci": metadata, "artifact": artifact,
                        "source_seal_sha256": ci.digest(target / "SHA256SUMS")})
    policy = ci.read(Path(__file__).parent / "mvp/example-uncalibrated.policy.json")
    comparison = compare_runs(*directories, policy=policy)
    comparison["producer"] = {"git_commit": sha, "run_id": run_id, "run_attempt": attempt,
                              "run_url": f"https://github.com/{export_ci.REPOSITORY}/actions/runs/{run_id}",
                              "mode": "CPU-only comparison of original C1 media; no new generation"}
    comparison["source_artifacts"] = sources
    ci.write(output / "comparison.json", comparison)
    write_report(comparison, output / "report/index.html")
    ci.write(output / "reprocessing.json", {"status": "complete", "generation_executed": False,
             "calibration_status": policy["calibration_status"], "release_qualified": False,
             "threshold_outcome": comparison["overall_status"], "matched_pairs": comparison["summary"]["matched_valid_pairs"],
             "interpretation": "CI success means report generation succeeded, not that fidelity or regression passed."})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-ids", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new directory")
    status = 0
    try:
        publish(export_ci.source_ids(args.source_run_ids), args.output)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        ci.write(args.output / "reprocessing-error.json", {"status": "failed", "error": str(error),
                 "generation_executed": False, "release_qualified": False})
        status = 2
    finally:
        files = export_ci.files_with_nested_seals(args.output)
        (args.output / "SHA256SUMS").write_text("".join(f"{sha}  {path}\n" for path, sha in files.items()))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
