#!/usr/bin/env python3
"""Publish new result contracts from immutable, GitHub-verified H3 executions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.request import Request, urlopen

import ci
from evaluator.mvp_result import write_result


REPOSITORY = "SemiAnalysisAI/InferenceX"


def files_with_nested_seals(root: Path) -> dict[str, str]:
    files = ci.inventory(root)
    for path in root.rglob("SHA256SUMS"):
        if path != root / "SHA256SUMS":
            ci.need(path.is_file() and not path.is_symlink(), "Nonregular nested checksum file")
            files[path.relative_to(root).as_posix()] = ci.digest(path)
    return dict(sorted(files.items()))


def source_ids(value: str) -> list[str]:
    values = value.split(",")
    ci.need(1 <= len(values) <= 2 and len(set(values)) == len(values)
            and all(re.fullmatch(r"[1-9][0-9]{0,19}", item) for item in values),
            "Supply one or two distinct numeric source run IDs")
    return values


def api(path: str) -> dict:
    request = Request("https://api.github.com/repos/" + REPOSITORY + "/" + path,
                      headers={"Authorization": "Bearer " + os.environ["GH_TOKEN"],
                               "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
    with urlopen(request, timeout=30) as response:
        payload = response.read(16 * 1024 * 1024 + 1)
    ci.need(len(payload) <= 16 * 1024 * 1024, "GitHub response exceeds metadata limit")
    return json.loads(payload)


def verified_execution(run_id: str, *, inventory: bool = False) -> tuple[dict, dict]:
    run = api("actions/runs/" + run_id)
    current_export = (run_id == os.environ.get("GITHUB_RUN_ID")
                      and str(run["run_attempt"]) == os.environ.get("GITHUB_RUN_ATTEMPT")
                      and run["head_sha"] == os.environ.get("GITHUB_SHA") and run["status"] == "in_progress")
    ci.need(str(run["id"]) == run_id and run["repository"]["full_name"] == REPOSITORY
            and run["head_repository"]["full_name"] == REPOSITORY
            and run["event"] == "workflow_dispatch"
            and ((run["status"] == "completed" and run["conclusion"] == "success") or current_export),
            "Source must be a successful manual InferenceX execution or this run's completed H3 job")
    jobs = api(f"actions/runs/{run_id}/attempts/{run['run_attempt']}/jobs")
    job_name = "H3 H200 hardware inventory" if inventory else "H3 video H200 smoke"
    selected = [job for job in jobs["jobs"] if re.fullmatch(
        r"(?:h3-video / )?p[0-9]+(?:\.[0-9]+)? \| " + re.escape(job_name), job["name"])]
    ci.need(len(selected) == 1 and selected[0]["status"] == "completed"
            and selected[0]["conclusion"] == "success", "Source lacks a successful H3 Slurm job")
    name = f"h3-{'hardware' if inventory else 'video'}-{run_id}-{run['run_attempt']}"
    artifacts = api(f"actions/runs/{run_id}/artifacts")
    selected_artifacts = [item for item in artifacts["artifacts"] if item["name"] == name]
    ci.need(len(selected_artifacts) == 1, "Expected exactly one original H3 artifact")
    artifact = selected_artifacts[0]
    ci.need(not artifact["expired"] and 0 < artifact["size_in_bytes"] <= 2 * 1024**3
            and artifact["workflow_run"]["id"] == run["id"]
            and artifact["workflow_run"]["head_sha"] == run["head_sha"], "Artifact identity, size or retention invalid")
    public_artifact = {key: artifact.get(key) for key in
                       ("id", "name", "digest", "size_in_bytes", "expired")}
    public_artifact["workflow_run"] = {key: artifact["workflow_run"][key] for key in ("id", "head_sha")}
    return ({"databaseId": run["id"], "headSha": run["head_sha"], "runAttempt": run["run_attempt"],
             "event": run["event"], "status": run["status"], "conclusion": run["conclusion"],
             "url": run["html_url"], "jobs": [{key: job[key] for key in ("id", "name", "status", "conclusion")}
                                             for job in jobs["jobs"]]}, public_artifact)


def verified_hardware(root: Path, run_id: str, attempt: str, sha: str) -> dict:
    """Join the downloaded inventory seal, CI identity, Slurm step and teardown."""
    expected = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        ci.need(match is not None, "Malformed hardware checksum entry")
        digest, name = match.groups()
        ci.need(name not in expected, "Duplicate hardware checksum entry")
        expected[name] = digest
    ci.need(expected and ci.inventory(root) == expected, "Hardware artifact checksum mismatch")
    profile, state, manifest, binding, step = (ci.read(root / name) for name in (
        "hardware-profile.json", "ci.json", "manifest.json", "binding.json", "step-result.json"))
    for record in (profile, state, manifest):
        ci.need(str(record["run_id"]) == run_id and str(record["run_attempt"]) == attempt
                and record["source_sha"] == sha, "Hardware observation belongs to a different CI producer")
    ci.need(profile["git_commit"] == sha and profile["ci"]["repository"] == REPOSITORY
            and str(profile["ci"]["run_id"]) == run_id and str(profile["ci"]["run_attempt"]) == attempt,
            "Hardware profile CI identity mismatch")
    ci.need(state["phase"] == "complete" and state["exit_code"] == manifest["exit_code"] == step["exit_code"] == 0
            and step["inventory_completed"] is True and state["step_cleanup"]["status"] == "ended"
            and state["allocation_cleanup"]["status"] == ("retained" if state["allocation_reused"] else "released"),
            "Hardware inventory or cleanup did not complete")
    ci.need(profile["slurm"] == binding and binding["job_id"] == manifest["slurm_allocation"]["identity"]["JobId"]
            == state["allocation"]["identity"]["JobId"]
            and state["step_cleanup"]["step_id"] == binding["job_id"] + "." + binding["step_id"]
            and set(profile["gpu_uuids"]) == set(binding["gpu_uuids"]), "Hardware Slurm/GPU identity mismatch")
    from inventory_ci import classify_tdp
    classified = classify_tdp((root / "nvidia-smi.xml").read_text(), profile["gpu_uuids"])
    ci.need(profile["tdp"] == classified or (profile["tdp"]["status"] == "unknown" and profile["tdp"]["watts_per_gpu"] is None),
            "Hardware TDP profile differs from raw PCI identity")
    ci.need(all(path in expected for path in profile["raw"].values()), "Hardware raw evidence is missing")
    profile["recorded_tdp_classification"] = profile["tdp"]
    profile["tdp"] = classified
    profile["hardware_variant"] = classified["hardware_variant"]
    profile["variant_status"] = classified["status"]
    profile["tdp_classifier_git_commit"] = os.environ.get("GITHUB_SHA")
    profile["evidence_root"] = "hardware"
    profile["tdp"]["evidence"]["raw_path"] = "hardware/nvidia-smi.xml"
    profile["raw"] = {name: "hardware/" + path for name, path in profile["raw"].items()}
    return profile


def publish(run_ids: list[str], output: Path, hardware: Path | None, *, hardware_run_id: str | None = None) -> int:
    output.mkdir(parents=True, exist_ok=False)
    sha = os.environ.get("GITHUB_SHA", "")
    ci.need(re.fullmatch(r"[0-9a-f]{40}", sha)
            and ci.command(["git", "rev-parse", "HEAD"]).strip() == sha, "Exact exporter checkout required")
    run_id, attempt = os.environ.get("GITHUB_RUN_ID", ""), os.environ.get("GITHUB_RUN_ATTEMPT", "")
    ci.need(run_id.isdigit() and attempt.isdigit() and os.environ.get("GITHUB_REPOSITORY") == REPOSITORY,
            "GitHub export identity required")
    ci.need(hardware is None or hardware_run_id is None, "Choose a local hardware artifact or a verified inventory run")
    if hardware_run_id is not None:
        ci.need(source_ids(hardware_run_id) == [hardware_run_id], "Expected one inventory run ID")
        inventory_ci, inventory_artifact = verified_execution(hardware_run_id, inventory=True)
        hardware = output / "hardware"
        subprocess.run(["gh", "run", "download", hardware_run_id, "--repo", REPOSITORY,
                        "--name", inventory_artifact["name"], "--dir", str(hardware)], check=True, timeout=180)
        profile = verified_hardware(hardware, hardware_run_id, str(inventory_ci["runAttempt"]), inventory_ci["headSha"])
        profile["verified_ci"] = inventory_ci
        profile["source_artifact"] = inventory_artifact
    else:
        profile = verified_hardware(hardware, run_id, attempt, sha) if hardware is not None else None
    producer = {"git_commit": sha, "ci": {"repository": REPOSITORY, "run_id": run_id,
                "run_attempt": attempt, "run_url": f"https://github.com/{REPOSITORY}/actions/runs/{run_id}"},
                "mode": "same_run_export" if run_ids == [run_id] else "verified_artifact_reprocessing; no_new_H3_generation"}
    catalog = {"schema_version": "1.0.0", "producer": producer, "results": [], "status": "complete"}
    exit_code = 0
    for source in run_ids:
        target = output / ("source-" + source)
        try:
            source_ci, artifact = verified_execution(source)
            subprocess.run(["gh", "run", "download", source, "--repo", REPOSITORY,
                            "--name", artifact["name"], "--dir", str(target)], check=True, timeout=180)
            original_checksums = (target / "SHA256SUMS").read_bytes()
            result = write_result(target, producer={**producer, "source_artifact": artifact},
                                  source_ci=source_ci, hardware_profile=profile)
            (target / "source-SHA256SUMS").write_bytes(original_checksums)
            ci.write(target / "source-ci.json", source_ci)
            ci.write(target / "source-artifact.json", artifact)
            if hardware is not None:
                shutil.copytree(hardware, target / "hardware")
            result["files"] = [{"path": name, "sha256": digest} for name, digest in files_with_nested_seals(target).items()
                               if name != "result.json"]
            ci.write(target / "result.json", result)
            catalog["results"].append({"source_run_id": source, "manifest": f"source-{source}/result.json",
                                       "status": result.get("status", "complete")})
        except (Exception, KeyboardInterrupt) as error:
            target.mkdir(parents=True, exist_ok=True)
            ci.write(target / "export-error.json", {"error": str(error), "source_run_id": source, "exit_code": 2})
            catalog["results"].append({"source_run_id": source, "status": "failed", "error": str(error)})
            catalog["status"], exit_code = "partial", 2
        finally:
            if target.exists():
                if (target / "SHA256SUMS").exists() and not (target / "source-SHA256SUMS").exists():
                    (target / "source-SHA256SUMS").write_bytes((target / "SHA256SUMS").read_bytes())
                (target / "SHA256SUMS").unlink(missing_ok=True)
                files = files_with_nested_seals(target)
                (target / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(files.items())))
    ci.write(output / "index.json", catalog)
    for name in ("result.schema.json", "RESULTS.md", "RESULTS_zh.md"):
        shutil.copyfile(Path(__file__).parent / name, output / name)
    (output / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in files_with_nested_seals(output).items()))
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-ids", required=True)
    parser.add_argument("--hardware", type=Path)
    parser.add_argument("--hardware-run-id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        return publish(source_ids(args.source_run_ids), args.output, args.hardware, hardware_run_id=args.hardware_run_id)
    except (Exception, KeyboardInterrupt) as error:
        args.output.mkdir(parents=True, exist_ok=True)
        ci.write(args.output / "export-error.json", {"error": str(error), "exit_code": 2})
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
