"""Nonpublishing PR qualification: full workload evidence, never a measurement receipt."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PURPOSE = "pr-qualification"
PREFIX = "native-qualification-"
CONCURRENCIES = (1, 2, 4, 8, 16, 20, 24, 28)


def require_pr_event(environment: Mapping[str, str]) -> dict[str, Any]:
    """A reusable-workflow input cannot turn a dispatch/push/fork into qualification."""
    if environment.get("GITHUB_EVENT_NAME") != "pull_request":
        raise ValueError("nonpublishing qualification requires a pull_request event")
    event = json.loads(Path(environment["GITHUB_EVENT_PATH"]).read_text())
    pr = event.get("pull_request", {})
    repository = environment.get("GITHUB_REPOSITORY")
    number = event.get("number")
    if (
        not repository
        or type(number) is not int
        or number <= 0
        or environment.get("GITHUB_REF") != f"refs/pull/{number}/merge"
        or pr.get("head", {}).get("repo", {}).get("full_name") != repository
        or pr.get("base", {}).get("repo", {}).get("full_name") != repository
    ):
        raise ValueError("nonpublishing qualification requires an actual same-repository PR")
    return event


def qualification_artifacts(names: Any) -> bool:
    return any(name.startswith(PREFIX) for name in names)


def matrix_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for group in ("single_node", "multi_node"):
        for values in (plan.get(group) or {}).values():
            rows.extend(
                dict(row, **{"run-eval": False, "eval-only": False}) for row in (values or [])
            )
    for group in ("evals", "agentic_evals", "multinode_evals", "multinode_agentic_evals"):
        rows.extend(
            dict(row, **{"run-eval": True, "eval-only": True, "eval-framework": "lm-eval"})
            for row in (plan.get(group) or [])
        )
    return rows


def qualification_jobs(plan: dict[str, Any], root: Path) -> dict[str, Any]:
    from infx.srt_slurm.job import parse_job

    rows = matrix_rows(plan)
    if not rows or any(row.get("execution", {}).get("runtime") != "srt-slurm" for row in rows):
        raise ValueError("qualification requires an isolated native H100 matrix")
    jobs = [parse_job(row, root, {"node-count": 1}) for row in rows]
    expected = {("throughput", c) for c in CONCURRENCIES} | {("eval", 28)}
    if len(jobs) != 9 or {(job.mode, job.row.conc) for job in jobs} != expected:
        raise ValueError("qualification requires all eight throughput points and real c28 eval")
    return {job.point_id: job for job in jobs}


def select_qualification(plan: dict[str, Any], root: Path, environment: Mapping[str, str]) -> bool:
    rows = matrix_rows(plan)
    native = any(row.get("execution", {}).get("runtime") == "srt-slurm" for row in rows)
    if not native or environment.get("GITHUB_EVENT_NAME") != "pull_request":
        return False
    require_pr_event(environment)
    qualification_jobs(plan, root)
    return True


def validate_point(
    directory: Path, jobs: dict[str, Any], source: dict[str, Any], root: Path
) -> dict:
    """Recheck downloaded evidence against the caller matrix and committed client policy."""
    from infx.benchmarks import agentx, eval as real_eval
    from infx.benchmarks.common import child_failed, read_json, require_finite
    from infx.benchmarks.spec import AgentXSpec, EvalSpec, PreparedFile, RuntimeSpec
    from infx.srt_slurm.contracts import digest, load_mapping
    from infx.srt_slurm.job import JobSpec, file_digest
    from infx.srt_slurm.launch import RuntimeLock, effective_identity, verify_wrapper_source
    from infx.srt_slurm.render import ClientPolicy, PreparedSite, client_spec

    diagnostics = directory / "native-execution"
    bundle = read_json(diagnostics / "bundle.json")
    if bundle.get("bundle_digest") != digest(
        {k: v for k, v in bundle.items() if k != "bundle_digest"}
    ):
        raise ValueError("qualification bundle digest differs")
    if bundle.get("source") != source:
        raise ValueError("qualification source identity or purpose differs")
    requested = bundle.get("requested_point_id")
    if requested not in jobs:
        raise ValueError("qualification point is not in the expected matrix")
    job = jobs[requested]
    if JobSpec.model_validate(bundle["job"]).semantic_inputs() != job.semantic_inputs():
        raise ValueError("qualification matrix semantics differ")
    lock = RuntimeLock.model_validate(read_json(root / job.row.execution.runtime_lock))
    if bundle["identity"]["runtime_lock"] != lock.model_dump():
        raise ValueError("qualification native runtime differs from the committed pin")
    verify_wrapper_source(bundle["identity"]["wrapper_identity"], root)
    point_id = bundle["point_id"]
    if not re.fullmatch(r"[a-f0-9]{64}", point_id) or directory.name != PREFIX + point_id:
        raise ValueError("qualification artifact identity differs")
    original = Path(bundle["directory"])

    def retained(name: str) -> Path:
        relative = Path(name).relative_to(original)
        if ".." in relative.parts:
            raise ValueError("qualification prepared path escapes bundle")
        path = diagnostics / "prepared" / relative
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("qualification prepared path escapes artifact")
        if name not in bundle["files"] or file_digest(path) != bundle["files"][name]:
            raise ValueError("qualification prepared file digest differs")
        return path

    for name in bundle["files"]:
        retained(name)
    spec_value = read_json(retained(str(original / "client.json")))
    runtime = RuntimeSpec.model_validate(spec_value["runtime"])
    resources = read_json(retained(str(original / "client/prepared-resources.json")))
    policy = ClientPolicy.model_validate(load_mapping(root / job.row.execution.client_policy))
    site = PreparedSite.model_validate(bundle["site"])
    if site.image_reference != job.row.image:
        raise ValueError("qualification serving image differs")
    computed, curve = effective_identity(
        job,
        site,
        bundle["identity"],
        runtime,
        resources,
        client_identity=read_json(retained(runtime.identity.path)),
    )
    if (point_id, bundle["effective_curve_id"]) != (computed, curve):
        raise ValueError("qualification effective identity differs")
    spec = client_spec(job, policy, runtime, resources, point_id)
    if spec.model_dump(mode="json") != spec_value:
        raise ValueError("qualification client policy differs")
    prepared = bundle["prepared"]
    if not set(lock.capabilities) <= set(prepared.get("capabilities", [])):
        raise ValueError("qualification native capabilities differ")
    if (
        prepared.get("resources")
        != {
            "nodes": 1,
            "gpus_per_node": 8,
            "serving_gpus": 8,
            "workers": 1,
            "cardinality": 1,
        }
        or file_digest(retained(str(original / "native/manifest.json")))
        != prepared["manifest_sha256"]
    ):
        raise ValueError("qualification native allocation or manifest differs")
    execution = read_json(diagnostics / "execution.json")
    if (
        execution.get("source") != source
        or any(
            execution.get(key) != bundle[key]
            for key in ("point_id", "execution_id", "bundle_digest")
        )
        or execution.get("mode") != job.mode
        or execution.get("client_exit_code") != 0
        or execution.get("native_receipt", {}).get("state") != "COMPLETED"
        or execution["native_receipt"].get("manifest_sha256") != prepared["manifest_sha256"]
        or not re.fullmatch(r"[1-9][0-9]*", str(execution["native_receipt"].get("job_id", "")))
        or read_json(diagnostics / "output-state.json").get("complete") is not True
    ):
        raise ValueError("qualification lacks successful native/client closure")
    audit = read_json(directory / "results/diagnostics/client-audit.json")
    if audit.get("errors") != [] or child_failed(audit["status"]):
        raise ValueError("qualification client audit failed")
    endpoint = audit["endpoint"]
    if job.mode == "eval":
        if not isinstance(spec, EvalSpec):
            raise ValueError("qualification requires the real eval client")
        # Revalidate the full raw corpus with locally retained, digest-checked inputs.
        relocated = spec.model_copy(
            update={
                key: PreparedFile(
                    path=str(retained(getattr(spec, key).path)), sha256=getattr(spec, key).sha256
                )
                for key in ("task", "document_identities")
            }
        )
        errors = real_eval.validate_outputs(
            relocated,
            endpoint,
            sorted(directory.glob("results*.json")),
            sorted(directory.glob("samples*.jsonl")),
        )
        if errors:
            raise ValueError("qualification real eval failed: " + "; ".join(errors))
    else:
        if not isinstance(spec, AgentXSpec):
            raise ValueError("qualification requires the AgentX client")
        artifact_root = directory / "results"
        artifact_dir = agentx.resolve_artifact_dir(artifact_root)
        aggregate = read_json(artifact_dir / "profile_export_aiperf.json")
        require_finite(aggregate)
        errors = agentx.validate_scenario(aggregate, spec, endpoint)
        errors.extend(agentx.validate_result(artifact_dir, spec.failed_request_threshold))
        metrics = read_json(artifact_dir / "server_metrics_export.json")
        metrics_csv = artifact_dir / "server_metrics_export.csv"
        if (
            not metrics_csv.is_file()
            or not metrics_csv.stat().st_size
            or not agentx.has_metric(metrics, "vllm:")
        ):
            errors.append("vLLM server metrics are missing")
        if errors:
            raise ValueError("qualification AgentX raw validation failed: " + "; ".join(errors))
    validate_normalized(directory, job, spec, bundle)
    return {
        "requested_point_id": requested,
        "point_id": point_id,
        "mode": job.mode,
        "concurrency": job.row.conc,
        "execution_id": bundle["execution_id"],
        "bundle_digest": bundle["bundle_digest"],
        "slurm_job_id": execution["native_receipt"]["job_id"],
    }


def validate_normalized(directory: Path, job: Any, spec: Any, bundle: dict) -> None:
    """Use the existing independent content validator without creating/sealing a receipt."""
    from infx.results.publication_receipt import Point, Topology, validate_point_content

    evaluation = job.mode == "eval"
    results = (
        sorted(directory.glob("results*.json"))
        if evaluation
        else [directory / f"{bundle['point_id']}.json"]
    )
    samples = sorted(directory.glob("samples*.jsonl")) if evaluation else []
    if len(results) != 1 or (evaluation and len(samples) != 1):
        raise ValueError("qualification normalized output is missing or ambiguous")
    point = Point(
        point_id=bundle["point_id"],
        execution_id=bundle["execution_id"],
        bundle_digest=bundle["bundle_digest"],
        native_manifest_sha256=bundle["prepared"]["manifest_sha256"],
        kind=job.mode,
        concurrency=job.row.conc,
        source_run_id=str(bundle["source"]["run_id"]),
        source_attempt=bundle["source"]["attempt"],
        topology=Topology(kind="aggregate", nodes=1, serving_gpus=8, tp=8, ep=1),
        artifact_ids=[1],
        execution_artifact_id=1,
        execution_path="unused",
        normalized_artifact_id=1,
        normalized_path=results[0].name,
        normalized_format="lm-eval" if evaluation else "normalized",
        metadata_path="meta_env.json" if evaluation else None,
        required_metrics=["em_strict", "em_flexible", "n_eff"]
        if evaluation
        else ["output_tput_tps", "total_tput_tps", "duration_seconds"],
        config={
            "model": "dsv41flash",
            "hardware": "h100",
            "framework": "vllm",
            "precision": "fp4",
            "specMethod": "mtp",
        },
        task="gsm8k" if evaluation else None,
        filters=["strict-match", "flexible-extract"] if evaluation else [],
        sample_count=1319 if evaluation else 0,
        samples_artifact_id=1 if evaluation else None,
        samples_path=samples[0].name if evaluation else None,
        dataset={}
        if evaluation
        else {
            "source_type": "public_dataset",
            "loader": spec.dataset_loader,
            "hf_dataset_name": spec.dataset_repository,
            "hf_split": "train",
            "num_dataset_entries": 393,
            "hf_revision": spec.dataset_revision,
        },
    )
    with tempfile.TemporaryDirectory(prefix="infx-qualification-content-") as temporary:
        files = [*results, *samples] + ([directory / "meta_env.json"] if evaluation else [])
        with zipfile.ZipFile(Path(temporary) / "1.zip", "w") as archive:
            for path in files:
                archive.write(path, path.name)
        validate_point_content(point, Path(temporary))


def summarize(artifacts: Path, plan: dict, root: Path, environment: Mapping[str, str]) -> dict:
    event = require_pr_event(environment)
    jobs = qualification_jobs(plan, root)
    source = {
        "repository": environment["GITHUB_REPOSITORY"],
        "run_id": int(environment["GITHUB_RUN_ID"]),
        "attempt": int(environment["GITHUB_RUN_ATTEMPT"]),
        "head_sha": environment["GITHUB_SHA"],
        "purpose": PURPOSE,
        "event": "pull_request",
        "pull_request": event["number"],
    }
    intent = json.loads((artifacts / (PREFIX + "run") / "qualification-intent.json").read_text())
    if intent != {
        "purpose": PURPOSE,
        "publication_eligible": False,
        "run_id": environment["GITHUB_RUN_ID"],
        "attempt": environment["GITHUB_RUN_ATTEMPT"],
    }:
        raise ValueError("qualification run intent differs")
    directories = sorted(path for path in artifacts.iterdir() if path.name != PREFIX + "run")
    if len(directories) != 9 or any(not path.is_dir() or path.is_symlink() for path in directories):
        raise ValueError("qualification requires exactly nine distinct point artifacts")
    points = [validate_point(path, jobs, source, root) for path in directories]
    if {point["requested_point_id"] for point in points} != set(jobs):
        raise ValueError("qualification has duplicate or missing matrix points")
    if len({point["execution_id"] for point in points}) != 9:
        raise ValueError("qualification execution identities must be distinct")
    return {
        "schema_version": 1,
        "purpose": PURPOSE,
        "publication_eligible": False,
        "source": source,
        "complete": True,
        "points": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan", "summary"))
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    root = Path(os.environ["GITHUB_WORKSPACE"])
    plan = json.loads(os.environ["SWEEP_MATRIX"])
    if args.mode == "plan":
        selected = select_qualification(plan, root, os.environ)
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
            stream.write(f"native-qualification={str(selected).lower()}\n")
        if selected:
            (root / "qualification-intent.json").write_text(
                json.dumps(
                    {
                        "purpose": PURPOSE,
                        "publication_eligible": False,
                        "run_id": os.environ["GITHUB_RUN_ID"],
                        "attempt": os.environ["GITHUB_RUN_ATTEMPT"],
                    }
                )
                + "\n"
            )
    else:
        result = summarize(args.artifacts, plan, root, os.environ)
        (root / "qualification-summary.json").write_text(json.dumps(result, indent=2) + "\n")
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
            stream.write(
                "## Nonpublishing H100 qualification\n\nEight full-duration throughput points and the complete real c28 GSM8K eval passed. "
                "Diagnostic evidence only; ineligible for receipt issuance, reuse or app import.\n"
            )


if __name__ == "__main__":
    main()
