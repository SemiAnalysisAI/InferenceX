"""Immutable measurement receipts issued by independently selected hosted tooling.

The issuer identity is supplied by the trusted workflow, never inferred from worker files.
The caller must authorize the source run and generate the expected contract from approved
immutable configuration. This module verifies API ownership and archive bytes, not actor policy.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import stat
import subprocess
import zipfile
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from infx.benchmarks.common import decode_json, require_finite
from infx.results.evals import build_rows

Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
GitSha = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{40}$")]
Positive = Annotated[int, Field(strict=True, gt=0)]
RunId = Annotated[str, StringConstraints(pattern=r"^[1-9][0-9]*$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Member(StrictModel):
    path: str
    sha256: Sha256
    size: Annotated[int, Field(strict=True, ge=0)]


class Artifact(StrictModel):
    id: Positive
    name: str
    sha256: Sha256
    run_id: RunId
    members: list[Member]


class Issuer(StrictModel):
    repository: str
    run_id: RunId
    job: str
    workflow_sha: GitSha
    collector_sha: GitSha


class Topology(StrictModel):
    kind: Literal["aggregate"]
    nodes: Literal[1]
    serving_gpus: Positive
    tp: Positive
    ep: Positive


class Point(StrictModel):
    point_id: Sha256
    bundle_digest: Sha256
    execution_id: str
    source_run_id: RunId
    source_attempt: Positive
    kind: Literal["throughput", "eval"]
    concurrency: Positive
    topology: Topology
    artifact_ids: list[Positive]
    execution_artifact_id: Positive
    execution_path: str
    native_manifest_sha256: Sha256
    normalized_artifact_id: Positive
    normalized_path: str
    normalized_format: Literal["normalized", "lm-eval"] = "normalized"
    metadata_path: str | None = None
    required_metrics: list[str]
    config: dict[str, str]
    task: str | None = None
    filters: list[str] = Field(default_factory=list)
    sample_count: Annotated[int, Field(strict=True, ge=0)] = 0
    samples_artifact_id: Positive | None = None
    samples_path: str | None = None
    dataset: dict[str, str | int] = Field(default_factory=dict)


class Contracts(StrictModel):
    raw: Literal["aiperf-1.4"]
    normalized: Literal["agentx-v1"]
    publication: Literal[1]


class ExpectedContract(StrictModel):
    repository: str
    source_run_id: RunId
    source_attempt: Positive
    source_head_sha: GitSha
    bundle_digest: Sha256
    contracts: Contracts
    points: list[Point]

    @model_validator(mode="after")
    def unique_points(self) -> ExpectedContract:
        if not self.points or len({point.point_id for point in self.points}) != len(self.points):
            raise ValueError("Expected points must be nonempty and unique")
        for point in self.points:
            if not point.config or not point.required_metrics:
                raise ValueError("Expected config and metric requirements cannot be empty")
            if not point.execution_id or len(point.artifact_ids) != len(set(point.artifact_ids)):
                raise ValueError("Missing execution identity or duplicate artifact binding")
            if point.execution_artifact_id not in point.artifact_ids:
                raise ValueError("Execution artifact must belong to its point")
            if point.normalized_artifact_id not in point.artifact_ids:
                raise ValueError("Normalized artifact must belong to its point")
            if point.normalized_format == "lm-eval" and (
                point.kind != "eval" or not point.metadata_path
            ):
                raise ValueError("Raw lm-eval format requires metadata path and eval mode")
            if point.kind == "eval" and (
                not point.task
                or not point.filters
                or point.sample_count <= 0
                or point.samples_artifact_id not in point.artifact_ids
                or not point.samples_path
            ):
                raise ValueError("Eval contract must require task, filters, count and raw samples")
        return self


class SourceReceipt(ExpectedContract):
    kind: Literal["source-measurement-receipt"]
    version: Literal[1]
    receipt_id: Sha256
    issuer: Issuer
    artifacts: list[Artifact]


class PublicationRecord(StrictModel):
    kind: Literal["publication-record"]
    version: Literal[1]
    receipt_id: Sha256
    receipt_artifact_id: Positive
    receipt_artifact_sha256: Sha256
    source_run_id: RunId
    merge_run_id: RunId
    merge_sha: GitSha
    changelog_artifact_id: Positive
    changelog_artifact_sha256: Sha256
    ingest_sha: GitSha
    app_sha: GitSha


def canonical_bytes(value: dict[str, Any]) -> bytes:
    """Receipt fields contain strings and integers; no floating-point canonicalization."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def safe_member(name: str) -> str:
    parts = name.removesuffix("/").split("/")
    if (
        not name
        or "\\" in name
        or "\0" in name
        or name.startswith("/")
        or re.match(r"^[A-Za-z]:", name)
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"Unsafe archive member: {name}")
    return str(PurePosixPath(*parts))


def inspect_archive(archive: Path) -> list[Member]:
    members: list[Member] = []
    names: set[str] = set()
    files: set[str] = set()
    total = 0
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            name = safe_member(item.filename)
            kind = stat.S_IFMT(item.external_attr >> 16)
            if name in names or kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise ValueError(f"Duplicate/link/special archive member: {name}")
            names.add(name)
            if item.is_dir():
                continue
            total += item.file_size
            if total > 20 * 1024**3 or item.file_size > 10 * 1024**3:
                raise ValueError("Artifact exceeds extraction budget")
            digest = hashlib.sha256()
            size = 0
            with source.open(item) as payload:
                while chunk := payload.read(1024**2):
                    size += len(chunk)
                    if size > item.file_size:
                        raise ValueError(f"Archive member exceeds declared size: {name}")
                    digest.update(chunk)
            if size != item.file_size:
                raise ValueError(f"Archive member differs from declared size: {name}")
            members.append(Member(path=name, size=size, sha256=digest.hexdigest()))
            files.add(name)
    for name in files:
        if any(str(parent) in files for parent in PurePosixPath(name).parents):
            raise ValueError(f"File/directory collision: {name}")
    return sorted(members, key=lambda member: member.path)


def verify_receipt(data: dict[str, Any]) -> SourceReceipt:
    receipt = SourceReceipt.model_validate(data)
    payload = receipt.model_dump(exclude={"receipt_id"})
    if hashlib.sha256(canonical_bytes(payload)).hexdigest() != receipt.receipt_id:
        raise ValueError("Receipt digest mismatch")
    required_ids = {artifact_id for point in receipt.points for artifact_id in point.artifact_ids}
    actual_ids = [artifact.id for artifact in receipt.artifacts]
    if len(actual_ids) != len(set(actual_ids)) or required_ids != set(actual_ids):
        raise ValueError("Receipt artifact set mismatch")
    return receipt


def validate_point_content(point: Point, archives: Path) -> None:
    """Validate normalized/raw eval semantics independently before receipt issuance."""
    with zipfile.ZipFile(archives / f"{point.normalized_artifact_id}.zip") as archive:
        value = decode_json(archive.read(safe_member(point.normalized_path)).decode())
        require_finite(value)
        if point.normalized_format == "lm-eval":
            meta = decode_json(archive.read(safe_member(point.metadata_path or "")).decode())
            require_finite(meta)
            if point.task == "gsm8k" and point.sample_count == 1319:
                config = value.get("config", {})
                model_args = config.get("model_args", {})
                if (
                    config.get("model") != "local-chat-completions"
                    or config.get("limit") is not None
                    or model_args.get("num_concurrent") != point.concurrency
                    or model_args.get("max_length") != 16384
                    or model_args.get("tokenized_requests") is not False
                    or config.get("gen_kwargs")
                    != {"max_tokens": 12288, "temperature": 0, "top_p": 1}
                    or set(value.get("results", {})) != {"gsm8k"}
                ):
                    raise ValueError(
                        "Pilot eval task, concurrency or context/generation contract differs"
                    )
                if (
                    point.config.get("model") == "dsv41flash"
                    and model_args.get("model") != "deepseek-ai/DeepSeek-V4.1-Flash"
                ):
                    raise ValueError("Pilot eval served model differs from independent expectation")
            rows = build_rows(value, meta, source=point.normalized_path)
        else:
            rows = value if isinstance(value, list) else [value]
    matching = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("conc", row.get("users")) == point.concurrency
        and (point.kind == "throughput" or row.get("task") == point.task)
    ]
    if len(matching) != 1:
        raise ValueError("Expected exactly one normalized point at required concurrency/task")
    row = matching[0]
    aliases = {
        "model": "infmax_model_prefix",
        "hardware": "hw",
        "specMethod": "spec_decoding",
        "recipeFingerprint": "recipe_fingerprint",
    }
    for key, expected in point.config.items():
        value = row.get(aliases.get(key, key))
        if key == "model":
            value = row.get("infmax_model_prefix", row.get("model_prefix"))
        if key == "hardware" and isinstance(value, str):
            value = value.lower().split("-", 1)[0]
        if value != expected:
            raise ValueError(f"Normalized configuration mismatch: {key}")
    topology = row.get("deployment")
    if topology is not None and topology != point.topology.model_dump():
        raise ValueError("Normalized deployment differs from expected topology")
    if row.get("disagg") is not False or row.get("is_multinode") is not False:
        raise ValueError("Pilot output must explicitly identify aggregate single-node topology")
    if (
        row.get("tp") != point.topology.tp
        or row.get("ep") != point.topology.ep
        or row.get("num_gpus") != point.topology.serving_gpus
    ):
        raise ValueError("Normalized GPU/parallelism mismatch")
    metric_paths = {
        "output_tput_tps": ("request_metrics", "throughput", "output", "tokens_per_second"),
        "total_tput_tps": ("request_metrics", "throughput", "total", "tokens_per_second"),
        "duration_seconds": ("request_metrics", "throughput", "duration_seconds"),
    }
    for key in point.required_metrics:
        value = row.get(key)
        if value is None and key in metric_paths:
            value = row
            for field in metric_paths[key]:
                value = value.get(field) if isinstance(value, dict) else None
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"Required finite non-negative metric missing: {key}")
    if point.dataset and row.get("dataset") != point.dataset:
        raise ValueError("Dataset identity differs from independent expectation")
    if point.kind == "eval":
        identities = None
        if point.task == "gsm8k" and point.sample_count == 1319:
            identities = decode_json(
                files("infx.benchmarks")
                .joinpath("resources/gsm8k-test-doc-hashes.json")
                .read_text()
            )
        observed: set[tuple[int, str]] = set()
        strict_passed = 0
        with (
            zipfile.ZipFile(archives / f"{point.samples_artifact_id}.zip") as archive,
            archive.open(safe_member(point.samples_path or "")) as source,
            io.TextIOWrapper(source, encoding="utf-8") as samples,
        ):
            for line in samples:
                if not line.strip():
                    continue
                sample = decode_json(line)
                require_finite(sample)
                doc_id, filter_name = sample.get("doc_id"), sample.get("filter")
                if (
                    type(doc_id) is not int
                    or doc_id < 0
                    or filter_name not in point.filters
                    or sample.get("task_name", point.task) != point.task
                    or (doc_id, filter_name) in observed
                ):
                    raise ValueError("Invalid/duplicate evaluation sample identity")
                if identities is not None:
                    document_hash = hashlib.sha256(
                        json.dumps(sample.get("doc"), indent=2, ensure_ascii=False).encode()
                    ).hexdigest()
                    target = sample.get("target")
                    if (
                        identities.get(str(doc_id)) != document_hash
                        or sample.get("doc_hash") != document_hash
                        or target != sample.get("doc", {}).get("answer")
                        or sample.get("target_hash")
                        != hashlib.sha256(str(target).encode()).hexdigest()
                    ):
                        raise ValueError(
                            "Pilot eval document/target differs from prepared full split"
                        )
                observed.add((doc_id, filter_name))
                if filter_name == "strict-match":
                    score = sample.get("exact_match,strict-match", sample.get("exact_match"))
                    if type(score) not in (int, float) or score not in (0, 1):
                        raise ValueError("GSM8K strict sample requires a binary score")
                    strict_passed += int(score)
        documents = {doc for doc, _ in observed}
        if (
            documents != set(range(point.sample_count))
            or len(observed) != point.sample_count * len(point.filters)
            or row.get("n_eff") != point.sample_count
        ):
            raise ValueError("Incomplete evaluation sample/filter coverage")
        if not math.isclose(
            row["em_strict"], strict_passed / point.sample_count, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("Evaluation strict summary differs from raw samples")


def seal_receipt(
    expected: ExpectedContract, issuer: Issuer, inventory: list[dict[str, Any]], archives: Path
) -> SourceReceipt:
    from infx.srt_slurm.qualification import qualification_artifacts

    if qualification_artifacts(row["name"] for row in inventory):
        raise ValueError("nonpublishing qualification artifacts cannot be sealed")
    required = {artifact_id for point in expected.points for artifact_id in point.artifact_ids}
    rows = {int(row["id"]): row for row in inventory}
    if len(rows) != len(inventory) or set(rows) != required:
        raise ValueError("Requested artifact set differs from independently fetched API inventory")
    artifacts: list[Artifact] = []
    for artifact_id in sorted(required):
        row = rows[artifact_id]
        run_ids = {
            point.source_run_id for point in expected.points if artifact_id in point.artifact_ids
        }
        if (
            len(run_ids) != 1
            or str(row.get("workflow_run", {}).get("id")) not in run_ids
            or row.get("expired")
        ):
            raise ValueError(f"Wrong-run/expired artifact {artifact_id}")
        archive = archives / f"{artifact_id}.zip"
        with archive.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if row.get("digest") != f"sha256:{digest}":
            raise ValueError(f"API/archive digest mismatch: {artifact_id}")
        artifacts.append(
            Artifact(
                id=artifact_id,
                name=row["name"],
                sha256=digest,
                run_id=next(iter(run_ids)),
                members=inspect_archive(archive),
            )
        )
    for point in expected.points:
        validate_point_content(point, archives)
        with zipfile.ZipFile(archives / f"{point.execution_artifact_id}.zip") as archive:
            execution = decode_json(archive.read(safe_member(point.execution_path)).decode())
        if not isinstance(execution, dict):
            raise ValueError("Execution evidence must be an object")
        native = execution.get("native_receipt", {})
        source = execution.get("source", {})
        if isinstance(source, dict) and source.get("purpose", "publication") != "publication":
            raise ValueError("nonpublishing execution cannot be sealed")
        if (
            not isinstance(native, dict)
            or not isinstance(source, dict)
            or type(execution.get("schema_version")) is not int
            or execution.get("schema_version") != 1
            or execution.get("point_id") != point.point_id
            or execution.get("execution_id") != point.execution_id
            or execution.get("bundle_digest") != point.bundle_digest
            or execution.get("mode") != point.kind
            or type(execution.get("client_exit_code")) is not int
            or execution.get("client_exit_code") != 0
            or source.get("repository") != expected.repository
            or str(source.get("run_id")) != point.source_run_id
            or type(source.get("attempt")) is not int
            or source.get("attempt") != point.source_attempt
            or source.get("head_sha") != expected.source_head_sha
            or native.get("state") != "COMPLETED"
            or not re.fullmatch(r"[1-9][0-9]*", str(native.get("job_id", "")))
            or native.get("manifest_sha256") != point.native_manifest_sha256
        ):
            raise ValueError(
                f"Execution evidence differs from independently expected point: {point.point_id}"
            )
    payload = expected.model_dump() | {
        "kind": "source-measurement-receipt",
        "version": 1,
        "issuer": issuer.model_dump(),
        "artifacts": [artifact.model_dump() for artifact in artifacts],
    }
    return verify_receipt(
        payload | {"receipt_id": hashlib.sha256(canonical_bytes(payload)).hexdigest()}
    )


def fetch_and_seal(expected: ExpectedContract, issuer: Issuer, archives: Path) -> SourceReceipt:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", expected.repository):
        raise ValueError("Invalid repository")
    archives.mkdir(parents=True, exist_ok=True)
    inventory = []
    for artifact_id in sorted({value for point in expected.points for value in point.artifact_ids}):
        endpoint = f"repos/{expected.repository}/actions/artifacts/{artifact_id}"
        row = json.loads(subprocess.check_output(["gh", "api", endpoint]))
        inventory.append(row)
        with (archives / f"{artifact_id}.zip").open("xb") as output:
            subprocess.run(["gh", "api", f"{endpoint}/zip"], stdout=output, check=True)
    return seal_receipt(expected, issuer, inventory, archives)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--issuer", type=Path, required=True)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = ExpectedContract.model_validate_json(args.expected.read_text())
    issuer = Issuer.model_validate_json(args.issuer.read_text())
    receipt = fetch_and_seal(expected, issuer, args.archives)
    with args.output.open("x") as output:
        output.write(json.dumps(receipt.model_dump(), indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
