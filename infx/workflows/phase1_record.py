"""Bind an immutable source receipt to a later publication without changing the source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from infx.benchmarks.common import decode_json, read_json
from infx.results.publication_receipt import PublicationRecord, inspect_archive, verify_receipt
from infx.workflows.phase1_publication import api


def verified_artifact(
    repository: str, artifact_id: int, expected_digest: str, *, name: str
) -> tuple[dict[str, Any], bytes]:
    endpoint = f"repos/{repository}/actions/artifacts/{artifact_id}"
    metadata = api(endpoint)
    if (
        not isinstance(metadata, dict)
        or metadata.get("id") != artifact_id
        or metadata.get("expired")
        or metadata.get("name") != name
    ):
        raise ValueError("required immutable artifact is missing, expired or misnamed")
    if metadata.get("digest") != "sha256:" + expected_digest:
        raise ValueError("artifact API digest disagrees with approved publication")
    payload = subprocess.run(
        ["gh", "api", endpoint + "/zip"], capture_output=True, check=True
    ).stdout
    if hashlib.sha256(payload).hexdigest() != expected_digest:
        raise ValueError("downloaded artifact differs from approved digest")
    return metadata, payload


def validate_record(
    record: PublicationRecord, repository: str, issuer_shas: set[str], reader_sha: str
) -> PublicationRecord:
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)
        or not issuer_shas
        or any(not re.fullmatch(r"[a-f0-9]{40}", value) for value in issuer_shas)
        or not re.fullmatch(r"[a-f0-9]{40}", reader_sha)
    ):
        raise ValueError("publication requires valid deployed repository/issuer/reader policy")
    metadata, payload = verified_artifact(
        repository,
        record.receipt_artifact_id,
        record.receipt_artifact_sha256,
        name="measurement-receipt",
    )
    issuer_id = metadata.get("workflow_run", {}).get("id")
    if type(issuer_id) is not int or issuer_id <= 0:
        raise ValueError("source receipt has no valid API issuer identity")
    issuer = api(f"repos/{repository}/actions/runs/{issuer_id}")
    if (
        not isinstance(issuer, dict)
        or issuer.get("id") != issuer_id
        or issuer.get("head_sha") not in issuer_shas
        or issuer.get("status") != "completed"
        or issuer.get("conclusion") != "success"
        or issuer.get("head_branch") != "main"
        or issuer.get("event") != "workflow_dispatch"
        or issuer.get("path") != ".github/workflows/phase1-receipt.yml"
    ):
        raise ValueError("source receipt issuer is not an approved completed trusted workflow")
    with tempfile.TemporaryDirectory(prefix="infx-source-receipt-") as temporary:
        archive = Path(temporary) / "receipt.zip"
        archive.write_bytes(payload)
        if [member.path for member in inspect_archive(archive)] != ["receipt.json"]:
            raise ValueError("source receipt artifact must contain only receipt.json")
        with zipfile.ZipFile(archive) as stream:
            receipt = verify_receipt(decode_json(stream.read("receipt.json").decode()))
    if (
        receipt.repository != repository
        or receipt.issuer.repository != repository
        or receipt.receipt_id != record.receipt_id
        or receipt.source_run_id != record.source_run_id
        or receipt.issuer.run_id != str(issuer["id"])
        or receipt.issuer.workflow_sha != issuer["head_sha"]
    ):
        raise ValueError("publication attempts to replace source measurement identity")
    source = api(
        f"repos/{repository}/actions/runs/{record.source_run_id}/attempts/{receipt.source_attempt}"
    )
    if (
        not isinstance(source, dict)
        or str(source.get("id")) != record.source_run_id
        or source.get("run_attempt") != receipt.source_attempt
        or source.get("head_sha") != receipt.source_head_sha
        or source.get("status") != "completed"
        or source.get("conclusion") != "success"
    ):
        raise ValueError("source receipt does not match its completed original attempt")
    merge = api(f"repos/{repository}/actions/runs/{record.merge_run_id}")
    if (
        not isinstance(merge, dict)
        or str(merge.get("id")) != record.merge_run_id
        or merge.get("head_sha") != record.merge_sha
        or merge.get("head_branch") != "main"
        or merge.get("event") != "push"
        or merge.get("status") != "completed"
        or merge.get("conclusion") != "success"
        or merge.get("path") != ".github/workflows/run-sweep.yml"
    ):
        raise ValueError("publication merge must be the completed successful main sweep")
    changelog, _ = verified_artifact(
        repository,
        record.changelog_artifact_id,
        record.changelog_artifact_sha256,
        name="changelog-metadata",
    )
    if str(changelog["workflow_run"]["id"]) != record.merge_run_id:
        raise ValueError("publication changelog belongs to a different merge run")
    if record.app_sha != reader_sha or record.ingest_sha != reader_sha:
        raise ValueError("publication reader revision does not match the deployed reader pin")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = validate_record(
        PublicationRecord.model_validate(read_json(args.approval)),
        os.environ["GITHUB_REPOSITORY"],
        {value.strip() for value in os.environ["TRUSTED_ISSUER_SHAS"].split(",") if value.strip()},
        os.environ["DEPLOYED_READER_SHA"],
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(record.model_dump(), indent=2) + "\n")


if __name__ == "__main__":
    main()
