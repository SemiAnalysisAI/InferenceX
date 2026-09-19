"""Resolve immutable receipts using read-only APIs and deployed issuer policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from infx import github
from infx.benchmarks.common import decode_json
from infx.results.publication_receipt import (
    PublicationRecord,
    SourceReceipt,
    inspect_archive,
    verify_receipt,
)


class PendingReceiptError(ValueError):
    """The native source has not yet passed its separate trusted publication stage."""


@dataclass(frozen=True)
class SealedArtifact:
    artifact_id: int
    archive_sha256: str
    content_sha256: str
    issuer_run_id: str
    issuer_sha: str
    document: dict[str, Any]


def read_sealed(
    repo: str, artifact: dict[str, Any], issuer: dict[str, Any], member: str
) -> SealedArtifact:
    artifact_id = artifact.get("id")
    if type(artifact_id) is not int or artifact_id <= 0 or artifact.get("expired"):
        raise ValueError("Invalid/expired receipt artifact")
    metadata = github.api(repo, f"/actions/artifacts/{artifact_id}")
    if (
        metadata.get("id") != artifact_id
        or metadata.get("workflow_run", {}).get("id") != issuer["id"]
        or metadata.get("expired")
    ):
        raise ValueError("Receipt API artifact ownership mismatch")
    with tempfile.TemporaryDirectory(prefix="infx-sealed-artifact-") as temporary:
        archive = Path(temporary) / f"{artifact_id}.zip"
        with archive.open("xb") as stream:
            subprocess.run(
                ["gh", "api", f"repos/{repo}/actions/artifacts/{artifact_id}/zip"],
                stdout=stream,
                check=True,
            )
        if archive.stat().st_size > 10 * 1024**2:
            raise ValueError("Compact receipt archive exceeds size budget")
        with archive.open("rb") as stream:
            archive_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if metadata.get("digest") != f"sha256:{archive_sha}":
            raise ValueError("Receipt archive differs from API digest")
        members = inspect_archive(archive)
        if len(members) != 1 or members[0].path != member or members[0].size > 10 * 1024**2:
            raise ValueError("Receipt archive must contain exactly its versioned JSON document")
        with zipfile.ZipFile(archive) as source:
            content = source.read(member)
    document = decode_json(content.decode())
    if not isinstance(document, dict):
        raise ValueError("Sealed receipt must be an object")
    return SealedArtifact(
        artifact_id,
        archive_sha,
        hashlib.sha256(content).hexdigest(),
        str(issuer["id"]),
        issuer["head_sha"],
        document,
    )


def resolve_transport(
    repo: str,
    source_run_id: str,
    merge_run_id: str,
    allowed_shas: set[str],
    *,
    workflow: str = ".github/workflows/phase1-receipt.yml",
    publication_required: bool = False,
) -> dict[str, str]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or any(
        not re.fullmatch(r"[1-9][0-9]*", value) for value in (source_run_id, merge_run_id)
    ):
        raise ValueError("Invalid repository/run identity")
    inventory = github.paginate(
        repo, f"/actions/runs/{source_run_id}/artifacts", item_key="artifacts"
    )
    if not any(str(item.get("name", "")).startswith("native-execution-") for item in inventory):
        return {"receipt-required": "false"}
    if not allowed_shas or any(not re.fullmatch(r"[a-f0-9]{40}", value) for value in allowed_shas):
        raise ValueError("Native receipt requires deployed issuer revision allowlist")
    if not re.fullmatch(r"\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml", workflow):
        raise ValueError("Invalid deployed issuer workflow")
    source_receipts: list[tuple[SealedArtifact, SourceReceipt]] = []
    publications: list[SealedArtifact] = []
    runs = github.paginate(
        repo,
        f"/actions/workflows/{workflow.rsplit('/', 1)[-1]}/runs",
        item_key="workflow_runs",
        params={"status": "completed"},
    )
    for run in runs:
        if (
            run.get("head_sha") not in allowed_shas
            or run.get("status") != "completed"
            or run.get("conclusion") != "success"
            or run.get("path") != workflow
            or run.get("head_branch") != "main"
            or run.get("event") != "workflow_dispatch"
        ):
            continue
        artifacts = github.paginate(
            repo, f"/actions/runs/{run['id']}/artifacts", item_key="artifacts"
        )
        for artifact in artifacts:
            if artifact.get("expired"):
                continue
            if artifact.get("name") == "measurement-receipt":
                sealed = read_sealed(repo, artifact, run, "receipt.json")
                if (
                    sealed.document.get("repository") != repo
                    or sealed.document.get("source_run_id") != source_run_id
                ):
                    continue
                receipt = verify_receipt(sealed.document)
                if (
                    receipt.issuer.repository != repo
                    or receipt.issuer.run_id != sealed.issuer_run_id
                    or receipt.issuer.workflow_sha != sealed.issuer_sha
                ):
                    raise ValueError("Receipt claims an issuer different from its actual API owner")
                attempt = github.api(
                    repo, f"/actions/runs/{source_run_id}/attempts/{receipt.source_attempt}"
                )
                if (
                    str(attempt.get("id")) != source_run_id
                    or attempt.get("run_attempt") != receipt.source_attempt
                    or attempt.get("head_sha") != receipt.source_head_sha
                    or attempt.get("status") != "completed"
                    or attempt.get("conclusion") != "success"
                ):
                    raise ValueError(
                        "Receipt source attempt/head is not a completed successful execution"
                    )
                source_receipts.append((sealed, receipt))
            elif artifact.get("name") == "publication-record":
                publications.append(read_sealed(repo, artifact, run, "publication.json"))
    if not source_receipts:
        raise PendingReceiptError(
            "Native source awaits an independently sealed measurement receipt; no ingest dispatched"
        )
    if len(source_receipts) != 1:
        raise ValueError(
            "Multiple source receipts require explicit accepted-snapshot resolution; refusing newest-by-name selection"
        )
    sealed, receipt = source_receipts[0]
    for artifact in receipt.artifacts:
        metadata = github.api(repo, f"/actions/artifacts/{artifact.id}")
        if (
            metadata.get("id") != artifact.id
            or metadata.get("expired")
            or str(metadata.get("workflow_run", {}).get("id")) != artifact.run_id
            or metadata.get("digest") != f"sha256:{artifact.sha256}"
        ):
            raise ValueError("Accepted artifact set is missing, expired, wrong-run or changed")
    payload = {
        "receipt-required": "true",
        "receipt-artifact-id": str(sealed.artifact_id),
        "receipt-artifact-sha256": sealed.archive_sha256,
        "receipt-sha256": sealed.content_sha256,
        "receipt-issuer-run-id": sealed.issuer_run_id,
        "receipt-issuer-sha": sealed.issuer_sha,
    }
    if publication_required or source_run_id != merge_run_id:
        matches: list[tuple[SealedArtifact, PublicationRecord]] = []
        for publication in publications:
            if (
                publication.document.get("source_run_id") != source_run_id
                or publication.document.get("merge_run_id") != merge_run_id
            ):
                continue
            record = PublicationRecord.model_validate(publication.document)
            if (
                record.source_run_id == source_run_id
                and record.merge_run_id == merge_run_id
                and record.receipt_id == receipt.receipt_id
            ):
                matches.append((publication, record))
        if not matches:
            raise PendingReceiptError(
                "Accepted source awaits its later publication record; no ingest dispatched"
            )
        if len(matches) != 1:
            raise ValueError(
                "Multiple publication records require explicit accepted-snapshot resolution"
            )
        publication, record = matches[0]
        merge = github.api(repo, f"/actions/runs/{merge_run_id}")
        changelog = github.api(repo, f"/actions/artifacts/{record.changelog_artifact_id}")
        if (
            record.receipt_artifact_id != sealed.artifact_id
            or record.receipt_artifact_sha256 != sealed.archive_sha256
            or merge.get("head_sha") != record.merge_sha
            or merge.get("head_branch") != "main"
            or merge.get("event") != "push"
            or merge.get("status") != "completed"
            or merge.get("conclusion") != "success"
            or changelog.get("expired")
            or str(changelog.get("workflow_run", {}).get("id")) != merge_run_id
            or changelog.get("name") != "changelog-metadata"
            or changelog.get("digest") != f"sha256:{record.changelog_artifact_sha256}"
        ):
            raise ValueError("Publication record differs from accepted source/merge/changelog")
        payload.update(
            {
                "publication-artifact-id": str(publication.artifact_id),
                "publication-artifact-sha256": publication.archive_sha256,
                "publication-sha256": publication.content_sha256,
                "publication-issuer-run-id": publication.issuer_run_id,
                "publication-issuer-sha": publication.issuer_sha,
            }
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"), required=False)
    parser.add_argument("--source-run-id", default=os.environ.get("SOURCE_RUN_ID"), required=False)
    parser.add_argument("--merge-run-id", default=os.environ.get("MERGE_RUN_ID"), required=False)
    parser.add_argument("--publication-required", action="store_true")
    parser.add_argument("--defer-unsealed", action="store_true")
    args = parser.parse_args()
    if not all((args.repository, args.source_run_id, args.merge_run_id)):
        parser.error("repository, source run ID and merge run ID are required")
    allowed = {
        value.strip()
        for value in os.environ.get("INFX_RECEIPT_ISSUER_SHAS", "").split(",")
        if value.strip()
    }
    try:
        payload = resolve_transport(
            args.repository,
            args.source_run_id,
            args.merge_run_id,
            allowed,
            workflow=os.environ.get("INFX_RECEIPT_ISSUER_WORKFLOW", ""),
            publication_required=args.publication_required,
        )
        outputs = {"ready": "true", "payload": json.dumps(payload, separators=(",", ":"))}
    except PendingReceiptError as exc:
        if not args.defer_unsealed:
            raise
        outputs = {"ready": "false", "payload": "{}"}
        print(str(exc))
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a") as stream:
            stream.write("".join(f"{key}={value}\n" for key, value in outputs.items()))
    print(json.dumps(outputs))


if __name__ == "__main__":
    main()
