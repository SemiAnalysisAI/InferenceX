"""Seal the H100 qualification against a separately reviewed prepared expectation.

Approval JSON belongs to trusted default-branch control code. Never populate its
identities from worker execution.json, infer approval from artifact names, or run
candidate code in this collector.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from infx.benchmarks.common import decode_json, read_json
from infx.results.publication_receipt import (
    Contracts,
    ExpectedContract,
    Issuer,
    Point,
    Topology,
    fetch_and_seal,
    inspect_archive,
)
from infx.srt_slurm.contracts import digest
from infx.srt_slurm.qualification import qualification_artifacts

Sha = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
GitSha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


class ApprovedPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    point_id: Sha
    execution_id: Sha
    bundle_digest: Sha
    native_manifest_sha256: Sha
    kind: Literal["throughput", "eval"]
    concurrency: int = Field(gt=0)
    dataset_revision: GitSha | None


class Approval(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    repository: Literal["SemiAnalysisAI/InferenceX"]
    source_run_id: str = Field(pattern=r"^[1-9][0-9]*$")
    source_attempt: int = Field(gt=0)
    source_head_sha: GitSha
    points: list[ApprovedPoint]

    @model_validator(mode="after")
    def complete_pilot(self) -> Approval:
        expected = {("throughput", c) for c in (1, 2, 4, 8, 16, 20, 24, 28)} | {("eval", 28)}
        if len(self.points) != 9 or {(p.kind, p.concurrency) for p in self.points} != expected:
            raise ValueError(
                "Phase 1 requires exactly eight throughput points and the real c28 eval"
            )
        if len({p.point_id for p in self.points}) != len(self.points):
            raise ValueError("approved points must have distinct semantic identities")
        if any(p.kind == "throughput" and p.dataset_revision is None for p in self.points):
            raise ValueError("throughput approval requires its actual immutable corpus revision")
        return self


def api(path: str, *, paginate: bool = False) -> object:
    argv = ["gh", "api", path]
    if paginate:
        argv.extend(("--paginate", "--slurp"))
    return decode_json(subprocess.run(argv, capture_output=True, text=True, check=True).stdout)


def expected_contract(approval: Approval) -> ExpectedContract:
    run = api(f"repos/{approval.repository}/actions/runs/{approval.source_run_id}")
    if (
        not isinstance(run, dict)
        or str(run.get("id")) != approval.source_run_id
        or run.get("head_sha") != approval.source_head_sha
        or run.get("run_attempt") != approval.source_attempt
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
        or run.get("path", "").split("@", 1)[0] != ".github/workflows/run-sweep.yml"
    ):
        raise ValueError("approved source is not the completed successful sweep/attempt/head")
    pages = api(
        f"repos/{approval.repository}/actions/runs/{approval.source_run_id}/artifacts?per_page=100",
        paginate=True,
    )
    inventory = [item for page in pages for item in page["artifacts"] if not item["expired"]]
    if qualification_artifacts(item["name"] for page in pages for item in page["artifacts"]):
        raise ValueError("nonpublishing qualification cannot be approved for receipt issuance")

    def artifact(name: str) -> int:
        matching = [item for item in inventory if item["name"] == name]
        if len(matching) != 1:
            raise ValueError(f"expected one unexpired artifact named {name}")
        return int(matching[0]["id"])

    points = []
    for approved in approval.points:
        execution = artifact(f"native-execution-{approved.point_id}")
        normalized_path = f"{approved.point_id}.json"
        samples_path = None
        metadata_path = None
        if approved.kind == "throughput":
            normalized = artifact(f"bmk_agentic_{approved.point_id}")
            ids = [execution, normalized, artifact(f"agentic_{approved.point_id}")]
            metrics = ["output_tput_tps", "total_tput_tps", "duration_seconds"]
        else:
            normalized = artifact(f"eval_{approved.point_id}_lm-eval__{approval.source_attempt}")
            ids = [execution, normalized]
            with tempfile.TemporaryDirectory(prefix="infx-eval-members-") as temporary:
                archive = Path(temporary) / "eval.zip"
                with archive.open("xb") as stream:
                    subprocess.run(
                        [
                            "gh",
                            "api",
                            f"repos/{approval.repository}/actions/artifacts/{normalized}/zip",
                        ],
                        stdout=stream,
                        check=True,
                    )
                members = [member.path for member in inspect_archive(archive)]
                results = [
                    name
                    for name in members
                    if name.startswith("results")
                    and name.endswith("_conc28.json")
                    and "/" not in name
                ]
                samples = [
                    name
                    for name in members
                    if name.startswith("samples")
                    and name.endswith("_conc28.jsonl")
                    and "/" not in name
                ]
                if len(results) != 1 or len(samples) != 1 or "meta_env.json" not in members:
                    raise ValueError(
                        "eval artifact lacks one complete c28 result/sample/metadata set"
                    )
                normalized_path, samples_path, metadata_path = (
                    results[0],
                    samples[0],
                    "meta_env.json",
                )
            metrics = ["em_strict", "em_strict_se", "em_flexible", "em_flexible_se", "n_eff"]
        points.append(
            Point(
                **approved.model_dump(exclude={"dataset_revision"}),
                source_run_id=approval.source_run_id,
                source_attempt=approval.source_attempt,
                topology=Topology(kind="aggregate", nodes=1, serving_gpus=8, tp=8, ep=1),
                artifact_ids=ids,
                execution_artifact_id=execution,
                execution_path="execution.json",
                normalized_artifact_id=normalized,
                normalized_path=normalized_path,
                normalized_format="lm-eval" if approved.kind == "eval" else "normalized",
                metadata_path=metadata_path,
                required_metrics=metrics,
                config={
                    "model": "dsv41flash",
                    "hardware": "h100",
                    "framework": "vllm",
                    "precision": "fp4",
                    "specMethod": "mtp",
                },
                task="gsm8k" if approved.kind == "eval" else None,
                filters=["strict-match", "flexible-extract"] if approved.kind == "eval" else [],
                sample_count=1319 if approved.kind == "eval" else 0,
                samples_artifact_id=normalized if approved.kind == "eval" else None,
                samples_path=samples_path,
                dataset={
                    "source_type": "public_dataset",
                    "loader": "semianalysis_cc_traces_weka_062126",
                    "hf_dataset_name": "semianalysisai/cc-traces-weka-062126",
                    "hf_split": "train",
                    "num_dataset_entries": 393,
                    "hf_revision": approved.dataset_revision,
                }
                if approved.kind == "throughput"
                else {},
            )
        )
    return ExpectedContract(
        repository=approval.repository,
        source_run_id=approval.source_run_id,
        source_attempt=approval.source_attempt,
        source_head_sha=approval.source_head_sha,
        bundle_digest=digest({point.point_id: point.bundle_digest for point in approval.points}),
        contracts=Contracts(raw="aiperf-1.4", normalized="agentx-v1", publication=1),
        points=points,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archives", type=Path, required=True)
    args = parser.parse_args()
    approval = Approval.model_validate(read_json(args.approval))
    issuer = Issuer(
        repository=os.environ["GITHUB_REPOSITORY"],
        run_id=os.environ["GITHUB_RUN_ID"],
        job=os.environ["GITHUB_JOB"],
        workflow_sha=os.environ["TRUSTED_WORKFLOW_SHA"],
        collector_sha=os.environ["TRUSTED_WORKFLOW_SHA"],
    )
    receipt = fetch_and_seal(expected_contract(approval), issuer, args.archives)
    with args.output.open("x") as stream:
        stream.write(json.dumps(receipt.model_dump(), indent=2) + "\n")


if __name__ == "__main__":
    main()
