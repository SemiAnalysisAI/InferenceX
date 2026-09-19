"""Render the explicitly selected aggregate recipe and prepared Python client."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from infx.benchmarks.spec import AgentXSpec, EvalSpec, PreparedFile, ResultMetadata, RuntimeSpec
from infx.srt_slurm.contracts import load_mapping
from infx.srt_slurm.job import JobSpec


class ClientPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    golden_curve: str
    golden_model: str
    thinking_mode: Literal["thinking_on"]
    dataset_repository: Literal["semianalysisai/cc-traces-weka-062126"]
    dataset_loader: Literal["semianalysis_cc_traces_weka_062126"]
    dataset_entries: Literal[393]
    duration_seconds: Literal[3600]
    warmup_requests_per_lane: Literal[10]
    warmup_grace_seconds: Literal[1800]
    trace_idle_gap_cap_seconds: Literal[300]
    live_failed_request_threshold: Literal[0.1]
    failed_request_threshold: Literal[0.1]
    random_seed: Literal[42]
    required_server_metric_prefix: Literal["vllm:"]
    eval_task: Literal["gsm8k"]
    eval_documents: Literal[1319]
    eval_max_length: Literal[16384]
    eval_max_tokens: Literal[12288]
    eval_minimum_score: Literal[0.9]
    telemetry: Literal["temporary-parity-exception-no-native-power"]


class PilotSite(BaseModel):
    """Provisioned shared paths; no implicit host/environment fallback."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    cluster: Literal["h100-dgxc"]
    native_python: str
    native_source: str
    wrapper_python: str
    shared_root: str
    model_snapshot: str
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    image: PreparedFile
    image_reference: str
    client_sites: dict[Literal["agentx", "eval"], str]
    mounts: dict[str, str]
    # Receipt reader deployment is a release prerequisite, not inferred from code presence.
    reader_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    collector_revision: str = Field(pattern=r"^[0-9a-f]{40}$")

    @field_validator(
        "native_python", "native_source", "wrapper_python", "shared_root", "model_snapshot"
    )
    @classmethod
    def absolute(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("site paths must be absolute and shared with compute nodes")
        return value

    @field_validator("shared_root")
    @classmethod
    def writable_root(cls, value: str) -> str:
        if Path(value).resolve().is_relative_to("/workspace"):
            raise ValueError("pilot output must not create directories under /workspace")
        return value

    @field_validator("mounts")
    @classmethod
    def identity_mounts(cls, value: dict[str, str]) -> dict[str, str]:
        if not value or any(not Path(k).is_absolute() or k != v for k, v in value.items()):
            raise ValueError("prepared interpreters and assets require explicit same-path mounts")
        if any(str(Path(path).resolve()) != path for path in value):
            raise ValueError(
                "same-path mount roots must be canonical paths without symlink aliases"
            )
        return value

    def require_visible(self, value: str, *, writable: bool = False) -> None:
        path = Path(value)
        if not path.is_absolute() or not all(
            any(candidate.is_relative_to(Path(mount)) for mount in self.mounts)
            for candidate in (path, path.resolve())
        ):
            raise ValueError(f"prepared path is not mounted at its absolute location: {value}")
        if writable and path.resolve().is_relative_to("/workspace"):
            raise ValueError("pilot caches must not create directories under /workspace")

    def require_interpreter(
        self, identity: dict[str, Any], *, python_minor: str | None = None
    ) -> None:
        paths = identity.get("python_paths", {})
        for name in ("executable", "executable_resolved", "prefix", "base_prefix"):
            value = paths.get(name)
            if not isinstance(value, str):
                raise ValueError(f"prepared interpreter identity is missing {name}")
            self.require_visible(value)
        if python_minor is not None and not identity.get("python_version", "").startswith(
            python_minor + "."
        ):
            raise ValueError(f"pilot interpreter must use Python {python_minor}")


def golden_acceptance(root: Path, policy: ClientPolicy, draft_tokens: int) -> float:
    path = (root / policy.golden_curve).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError("golden curve must be a committed checkout resource")
    curve = load_mapping(path)
    value = curve[policy.golden_model][policy.thinking_mode][draft_tokens]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("golden acceptance must be numeric")
    if not math.isfinite(value) or not 1 <= value <= draft_tokens + 1:
        raise ValueError("golden acceptance is outside the draft/target token range")
    return float(value)


def client_spec(
    job: JobSpec,
    policy: ClientPolicy,
    runtime: RuntimeSpec,
    resources: dict[str, Any],
    result_filename: str,
) -> AgentXSpec | EvalSpec:
    row = job.row
    metadata = ResultMetadata(
        hw="h100",
        model=row.model,
        model_prefix=row.model_prefix,
        image=row.image,
        framework=row.framework,
        precision=row.precision,
        spec_decoding=row.spec_decoding,
        tp=row.tp,
        pp=row.pp,
        dcp_size=row.dcp_size,
        pcp_size=row.pcp_size,
        ep=row.ep,
        dp_attention=row.dp_attn,
        total_cpu_dram_gb=row.total_cpu_dram_gb,
        recipe_fingerprint=row.recipe_fingerprint or job.point_id,
    )
    shared = {
        "schema_version": 1,
        "runtime": runtime,
        "metadata": metadata,
        "concurrency": row.conc,
    }
    if job.mode == "eval":
        return EvalSpec(
            **shared,
            task=PreparedFile.model_validate(resources["task"]),
            document_identities=PreparedFile.model_validate(resources["document_identities"]),
            task_name=policy.eval_task,
            expected_documents=policy.eval_documents,
            max_length=policy.eval_max_length,
            max_tokens=policy.eval_max_tokens,
            minimum_score=policy.eval_minimum_score,
        )
    names = (
        "dataset_repository",
        "dataset_loader",
        "dataset_entries",
        "duration_seconds",
        "warmup_requests_per_lane",
        "warmup_grace_seconds",
        "trace_idle_gap_cap_seconds",
        "live_failed_request_threshold",
        "failed_request_threshold",
        "random_seed",
        "required_server_metric_prefix",
    )
    return AgentXSpec(
        **shared,
        **{key: getattr(policy, key) for key in names},
        result_filename=result_filename,
        tokenizer=row.model,
        dataset_revision=resources["dataset_revision"],
    )


def render_recipe(
    job: JobSpec,
    root: Path,
    site: PilotSite,
    policy: ClientPolicy,
    spec_path: Path,
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference = job.row.execution
    if reference is None:
        raise ValueError("explicit execution reference is required")
    recipe = copy.deepcopy(load_mapping(root / reference.recipe))
    if recipe["model"]["path"] != job.row.model or recipe["model"]["container"] != job.row.image:
        raise ValueError("master and recipe model/image identities disagree")
    if site.image_reference != job.row.image:
        raise ValueError("prepared image identity does not match the selected image")
    if set(recipe["roles"]) != {"agg"} or recipe["frontend"]["type"] != "vllm":
        raise ValueError("pilot requires one aggregate direct vLLM frontend")
    role = recipe["roles"]["agg"]
    if (role["nodes"], role["workers"], role["gpus"]) != (1, 1, 8):
        raise ValueError("pilot requires one physical node and one TP8 worker")
    args = role["args"]
    args["max-num-seqs"] = 2 * job.row.conc
    args["max-cudagraph-capture-size"] = min(2048, 1 << (12 * job.row.conc - 1).bit_length())
    spec = args["speculative-config"]
    if spec.get("synthetic_acceptance_length") is not None:
        raise ValueError("recipe must not hard-code a synthetic acceptance value")
    spec["enable_adaptive_verification"] = job.mode == "eval"
    spec["rejection_sample_method"] = "block" if job.mode == "eval" else "synthetic"
    if job.mode != "eval":
        spec["synthetic_acceptance_length"] = golden_acceptance(
            root, policy, spec["num_speculative_tokens"]
        )
    recipe["model"]["path"] = site.model_snapshot
    recipe["model"]["container"] = site.image.path
    recipe["identity"] = {
        "model": {"repo": job.row.model, "revision": site.model_revision},
        "container": {"image": job.row.image},
    }
    module = "eval" if job.mode == "eval" else "agentx"
    recipe["benchmark"] = {
        "type": "custom",
        "argv": [
            site.wrapper_python,
            "-I",
            "-m",
            f"infx.benchmarks.{module}",
            "--spec",
            str(spec_path),
            "--artifact-root",
            str(output),
        ],
        "cwd": str(spec_path.parent),
        "env": {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
        "env_unset": ["PYTHONPATH", "PYTHONHOME", "BASH_ENV", "ENV"],
        "container_image": site.image.path,
    }
    profile = load_mapping(root / reference.profile)
    if profile.get("use_exclusive_sbatch_directive") is not True:
        raise ValueError("the direct port 8000 policy requires an exclusive node")
    profile.update(
        srtctl_root=site.native_source,
        output_dir=str(spec_path.parent / "native-output"),
        default_mounts=site.mounts,
    )
    return recipe, profile
