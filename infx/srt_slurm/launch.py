"""Prepare an immutable one-point bundle and delegate allocation to native srtctl."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from infx.benchmarks.common import (
    child_failed,
    verify_file,
    verify_model_snapshot_assets,
    write_json,
)
from infx.benchmarks.identity import capture_identity, verify_runtime
from infx.benchmarks.spec import RuntimeSpec
from infx.srt_slurm.contracts import digest, load_mapping
from infx.srt_slurm.job import JobSpec, file_digest, intent_id, parse_job, read_json
from infx.srt_slurm.render import ClientPolicy, PilotSite, client_spec, render_recipe


class RuntimeLock(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    repository: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    uv_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capabilities: list[str]


class NativeCommandError(RuntimeError):
    def __init__(self, output: dict[str, Any], detail: str) -> None:
        self.output = output
        super().__init__(detail)


def checked_json(argv: list[str], *, timeout: int = 600) -> dict[str, Any]:
    with subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException:
            from contextlib import suppress

            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=3)
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
            raise
        result = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"native command did not return JSON: {result.stderr}") from error
    if not isinstance(value, dict):
        raise ValueError("native command must return a JSON object")
    if result.returncode:
        raise NativeCommandError(
            value, f"native command failed ({result.returncode}): {value}\n{result.stderr}"
        )
    return value


def native(site: PilotSite, *args: str, timeout: int = 600) -> dict[str, Any]:
    return checked_json(
        [site.native_python, "-I", "-m", "srtctl.cli.submit", *args, "--json"], timeout=timeout
    )


def verify_site(job: JobSpec, site: PilotSite, root: Path) -> dict[str, Any]:
    reference = job.row.execution
    if reference is None:
        raise ValueError("missing native execution reference")
    lock = RuntimeLock.model_validate(read_json(root / reference.runtime_lock))
    source = Path(site.native_source)
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if revision != lock.revision or dirty or file_digest(source / "uv.lock") != lock.uv_lock_sha256:
        raise ValueError("native runtime checkout or dependency lock differs from pilot pin")
    verify_file(site.image)
    if (
        not Path(site.model_snapshot).is_dir()
        or Path(site.model_snapshot).name != site.model_revision
    ):
        raise ValueError("model must be a prepared immutable Hugging Face snapshot")
    for value in (
        site.native_python,
        site.native_source,
        site.wrapper_python,
        site.shared_root,
        site.model_snapshot,
    ):
        site.require_visible(value)
    if set(site.client_sites) != {"agentx", "eval"}:
        raise ValueError("both throughput and real eval client environments must be provisioned")
    wrapper_identity = capture_identity(site.wrapper_python, ["infx"], dataset_loader=None)
    site.require_interpreter(wrapper_identity, python_minor="3.12")
    verify_wrapper_source(wrapper_identity, root)
    native_identity = capture_identity(site.native_python, ["srtctl"], dataset_loader=None)
    site.require_interpreter(native_identity, python_minor="3.12")
    return {
        "runtime_lock": lock.model_dump(),
        "wrapper_identity": wrapper_identity,
        "native_identity": native_identity,
    }


def verify_wrapper_source(identity: dict[str, Any], root: Path) -> None:
    distribution = identity["distributions"]["infx"]
    if (distribution.get("direct_url") or {}).get("dir_info", {}).get("editable"):
        raise ValueError("prepared wrapper must be installed noneditable")
    installed = {
        name: value for name, value in distribution["files"].items() if name.startswith("infx/")
    }
    expected_python = {
        str(path.relative_to(root))
        for path in (root / "infx").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    if not expected_python <= installed.keys():
        raise ValueError("installed wrapper is missing candidate Python modules")
    for name, value in installed.items():
        path = root / name
        if not path.is_file() or file_digest(path) != value:
            raise ValueError(f"installed wrapper differs from candidate checkout: {name}")


def effective_identity(
    job: JobSpec,
    site: PilotSite,
    identity: dict[str, Any],
    runtime: RuntimeSpec,
    resources: dict[str, Any],
) -> tuple[str, str]:
    semantics = job.semantic_inputs()
    inputs = {
        "requested": semantics,
        "installed": identity,
        "client_identity": read_json(Path(runtime.identity.path)),
        "client_assets": {asset.path: asset.sha256 for asset in runtime.assets},
        "client_env": runtime.env,
        "client_env_unset": sorted(runtime.env_unset),
        "model_revision": site.model_revision,
        "image_sha256": site.image.sha256,
        "dataset_revision": resources.get("dataset_revision"),
        "task_sha256": resources.get("task", {}).get("sha256"),
        "document_identities_sha256": resources.get("document_identities", {}).get("sha256"),
    }
    point = digest(inputs)
    semantics["row"].pop("conc", None)
    return point, digest(inputs)


def prepare(
    job: JobSpec,
    site: PilotSite,
    root: Path,
    source: dict[str, Any],
) -> dict[str, Any]:
    """A local preparation lock protects files; only native srtctl may claim/submit Slurm."""
    from infx.benchmarks.prepare import ClientSite, prepare as prepare_client

    execution = intent_id(
        source["repository"], str(source["run_id"]), str(source["attempt"]), job.point_id
    )
    directory = Path(site.shared_root) / "runs" / execution
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "prepare.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        index = directory / "bundle.json"
        if index.exists():
            bundle = read_json(index)
            old_job = JobSpec.model_validate(bundle["job"])
            if old_job.semantic_inputs() != job.semantic_inputs() or bundle["source"] != source:
                raise ValueError("existing execution bundle belongs to different inputs")
            verify_bundle(bundle)
            return bundle
        if (directory / "client").exists() or (directory / "native").exists():
            raise ValueError(
                "incomplete preparation requires inspection; refusing to overwrite or allocate"
            )
        identity = verify_site(job, site, root)
        kind = "eval" if job.mode == "eval" else "agentx"
        client_site = ClientSite.model_validate(read_json(Path(site.client_sites[kind])))
        if client_site.model_path != site.model_snapshot:
            raise ValueError("serving and client model snapshots disagree")
        prepare_client(client_site, kind=kind, output=directory / "client")
        runtime = RuntimeSpec.model_validate(read_json(directory / "client" / "runtime.json"))
        verify_model_snapshot_assets(
            runtime,
            job.row.model,
            expected_revision=site.model_revision,
            expected_snapshot=Path(site.model_snapshot),
        )
        site.require_interpreter(
            read_json(Path(runtime.identity.path)),
            python_minor="3.11" if kind == "agentx" else None,
        )
        for path in (
            runtime.python,
            runtime.identity.path,
            *(asset.path for asset in runtime.assets),
        ):
            site.require_visible(path)
        for key in (
            "HF_HUB_CACHE",
            "HF_DATASETS_CACHE",
            "HF_MODULES_CACHE",
            "AIPERF_DATASET_MMAP_CACHE_DIR",
        ):
            if runtime.env.get(key):
                site.require_visible(runtime.env[key], writable=True)
        resources = read_json(directory / "client" / "prepared-resources.json")
        reference = job.row.execution
        if reference is None:
            raise ValueError("missing execution reference")
        policy = ClientPolicy.model_validate(load_mapping(root / reference.client_policy))
        point_id, curve_id = effective_identity(job, site, identity, runtime, resources)
        spec = client_spec(job, policy, runtime, resources, point_id)
        spec_path = directory / "client.json"
        write_json(spec_path, spec.model_dump(mode="json"))
        recipe, profile = render_recipe(
            job, root, site, policy, spec_path, directory / "client-output"
        )
        recipe["benchmark"]["argv"][3:4] = [
            "infx.srt_slurm.client_guard",
            "--bundle",
            str(index),
            "--client",
            kind,
        ]
        # Retain --spec/--artifact-root after the guarded module's explicit options.
        for name, data in (("recipe.yaml", recipe), ("profile.yaml", profile)):
            (directory / name).write_text(yaml.safe_dump(data, sort_keys=False))
        prepared = native(
            site,
            "prepare",
            "--recipe",
            str(directory / "recipe.yaml"),
            "--profile",
            str(directory / "profile.yaml"),
            "--output",
            str(directory / "native"),
            "--expected-nodes",
            str(job.scheduling.node_count),
            "--runtime-python",
            site.native_python,
        )
        if prepared.get("state") != "prepared" or prepared["resources"] != {
            "nodes": 1,
            "gpus_per_node": 8,
            "serving_gpus": 8,
            "workers": 1,
            "cardinality": 1,
        }:
            raise ValueError(
                "native resolved allocation differs from the queued TP8 aggregate point"
            )
        if not set(identity["runtime_lock"]["capabilities"]) <= set(
            prepared.get("capabilities", [])
        ):
            raise ValueError("native runtime lacks required Phase 1 capabilities")
        files = {
            str(path): file_digest(path)
            for path in directory.rglob("*")
            if path.is_file() and path.name != "prepare.lock"
        }
        bundle = {
            "schema_version": 1,
            "point_id": point_id,
            "requested_point_id": job.point_id,
            "effective_curve_id": curve_id,
            "execution_id": execution,
            "job": job.model_dump(mode="json", by_alias=True),
            "source": source,
            "site": site.model_dump(mode="json"),
            "identity": identity,
            "prepared": prepared,
            "files": files,
            "directory": str(directory),
            "telemetry": policy.telemetry,
        }
        bundle["bundle_digest"] = digest(bundle)
        write_json(index, bundle)
        for path in (*files, str(index)):
            Path(path).chmod(0o444)
        return bundle


def verify_bundle(bundle: dict[str, Any]) -> None:
    if bundle["bundle_digest"] != digest(
        {key: value for key, value in bundle.items() if key != "bundle_digest"}
    ):
        raise ValueError("prepared bundle identity changed")
    for name, expected in bundle["files"].items():
        if file_digest(Path(name)) != expected:
            raise ValueError(f"prepared input changed: {name}")


def verify_execution_clients(bundle: dict[str, Any], site: PilotSite) -> None:
    verify_file(site.image)
    actual = capture_identity(site.wrapper_python, ["infx"], dataset_loader=None)
    if actual != bundle["identity"]["wrapper_identity"]:
        raise ValueError("installed wrapper changed before allocation")
    spec = read_json(Path(bundle["directory"]) / "client.json")
    verify_runtime(
        RuntimeSpec.model_validate(spec["runtime"]), dataset_loader=spec.get("dataset_loader")
    )


def copy_diagnostics(bundle: dict[str, Any], workspace: Path, *, complete: bool) -> None:
    directory = Path(bundle["directory"])
    diagnostics = workspace / "native-execution"
    diagnostics.mkdir(exist_ok=True)
    for name in bundle["files"]:
        source = Path(name)
        relative = source.relative_to(directory)
        target = diagnostics / "prepared" / relative
        if source.is_symlink():
            raise ValueError("prepared diagnostic input is a symlink")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(directory / "bundle.json", diagnostics / "bundle.json")
    client = directory / "client-output"
    if client.exists():
        if any(path.is_symlink() for path in client.rglob("*")):
            raise ValueError("client diagnostics contain a symlink")
        shutil.copytree(client, workspace / "results", dirs_exist_ok=True)
    logs = directory / "native-output"
    for path in logs.rglob("*"):
        if path.is_file() and not path.is_symlink() and path.suffix in {".log", ".json", ".yaml"}:
            target = workspace / "results" / "native" / path.relative_to(logs)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    write_json(diagnostics / "output-state.json", {"schema_version": 1, "complete": complete})


def publish_outputs(bundle: dict[str, Any], receipt: dict[str, Any], workspace: Path) -> None:
    directory = Path(bundle["directory"])
    client = directory / "client-output"
    audit = read_json(client / "diagnostics" / "client-audit.json")
    if audit["errors"] or child_failed(audit["status"]):
        raise ValueError("client did not produce a successful closed result")
    if bundle["job"]["row"].get("eval-only"):
        names = [
            *client.glob("results*.json"),
            *client.glob("samples*.jsonl"),
            client / "meta_env.json",
        ]
    else:
        names = [client / f"{bundle['point_id']}.json"]
    for source_path in names:
        destination = workspace / source_path.name
        if destination.exists() and file_digest(destination) != file_digest(source_path):
            raise ValueError(f"refusing to overwrite different result: {destination}")
        shutil.copyfile(source_path, destination)
    diagnostics = workspace / "native-execution"
    diagnostics.mkdir(exist_ok=True)
    shutil.copyfile(directory / "bundle.json", diagnostics / "bundle.json")
    write_json(
        diagnostics / "execution.json",
        {
            "schema_version": 1,
            "point_id": bundle["point_id"],
            "execution_id": bundle["execution_id"],
            "bundle_digest": bundle["bundle_digest"],
            "source": bundle["source"],
            "mode": "eval" if bundle["job"]["row"].get("eval-only") else "throughput",
            "native_receipt": {
                "job_id": receipt["job_id"],
                "state": "COMPLETED",
                "manifest_sha256": bundle["prepared"]["manifest_sha256"],
            },
            "client_exit_code": 0,
        },
    )


def execute(bundle: dict[str, Any], workspace: Path, *, reconcile_timeout: int = 120) -> None:
    verify_bundle(bundle)
    site = PilotSite.model_validate(bundle["site"])
    verify_execution_clients(bundle, site)
    journal = Path(site.shared_root) / "journal"
    receipt_path = Path(
        native(
            site,
            "intent-path",
            "--intent",
            bundle["execution_id"],
            "--cluster",
            site.cluster,
            "--journal-dir",
            str(journal),
        )["receipt_path"]
    )
    receipt: dict[str, Any] | None = None
    completed = False
    previous = {}

    def interrupted(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"workflow interrupted by signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, interrupted)
    try:
        try:
            receipt = native(
                site,
                "submit-prepared",
                "--prepared-dir",
                bundle["prepared"]["prepared_dir"],
                "--intent",
                bundle["execution_id"],
                "--cluster",
                site.cluster,
                "--journal-dir",
                str(journal),
            )
        except NativeCommandError as error:
            if error.output.get("receipt_path"):
                receipt_path = Path(error.output["receipt_path"])
            raise
        receipt_path = Path(receipt["receipt_path"])
        if receipt.get("state") != "accepted" or len(receipt["accepted_ids"]) != 1:
            raise ValueError(
                "submission was not uniquely accepted; reconcile the native journal before retry"
            )
        state = native(
            site,
            "wait",
            "--receipt",
            str(receipt_path),
            "--timeout",
            "28800",
            "--poll",
            "10",
            timeout=28920,
        )
        if state.get("state") != "completed" or state.get("terminal") is not True:
            raise ValueError(f"native job has not completed successfully: {state}")
        publish_outputs(bundle, receipt, workspace)
        completed = True
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        try:
            if not completed and receipt_path is not None and receipt_path.exists():
                deadline = time.monotonic() + reconcile_timeout
                while True:
                    try:
                        recovery = native(
                            site, "reconcile", "--receipt", str(receipt_path), timeout=180
                        )
                    except NativeCommandError as error:
                        recovery = error.output
                    if recovery.get("accepted_ids") or time.monotonic() >= deadline:
                        break
                    time.sleep(min(2, max(0, deadline - time.monotonic())))
                if recovery.get("accepted_ids"):
                    native(site, "cancel-known", "--receipt", str(receipt_path), timeout=180)
                    try:
                        closure = native(
                            site,
                            "wait-known",
                            "--receipt",
                            str(receipt_path),
                            "--timeout",
                            "120",
                            "--poll",
                            "5",
                            timeout=150,
                        )
                    except NativeCommandError as error:
                        closure = error.output
                    if closure.get("terminal") is not True:
                        raise RuntimeError(
                            f"owned allocation cleanup is unresolved: {receipt_path}"
                        )
                else:
                    raise RuntimeError(
                        f"submission ownership remains unresolved; intent stays fenced: {receipt_path}"
                    )
        finally:
            copy_diagnostics(bundle, workspace, complete=completed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    job = parse_job(read_json(args.job), args.root)
    site = PilotSite.model_validate(read_json(args.site))
    bundle = prepare(job, site, args.root, read_json(args.source))
    print(
        json.dumps(
            {
                "point_id": bundle["point_id"],
                "bundle_digest": bundle["bundle_digest"],
                "bundle": str(Path(bundle["directory"]) / "bundle.json"),
            }
        )
    )
    if not args.prepare_only:
        execute(bundle, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
