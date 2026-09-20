"""Disposable, owned native lifecycle probes; never an accepted benchmark execution."""

from __future__ import annotations

import argparse
import copy
import json
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import model_validator

from infx.benchmarks.common import verify_file, write_json
from infx.srt_slurm.contracts import load_mapping
from infx.srt_slurm.job import file_digest, read_json
from infx.srt_slurm.launch import NativeCommandError, RuntimeLock, checked_json
from infx.srt_slurm.provision_runtime import NATIVE_LOCK
from infx.srt_slurm.render import PreparedSite

RECIPE = (
    "benchmarks/multi_node/srt-slurm-recipes/dsv41flash/vllm/h100-fp4/agentx/agg-tp8-dspark5.yaml"
)
PROFILE = "runners/srt-slurm/h100-phase1.yaml"
OWNERSHIP_CAPABILITY = "prepared-direct-listener-ownership-v1"
RESOURCES = {"nodes": 1, "gpus_per_node": 8, "serving_gpus": 8, "workers": 1, "cardinality": 1}

# This literal stdlib client sends no requests. A closed marker is written only
# after its heartbeat file has been flushed, fsynced and closed.
CLIENT_WRITER = r"""
import argparse, json, os, signal, sys, time
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--token", required=True)
parser.add_argument("--endpoint")
args = parser.parse_args()
root = Path(args.output)
root.mkdir(exist_ok=True)
endpoint = args.endpoint or os.environ["SRT_ENDPOINT"]
identity = {"token": args.token, "job_id": os.environ["SRT_JOB_ID"],
            "pid": os.getpid(), "endpoint": endpoint, "cwd": str(Path.cwd())}
stopped = None
def stop(number, frame):
    global stopped
    stopped = number
for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(number, stop)
def publish(name, value):
    temporary = root / (name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(root / name)
count = 0
with (root / "heartbeat.jsonl").open("x") as stream:
    while stopped is None:
        stream.write(json.dumps({"sequence": count, "token": args.token}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        count += 1
        if count == 1:
            publish("started.json", identity)
        time.sleep(0.1)
publish("closed.json", {**identity, "signal": stopped, "records": count,
        "bytes": (root / "heartbeat.jsonl").stat().st_size, "writer_closed": True})
sys.exit(128 + stopped)
"""

# Scheduler parsing and allocation ownership stay in the pinned native runtime.
# A worker PID is evidence of container entry, not server readiness.
WORKER_PROBE = r"""
import json, sys
from pathlib import Path
from srtctl.core.prepared import validate_receipt
from srtctl.core.observation import observe_job
from srtctl.core.processes import list_step_ids
receipt = validate_receipt(Path(sys.argv[1]))
observed = observe_job(receipt["job_id"], command_timeout=5,
                       expected_comment=receipt["scheduler_comment"])
steps = list_step_ids(receipt["job_id"], timeout=5) if observed["state"] == "active" else None
print(json.dumps({"observation": observed, "steps": steps}))
"""


class QualificationInterruptedError(RuntimeError):
    """Escape subprocess selectors, which deliberately swallow InterruptedError."""


class DraftSite(PreparedSite):
    """A provisioned runtime without any deployment declaration."""

    @model_validator(mode="after")
    def qualification_paths(self) -> Self:
        shared = Path(self.shared_root)
        if shared.resolve() != shared:
            raise ValueError("qualification requires canonical shared storage")
        for value in (
            self.native_python,
            self.native_source,
            self.wrapper_python,
            self.shared_root,
            self.model_snapshot,
        ):
            self.require_visible(value)
        if set(self.client_sites) != {"agentx", "eval"}:
            raise ValueError("draft must retain both provisioned client-site references")
        return self


class Native:
    def __init__(self, site: DraftSite, directory: Path) -> None:
        self.site = site
        self.directory = directory
        self.number = 0

    def run(self, command: str, *args: str, timeout: int = 60) -> dict[str, Any]:
        return self.invoke(
            command,
            [self.site.native_python, "-I", "-m", "srtctl.cli.submit", command, *args, "--json"],
            timeout=timeout,
        )

    def probe(self, receipt: Path, *, timeout: int) -> dict[str, Any]:
        return self.invoke(
            "worker-observation",
            [self.site.native_python, "-I", "-c", WORKER_PROBE, str(receipt)],
            timeout=timeout,
        )

    def invoke(self, name: str, argv: list[str], *, timeout: int) -> dict[str, Any]:
        self.number += 1
        path = self.directory / f"{self.number:04d}-{name}.json"
        try:
            result = checked_json(argv, timeout=timeout)
        except NativeCommandError as error:
            result = error.output
        except BaseException as error:
            write_json(path, {"state": "interrupted", "error_type": type(error).__name__})
            raise
        write_json(path, result)
        if result.get("state") == "error":
            raise NativeCommandError(result, f"native {name} failed: {result}")
        return result


def verify_inputs(root: Path, site: DraftSite) -> RuntimeLock:
    lock = RuntimeLock.model_validate(read_json(root / NATIVE_LOCK))
    source = Path(site.native_source)
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    if revision != lock.revision or dirty or file_digest(source / "uv.lock") != lock.uv_lock_sha256:
        raise ValueError("native runtime checkout or dependency lock differs from selected pin")
    if OWNERSHIP_CAPABILITY not in lock.capabilities:
        raise ValueError("runtime lock does not qualify direct listener ownership")
    verify_file(site.image)
    snapshot = Path(site.model_snapshot)
    if (
        snapshot.name != site.model_revision
        or not snapshot.is_dir()
        or snapshot.resolve() != snapshot
    ):
        raise ValueError("draft model must be the canonical immutable serving snapshot")
    return lock


def render_probe(
    root: Path, site: DraftSite, directory: Path, namespace: str, walltime_seconds: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    recipe = copy.deepcopy(load_mapping(root / RECIPE))
    profile = copy.deepcopy(load_mapping(root / PROFILE))
    role = recipe.get("roles", {}).get("agg", {})
    if (
        set(recipe.get("roles", {})) != {"agg"}
        or recipe.get("engine") != "vllm"
        or recipe.get("frontend", {}).get("type") != "vllm"
        or (role.get("nodes"), role.get("workers"), role.get("gpus")) != (1, 1, 8)
        or role.get("args", {}).get("tensor-parallel-size") != 8
        or profile.get("use_exclusive_sbatch_directive") is not True
        or recipe["model"]["container"] != site.image_reference
    ):
        raise ValueError("qualification requires the selected exclusive H100 aggregate TP8 recipe")
    hours, remainder = divmod(walltime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    walltime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    recipe["slurm"]["time_limit"] = walltime
    recipe["name"] = "cancellation-" + namespace
    recipe["identity"] = {
        "model": {"repo": recipe["model"]["path"], "revision": site.model_revision},
        "container": {"image": site.image_reference},
    }
    recipe["model"].update(path=site.model_snapshot, container=site.image.path)
    recipe["benchmark"] = {
        "type": "custom",
        "argv": [
            site.native_python,
            "-I",
            "-c",
            CLIENT_WRITER,
            "--output",
            str(directory / "writer"),
            "--token",
            namespace,
        ],
        "cwd": str(directory),
        "env": {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
        "env_unset": [
            "PYTHONPATH",
            "PYTHONHOME",
            "BASH_ENV",
            "ENV",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
        ],
        "container_image": site.image.path,
    }
    profile.update(
        default_time_limit=walltime,
        srtctl_root=site.native_source,
        output_dir=str(directory / "native-output"),
        default_mounts=site.mounts,
    )
    return recipe, profile


def worker_evidence(receipt: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any] | None:
    observed = observation.get("observation", {})
    if observed.get("terminal") or observed.get("state") == "failed":
        raise RuntimeError("allocation ended or changed generation before the cancellation trigger")
    if observed.get("state") != "active" or observed.get("identity_mismatch"):
        return None
    job_id = receipt["job_id"]
    steps = observation.get("steps") or {}
    aggregate = {
        name: step
        for name, step in steps.items()
        if re.fullmatch(r"agg_0_.+", name) and re.fullmatch(re.escape(job_id) + r"\.\d+", step)
    }
    if len(aggregate) != 1:
        return None
    path = Path(receipt["output_dir"]) / "logs/direct-vllm-worker.json"
    try:
        identity = read_json(path)
    except (OSError, ValueError):
        return None
    if (
        not isinstance(identity, dict)
        or any(
            type(identity.get(name)) is not int or identity[name] <= 0
            for name in ("pid", "start_ticks")
        )
        or any(
            not isinstance(identity.get(name), list)
            or len(identity[name]) != 2
            or any(type(value) is not int or value < 0 for value in identity[name])
            for name in ("pid_namespace", "net_namespace")
        )
    ):
        return None
    return {"allocation": observed, "aggregate_steps": aggregate, "worker": identity}


def writer_record(
    directory: Path, name: str, receipt: dict[str, Any], namespace: str
) -> dict[str, Any]:
    record = read_json(directory / "writer" / name)
    if (
        record.get("token") != namespace
        or record.get("job_id") != receipt["job_id"]
        or type(record.get("pid")) is not int
        or record["pid"] <= 0
        or record.get("cwd") != str(directory)
        or not isinstance(record.get("endpoint"), str)
        or not record["endpoint"].startswith("http://")
    ):
        raise ValueError("diagnostic writer identity differs from the owned allocation")
    return record


def await_trigger(
    native: Native,
    receipt_path: Path,
    receipt: dict[str, Any],
    directory: Path,
    namespace: str,
    mode: str,
    timeout: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = native.run(
            "wait",
            "--receipt",
            str(receipt_path),
            "--timeout",
            "0.2",
            "--poll",
            "0.2",
            timeout=min(15, max(1, int(deadline - time.monotonic()))),
        )
        if (
            state.get("terminal")
            or state.get("identity_mismatch")
            or state.get("state") == "failed"
        ):
            raise RuntimeError("native allocation failed before the cancellation trigger")
        observation = native.probe(
            receipt_path, timeout=min(20, max(1, int(deadline - time.monotonic())))
        )
        evidence = worker_evidence(receipt, observation)
        if evidence is not None:
            started = directory / "writer/started.json"
            if mode == "startup":
                if started.exists():
                    raise RuntimeError("startup probe missed the pre-client cancellation window")
                return evidence
            client_step = (observation.get("steps") or {}).get("benchmark-client", "")
            if started.exists() and re.fullmatch(
                re.escape(receipt["job_id"]) + r"\.\d+", client_step
            ):
                evidence["writer"] = writer_record(directory, "started.json", receipt, namespace)
                return evidence
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    raise TimeoutError("owned allocation did not reach the requested cancellation trigger")


def close_owned(native: Native, receipt_path: Path, timeout: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    recovery: dict[str, Any] = {}
    while receipt_path.exists() and time.monotonic() < deadline:
        recovery = native.run(
            "reconcile",
            "--receipt",
            str(receipt_path),
            timeout=min(65, max(1, int(deadline - time.monotonic()))),
        )
        if recovery.get("accepted_ids"):
            break
        time.sleep(min(1, max(0, deadline - time.monotonic())))
    if not recovery.get("accepted_ids"):
        raise RuntimeError(f"submission remains unresolved and fenced: {receipt_path}")
    cancelled = native.run(
        "cancel-known",
        "--receipt",
        str(receipt_path),
        timeout=max(1, int(deadline - time.monotonic())),
    )
    remaining = max(0.1, deadline - time.monotonic())
    closure = native.run(
        "wait-known",
        "--receipt",
        str(receipt_path),
        "--timeout",
        str(remaining),
        "--poll",
        "1",
        timeout=max(1, int(remaining) + 5),
    )
    if closure.get("terminal") is not True or closure.get("state") != "closed":
        raise RuntimeError(
            f"owned allocations have not reached physical terminal closure: {receipt_path}"
        )
    return {"reconciled": recovery, "cancellation": cancelled, "closure": closure}


def verify_closed_writer(
    directory: Path, receipt: dict[str, Any], namespace: str
) -> dict[str, Any]:
    started = writer_record(directory, "started.json", receipt, namespace)
    closed = writer_record(directory, "closed.json", receipt, namespace)
    if (
        any(started[name] != closed[name] for name in ("pid", "token", "job_id", "endpoint"))
        or closed.get("writer_closed") is not True
        or closed.get("signal") not in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    ):
        raise ValueError("diagnostic writer did not close after interruption")
    heartbeat = directory / "writer/heartbeat.jsonl"
    before = file_digest(heartbeat)
    count = 0
    with heartbeat.open() as stream:
        for count, line in enumerate(stream, 1):
            if json.loads(line) != {"sequence": count - 1, "token": namespace}:
                raise ValueError("diagnostic writer output is incomplete or foreign")
    if (
        count < 1
        or closed.get("records") != count
        or closed.get("bytes") != heartbeat.stat().st_size
    ):
        raise ValueError("diagnostic writer closure does not match its final output")
    time.sleep(1)
    if file_digest(heartbeat) != before:
        raise ValueError("diagnostic writer output changed after terminal allocation closure")
    return {**closed, "heartbeat_sha256": before}


def retain_evidence(directory: Path, output: Path, report: dict[str, Any]) -> None:
    evidence = output / "evidence"
    evidence.mkdir(exist_ok=False)
    copied = []
    for source in sorted(directory.rglob("*")):
        if (
            source.is_symlink()
            or not source.is_file()
            or not source.resolve().is_relative_to(directory)
        ):
            continue
        relative = source.relative_to(directory)
        target = evidence / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        size = source.stat().st_size
        with source.open("rb") as stream:
            stream.seek(max(0, size - 16 * 1024 * 1024))
            target.write_bytes(stream.read(16 * 1024 * 1024))
        copied.append(
            {"path": str(relative), "source_bytes": size, "retained_bytes": target.stat().st_size}
        )
    report["retained_files"] = copied
    write_json(output / "qualification.json", report)


def qualify(
    root: Path,
    site_draft: Path,
    output: Path,
    namespace: str,
    *,
    mode: Literal["startup", "client"],
    walltime_seconds: int,
    observation_timeout_seconds: int,
    cleanup_timeout_seconds: int,
) -> dict[str, Any]:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", namespace) is None:
        raise ValueError("qualification namespace must be one unique safe path component")
    if (
        (mode == "startup" and walltime_seconds != 300)
        or (mode == "client" and not 1800 <= walltime_seconds <= 7200)
        or mode not in {"startup", "client"}
    ):
        raise ValueError(
            "startup walltime must be 300s; client walltime must be explicit 1800-7200s"
        )
    if (
        not 0 < observation_timeout_seconds < walltime_seconds
        or not 0 < cleanup_timeout_seconds <= 600
    ):
        raise ValueError("observation/cleanup deadlines must be positive and explicitly bounded")
    root = root.resolve(strict=True)
    site = DraftSite.model_validate(read_json(site_draft))
    lock = verify_inputs(root, site)
    base = Path(site.shared_root) / "cancellation-qualification"
    base.mkdir(exist_ok=True)
    if base.resolve() != base:
        raise ValueError("qualification generation parent cannot be a symlink")
    directory = base / namespace
    output = output.resolve()
    if output.is_relative_to("/workspace"):
        raise ValueError("qualification artifacts must stay outside /workspace")
    if output.is_relative_to(directory) or directory.is_relative_to(output):
        raise ValueError("artifact output and owned runtime generation must be disjoint")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "qualification.json").exists() or (output / "evidence").exists():
        raise FileExistsError("qualification artifact output is already owned")
    directory.mkdir(exist_ok=False)
    (directory / "commands").mkdir()
    native = Native(site, directory / "commands")
    report: dict[str, Any] = {
        "schema_version": 1,
        "state": "failed",
        "mode": mode,
        "namespace": namespace,
        "qualification_complete": False,
        "accepted_benchmark": False,
        "lifecycle_qualified": False,
        "directory": str(directory),
        "native_revision": lock.revision,
        "walltime_seconds": walltime_seconds,
        "observation_timeout_seconds": observation_timeout_seconds,
        "cleanup_timeout_seconds": cleanup_timeout_seconds,
    }
    receipt_path: Path | None = None
    receipt: dict[str, Any] | None = None
    attempted = False
    error: BaseException | None = None
    handlers = {}

    def interrupted(signum: int, _frame: Any) -> None:
        raise QualificationInterruptedError(f"qualification interrupted by signal {signum}")

    for number in (signal.SIGINT, signal.SIGTERM):
        handlers[number] = signal.signal(number, interrupted)
    try:
        capabilities = native.run("capabilities")
        if not set(lock.capabilities).issubset(capabilities.get("capabilities", [])):
            raise ValueError("installed native runtime lacks pinned qualification capabilities")
        recipe, profile = render_probe(root, site, directory, namespace, walltime_seconds)
        for name, value in (("recipe.yaml", recipe), ("profile.yaml", profile)):
            (directory / name).write_text(yaml.safe_dump(value, sort_keys=False))
        write_json(directory / "site-draft.json", site.model_dump())
        write_json(directory / "runtime-lock.json", lock.model_dump())
        prepared = native.run(
            "prepare",
            "--recipe",
            str(directory / "recipe.yaml"),
            "--profile",
            str(directory / "profile.yaml"),
            "--output",
            str(directory / "prepared"),
            "--expected-nodes",
            "1",
            "--runtime-python",
            site.native_python,
            timeout=600,
        )
        if prepared.get("state") != "prepared" or prepared.get("resources") != RESOURCES:
            raise ValueError("native preflight did not resolve exactly one TP8 aggregate worker")
        if prepared.get("prepared_dir") != str(directory / "prepared") or prepared.get(
            "output_root"
        ) != str(directory / "native-output"):
            raise ValueError("native prepared paths escaped the owned qualification generation")
        report["prepared"] = prepared
        intent = "cancellation-qualification-" + namespace
        location = native.run(
            "intent-path",
            "--intent",
            intent,
            "--cluster",
            site.cluster,
            "--journal-dir",
            str(directory / "journal"),
        )
        if location.get("state") != "intent" or not isinstance(location.get("receipt_path"), str):
            raise ValueError(f"native intent-path did not return an ownership path: {location}")
        receipt_path = Path(location["receipt_path"])
        if receipt_path.resolve() != receipt_path or not receipt_path.is_relative_to(
            directory / "journal"
        ):
            raise ValueError("native intent journal escaped the owned qualification generation")
        report["receipt_path"] = str(receipt_path)
        attempted = True
        receipt = native.run(
            "submit-prepared",
            "--prepared-dir",
            prepared["prepared_dir"],
            "--intent",
            intent,
            "--cluster",
            site.cluster,
            "--journal-dir",
            str(directory / "journal"),
            timeout=150,
        )
        report["submission"] = receipt
        if receipt.get("state") != "accepted" or receipt.get("accepted_ids") != [
            receipt.get("job_id")
        ]:
            raise RuntimeError(
                "submission was not uniquely accepted; intent stays fenced without retry"
            )
        if not re.fullmatch(r"[0-9]+", str(receipt.get("job_id", ""))) or receipt.get(
            "output_dir"
        ) != str(directory / "native-output" / receipt["job_id"]):
            raise ValueError(
                "accepted native output differs from the owned qualification generation"
            )
        report["trigger"] = await_trigger(
            native, receipt_path, receipt, directory, namespace, mode, observation_timeout_seconds
        )
    except BaseException as caught:  # noqa: BLE001 - signals must retain evidence and close owned jobs
        error = caught
        report["error_type"] = type(caught).__name__
        report["error"] = str(caught)
    finally:
        # A second workflow signal must not skip the bounded owned cleanup.
        for number in handlers:
            signal.signal(number, signal.SIG_IGN)
        try:
            if attempted and receipt_path is not None:
                report["cleanup"] = close_owned(native, receipt_path, cleanup_timeout_seconds)
            if error is None and receipt is not None:
                cancellation = report["cleanup"]["cancellation"]
                requests = cancellation.get("jobs", [])
                if (
                    cancellation.get("state") != "cancellation_requested"
                    or [job.get("job_id") for job in requests] != [receipt["job_id"]]
                    or any(job.get("already_terminal") for job in requests)
                ):
                    raise RuntimeError("native runtime did not confirm a new cancellation request")
                closure = report["cleanup"]["closure"]
                jobs = closure.get("jobs", [])
                if not jobs or any(job.get("slurm_state") != "CANCELLED" for job in jobs):
                    raise RuntimeError("allocation closed without observed explicit cancellation")
                if mode == "client":
                    report["writer_closure"] = verify_closed_writer(directory, receipt, namespace)
                report.update(state="passed", lifecycle_qualified=True)
        except BaseException as cleanup_error:  # noqa: BLE001 - preserve both execution and cleanup failures
            report["cleanup_error_type"] = type(cleanup_error).__name__
            report["cleanup_error"] = str(cleanup_error)
            error = error or cleanup_error
        finally:
            for number, handler in handlers.items():
                signal.signal(number, handler)
            write_json(directory / "qualification.json", report)
            retain_evidence(directory, output, report)
    if error is not None:
        raise RuntimeError(
            f"cancellation qualification failed; inspect {output / 'qualification.json'}"
        ) from error
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "site-draft", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--mode", choices=("startup", "client"), required=True)
    for name in ("walltime-seconds", "observation-timeout-seconds", "cleanup-timeout-seconds"):
        parser.add_argument("--" + name, type=int, required=True)
    arguments = vars(parser.parse_args())
    print(json.dumps(qualify(**arguments), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
