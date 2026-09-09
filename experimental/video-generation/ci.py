#!/usr/bin/env python3
"""InferenceX H200 Slurm adapter for a prepared, trusted H3 runtime.

No SSH, image import, dependency installation, or model download. The submit
host and compute node share workspace.host, mounted at workspace.container by
an immutable entry-only wrapper. A receipt, not a username, identifies reuse.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from typing import Any, Iterator

from evaluator.mvp_gpu_job import cuda_devices

PARTITION = "main"
ACCOUNT = "sa-shared"
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
SHA = re.compile(r"[0-9a-f]{64}")
IDENTITY = ("JobId", "JobName", "Comment", "WorkDir", "Account", "Partition", "UserId")
CACHE_PATHS = ("gpu/supervisor/baseline/cache", "gpu/supervisor/candidate/cache", "gpu/supervisor/compare-cache")
CACHE_PATHS += tuple(f"gpu/c{concurrency}/supervisor/baseline/cache" for concurrency in (1, 2, 4))
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def write(path: Path | str, value: Any) -> None:
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def digest(path: Path | str) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def absolute(value: str) -> Path:
    need(isinstance(value, str) and re.fullmatch(r"/[A-Za-z0-9_./-]+", value)
         and ".." not in Path(value).parts and value != "/" and str(Path(value)) == value,
         "Expected a normalized absolute cluster path without whitespace")
    return Path(value)


def mapped(config: dict, host: Path) -> Path:
    return Path(config["workspace"]["container"]) / Path(host).relative_to(config["workspace"]["host"])


def host_path(config: dict, container_path: str) -> Path:
    path = absolute(container_path)
    mount = Path(config["workspace"]["container"])
    if path.is_relative_to(mount):
        return Path(config["workspace"]["host"]) / path.relative_to(mount)
    return Path(config["runtime"]["rootfs"]) / path.relative_to("/")


def validate_config(config: dict) -> dict:
    need(set(config) == {"schema_version", "task_id", "workspace", "runtime", "spec", "resources", "allocation_receipts", "mode"}, "Unknown or missing site configuration fields")
    need(config["schema_version"] == 1 and NAME.fullmatch(config["task_id"]), "Invalid schema_version/task_id")
    need(config["mode"] in {"smoke", "regression", "serving-smoke"}, "mode must be smoke, regression or serving-smoke")
    need(set(config["workspace"]) == {"host", "container"}, "Invalid workspace mapping")
    for value in config["workspace"].values():
        path = absolute(value)
        need(not path.is_relative_to("/workspace"), "Use the declared persistent mount, not /workspace")
    runtime = config["runtime"]
    need(set(runtime) == {"entry", "entry_sha256", "rootfs", "ready_marker", "python"}, "Invalid runtime contract")
    for key in ("entry", "rootfs", "ready_marker", "python"):
        absolute(runtime[key])
    need(SHA.fullmatch(runtime["entry_sha256"]), "Runtime entry SHA256 required")
    need(set(config["spec"]) == {"path", "sha256"} and SHA.fullmatch(config["spec"]["sha256"]), "Pinned prepared spec required")
    absolute(config["spec"]["path"])
    resources = config["resources"]
    need(set(resources) == {"gpus", "cpus", "memory_gb", "minutes"}, "Invalid resource request")
    for key, low, high in (("gpus", 1, 8), ("cpus", 1, 128), ("memory_gb", 1, 1400), ("minutes", 10, 90)):
        need(type(resources[key]) is int and low <= resources[key] <= high, "Resource outside bounded H200 budget: " + key)
    need(isinstance(config["allocation_receipts"], list), "allocation_receipts must be a list")
    for path in config["allocation_receipts"]:
        absolute(path)
    return config


def environment() -> dict[str, str]:
    # Slurm defaults inherited from the runner must not change this request.
    # The payload receives an explicit environment allowlist at the srun edge.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("SLURM_", "SBATCH_", "SALLOC_", "SRUN_", "SQUEUE_", "SCANCEL_"))}
    env.update(TZ="UTC", LC_ALL="C", PYTHONDONTWRITEBYTECODE="1")
    return env


def command(argv: list[str], timeout: float = 30) -> str:
    return subprocess.run(argv, text=True, capture_output=True, check=True,
                          timeout=timeout, env=environment()).stdout


def fields(raw: str) -> dict[str, str]:
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_/:]*)=(\S+)", raw))


def job_record(job_id: str) -> dict[str, str]:
    need(re.fullmatch(r"[0-9]+", str(job_id)), "Invalid Slurm job ID")
    return fields(command(["scontrol", "show", "job", "-o", str(job_id)]))


def verify_identity(receipt: dict, record: dict, task_id: str) -> None:
    need(receipt.get("task_id") == task_id, "Allocation belongs to another task")
    expected = receipt["identity"]
    need(set(expected) == set(IDENTITY), "Incomplete allocation ownership receipt")
    need(all(record.get(key) == expected[key] for key in IDENTITY), "Slurm allocation identity differs from receipt")
    need(record["Account"] == ACCOUNT and record["Partition"] == PARTITION, "Allocation is not in the SemiAnalysis H200 pool")
    need(re.fullmatch(r"[^()]+\(" + str(os.getuid()) + r"\)", record["UserId"]), "Allocation Unix owner differs")


def capacity(record: dict, resources: dict, timestamp: datetime | None = None) -> str | None:
    need(record.get("NumNodes") == "1" and NAME.fullmatch(record.get("NodeList", "")), "Reuse requires one explicit node")
    tres = dict(item.split("=", 1) for item in record["AllocTRES"].split(","))
    memory = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGT]?)", tres.get("mem", ""))
    need(memory is not None, "Unrecognized allocated memory")
    memory_gb = float(memory[1]) * {"K": 1 / 1048576, "M": 1 / 1024, "": 1 / 1024, "G": 1, "T": 1024}[memory[2]]
    end = datetime.fromisoformat(record["EndTime"]).replace(tzinfo=timezone.utc)
    remaining = (end - (timestamp or datetime.now(timezone.utc))).total_seconds()
    if remaining < resources["minutes"] * 60 - 300 + 30:
        return "insufficient remaining allocation time"
    if int(tres.get("gres/gpu", "0")) < resources["gpus"] or int(record["NumCPUs"]) < resources["cpus"] or memory_gb < resources["memory_gb"]:
        return "insufficient allocated GPU/CPU/memory capacity"
    return None


def recover(config: dict, result_root: Path, *, node: str | None = None) -> dict:
    paths = set(result_root.glob("*/allocation.json")) | {Path(p) for p in config["allocation_receipts"]}
    # A crash between intent and acknowledgment must be reconciled, not retried.
    for intent in result_root.glob("*/allocation-intent.json"):
        need(intent.with_name("allocation.json").exists(), f"Unresolved allocation intent: {intent}; reconcile Slurm before another submission")
    active = set(command(["squeue", "--all", "--noheader", "--user=" + str(os.getuid()), "--format=%i"]).split())
    reasons = []
    waiting = []
    for path in sorted(paths):
        receipt = read(path)
        need(receipt.get("task_id") == config["task_id"], "Saved allocation receipt belongs to another task")
        identity = receipt["identity"]
        need(set(identity) == set(IDENTITY), "Incomplete allocation ownership receipt")
        job = identity["JobId"]
        need(re.fullmatch(r"[0-9]+", job), "Invalid saved job ID")
        if job not in active:
            reasons.append({"receipt": str(path), "job_id": job, "reason": "inactive in successful scheduler snapshot"})
            continue
        record = job_record(job)
        verify_identity(receipt, record, config["task_id"])
        state = record["JobState"]
        if state in TERMINAL:
            reasons.append({"job_id": job, "reason": state})
            continue
        if state != "RUNNING":
            waiting.append({"job_id": job, "state": state})
            continue
        if node is not None and record["NodeList"] != node:
            reasons.append({"job_id": job, "reason": "allocation is on a different physical node"})
            continue
        reason = capacity(record, config["resources"])
        if reason:
            reasons.append({"job_id": job, "reason": reason})
            continue
        steps = command(["squeue", "--steps", "--noheader", "--jobs=" + job, "--format=%i|%N"])
        return {"action": "reuse", "receipt": receipt, "record": record, "active_steps": steps, "reasons": reasons}
    if waiting:
        return {"action": "wait", "jobs": waiting, "reasons": reasons}
    return {"action": "allocate", "reasons": reasons or [{"reason": "no saved allocations for this task"}]}


def allocate(config: dict, run_dir: Path, *, node: str | None = None) -> dict:
    need(node is None or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,252}", node), "Invalid target node")
    nonce = uuid.uuid4().hex
    job_name = os.environ.get("RUNNER_NAME", "h3-" + config["task_id"])
    need(NAME.fullmatch(job_name), "Invalid runner/job name")
    comment = "h3:" + nonce
    request = config["resources"]
    intent = {"task_id": config["task_id"], "job_name": job_name, "comment": comment,
              "work_dir": str(run_dir), "user_id": os.getuid(), "created_at": now()}
    write(run_dir / "allocation-intent.json", intent)
    placement = ["--gres=gpu:" + str(request["gpus"])] if config["mode"] == "serving-smoke" else ["--exclusive", "--gres=gpu:8"]
    argv = ["salloc", "--no-shell", "--no-bell", "--partition=" + PARTITION, "--account=" + ACCOUNT,
            "--nodes=1", "--ntasks=1", *placement,
            "--cpus-per-task=" + str(request["cpus"]), "--mem=" + str(request["memory_gb"]) + "G",
            "--time=" + str(request["minutes"]), "--immediate=30",
            "--job-name=" + job_name, "--comment=" + comment, "--chdir=" + str(run_dir)]
    if node is not None:
        argv.append("--nodelist=" + node)
    write(run_dir / "allocation-command.json", argv)
    # With --no-shell, Slurm records the caller's cwd rather than --chdir.
    result = subprocess.run(argv, text=True, capture_output=True, timeout=45, env=environment(), cwd=run_dir)
    (run_dir / "salloc.log").write_text(result.stdout + result.stderr)
    granted = re.findall(r"Granted job allocation ([0-9]+)", result.stdout + result.stderr)
    need(len(set(granted)) == 1, "No unambiguous Slurm acknowledgment; allocation intent retained for reconciliation")
    job = granted[0]
    # Save the expected identity before querying, so a lost query can be recovered.
    user = command(["id", "-un"]).strip()
    receipt = {"task_id": config["task_id"], "created_at": now(), "identity": {
        "JobId": job, "JobName": job_name, "Comment": comment, "WorkDir": str(run_dir),
        "Account": ACCOUNT, "Partition": PARTITION, "UserId": f"{user}({os.getuid()})"}}
    write(run_dir / "allocation.json", receipt)
    need(result.returncode == 0, "Slurm returned a failure after granting an allocation; reconcile receipt")
    return receipt


def stop_allocation(receipt: dict, task_id: str) -> dict:
    record = job_record(receipt["identity"]["JobId"])
    verify_identity(receipt, record, task_id)
    if record["JobState"] not in TERMINAL:
        command(["scancel", receipt["identity"]["JobId"]])
    # scancel success is a request, not a terminal-state observation.
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        active = command(["squeue", "--noheader", "--jobs=" + receipt["identity"]["JobId"], "--format=%T"]).strip()
        if not active or all(state in TERMINAL for state in active.split()):
            return {"status": "released", "job_id": receipt["identity"]["JobId"]}
        time.sleep(1)
    raise RuntimeError("Owned allocation has not reached terminal state after cancellation")


def drain_step(receipt: dict, task_id: str, run_dir: Path) -> dict:
    binding_path = run_dir / "binding.json"
    if not binding_path.exists():
        return {"status": "not_observed", "reason": "payload did not write a step binding"}
    binding = read(binding_path)
    job, step = receipt["identity"]["JobId"], binding["step_id"]
    need(binding["job_id"] == job and re.fullmatch(r"[0-9]+", step), "Step binding differs from owned allocation")
    step_id = job + "." + step
    def active():
        return step_id in command(["squeue", "--steps", "--noheader", "--jobs=" + job, "--format=%i"]).split()
    if active():
        verify_identity(receipt, job_record(job), task_id)
        # This exact step is ours; the parent of an attachment is never canceled.
        command(["scancel", step_id])
        for _ in range(15):
            if not active():
                break
            time.sleep(1)
        else:
            raise RuntimeError("Owned Slurm step remains active; preserve evidence and reconcile")
    return {"status": "ended", "step_id": step_id}


def inventory(root: Path, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    files = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        dirs[:] = [name for name in dirs if name != "__pycache__" and (parent / name).relative_to(root).as_posix() not in exclude]
        need(not any((parent / name).is_symlink() for name in dirs), "Directory symlink in evidence or staged source")
        for name in sorted(names):
            path = parent / name
            if name == "SHA256SUMS" or path.relative_to(root).as_posix() in exclude:
                continue
            need(not path.is_symlink() and path.is_file(), "Nonregular file in evidence or staged source: " + str(path))
            files[path.relative_to(root).as_posix()] = digest(path)
    return dict(sorted(files.items()))


def collect(run_dir: Path, output: Path) -> None:
    files = inventory(run_dir, CACHE_PATHS)
    sums = "".join(f"{value}  {path}\n" for path, value in files.items())
    (run_dir / "SHA256SUMS").write_text(sums)
    output.mkdir(parents=True, exist_ok=True)
    for name in files:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run_dir / name, target)
    (output / "SHA256SUMS").write_text(sums)
    need(inventory(output) == files, "Artifact collection hash mismatch")


def stage_package(source: Path, destination: Path) -> dict[str, str]:
    selected = {p.relative_to(source).as_posix(): p for p in [*source.glob("*.py"), *(source / "evaluator").glob("*.py")]}
    shared_power = source.parents[1] / "utils" / "aggregate_power.py"
    if shared_power.is_file():
        selected["utils/aggregate_power.py"] = shared_power
    expected = {name: digest(path) for name, path in selected.items()}
    if destination.exists():
        need(inventory(destination) == expected, "Staged source differs from this GitHub commit")
    else:
        destination.mkdir(parents=True)
        for name, path in selected.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    return expected


def step_argv(config: dict, receipt: dict, record: dict, run_dir: Path, package: Path) -> list[str]:
    request = config["resources"]
    return ["srun", "--jobid=" + receipt["identity"]["JobId"], "--nodelist=" + record["NodeList"],
            "--exclusive", "--exact", "--nodes=1", "--ntasks=1", "--immediate=30", "--kill-on-bad-exit=1",
            "--cpus-per-task=" + str(request["cpus"]), "--cpu-bind=verbose,cores",
            "--gpus-per-task=" + str(request["gpus"]), "--gpu-bind=verbose,per_task:" + str(request["gpus"]),
            "--mem=" + str(request["memory_gb"]) + "G", "--time=" + str(request["minutes"] - 5),
            "--chdir=" + str(run_dir), "--export=PATH,PYTHONDONTWRITEBYTECODE,TZ,LC_ALL",
            "python3", str(package / "ci.py"), "--enter", str(run_dir)]


def run_step(argv: list[str], log: Path, seconds: float) -> int:
    process = None
    def cancelled(signum, frame):
        raise InterruptedError("CI canceled the owned Slurm step")
    old = {sig: signal.signal(sig, cancelled) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with log.open("w") as stream:
            process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, env=environment(), start_new_session=True)
            return process.wait(timeout=seconds)
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


@contextmanager
def task_lock(path: Path) -> Iterator[None]:
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def prepared_spec(config: dict) -> dict:
    runtime = config["runtime"]
    need(Path(runtime["rootfs"]).is_dir() and Path(runtime["ready_marker"]).is_file(), "Existing persistent runtime or readiness record missing; no allocation requested")
    need(digest(runtime["entry"]) == runtime["entry_sha256"], "Persistent entry script changed")
    need(digest(config["spec"]["path"]) == config["spec"]["sha256"], "Prepared GPU specification changed")
    spec = read(config["spec"]["path"])
    # Slurm assigns physical devices later. Validate the rest without a GPU call.
    spec["gpu_uuids"] = [f"GPU-00000000-0000-0000-0000-{i:012d}" for i in range(config["resources"]["gpus"])]
    from evaluator.mvp_gpu_job import validate_gpu_job
    spec = validate_gpu_job(spec)
    if config["mode"] == "serving-smoke":
        from evaluator.mvp_serving_smoke import validate_spec
        validate_spec(spec)
    need(spec["authorization"]["compute_approved"] and spec["authorization"]["model_license_reviewed"]
         and spec["authorization"]["approval_reference"].strip(), "Prepared spec must record compute and model approval")
    need(spec["limits"]["job_seconds"] + 600 <= config["resources"]["minutes"] * 60,
         "Allocation must leave ten minutes beyond supervisor budget for step entry/report/cleanup")
    # Cheap inventory checks happen before salloc; full pinned source/weight
    # hashing remains in the existing supervisor immediately before execution.
    for role in ("baseline", "candidate"):
        need(host_path(config, spec[role]["source"]).is_dir(), "Prepared runtime source missing: " + role)
    model = host_path(config, spec["model"]["path"])
    for item in spec["model"]["files"]:
        path = model / item["path"]
        need(path.is_file() and path.stat().st_size == item["size_bytes"],
             "Prepared model file missing or wrong size; stage weights before allocating: " + item["path"])
    return spec


def launch(config: dict, output: Path) -> int:
    config = validate_config(config)
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "")
    sha = os.environ.get("H3_SOURCE_SHA", "")
    need(re.fullmatch(r"[0-9]+", run_id) and re.fullmatch(r"[0-9]+", attempt), "GitHub run ID and attempt required")
    need(re.fullmatch(r"[0-9a-f]{40}", sha), "Exact H3_SOURCE_SHA required")
    source = Path(__file__).resolve().parent
    need(command(["git", "-C", str(source), "rev-parse", "HEAD"]).strip() == sha, "Checkout differs from admitted source SHA")
    need(not command(["git", "-C", str(source), "status", "--porcelain"]).strip(), "Harness checkout must be clean and committed")
    spec = prepared_spec(config)
    workspace = Path(config["workspace"]["host"])
    need(workspace.is_dir(), "Persistent workspace missing")
    results = workspace / "results" / config["task_id"]
    control = workspace / "campaigns" / config["task_id"] / "control"
    results.mkdir(parents=True, exist_ok=True)
    control.mkdir(parents=True, exist_ok=True)
    run_dir = results / f"github-{run_id}-{attempt}"
    reserved_gpus = config["resources"]["gpus"] if config["mode"] == "serving-smoke" else 8
    state = {"schema_version": 1, "task_id": config["task_id"], "run_id": run_id, "run_attempt": attempt,
             "source_sha": sha, "started_at": now(), "phase": "preparing", "mode": config["mode"],
             "ci_accepted": False, "release_qualified": False, "persistent_output": str(run_dir),
             "excluded_cache_paths": list(CACHE_PATHS),
             "ci": {"repository": os.environ.get("GITHUB_REPOSITORY"),
                    "workflow_ref": os.environ.get("GITHUB_WORKFLOW_REF"),
                    "workflow_sha": os.environ.get("GITHUB_WORKFLOW_SHA"),
                    "actor": os.environ.get("GITHUB_ACTOR"),
                    "triggering_actor": os.environ.get("GITHUB_TRIGGERING_ACTOR"),
                    "run_url": f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/{run_id}"},
             "resources": {"requested": config["resources"], "new_allocation_gpus": reserved_gpus,
                           "new_allocation_gpu_hours_cap": reserved_gpus * config["resources"]["minutes"] / 60}}
    with task_lock(control / "ci.lock"):
        run_dir.mkdir(exist_ok=False)
        write(run_dir / "ci.json", state)
        receipt = None
        reused = False
        code = 2
        try:
            shutil.copyfile(config["runtime"]["entry"], run_dir / "runtime-entry.sh")
            shutil.copyfile(config["runtime"]["ready_marker"], run_dir / "runtime-readiness.record")
            package = workspace / "campaigns" / config["task_id"] / "packages" / sha
            package_files = stage_package(source, package)
            decision = recover(config, results)
            write(run_dir / "recovery.json", decision)
            need(decision["action"] != "wait", "A task-owned allocation is pending or suspended; no duplicate submitted")
            if decision["action"] == "reuse":
                receipt, reused = decision["receipt"], True
                write(run_dir / "allocation.json", receipt)
            else:
                receipt = allocate(config, run_dir)
            record = job_record(receipt["identity"]["JobId"])
            verify_identity(receipt, record, config["task_id"])
            need(record["JobState"] == "RUNNING", "Owned allocation is not RUNNING")
            need(capacity(record, config["resources"]) is None, "Allocation cannot serve bounded step")
            if config["mode"] == "serving-smoke":
                tres = dict(item.split("=", 1) for item in record["AllocTRES"].split(","))
                need(int(tres.get("gres/gpu", "0")) == reserved_gpus, "Serving allocation GPU count exceeds the declared budget")
            need(digest(config["runtime"]["entry"]) == config["runtime"]["entry_sha256"], "Entry changed after preflight")
            state.update(phase="starting", allocation_reused=reused, allocation=receipt, slurm_job=record)
            write(run_dir / "ci.json", state)
            # Any concurrent active measured step means cooperative sharing.
            active_steps = command(["squeue", "--steps", "--noheader", "--jobs=" + record["JobId"], "--format=%i|%N"])
            write(run_dir / "context.json", {"config": config, "spec": spec, "allocation": receipt,
                "node": record["NodeList"], "active_steps": active_steps, "source_sha": sha,
                "exclusive_node": record.get("OverSubscribe") == "NO" and "gres/gpu=8" in record.get("AllocTRES", "").split(","),
                "package_files": package_files, "run_id": f"github-{run_id}-{attempt}"})
            argv = step_argv(config, receipt, record, run_dir, package)
            write(run_dir / "step-command.json", argv)
            code = run_step(argv, run_dir / "srun.log", config["resources"]["minutes"] * 60 - 300 + 60)
            inside = read(run_dir / "step-result.json")
            need(code == inside["exit_code"], "Slurm exit and workload receipt differ")
            state.update(phase="complete" if code == 0 else "failed", **inside)
        except (Exception, KeyboardInterrupt) as error:
            state.update(phase="failed", error=str(error), exit_code=2)
            code = 2
        finally:
            if receipt is None and (run_dir / "allocation.json").is_file():
                receipt = read(run_dir / "allocation.json")
            try:
                if receipt is not None:
                    state["step_cleanup"] = drain_step(receipt, config["task_id"], run_dir)
            except Exception as error:
                state.update(phase="failed", step_cleanup_error=str(error), ci_accepted=False)
                code = 2
            try:
                if receipt is not None and not reused:
                    state["allocation_cleanup"] = stop_allocation(receipt, config["task_id"])
                elif reused:
                    state["allocation_cleanup"] = {"status": "retained", "reason": "attached step does not own the parent allocation"}
            except Exception as error:
                state.update(phase="failed", cleanup_error=str(error), ci_accepted=False)
                code = 2
            state.update(finished_at=now(), exit_code=code)
            write(run_dir / "ci.json", state)
            links = ("ci.json", "runtime-entry.sh", "runtime-readiness.record", "allocation.json", "recovery.json", "binding.json", "context.json", "step-result.json",
                     "gpu/spec.json", "gpu/gpu-job.json", "gpu/baseline/run.json", "gpu/candidate/run.json",
                     "gpu/comparison.json", "report/index.html")
            if config["mode"] == "serving-smoke":
                links += ("serving-smoke.json",)
                links += tuple(f"gpu/c{concurrency}/{path}" for concurrency in (1, 2, 4)
                               for path in ("spec.json", "gpu-job.json", "baseline/run.json", "power.json"))
            write(run_dir / "manifest.json", {"schema_version": 1, "task_id": config["task_id"],
                "git_commit": sha, "ci": state["ci"], "run_id": run_id, "run_attempt": attempt,
                "slurm_allocation": receipt, "runtime": config["runtime"], "prepared_spec": config["spec"],
                "workload_plan": spec.get("plan"), "mode": config["mode"], "resources": state["resources"], "exit_code": code,
                "evidence": {path: digest(run_dir / path) for path in links if (run_dir / path).is_file()},
                "artifact_checksums": "SHA256SUMS", "excluded_persistent_caches": list(CACHE_PATHS)})
            collect(run_dir, output)
    return code


def enter(run_dir: Path) -> None:
    """First command in the allocated step: retain identity before Enroot/CUDA."""
    context = read(run_dir / "context.json")
    config = validate_config(context["config"])
    job, step = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_STEP_ID", "")
    need(job == context["allocation"]["identity"]["JobId"] and re.fullmatch(r"[0-9]+", step)
         and os.environ.get("SLURMD_NODENAME") == context["node"], "Wrong Slurm step assignment")
    write(run_dir / "binding.json", {"job_id": job, "step_id": step, "node": context["node"],
          "cpu_affinity": sorted(os.sched_getaffinity(0)), "observed_at": now(), "phase": "entering_runtime"})
    need(digest(config["runtime"]["entry"]) == config["runtime"]["entry_sha256"], "Entry changed on compute node")
    argv = ["/bin/bash", config["runtime"]["entry"], config["runtime"]["python"],
            str(mapped(config, Path(__file__).parent) / "ci.py"), "--inside", str(mapped(config, run_dir))]
    os.execv(argv[0], argv)


def workload_complete(verified: dict) -> bool:
    runs, comparison = verified["runs"], verified["comparison"]
    return (all(run["summary"]["valid"] == run["summary"]["scheduled"] > 0 for run in runs.values())
            and bool(comparison["slots"])
            and all(slot[role]["status"] == "succeeded" and (slot[role].get("media") or {}).get("valid") is True
                    and not slot[role].get("analysis_error")
                    for slot in comparison["slots"] for role in ("baseline", "candidate"))
            and all(check["status"] == "pass" for check in comparison["checks"]
                    if check["name"] in {"baseline.warmup", "candidate.warmup"}))


def smoke_exit(verified: dict, receipt: dict, mode: str) -> int:
    if mode == "regression":
        if receipt.get("regression_status") == "fail":
            return 1
        return 0 if receipt.get("ci_accepted") is True else 2
    # Raw verification establishes identity, timing and cleanup. A smoke tests
    # execution and fresh media validity independently of regression thresholds.
    return 0 if workload_complete(verified) else 1


def inside(run_dir: Path) -> int:
    context = read(run_dir / "context.json")
    config = validate_config(context["config"])
    expected = context["allocation"]["identity"]["JobId"]
    step = os.environ.get("SLURM_STEP_ID", "")
    result = {"exit_code": 2, "measurement_status": "incomplete", "regression_status": "inconclusive", "ci_accepted": False, "release_qualified": False}
    try:
        need(os.environ.get("SLURM_JOB_ID") == expected and re.fullmatch(r"[0-9]+", step)
             and os.environ.get("SLURMD_NODENAME") == context["node"]
             and os.environ.get("SLURM_PROCID") == "0" and os.environ.get("SLURM_NTASKS") == "1",
             "Payload is not the exact single-node Slurm task")
        need(inventory(Path(__file__).parent) == context["package_files"], "Staged harness bytes changed")
        devices = cuda_devices()
        assigned = os.environ.get("H3_ASSIGNED_GPU_UUIDS", "").split(",")
        need(set(devices) == set(assigned), "CUDA-visible UUIDs differ from the Slurm global GPU assignment")
        cpus = sorted(os.sched_getaffinity(0))
        binding = read(run_dir / "binding.json")
        need(binding["job_id"] == expected and binding["step_id"] == step and binding["cpu_affinity"] == cpus,
             "Container changed its assigned step identity or CPU binding")
        need(len(devices) == config["resources"]["gpus"] and len(set(devices)) == len(devices), "Slurm step CUDA device count/UUIDs differ")
        need(len(cpus) >= config["resources"]["cpus"], "Bound step CPU set is too small")
        write(run_dir / "binding.json", {"job_id": expected, "step_id": step, "node": context["node"],
              "gpu_uuids": devices, "cpu_affinity": cpus,
              "slurm": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "H3_ORIGINAL_CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_CPU_BIND", "SLURM_CPUS_PER_TASK")}, "observed_at": now()})
        spec = context["spec"]
        spec["gpu_uuids"], spec["job_id"] = devices, context["run_id"]
        active = [line for line in context["active_steps"].splitlines() if not re.match(r"[0-9]+\.(batch|extern)\|", line)]
        spec["allocation"] = {"mode": "dedicated_ci" if not active and context["exclusive_node"] else "cooperative_shared", "label": f"Slurm {expected}.{step} on {context['node']}"}
        if config["mode"] == "serving-smoke":
            from evaluator.mvp_serving_smoke import run_matrix
            matrix = run_matrix(spec, run_dir)
            complete = matrix["status"] == "complete"
            result.update(exit_code=0 if complete else 1, smoke_completed=complete,
                          measurement_status="complete" if complete else "incomplete",
                          serving_summary="serving-smoke.json")
        else:
            from evaluator.mvp_gpu_job import run_gpu_job
            from evaluator.mvp_gpu_evidence import verify_measurement_job
            receipt = run_gpu_job(spec, run_dir / "gpu")
            result.update({key: receipt[key] for key in ("measurement_status", "regression_status", "ci_accepted", "release_qualified")})
            verified = verify_measurement_job(run_dir / "gpu", deadline=time.monotonic() + 120)
            result["exit_code"] = smoke_exit(verified, receipt, config["mode"])
            result["smoke_completed"] = workload_complete(verified)
    except (Exception, KeyboardInterrupt) as error:
        result.update(exit_code=2, error=str(error), smoke_completed=False, ci_accepted=False)
    finally:
        if (run_dir / "gpu" / "gpu-job.json").is_file():
            try:
                from evaluator.mvp_gpu_report import write_gpu_report
                write_gpu_report(run_dir / "gpu", run_dir / "report")
            except Exception as error:
                result.update(exit_code=2, report_error=str(error), ci_accepted=False)
        write(run_dir / "step-result.json", result)
    return result["exit_code"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inside", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--enter", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.enter:
        enter(args.enter)
        return 2
    if args.inside:
        return inside(args.inside)
    need(args.config is not None and args.output is not None, "--config and --output required")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        return launch(read(args.config), args.output)
    except (Exception, KeyboardInterrupt) as error:
        write(args.output / "adapter-error.json", {"error": str(error), "exit_code": 2, "recorded_at": now(), "ci_accepted": False})
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
