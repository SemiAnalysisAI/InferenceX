#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import threading
import time
import traceback
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "operatorx_batch_base_campaign", HERE / "operatorx_campaign.py"
)
c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c)
HEALTH_QUERIES = {"health", "current-fault-event", "ecc", "ecc-enable", "pcie-err"}
EXECUTION_MODE = "persistent_worker_process"


def worker_script(args):
    return Path(
        getattr(args, "worker_script", None) or HERE / "operatorx_ascend_worker.py"
    ).resolve()


def preflight_subprocess(args):
    """Do not initialize ACL in the persistent process before torch_npu.

    CANN can retain the first aclInit configuration even after aclFinalize. The
    identity helper's aclInit(None) must therefore occur in a disposable child.
    A report file keeps native ACL stdout chatter separate from structured data.
    """
    settings = {
        key: getattr(args, key)
        for key in ("board_id", "chip_id", "device", "expected_name", "expect_pci")
    }
    with tempfile.TemporaryDirectory(prefix="operatorx-preflight-") as temporary:
        report_path = Path(temporary) / "report.json"
        capture = c.command(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "_preflight-child",
                str(report_path),
                c.canonical(settings),
            ],
            timeout=180,
        )
        if capture["returncode"] != 0 or not report_path.is_file():
            raise RuntimeError("Preflight subprocess failed: " + c.canonical(capture))
        report = json.loads(report_path.read_text())
    before = report["before"]
    before["preflight_subprocess"] = {
        "pid": report["pid"],
        "returncode": capture["returncode"],
        "elapsed_s": capture["elapsed_s"],
        "stdout": capture.get("stdout", ""),
        "stderr": capture.get("stderr", ""),
    }
    args.acl_runtime_mapping = report["acl_runtime_mapping"]
    # Loading the helper is pure; its ACL mapping function is never called here.
    helper = c.load_module(c.idle_helper_path(), "operatorx_batch_pure_health_helper")
    return before, helper, report["mapped"]


def preflight_child(report_path, settings_json):
    args = argparse.Namespace(**json.loads(settings_json))
    before, _helper, mapped = c.preflight(args)
    c.write_json(
        Path(report_path),
        {
            "before": before,
            "mapped": mapped,
            "pid": os.getpid(),
            "acl_runtime_mapping": getattr(args, "acl_runtime_mapping", None),
        },
    )
    return 0


def latest_records(connection):
    latest = {}
    for case_id, record in connection.execute(
        "SELECT case_id,record FROM results ORDER BY seq"
    ):
        latest[case_id] = json.loads(record)
    return latest


def completed_case_ids(connection):
    # A new running/failed attempt invalidates an older success for resume.
    return {
        case_id
        for case_id, record in latest_records(connection).items()
        if record["status"] == "ok"
    }


def require_smoke(plan, phase, completed):
    if phase != "smoke" and not set(plan["smoke_case_ids"]).issubset(completed):
        raise ValueError(
            "Run every smoke case successfully in this same batch run/configuration before the full phases"
        )


def process_snapshot(args, helper, mapped, own_pids):
    capture = c.command(["npu-smi", "info"])
    if capture["returncode"] != 0:
        raise RuntimeError("Cannot verify NPU process status")
    parsed = helper.status_rows(capture["stdout"], args.board_id)
    keywords = {}
    if getattr(args, "acl_runtime_mapping", None) is not None:
        keywords["acl_runtime_mapping"] = args.acl_runtime_mapping
    selected = helper.validate_identity(
        parsed,
        mapped,
        args.board_id,
        args.chip_id,
        args.device,
        args.expect_pci,
        **keywords,
    )
    foreign = [row for row in parsed["processes"] if row["pid"] not in own_pids]
    capture.update(parsed=parsed, selected=selected, own_pids=sorted(own_pids))
    if foreign:
        raise RuntimeError("Foreign NPU process detected: " + c.canonical(foreign))
    # Utilization can report the preceding case after its synchronization. The
    # process check, not stale utilization alone, distinguishes foreign work.
    capture["nonzero_utilization_alone_rejected"] = False
    return capture


@contextlib.contextmanager
def capture_native_output(folder):
    """Keep Python and native runtime diagnostics in durable per-attempt files."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    try:
        with (
            (folder / "stdout.txt").open("a") as out,
            (folder / "stderr.txt").open("a") as err,
        ):
            os.dup2(out.fileno(), 1)
            os.dup2(err.fileno(), 2)
            yield
            sys.stdout.flush()
            sys.stderr.flush()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])


class CaseWatchdog:
    """Exit 124 on expiry; the committed running marker remains the latest row."""

    def __init__(self, seconds, evidence_path, case_id):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Watchdog timeout must be finite and positive")
        self.seconds = seconds
        self.evidence_path = Path(evidence_path)
        self.case_id = case_id
        self.lock = threading.Lock()
        self.active = False
        self.timer = None

    def _expire_locked(self):
        try:
            c.write_json(
                self.evidence_path,
                {
                    "status": "timeout",
                    "case_id": self.case_id,
                    "utc": c.utc(),
                    "exit_code": 124,
                    "timeout_seconds": self.seconds,
                    "metrics": None,
                },
            )
        finally:
            # No terminal success can be committed after this hard process exit.
            os._exit(124)

    def _fire(self):
        with self.lock:
            if self.active:
                self._expire_locked()

    def start(self):
        with self.lock:
            self.active = True
            self.deadline = time.monotonic() + self.seconds
            self.timer = threading.Timer(self.seconds, self._fire)
            self.timer.daemon = True
            self.timer.start()

    def disarm(self):
        # Serialize against the callback and check the deadline ourselves, so a
        # delayed watchdog thread cannot allow an expired execution to become ok.
        with self.lock:
            if self.active and time.monotonic() >= self.deadline:
                self._expire_locked()
            self.active = False
            if self.timer is not None:
                self.timer.cancel()
        if self.timer is not None:
            self.timer.join(timeout=1)


def _runtime_versions(torch, torch_npu):
    packages = {}
    for name in ("torch", "torch-npu", "numpy"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    cann_home = os.environ.get("ASCEND_HOME_PATH")
    cann_versions = {}
    if cann_home:
        for name in (
            "version.cfg",
            "version.info",
            "compiler/version.info",
            "runtime/version.info",
        ):
            path = Path(cann_home) / name
            if path.is_file():
                cann_versions[name] = path.read_text(errors="replace")
    return {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "packages": packages,
        "torch_imported_version": str(torch.__version__),
        "torch_npu_imported_version": str(torch_npu.__version__),
        "cann_home": str(Path(cann_home).resolve()) if cann_home else None,
        "cann_versions": cann_versions,
    }


def load_runtime(cpu_threads):
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    torch = importlib.import_module("torch")
    torch_npu = importlib.import_module("torch_npu")
    torch.set_num_threads(cpu_threads)
    return torch, torch_npu


def make_fingerprint(args, before, config, torch, torch_npu):
    return {
        "execution_mode": EXECUTION_MODE,
        "timing_helper_sha256": c.digest(HERE / "operatorx_queued_timing.py"),
        "reference_helper_sha256": c.digest(HERE / "operatorx_ascend_worker.py"),
        "plan_sha256": c.digest(args.plan),
        "batch_controller_sha256": c.digest(__file__),
        "campaign_helper_sha256": c.digest(HERE / "operatorx_campaign.py"),
        "worker_sha256": c.digest(worker_script(args)),
        "worker_script": str(worker_script(args)),
        "idle_helper_sha256": c.digest(c.idle_helper_path()),
        "hardware_identity": before["identity_sha256"],
        "config": config,
        "runtime": _runtime_versions(torch, torch_npu),
    }


def cleanup_runtime(torch):
    gc.collect()
    torch.npu.synchronize()
    torch.npu.empty_cache()
    torch.npu.synchronize()


def _attempt_folder(run_dir, case_id):
    stem = hashlib.sha256(case_id.encode()).hexdigest()[:24]
    attempts = run_dir / "attempts"
    attempts.mkdir(exist_ok=True)
    number = 1
    while True:
        folder = attempts / f"{stem}-{number:04d}"
        try:
            folder.mkdir()
            return folder, number
        except FileExistsError:
            number += 1


def _invalidate(outcome, status, reason):
    if outcome.get("metrics") is not None:
        outcome["diagnostic_metrics_not_accepted"] = outcome["metrics"]
    outcome.update(status=status, reason=reason, metrics=None)


def execute_case(
    connection, case, config, args, worker, torch, helper, mapped, worker_sha
):
    own_pids = {os.getpid()}
    before = process_snapshot(args, helper, mapped, own_pids)
    folder, attempt = _attempt_folder(args.run_dir, case["case_id"])
    c.write_json(folder / "case.json", case)
    c.write_json(folder / "process-before.json", before)
    begun = time.monotonic()
    marker = {
        "status": "running",
        "metrics": None,
        "case_id": case["case_id"],
        "opType": case["opType"],
        "args": case["args"],
        "sources": case.get("sources", []),
        "phase": args.phase,
        "attempt": attempt,
        "started_at": c.utc(),
        "execution_mode": EXECUTION_MODE,
        "controller_pid": os.getpid(),
        "attempt_directory": str(folder.relative_to(args.run_dir)),
        "timeout_seconds": config["timeout_seconds"],
    }
    c.record_result(connection, case["case_id"], marker)
    watchdog = CaseWatchdog(
        config["timeout_seconds"], folder / "watchdog-timeout.json", case["case_id"]
    )
    state = {"lock": threading.Lock(), "own_pids": own_pids, "issues": []}
    stop = threading.Event()
    telemetry = folder / "telemetry.jsonl"

    def monitored():
        try:
            c.monitor(stop, telemetry, args.board_id, helper, state)
        except BaseException as exc:
            with state["lock"]:
                state["issues"].append(
                    {
                        "process_check_error": f"Monitor terminated: {type(exc).__name__}: {exc}"
                    }
                )

    monitor_thread = threading.Thread(target=monitored, daemon=True)
    outcome = {"status": "error", "metrics": None}
    interruption = None
    watchdog.start()
    monitor_thread.start()
    try:
        with capture_native_output(folder):
            try:
                kwargs = {
                    key: config[key]
                    for key in (
                        "memory_budget_bytes",
                        "device",
                        "samples",
                        "warmups",
                        "warmup_ms",
                        "seed",
                        "cache_write_bytes",
                        "workspace_reserve_bytes",
                    )
                }
                raw = worker.run_case(case, **kwargs)
                # Preserve even malformed/failed original output independently of
                # the acceptance record. No benchmark JSON is constructed here.
                (folder / "worker-return.json").write_text(
                    json.dumps(raw, indent=2, allow_nan=True, default=repr) + "\n"
                )
                try:
                    c.validate_worker_result(raw, case, config, worker_sha)
                    outcome = raw
                except (ValueError, TypeError, KeyError) as exc:
                    outcome = {
                        "status": "error",
                        "metrics": None,
                        "reason": "Invalid worker output: " + str(exc),
                    }
            except BaseException as exc:
                traceback.print_exc()
                outcome = {
                    "status": "interrupted"
                    if isinstance(exc, (KeyboardInterrupt, SystemExit))
                    else "error",
                    "metrics": None,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    interruption = exc
            finally:
                try:
                    cleanup_runtime(torch)
                except BaseException as exc:
                    traceback.print_exc()
                    _invalidate(
                        outcome,
                        "error",
                        f"NPU cleanup failed: {type(exc).__name__}: {exc}",
                    )
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        interruption = exc
    finally:
        stop.set()
        monitor_thread.join(timeout=15)
        if monitor_thread.is_alive():
            _invalidate(outcome, "error", "Cannot drain per-case process monitor")
        with state["lock"]:
            issues = list(state["issues"])
        if not telemetry.exists() or not telemetry.stat().st_size:
            issues.append(
                {"process_check_error": "No under-load monitor sample was captured"}
            )
        if issues:
            outcome["process_monitor_issues"] = issues
            if outcome.get("status") == "ok":
                _invalidate(
                    outcome,
                    "contaminated",
                    "Foreign NPU work or unverified process status observed during this case",
                )
        if outcome.get("status") != "ok" and outcome.get("metrics") is not None:
            _invalidate(
                outcome, outcome["status"], outcome.get("reason", "Unaccepted result")
            )
        # Disarm only after synchronization, cleanup, validation and monitor drain.
        # Expiry exits immediately, leaving the running marker and raw artifacts.
        watchdog.disarm()
    record = {
        **marker,
        **outcome,
        "status": outcome["status"],
        "finished_at": c.utc(),
        "elapsed_s": time.monotonic() - begun,
        "process_before": before,
        "raw_artifacts": {
            "worker_return": "worker-return.json",
            "stdout": "stdout.txt",
            "stderr": "stderr.txt",
            "telemetry": "telemetry.jsonl",
        },
    }
    c.write_json(folder / "accepted-result.json", record)
    c.record_result(connection, case["case_id"], record)
    if interruption is not None:
        raise interruption
    return record


def run_locked(args):
    plan = json.loads(args.plan.read_text())
    ids = [case["case_id"] for case in plan["cases"]]
    if len(ids) != len(set(ids)) or not set(plan["smoke_case_ids"]).issubset(ids):
        raise ValueError("Invalid duplicate cases or missing smoke references")
    names = {
        "unquantized": "operatorx_ascend_worker.py",
        "fp8": "operatorx_fp8_block_worker.py",
    }
    if (
        plan.get("kind") not in names
        or worker_script(args) != (HERE / names[plan["kind"]]).resolve()
    ):
        raise ValueError("Select the packaged worker matching the plan kind")
    dependencies = plan.get("dependency_hashes", {})
    if set(dependencies) != {
        "operatorx_queued_timing.py",
        "operatorx_ascend_worker.py",
    }:
        raise ValueError(
            "Plan lacks the timing/reference dependency fingerprint; prepare a new plan"
        )
    if any(
        c.digest(HERE / name) != expected for name, expected in dependencies.items()
    ):
        raise ValueError("Timing/reference dependency changed after planning")
    if plan["kind"] == "fp8":
        from fp8_plan import verify_plan

        verify_plan(plan)
    selected = c.select_cases(plan, args.phase)
    if not selected or any(
        case.get("phase") not in ("new", "remeasure") for case in selected
    ):
        raise ValueError(
            "Selected phase contains no ready cases or includes unprepared cases"
        )
    worker_path = worker_script(args)
    if plan.get("worker_sha256") != c.digest(worker_path):
        raise ValueError("Worker changed after planning; rebuild plan")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    before, helper, mapped = preflight_subprocess(args)
    if (
        len(before.get("idle_observations", [])) != 2
        or set(before.get("diagnostic_gate") or {}) != HEALTH_QUERIES
    ):
        raise ValueError(
            "Two idle checks and the complete health/fault/ECC/PCIe gate are required before torch import"
        )
    if (
        "fp8_l0a" in plan
        and plan["fp8_l0a"]["hardware_identity"] != before["identity_sha256"]
    ):
        raise ValueError("FP8 guard was recorded on different hardware")
    session = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    c.write_json(args.run_dir / f"preflight-{session}.json", before)
    config = {
        "warmups": args.warmups,
        "warmup_ms": args.warmup_ms,
        "samples": args.samples,
        "seed": args.seed,
        "device": args.device,
        "timeout_seconds": args.timeout_seconds,
        "cpu_threads": args.cpu_threads,
        **plan["config"],
    }
    with capture_native_output(args.run_dir / f"runtime-{session}"):
        torch, torch_npu = load_runtime(args.cpu_threads)
    fingerprint = make_fingerprint(args, before, config, torch, torch_npu)
    connection = c.open_journal(args.run_dir / "results.sqlite3", fingerprint)
    failed = False
    try:
        completed = completed_case_ids(connection)
        require_smoke(plan, args.phase, completed)
        worker = c.load_module(worker_path, "operatorx_persistent_worker")
        for case in selected:
            if case["case_id"] in completed:
                continue
            record = execute_case(
                connection,
                case,
                config,
                args,
                worker,
                torch,
                helper,
                mapped,
                fingerprint["worker_sha256"],
            )
            print(
                c.canonical(
                    {
                        "case_id": case["case_id"],
                        "status": record["status"],
                        "elapsed_s": record["elapsed_s"],
                        "execution_mode": EXECUTION_MODE,
                    }
                ),
                flush=True,
            )
            if record["status"] == "ok":
                completed.add(case["case_id"])
            elif record["status"] not in ("skipped_memory", "unsupported"):
                failed = True
                break
    finally:
        summary = c.export_journal(connection, args.run_dir / "results.json")
        connection.close()
        # Own context remains until this process exits; this capture is therefore
        # explicitly process status, not a claim that the board is idle again.
        c.write_json(
            args.run_dir / f"process-status-after-{session}.json",
            c.command(["npu-smi", "info"]),
        )
    print(
        c.canonical(
            {
                "counts": summary["counts"],
                "stopped_on_failure": failed,
                "execution_mode": EXECUTION_MODE,
            }
        ),
        flush=True,
    )
    return 1 if failed else 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 3 and argv[0] == "_preflight-child":
        return preflight_child(argv[1], argv[2])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for flag in ("plan", "run-dir"):
        run.add_argument("--" + flag, type=Path, required=True)
    run.add_argument(
        "--worker-script",
        type=Path,
        default=HERE / "operatorx_ascend_worker.py",
        help="Worker implementing run_case with the same validated result contract",
    )
    run.add_argument(
        "--phase", choices=("smoke", "new", "remeasure", "all-ready"), required=True
    )
    for flag in (
        "device",
        "board-id",
        "chip-id",
        "warmups",
        "samples",
        "seed",
        "timeout-seconds",
        "cpu-threads",
    ):
        run.add_argument("--" + flag, type=int, required=True)
    run.add_argument("--warmup-ms", type=float, required=True)
    run.add_argument("--expected-name", required=True)
    run.add_argument("--expect-pci")
    args = parser.parse_args(argv)
    if (
        min(args.device, args.board_id, args.chip_id, args.seed) < 0
        or args.seed >= 2**63
    ):
        parser.error("Device identifiers/seed are out of range")
    if (
        not 1 <= args.warmups <= 10000
        or not 10 <= args.samples <= 10000
        or not 0 <= args.warmup_ms <= 10000
    ):
        parser.error("Invalid warmup/sample configuration")
    if min(args.timeout_seconds, args.cpu_threads) <= 0 or args.timeout_seconds > 300:
        parser.error("Timeout must be in [1,300] seconds and CPU threads positive")
    if os.environ.get("LD_PRELOAD") or os.environ.get("TASK_QUEUE_ENABLE") not in (
        None,
        "1",
    ):
        raise ValueError(
            "Timing must not inherit the preflight preload or disabled task queue"
        )
    if "torch" in sys.modules or "torch_npu" in sys.modules:
        raise RuntimeError(
            "Launch this controller in a fresh process so preflight precedes torch import"
        )
    with open(
        Path("/tmp") / f"operatorx-ascend-board-{args.board_id}.lock", "a"
    ) as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Another OperatorX campaign holds this board lock"
            ) from None
        return run_locked(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
