#!/usr/bin/env python3
"""Bound cleanup of the single task-owned GLM-5.2 B200 C48 recovery job."""

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

GRACE_SECONDS = 300
CANCEL_WAIT_SECONDS = 120
POLL_SECONDS = 5
INSPECTION_FAILURE_LIMIT = 3
TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "BOOT_FAIL",
    "PREEMPTED",
    "DEADLINE",
    "REVOKED",
}
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# These are terminal control-flow messages in the pinned producer/AIPerf.
# A scored request warning, timeout, or error alone must never arm cleanup.
ABORT = re.compile(
    r"Terminal warmup failure for trace "
    r"|Run aborted \("
    r"|Benchmark failed with exit code [1-9][0-9]*"
    r"|Critical process '.+' exited with code -?[1-9][0-9]*"
    r"|\bINFO\]?\s+Cleanup(?:\s|$)"
    r"|Cleaning up [0-9]+ processes \([0-9]+ running\)"
)


def command(*args):
    result = subprocess.run(
        args, capture_output=True, text=True, timeout=15, check=False
    )
    if result.returncode:
        raise RuntimeError(
            f"{args[0]} failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def owned_job(job_id, log_path, job_name):
    raw = command("scontrol", "show", "job", "-o", job_id)
    fields = dict(item.split("=", 1) for item in shlex.split(raw) if "=" in item)
    uid = re.fullmatch(r"[^()]+\(([0-9]+)\)", fields.get("UserId", ""))
    if not (
        fields.get("JobId") == job_id
        and uid is not None
        and int(uid[1]) == os.getuid()
        and fields.get("JobName") == job_name
        and Path(fields.get("StdOut", "")).resolve() == log_path
    ):
        raise ValueError(
            "Slurm job/user/name/output ownership mismatch; refusing cancellation"
        )
    if not fields.get("JobState"):
        raise RuntimeError("Slurm inspection omitted JobState")
    return fields, raw


def watch(job_id, log_path, job_name):
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("Expected one numeric C48 Slurm job ID")
    log_path = Path(log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = log_path.parent / "c48-cleanup-watcher.json"
    receipt = {
        "job_id": job_id,
        "job_name": job_name,
        "uid": os.getuid(),
        "log": str(log_path),
        "grace_seconds": GRACE_SECONDS,
        "cancel_wait_seconds": CANCEL_WAIT_SECONDS,
        "status": "watching",
        "events": [],
    }

    def record(event, **details):
        receipt["events"].append(
            {
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": event,
                **details,
            }
        )
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"C48 cleanup watcher: {event}", flush=True)

    record("started")
    positions, tails = {}, {}
    armed_at = cancelled_at = terminal_at = None
    inspection_failures = 0
    verified_owner = False
    while True:
        now = time.monotonic()
        for path in (log_path, log_path.parent / "benchmark.out"):
            if not path.exists():
                continue
            with path.open(errors="replace") as handle:
                handle.seek(positions.get(path, 0))
                text = handle.read()
                positions[path] = handle.tell()
            if path == log_path and text:
                print(text, end="", flush=True)
            text = ANSI.sub("", tails.get(path, "") + text)
            match = ABORT.search(text)
            tails[path] = text[-4096:]
            if armed_at is None and match:
                armed_at = now
                record("cleanup_grace_armed", marker=match[0], source=str(path))
        try:
            fields, raw = owned_job(job_id, log_path, job_name)
            steps = command(
                "squeue", "--steps", f"--jobs={job_id}", "--noheader", "--format=%i"
            ).strip()
            inspection_failures = 0
            receipt["last_job"] = raw.strip()
            receipt["last_active_steps"] = steps.splitlines()
            if not verified_owner:
                verified_owner = True
                record("ownership_verified")
        except ValueError as exc:
            receipt["status"] = "ownership_mismatch"
            record("failed_closed", reason=str(exc))
            return 1
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            inspection_failures += 1
            record("inspection_failed", count=inspection_failures, reason=str(exc))
            if inspection_failures >= INSPECTION_FAILURE_LIMIT:
                receipt["status"] = "cleanup_unverified"
                record(
                    "failed_closed",
                    reason="Scheduler inspection unavailable; no cancellation without current ownership",
                )
                return 1
            time.sleep(POLL_SECONDS)
            continue

        state = fields.get("JobState", "").split("+", 1)[0]
        if state in TERMINAL:
            if not steps:
                receipt["status"] = "terminal_no_active_steps"
                record(
                    "cleanup_verified", state=state, exit_code=fields.get("ExitCode")
                )
                return (
                    0
                    if state == "COMPLETED"
                    and fields.get("ExitCode") == "0:0"
                    and cancelled_at is None
                    else 1
                )
            if terminal_at is None:
                terminal_at = now
                record("terminal_with_active_steps", state=state)
            if now - terminal_at >= CANCEL_WAIT_SECONDS:
                receipt["status"] = "cleanup_unverified"
                record(
                    "failed_closed", reason="Terminal allocation still has active steps"
                )
                return 1
        elif cancelled_at is not None:
            if now - cancelled_at >= CANCEL_WAIT_SECONDS:
                receipt["status"] = "cleanup_unverified"
                record(
                    "failed_closed",
                    reason="Exact-job cancellation did not reach terminal/no-active-steps",
                )
                return 1
        elif armed_at is not None and now - armed_at >= GRACE_SECONDS:
            # Revalidated this exact job and output immediately above. Normal
            # scancel cancels the allocation (not just its local srun client).
            try:
                command("scancel", job_id)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                receipt["status"] = "cleanup_unverified"
                record("failed_closed", reason=f"Exact-job cancellation failed: {exc}")
                return 1
            cancelled_at = time.monotonic()
            record(
                "exact_job_cancelled",
                reason="Terminal failure/cleanup exceeded 300-second operational grace",
            )
        time.sleep(POLL_SECONDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("job_name")
    args = parser.parse_args()
    return watch(args.job_id, args.log_path, args.job_name)


if __name__ == "__main__":
    raise SystemExit(main())
