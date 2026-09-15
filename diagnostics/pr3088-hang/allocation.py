"""Bound this diagnostic's salloc wait and preserve native failure before cleanup."""

import pwd
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import time

TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "OUT_OF_MEMORY",
    "REVOKED",
    "SPECIAL_EXIT",
}
POLL_SECONDS = 2
QUEUE_SECONDS = 600


def record(out, name, data):
    temporary = out / (name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(out / name)


def query(out, args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=5)
        row = {
            "command": args,
            "rc": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        row = {"command": args, "timeout_seconds": 5, "stdout": ""}
    with (out / "scheduler.jsonl").open("a") as log:
        log.write(json.dumps({"at": time.time(), **row}) + "\n")
    return row["stdout"] if row.get("rc") == 0 else ""


def native(out, job):
    accounting = query(
        out, ["sacct", "-X", "-n", "-P", "-j", job, "--format=JobIDRaw,State,ExitCode"]
    )
    for line in accounting.splitlines():
        fields = line.split("|")
        if len(fields) >= 3 and fields[0] == job:
            state = fields[1].split()[0].rstrip("+")
            if state in TERMINAL:
                record(
                    out,
                    "native-terminal.json",
                    {"job_id": job, "state": state, "exit_code": fields[2]},
                )
                return state, {}
    controller = query(out, ["scontrol", "show", "job", "-o", job])
    fields = dict(re.findall(r"(\w+)=([^\s]+)", controller))
    if fields.get("JobId") != job:
        return "", {}
    state = fields.get("JobState", "")
    if state in TERMINAL:
        record(
            out,
            "native-terminal.json",
            {"job_id": job, "state": state, "exit_code": fields.get("ExitCode")},
        )
    return state, fields


def owned(fields, job, name):
    return (
        fields.get("JobId") == job
        and fields.get("JobName") == name
        and fields.get("UserId")
        == f"{pwd.getpwuid(os.getuid()).pw_name}({os.getuid()})"
    )


def cleanup(out):
    receipt = out / "allocation.json"
    if not receipt.exists() or (out / "cleanup.json").exists():
        return
    data = json.loads(receipt.read_text())
    job = data["job_id"]
    terminal = out / "native-terminal.json"
    if terminal.exists():
        state, fields = json.loads(terminal.read_text())["state"], {}
    else:
        state, fields = native(out, job)
    if state in TERMINAL:
        result = {
            "job_id": job,
            "state": state,
            "cancelled": False,
            "reason": "already terminal; original state preserved",
        }
    elif owned(fields, job, data["job_name"]):
        output = query(out, ["scancel", job])
        result = {
            "job_id": job,
            "state_before_cancel": state,
            "cancel_requested": True,
            "stdout": output,
            "limit": "Request outcome is in scheduler.jsonl; terminal cleanup is not asserted.",
        }
    else:
        result = {
            "job_id": job,
            "cancelled": False,
            "reason": "cannot verify current job ID/name/owner; no external cancellation",
        }
    record(out, "cleanup.json", result)


def allocate(out, args):
    out.mkdir(parents=True, exist_ok=False)
    name = os.environ["RUNNER_NAME"]
    if f"--job-name={name}" not in args:
        raise RuntimeError("salloc job name must match this runner invocation")
    process = None
    selector = selectors.DefaultSelector()
    started = time.monotonic()
    next_query = started
    job = ""
    granted = False
    buffer = ""
    success = False
    try:
        process = subprocess.Popen(
            ["salloc", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "LC_ALL": "C"},
        )
        selector.register(process.stdout, selectors.EVENT_READ)
        with (out / "salloc.log").open("wb") as log:
            while time.monotonic() - started < QUEUE_SECONDS:
                for key, _ in selector.select(0.2):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    log.write(chunk)
                    log.flush()
                    sys.stderr.buffer.write(chunk)
                    sys.stderr.flush()
                    buffer = (buffer + chunk.decode(errors="replace"))[-8192:]
                    for kind, value in re.findall(
                        r"(Pending|Granted) job allocation (\d+)", buffer
                    ):
                        if job and job != value:
                            raise RuntimeError(
                                "salloc emitted inconsistent allocation IDs"
                            )
                        job = value
                        granted = granted or kind == "Granted"
                        record(
                            out,
                            "allocation.json",
                            {
                                "job_id": job,
                                "job_name": name,
                                "uid": os.getuid(),
                                "salloc_pid": process.pid,
                                "run": os.environ.get("GITHUB_RUN_ID"),
                                "attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
                            },
                        )
                rc = process.poll()
                if job and (time.monotonic() >= next_query or rc is not None):
                    state, fields = native(out, job)
                    next_query = time.monotonic() + POLL_SECONDS
                    if state in TERMINAL:
                        raise RuntimeError(
                            f"allocation {job} became {state} before workload launch"
                        )
                    if rc == 0 and granted:
                        if state != "RUNNING" or not owned(fields, job, name):
                            raise RuntimeError(
                                "granted allocation is not verifiably RUNNING with the expected owner/name"
                            )
                        success = True
                        return job
                if rc is not None:
                    raise RuntimeError(f"salloc exited {rc} without a verified grant")
            raise RuntimeError(
                f"allocation queue wait exceeded {QUEUE_SECONDS} seconds"
            )
    finally:
        selector.close()
        # The child has not been reaped if poll() is None, so its PID/session
        # cannot be recycled between this check and signalling its own group.
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
        if not success:
            cleanup(out)


def main():
    mode = sys.argv[1]
    out = Path(sys.argv[2])
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(130))
    signal.signal(signal.SIGHUP, lambda *_: sys.exit(143))
    try:
        if mode == "allocate":
            print(allocate(out, sys.argv[3:]))
        elif mode == "cleanup":
            cleanup(out)
        else:
            raise ValueError("unknown allocation operation")
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        if out.is_dir():
            record(out, "error.json", {"error": str(error)})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
