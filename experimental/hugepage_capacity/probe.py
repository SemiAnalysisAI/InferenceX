import argparse
import json
import os
import resource
import signal
import socket
import subprocess
import time
from pathlib import Path


def memory():
    return {
        line.split(":")[0]: int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
    }


def emit(event, **data):
    print(json.dumps({"time": time.time(), "event": event, **data}), flush=True)


def snapshot(event):
    emit(
        event,
        host=socket.gethostname(),
        uid=os.getuid(),
        memory=memory(),
        memlock=resource.getrlimit(resource.RLIMIT_MEMLOCK),
    )


def apply_hook(hook, target):
    env = dict(
        os.environ,
        HUGEPAGES_TARGET=str(target),
        SKIP_HUGEPAGE_CHECK="0",
        HUGEPAGE_GROW_TOLERANCE_PCT="0",
        HUGEPAGE_FAIL_THRESHOLD_PAGES="1",
    )
    started = time.monotonic()
    process = subprocess.Popen(["bash", str(hook)], env=env, start_new_session=True)
    try:
        rc = process.wait(timeout=180)
    except BaseException:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
        raise
    emit("hook", target=target, rc=rc, seconds=time.monotonic() - started)
    return rc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hook", required=True, type=Path)
    parser.add_argument("--target-pages", required=True, type=int)
    parser.add_argument("--min-headroom-gib", required=True, type=int)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--expected-host")
    scope.add_argument("--expected-job")
    args = parser.parse_args()
    if (
        not args.hook.is_file()
        or args.target_pages < 512
        or args.min_headroom_gib < 512
    ):
        raise RuntimeError("Invalid hook, target or headroom")
    host = socket.gethostname().split(".")[0]
    if args.expected_host:
        if host != args.expected_host:
            raise RuntimeError("Wrong host")
        containers = subprocess.check_output(["docker", "ps", "-q"], text=True).strip()
        if containers:
            raise RuntimeError("Running containers: refusing host reservation")
    else:
        if os.environ.get("SLURM_JOB_ID") != args.expected_job:
            raise RuntimeError("Wrong allocation")
        jobs = subprocess.check_output(
            ["squeue", "-h", "-w", host, "-o", "%i"], text=True
        ).split()
        if jobs != [args.expected_job]:
            raise RuntimeError(f"Node not dedicated to expected allocation: {jobs}")
        subprocess.run(["scontrol", "show", "job", args.expected_job], check=True)
    if list(Path("/sys/class/kfd/kfd/proc").glob("[0-9]*")):
        raise RuntimeError("GPU process present")
    initial = memory()
    snapshot("before")
    if any(
        initial[key] for key in ("HugePages_Total", "HugePages_Rsvd", "HugePages_Surp")
    ):
        raise RuntimeError("Existing hugepage reservation: refusing to modify it")
    if initial["Hugepagesize"] != 2048:
        raise RuntimeError("Expected 2 MiB hugepages")
    headroom_kib = initial["MemAvailable"] - args.target_pages * initial["Hugepagesize"]
    if headroom_kib < args.min_headroom_gib * 1024**2:
        raise RuntimeError("Insufficient ordinary-memory headroom")
    emit("plan", target_pages=args.target_pages, remaining_available_kib=headroom_kib)

    def interrupted(signum, frame):
        raise RuntimeError(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for target in (512, args.target_pages):
            rc = apply_hook(args.hook, target)
            snapshot("after_reserve")
            state = memory()
            if (
                rc
                or state["HugePages_Total"] != target
                or state["HugePages_Free"] != target
            ):
                raise RuntimeError(f"Exact reservation failed at {target} pages")
        emit(
            "capacity_pass",
            pages=args.target_pages,
            bytes=args.target_pages * 2 * 1024**2,
        )
    finally:
        state = memory()
        if (
            state["HugePages_Total"] != state["HugePages_Free"]
            or state["HugePages_Rsvd"]
        ):
            raise RuntimeError("Held pages detected: refusing automatic shrink")
        rc = apply_hook(args.hook, 0)
        snapshot("after_cleanup")
        if rc or memory()["HugePages_Total"] or memory()["HugePages_Surp"]:
            raise RuntimeError("Hugepage cleanup did not restore baseline")


if __name__ == "__main__":
    main()
