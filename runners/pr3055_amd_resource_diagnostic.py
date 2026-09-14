#!/usr/bin/env python3
"""Read only known PR3055 Slurm/node state; no allocation or remote execution."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

JOBS = (43980, 43990, 43992, 43996)
NODES = ("mia1-p01-g16", "mia1-p01-g18")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []

    def capture(name, command):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            item = {"name": name, "command": command, "exit_code": result.returncode,
                    "stdout": result.stdout[:250000], "stderr": result.stderr[:20000]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            item = {"name": name, "command": command, "error": str(exc)}
        results.append(item)
        print(name, item.get("exit_code", item.get("error")), flush=True)
        return item

    capture("accounting", ["sacct", "-j", ",".join(map(str, JOBS)), "--parsable2",
            "--noheader", "--format=JobIDRaw,JobName,User,State,ExitCode,Start,End,Elapsed,NodeList,AllocTRES,MaxRSS,ReqMem"])
    capture("active_jobs_on_target_nodes", ["squeue", "--nodes=" + ",".join(NODES),
            "--noheader", "--format=%i|%j|%u|%T|%M|%L|%N|%R"])
    for node in NODES:
        capture("node_" + node, ["scontrol", "show", "node", node])
    for job in JOBS:
        capture("job_" + str(job), ["scontrol", "show", "job", str(job)])
    hostname = socket.gethostname().split(".")[0]
    if hostname in NODES:
        capture("local_gpu_identity", ["amd-smi", "list", "--json"])
        capture("local_gpu_processes", ["amd-smi", "process", "--json"])
        # Slurm listpids operates on this host, not on arbitrary remote nodes.
        for job in JOBS:
            capture("local_pids_" + str(job), ["scontrol", "listpids", str(job)])
        capture("local_task_containers", ["docker", "ps", "-a", "--no-trunc",
                "--format", "{{.ID}}|{{.Names}}|{{.Status}}",
                "--filter", "name=_43980", "--filter", "name=_43990",
                "--filter", "name=_43992", "--filter", "name=_43996"])
    else:
        results.append({"name": "node_process_access", "status": "not_executed",
                        "reason": "Runner is not g16/g18; no established node SSH route. No allocation or guessed remote command."})
    receipt = {"at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "hostname": hostname, "uid": os.getuid(), "gid": os.getgid(),
               "github_run_id": os.environ.get("GITHUB_RUN_ID"),
               "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
               "workflow_sha": os.environ.get("GITHUB_SHA"),
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "gpu_allocations_requested": 0, "results": results,
               "limit": "Read-only infrastructure evidence; not benchmark/eval or power qualification."}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    failed = [r for r in results if r.get("name") in {"accounting", "active_jobs_on_target_nodes"}
              and r.get("exit_code") != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
