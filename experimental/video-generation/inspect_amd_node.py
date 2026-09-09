"""Bounded AMD node inventory using the existing Slurm ownership receipts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess

import ci


def observation(argv: list[str]) -> dict:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return {"argv": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": argv, "error": str(error)}


def inspect_node(run_dir: Path) -> None:
    context = ci.read(run_dir / "context.json")
    job = context["allocation"]["identity"]["JobId"]
    ci.need(os.environ.get("SLURM_JOB_ID") == job
            and os.environ.get("SLURMD_NODENAME") == context["node"], "Wrong AMD inventory allocation")
    binding = {"job_id": job, "step_id": os.environ.get("SLURM_STEP_ID"),
               "node": context["node"], "cpu_affinity": sorted(os.sched_getaffinity(0)),
               "slurm": {name: os.environ.get(name) for name in
                         ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}}
    ci.write(run_dir / "binding.json", binding)
    commands = [["uname", "-a"], ["enroot", "list", "-f"], ["srun", "--help"],
                ["bash", "-c", "command -v amd-smi rocm-smi rocminfo python3; ls -ld /opt/rocm* /var/lib/enroot /run/enroot /etc/enroot 2>/dev/null"]]
    smi = shutil.which("amd-smi")
    if not smi and Path("/opt/rocm/bin/amd-smi").is_file():
        smi = "/opt/rocm/bin/amd-smi"
    if smi:
        commands += [[smi, option, "--json"] for option in ("list", "static", "metric", "process")]
        commands += [[smi, "version", "--json"]]
    cache = Path("/var/lib/squash")
    images = [{"path": str(p), "size_bytes": p.stat().st_size} for p in sorted(cache.glob("*sglang*rocm*.sqsh"))]
    ci.write(run_dir / "node-inventory.json", {
        "schema_version": 1, "bundle_type": "h3_amd_node_inventory", "binding": binding,
        "cached_images": images, "observations": [observation(argv) for argv in commands],
        "generation_executed": False, "runtime_compatibility": "not_tested", "observed_at": ci.now(),
    })


def inspect(workspace: Path, output: Path) -> int:
    ci.need(workspace.is_absolute(), "Persistent workspace must be absolute")
    run_id, attempt = os.environ["H3_RUN_ID"], os.environ["H3_RUN_ATTEMPT"]
    ci.need(run_id.isdigit() and attempt.isdigit(), "Invalid CI identity")
    root = workspace / "results/h3-cross-hardware"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"github-{run_id}-{attempt}"
    run_dir.mkdir(exist_ok=False)
    account = pwd.getpwuid(os.getuid()).pw_name
    ci.need(account == "cameronamd@semianalysis.com", "Unexpected AMD scheduler identity")
    config = {"task_id": "h3-cross-hardware", "mode": "serving-smoke", "allocation_receipts": [],
              "site": {"cluster": "mi355x-amds", "partition": "compute", "account": account, "gpu_model": "MI355X"},
              "resources": {"gpus": 8, "allocated_gpus": 8, "cpus": 8, "memory_gb": 64, "minutes": 15}}
    state = {"status": "starting", "started_at": ci.now(), "config": config,
             "purpose": "Observe actual AMD device identities, telemetry formats and cached runtimes before adding a GPU adapter",
             "source_sha": os.environ.get("H3_SOURCE_SHA"), "generation_executed": False}
    receipt, reused, code = None, False, 2
    control = workspace / "campaigns/h3-cross-hardware"
    control.mkdir(parents=True, exist_ok=True)
    with ci.task_lock(control / ".node-inventory.lock"):
        try:
            recovery = ci.recover(config, root)
            ci.write(run_dir / "recovery.json", recovery)
            ci.need(recovery["action"] != "wait", "Task-owned AMD allocation is waiting; do not submit another")
            if recovery["action"] == "reuse":
                ci.need(not recovery["active_steps"].strip(), "Task-owned AMD allocation has active steps")
                receipt, reused = recovery["receipt"], True
                ci.write(run_dir / "allocation.json", receipt)
            else:
                receipt = ci.allocate(config, run_dir)
            record = ci.job_record(receipt["identity"]["JobId"])
            state.update(allocation=receipt, allocation_reused=reused, slurm_job=record)
            ci.write(run_dir / "slurm-job.json", record)
            ci.verify_identity(receipt, record, config["task_id"])
            ci.need(record["JobState"] == "RUNNING", "Owned AMD allocation is " + record["JobState"])
            reason = ci.capacity(record, config["resources"])
            ci.need(reason is None, "Owned AMD allocation: " + str(reason))
            ci.write(run_dir / "context.json", {"allocation": receipt, "node": record["NodeList"]})
            # The source checkout and result directory are on the shared filesystem.
            argv = ["srun", "--jobid=" + record["JobId"], "--nodelist=" + record["NodeList"],
                    "--nodes=1", "--ntasks=1", "--gres=gpu:8", "--cpus-per-task=8", "--cpu-bind=cores",
                    "--time=10", "--export=NONE", "/usr/bin/python3", str(Path(__file__).resolve()),
                    "--inside", str(run_dir)]
            ci.write(run_dir / "step-command.json", argv)
            code = ci.run_step(argv, run_dir / "srun.log", 660)
            ci.need(code == 0 and (run_dir / "node-inventory.json").is_file(), "AMD inventory step failed; inspect retained logs")
            state["status"] = "complete"
        except Exception as error:
            state.update(status="failed", error=str(error))
            code = 2
        finally:
            if receipt is None and (run_dir / "allocation.json").exists():
                receipt = ci.read(run_dir / "allocation.json")
            try:
                if receipt:
                    state["step_cleanup"] = ci.drain_step(receipt, config["task_id"], run_dir)
                    state["allocation_cleanup"] = {"status": "retained"} if reused else ci.stop_allocation(receipt, config["task_id"])
            except Exception as error:
                state.update(status="failed", cleanup_error=str(error))
                code = 2
            state.update(finished_at=ci.now(), exit_code=code)
            ci.write(run_dir / "inventory-status.json", state)
            ci.collect(run_dir, output)
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inside", type=Path)
    args = parser.parse_args()
    if args.inside:
        inspect_node(args.inside)
    else:
        raise SystemExit(inspect(args.workspace, args.output))
