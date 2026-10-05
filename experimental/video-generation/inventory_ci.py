#!/usr/bin/env python3
"""Bounded H200 hardware inventory using the prepared H3 CI allocation route.

This records current hardware and power settings. It never loads the model and
cannot establish the power limits of an earlier benchmark.
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import ci
from evaluator.mvp_gpu_job import GpuProbe, cuda_devices, validate_gpu_job

RESOURCES = {"gpus": 4, "cpus": 4, "memory_gb": 8, "minutes": 10}



def classify_tdp(xml_text: str, devices: list[str]) -> dict:
    """Identify an advertised maximum TDP from observed PCI IDs, never power draw."""
    sources = {
        "specification": "https://www.nvidia.com/en-us/data-center/h200/",
        "supported_products": "https://download.nvidia.com/XFree86/Linux-x86_64/580.173.02/README/supportedchips.html",
        "pci_variant_mapping": "https://raw.githubusercontent.com/NVIDIA/k8s-launch-kit/db32e4b98170/pkg/networkoperatorplugin/internal/pciids/nvidia.ids",
    }
    result = {"status": "unknown", "watts_per_gpu": None, "hardware_variant": None,
              "source_url": sources["specification"], "semantics": "manufacturer maximum configurable TDP; not an observed power limit",
              "evidence": {"sources": sources, "raw_path": "nvidia-smi.xml", "devices": []}}
    # Full NVIDIA subsystem IDs narrow the mapping to published supported boards.
    supported = {
        ("0x233510de", "0x18be10de", "NVIDIA H200"): ("H200 SXM", 700),
        ("0x233510de", "0x18bf10de", "NVIDIA H200"): ("H200 SXM", 700),
        ("0x233b10de", "0x199610de", "NVIDIA H200 NVL"): ("H200 NVL", 600),
    }
    try:
        root = ET.fromstring(xml_text)
        rows = root.findall("gpu")
        uuids = [row.findtext("uuid") for row in rows]
        ci.need(root.tag == "nvidia_smi_log" and devices and len(set(devices)) == len(devices)
                and len(uuids) == len(set(uuids)) and set(devices) <= set(uuids), "XML inventory lacks unique assigned UUIDs")
        variants = set()
        for row in rows:
            uuid = row.findtext("uuid")
            if uuid not in devices:
                continue
            device = {"uuid": uuid, "product_name": row.findtext("product_name"),
                      "pci_device_id": row.findtext("pci/pci_device_id"),
                      "pci_sub_system_id": row.findtext("pci/pci_sub_system_id")}
            result["evidence"]["devices"].append(device)
            pci_ids = [str(device[key]).strip().lower() for key in ("pci_device_id", "pci_sub_system_id")]
            ci.need(all(re.fullmatch(r"(?:0x)?[0-9a-f]{8}", value) for value in pci_ids), "Invalid PCI hexadecimal identity")
            identity = (*("0x" + value.removeprefix("0x") for value in pci_ids), device["product_name"])
            ci.need(identity in supported, "Unrecognized or inconsistent PCI/product identity")
            variants.add(supported[identity])
        ci.need(len(variants) == 1, "Selected GPUs have different hardware variants")
        variant, watts = variants.pop()
        result.update(status="verified", watts_per_gpu=watts, hardware_variant=variant)
    except (ET.ParseError, ValueError) as error:
        result["reason"] = str(error)
    return result

def prepare(config: dict) -> tuple[dict, dict]:
    config = copy.deepcopy(ci.validate_config(config))
    runtime = config["runtime"]
    ci.need(Path(runtime["rootfs"]).is_dir() and Path(runtime["ready_marker"]).is_file(), "Prepared persistent runtime/readiness missing")
    ci.need(ci.digest(runtime["entry"]) == runtime["entry_sha256"], "Persistent entry script changed")
    interpreter = ci.host_path(config, runtime["python"])
    # Absolute symlinks resolve inside Enroot, not against the submit host root.
    ci.need(interpreter.is_file() or interpreter.is_symlink(), "Prepared interpreter missing")
    ci.need(ci.digest(config["spec"]["path"]) == config["spec"]["sha256"], "Prepared specification changed")
    spec = validate_gpu_job(ci.read(config["spec"]["path"]))
    approval = spec["authorization"]
    ci.need(approval["compute_approved"] and approval["model_license_reviewed"] and approval["approval_reference"].strip(), "Prepared specification lacks approval")
    config["resources"] = dict(RESOURCES)
    return config, approval



def source_target(config: dict, run_ids: list[str]) -> dict:
    """Join verified GitHub runs to sealed task-owned persistent hardware receipts."""
    from export_ci import REPOSITORY, source_ids, verified_execution
    run_ids = source_ids(",".join(run_ids))
    results = Path(config["workspace"]["host"]) / "results" / config["task_id"]
    target = {"node": None, "gpu_uuids": None, "sources": []}
    for run_id in run_ids:
        source_ci, artifact = verified_execution(run_id)
        ci.need(source_ci["status"] == "completed" and source_ci["conclusion"] == "success", "Inventory requires completed source execution")
        attempt, sha = str(source_ci["runAttempt"]), source_ci["headSha"]
        ci.need(attempt.isdigit() and re.fullmatch(r"[0-9a-f]{40}", sha), "Invalid verified source identity")
        directory = results / f"github-{run_id}-{attempt}"
        ci.need(directory.is_dir() and not directory.is_symlink(), "Original task-owned persistent source missing")
        sealed = {}
        for line in (directory / "SHA256SUMS").read_text().splitlines():
            digest, separator, name = line.partition("  ")
            ci.need(separator and ci.SHA.fullmatch(digest) and name not in sealed, "Invalid source checksum seal")
            sealed[name] = digest
        names = ("manifest.json", "ci.json", "context.json", "binding.json", "gpu/spec.json")
        for name in names:
            path = directory / name
            ci.need(not path.is_symlink() and path.is_file() and ci.digest(path) == sealed.get(name), "Source receipt differs from its seal: " + name)
        manifest, state, context, binding, spec = (ci.read(directory / name) for name in names)
        ci.need(all(manifest["evidence"].get(name) == sealed[name] for name in names[1:]), "Source manifest receipt hashes differ")
        ci.need(manifest["task_id"] == state["task_id"] == context["config"]["task_id"] == config["task_id"]
                and str(manifest["run_id"]) == str(state["run_id"]) == run_id
                and str(manifest["run_attempt"]) == str(state["run_attempt"]) == attempt
                and context["run_id"] == directory.name
                and manifest["git_commit"] == state["source_sha"] == context["source_sha"] == sha,
                "Source CI/Git/task identities disagree")
        ci.need(manifest["ci"] == state["ci"] and manifest["ci"]["repository"] == REPOSITORY
                and manifest["ci"]["run_url"] == source_ci["url"] and manifest["exit_code"] == state["exit_code"] == 0
                and state["phase"] == "complete" and state.get("smoke_completed") is True, "Source execution did not complete cleanly")
        allocation = manifest["slurm_allocation"]
        ci.need(allocation == state["allocation"] == context["allocation"], "Source allocation receipts disagree")
        ci.verify_identity(allocation, state["slurm_job"], config["task_id"])
        node, devices = binding["node"], sorted(binding["gpu_uuids"])
        ci.need(ci.NAME.fullmatch(node) and node == context["node"] == state["slurm_job"]["NodeList"]
                and binding["job_id"] == allocation["identity"]["JobId"]
                and state["step_cleanup"].get("status") == "ended"
                and state["step_cleanup"]["step_id"] == binding["job_id"] + "." + binding["step_id"], "Source Slurm bindings disagree")
        ci.need(len(devices) == RESOURCES["gpus"] and len(set(devices)) == len(devices)
                and all(re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", device) for device in devices)
                and sorted(spec["gpu_uuids"]) == devices, "Source GPU assignments disagree")
        ci.need(target["node"] in (None, node) and target["gpu_uuids"] in (None, devices), "Source runs used different physical hardware")
        target.update(node=node, gpu_uuids=devices)
        target["sources"].append({"run_id": run_id, "run_attempt": attempt, "git_commit": sha,
                                  "verified_ci": source_ci, "artifact": artifact, "receipt_hashes": {name: sealed[name] for name in names}})
    return target

def collect_inventory(config: dict, output: Path, *, source_run_ids: list[str] | None = None) -> int:
    config, approval = prepare(config)
    run, attempt, sha = (os.environ.get(key, "") for key in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "H3_SOURCE_SHA"))
    ci.need(re.fullmatch(r"[0-9]+", run) and re.fullmatch(r"[0-9]+", attempt) and re.fullmatch(r"[0-9a-f]{40}", sha), "Exact CI run/attempt/source required")
    source = Path(__file__).resolve().parent
    ci.need(ci.command(["git", "-C", str(source), "rev-parse", "HEAD"]).strip() == sha, "Checkout differs from admitted source")
    ci.need(not ci.command(["git", "-C", str(source), "status", "--porcelain"]).strip(), "Inventory checkout must be clean and committed")
    workspace = Path(config["workspace"]["host"])
    ci.need(workspace.is_dir(), "Persistent workspace missing")
    results, control = workspace / "results" / config["task_id"], workspace / "campaigns" / config["task_id"] / "control"
    results.mkdir(parents=True, exist_ok=True)
    control.mkdir(parents=True, exist_ok=True)
    run_dir = results / f"inventory-{run}-{attempt}"
    ci_identity = {"run_id": run, "run_attempt": attempt, "repository": os.environ.get("GITHUB_REPOSITORY"),
                   "run_url": f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/{run}"}
    state = {"schema_version": 1, "operation": "hardware_inventory", "task_id": config["task_id"],
             "run_id": run, "run_attempt": attempt, "source_sha": sha, "started_at": ci.now(), "phase": "preparing",
             "resources": {"requested": config["resources"], "new_allocation_gpus": 8, "new_allocation_gpu_hours_cap": 8 * 10 / 60},
             "authorization": approval, "ci": {key: os.environ.get(key) for key in ("GITHUB_REPOSITORY", "GITHUB_WORKFLOW_REF", "GITHUB_WORKFLOW_SHA", "GITHUB_ACTOR", "GITHUB_TRIGGERING_ACTOR")}}
    with ci.task_lock(control / "ci.lock"):
        run_dir.mkdir(exist_ok=False)
        ci.write(run_dir / "ci.json", state)
        receipt, reused, code = None, False, 2
        try:
            shutil.copyfile(config["runtime"]["entry"], run_dir / "runtime-entry.sh")
            shutil.copyfile(config["runtime"]["ready_marker"], run_dir / "runtime-readiness.record")
            package = workspace / "campaigns" / config["task_id"] / "packages" / (sha + "-inventory")
            package_files = ci.stage_package(source, package)
            target = source_target(config, source_run_ids if source_run_ids is not None else os.environ.get("H3_SOURCE_RUN_IDS", "").split(","))
            ci.write(run_dir / "target-source.json", target)
            decision = ci.recover(config, results, node=target["node"])
            ci.write(run_dir / "recovery.json", decision)
            ci.need(decision["action"] != "wait", "Task-owned allocation pending; no duplicate submitted")
            if decision["action"] == "reuse":
                receipt, reused = decision["receipt"], True
                ci.write(run_dir / "allocation.json", receipt)
            else:
                receipt = ci.allocate(config, run_dir, node=target["node"])
            record = ci.job_record(receipt["identity"]["JobId"])
            ci.verify_identity(receipt, record, config["task_id"])
            ci.need(record["JobState"] == "RUNNING" and record["NodeList"] == target["node"]
                    and ci.capacity(record, config["resources"]) is None, "Allocation cannot serve pinned inventory step")
            state.update(phase="starting", allocation=receipt, allocation_reused=reused, slurm_job=record)
            ci.write(run_dir / "ci.json", state)
            ci.write(run_dir / "context.json", {"config": config, "allocation": receipt, "node": record["NodeList"],
                     "source_sha": sha, "package_files": package_files, "run_id": run, "run_attempt": attempt, "ci": ci_identity, "target": target,
                     "active_steps": ci.command(["squeue", "--steps", "--noheader", "--jobs=" + record["JobId"], "--format=%i|%N"])})
            argv = ci.step_argv(config, receipt, record, run_dir, package)
            argv[-3] = str(package / "inventory_ci.py")
            ci.write(run_dir / "step-command.json", argv)
            code = ci.run_step(argv, run_dir / "srun.log", 360)
            result = ci.read(run_dir / "step-result.json")
            ci.need(code == result["exit_code"], "Slurm exit and inventory receipt differ")
            ci.need(code != 0 or (result.get("inventory_completed") is True and (run_dir / "hardware-profile.json").is_file()), "Inventory success requires a completed hardware profile")
            state.update(phase="complete" if code == 0 else "failed", **result)
        except (Exception, KeyboardInterrupt) as error:
            state.update(phase="failed", error=str(error))
            code = 2
        finally:
            if receipt is None and (run_dir / "allocation.json").is_file():
                receipt = ci.read(run_dir / "allocation.json")
            try:
                if receipt is not None:
                    state["step_cleanup"] = ci.drain_step(receipt, config["task_id"], run_dir)
            except Exception as error:
                state.update(phase="failed", step_cleanup_error=str(error))
                code = 2
            try:
                if receipt is not None and not reused:
                    state["allocation_cleanup"] = ci.stop_allocation(receipt, config["task_id"])
                elif reused:
                    state["allocation_cleanup"] = {"status": "retained", "reason": "attached step does not own parent allocation"}
            except Exception as error:
                state.update(phase="failed", cleanup_error=str(error))
                code = 2
            state.update(finished_at=ci.now(), exit_code=code)
            ci.write(run_dir / "ci.json", state)
            ci.write(run_dir / "manifest.json", {"schema_version": 1, "operation": "hardware_inventory", "source_sha": sha,
                     "run_id": run, "run_attempt": attempt, "task_id": config["task_id"], "ci": state["ci"],
                     "slurm_allocation": receipt, "runtime": config["runtime"], "prepared_spec": config["spec"],
                     "evidence": ci.inventory(run_dir), "exit_code": code, "artifact_checksums": "SHA256SUMS"})
            ci.collect(run_dir, output)
    return code


def enter(run_dir: Path) -> None:
    context = ci.read(run_dir / "context.json")
    config = ci.validate_config(context["config"])
    job, step = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_STEP_ID", "")
    ci.need(job == context["allocation"]["identity"]["JobId"] and re.fullmatch(r"[0-9]+", step)
            and os.environ.get("SLURMD_NODENAME") == context["node"], "Wrong Slurm step assignment")
    ci.write(run_dir / "binding.json", {"job_id": job, "step_id": step, "node": context["node"],
             "cpu_affinity": sorted(os.sched_getaffinity(0)), "observed_at": ci.now(), "phase": "entering_runtime"})
    ci.need(ci.digest(config["runtime"]["entry"]) == config["runtime"]["entry_sha256"], "Entry changed on compute node")
    argv = ["/bin/bash", config["runtime"]["entry"], config["runtime"]["python"],
            str(ci.mapped(config, Path(__file__).parent) / "inventory_ci.py"), "--inside", str(ci.mapped(config, run_dir))]
    os.execv(argv[0], argv)


def capture(argv: list[str], path: Path) -> None:
    try:
        result = subprocess.run(argv, text=True, capture_output=True, timeout=20, env=ci.environment())
    except subprocess.TimeoutExpired as error:
        for target, value in ((path, error.stdout), (path.with_suffix(path.suffix + ".stderr.log"), error.stderr)):
            target.write_bytes(value.encode() if isinstance(value, str) else (value or b""))
        raise
    path.write_text(result.stdout)
    path.with_suffix(path.suffix + ".stderr.log").write_text(result.stderr)
    ci.need(result.returncode == 0, "Inventory query failed: " + argv[0])


def inside(run_dir: Path) -> int:
    result = {"exit_code": 2, "inventory_completed": False}
    try:
        context = ci.read(run_dir / "context.json")
        config = ci.validate_config(context["config"])
        ci.need(os.access(config["runtime"]["python"], os.X_OK)
                and Path(config["runtime"]["python"]).samefile(sys.executable), "Inventory is not running the configured container interpreter")
        interpreter = {"configured_path": config["runtime"]["python"], "executable": sys.executable,
                       "version": sys.version.split()[0], "sha256": ci.digest(sys.executable)}
        job, step = context["allocation"]["identity"]["JobId"], os.environ.get("SLURM_STEP_ID", "")
        ci.need(os.environ.get("SLURM_JOB_ID") == job and re.fullmatch(r"[0-9]+", step)
                and os.environ.get("SLURMD_NODENAME") == context["node"]
                and os.environ.get("SLURM_PROCID") == "0" and os.environ.get("SLURM_NTASKS") == "1", "Wrong single-node Slurm task")
        ci.need(ci.inventory(Path(__file__).parent) == context["package_files"], "Staged inventory bytes changed")
        devices, cpus = cuda_devices(), sorted(os.sched_getaffinity(0))
        assigned = os.environ.get("H3_ASSIGNED_GPU_UUIDS", "").split(",")
        ci.need(len(devices) == RESOURCES["gpus"] and len(set(devices)) == len(devices) and set(devices) == set(assigned), "CUDA UUIDs differ from assigned GPUs")
        ci.need(context["node"] == context["target"]["node"] and sorted(devices) == context["target"]["gpu_uuids"], "Inventory assignment differs from historical source hardware")
        binding = ci.read(run_dir / "binding.json")
        ci.need(binding["job_id"] == job and binding["step_id"] == step and binding["cpu_affinity"] == cpus
                and len(cpus) >= RESOURCES["cpus"], "Container changed assigned CPU/step binding")
        binding.update(gpu_uuids=devices, observed_at=ci.now(), phase="inventory",
                       slurm={key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "H3_ORIGINAL_CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_CPU_BIND", "SLURM_CPUS_PER_TASK")})
        ci.write(run_dir / "binding.json", binding)
        probe = GpuProbe(devices, timeout=15)
        before = probe.snapshot()
        ci.write(run_dir / "gpu-before.json", before)
        ci.need(not before["compute_apps"], "Assigned inventory GPUs have active compute; no workload started")
        capture(["nvidia-smi", "-q", "-x"], run_dir / "nvidia-smi.xml")
        capture(["nvidia-smi", "topo", "-m"], run_dir / "topology.txt")
        power = probe.power_configuration()
        ci.write(run_dir / "power-configuration.json", power)
        dmi = {}
        for name in ("product_name", "sys_vendor"):
            try:
                dmi[name] = {"value": (Path("/sys/class/dmi/id") / name).read_text().strip()}
            except OSError as error:
                dmi[name] = {"value": None, "reason": type(error).__name__}
        after = probe.snapshot()
        ci.write(run_dir / "gpu-after.json", after)
        ci.need(not after["compute_apps"], "Compute appeared on assigned inventory GPUs")
        tdp = classify_tdp((run_dir / "nvidia-smi.xml").read_text(), devices)
        ci.write(run_dir / "hardware-profile.json", {"schema_version": 1, "observation_kind": "read_only_inventory",
                 "observed_at": ci.now(), "source_sha": context["source_sha"], "git_commit": context["source_sha"],
                 "ci": context["ci"], "source_files": context["package_files"], "source_hardware": context["target"], "interpreter": interpreter,
                 "run_id": context["run_id"], "run_attempt": context["run_attempt"], "slurm": binding,
                 "gpu_uuids": devices, "gpus": after["gpus"], "dmi": dmi, "power_configuration": power,
                 "hardware_variant": tdp["hardware_variant"], "variant_status": tdp["status"], "tdp": tdp,
                 "historical_benchmark_power_limits": "not_observed", "raw": {"inventory": "nvidia-smi.xml", "topology": "topology.txt"}})
        result.update(exit_code=0, inventory_completed=True)
    except (Exception, KeyboardInterrupt) as error:
        result["error"] = str(error)
    finally:
        ci.write(run_dir / "step-result.json", result)
    return result["exit_code"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-run-ids", help="One or two successful original H3 CI run IDs")
    parser.add_argument("--enter", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--inside", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.enter:
        enter(args.enter)
        return 2
    if args.inside:
        return inside(args.inside)
    ci.need(args.config is not None and args.output is not None, "--config and --output required")
    try:
        from export_ci import source_ids
        return collect_inventory(ci.read(args.config), args.output, source_run_ids=source_ids(args.source_run_ids or os.environ.get("H3_SOURCE_RUN_IDS", "")))
    except (Exception, KeyboardInterrupt) as error:
        args.output.mkdir(parents=True, exist_ok=True)
        ci.write(args.output / "preflight-error.json", {"operation": "hardware_inventory", "error": str(error), "exit_code": 2, "recorded_at": ci.now()})
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
