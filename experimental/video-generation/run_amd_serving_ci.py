"""Keep AMD runtime inspection, warmup and twenty C1 requests in one lease."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import shutil
import signal

import ci
import inspect_amd_node
import stage_amd_site
from prepare_amd_runtime import prepare_source
from stage_model_ci import source_spec


def allocation_budget(job: dict) -> tuple[int, datetime]:
    end = datetime.fromisoformat(job["EndTime"]).replace(tzinfo=timezone.utc)
    minutes = min(110, int((end - datetime.now(timezone.utc)).total_seconds() // 60) - 1)
    ci.need(minutes >= 75, "Fewer than 75 serving minutes remain; do not start the twenty-request campaign")
    ready_by = datetime.fromisoformat(job["StartTime"]).replace(tzinfo=timezone.utc) + timedelta(minutes=15)
    ci.need(ready_by > datetime.now(timezone.utc), "Allocation-wide runtime readiness deadline expired")
    return minutes, ready_by


def ready_before(run_dir: Path, deadline: datetime) -> dict:
    path = run_dir / "gpu/c1/gpu-job.json"
    ci.need(path.is_file(), "AMD runtime was not ready within 15 minutes of allocation start")
    role = ci.read(path).get("roles", {}).get("baseline", {})
    seconds = role.get("startup_seconds")
    start = role.get("startup_timing_window", {}).get("start_utc")
    ci.need(type(seconds) in (int, float) and seconds >= 0 and isinstance(start, str),
            "AMD runtime readiness was not recorded before the deadline")
    ready_at = datetime.fromisoformat(start) + timedelta(seconds=seconds)
    ci.need(ready_at <= deadline, "AMD runtime exceeded the 15-minute readiness deadline")
    return {"ready_at": ready_at.isoformat(), "deadline": deadline.isoformat(),
            "evidence": "gpu/c1/gpu-job.json: roles.baseline.startup_timing_window and startup_seconds"}


def run(source_run_id: str, output: Path) -> int:
    workspace = stage_amd_site.WORKSPACE
    run_id = f"github-{os.environ['H3_RUN_ID']}-{os.environ['H3_RUN_ATTEMPT']}"
    control = workspace / "campaigns/h3-cross-hardware" / run_id
    control.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=True)
    preparation = control / "preparation"
    run_dir = workspace / "results/h3-cross-hardware" / run_id
    prep_dir = run_dir.with_name(run_id + "-runtime")
    record = {"status": "preparing", "started_at": ci.now(), "allocation_minutes_cap": 120,
              "allocated_gpus": 8, "participating_gpus": 4, "planned_measured_requests": 20,
              "warmups": 1, "runtime_probe_seconds_cap": 660, "allocation_to_readiness_seconds_cap": 900,
              "cleanup_reserve_minutes": 10, "replacement_allocation_allowed": False}
    receipt, owned, code, ready_by = None, False, 2, None
    def interrupted(signum, frame):
        raise InterruptedError("AMD serving orchestration interrupted")
    def readiness_expired(signum, frame):
        record["readiness"] = ready_before(run_dir, ready_by)
    previous = {sig: signal.signal(sig, handler) for sig, handler in
                ((signal.SIGTERM, interrupted), (signal.SIGALRM, readiness_expired))}
    try:
        source_output = control / "source"
        source_output.mkdir()
        spec, record["source_provenance"] = source_spec(source_run_id, source_output)
        ci.need(spec["plan"] == ci.read(stage_amd_site.INPUTS / "formal-8s-plan.json"),
                "Source workload differs from the frozen twenty-request plan; no allocation requested")
        stage_amd_site.timing_source(prepare_source(workspace))
        code = inspect_amd_node.inspect(workspace, preparation, prepare_runtime=True, serving_continuation=True)
        inspected = ci.read(preparation / "inventory-status.json")
        receipt = inspected.get("allocation")
        owned = receipt is not None and not inspected.get("allocation_reused", False)
        record["runtime_inspection"] = inspected
        ci.need(code == 0, "AMD runtime inspection failed; measured requests were not started")
        stage_amd_site.runtime_probe(ci.read(workspace / "campaigns/h3-cross-hardware/runtime-inspected.json"))
        job = ci.job_record(receipt["identity"]["JobId"])
        ci.verify_identity(receipt, job, "h3-cross-hardware")
        minutes, ready_by = allocation_budget(job)
        record["readiness_deadline"] = ready_by.isoformat()
        signal.setitimer(signal.ITIMER_REAL, (ready_by - datetime.now(timezone.utc)).total_seconds())
        staged = control / "prepared-site"
        staged.mkdir()
        with ci.task_lock(workspace / "campaigns/h3-cross-hardware/.site-preparation.lock"):
            site = stage_amd_site.stage(spec, staged, server_timing=True, allocation_minutes=minutes,
                                        destination=control / "formal-c1")
        config = ci.read(site["site_config"])
        record.update(status="running", prepared_site=site, allocation=receipt,
                      serving_minutes_cap=minutes, supervisor_seconds_cap=minutes * 60 - 600)
        ci.write(control / "amd-serving.json", record)
        code = ci.launch(config, output, required_allocation=receipt["identity"]["JobId"])
        if code == 0:
            record["readiness"] = ready_before(run_dir, ready_by)
        record["status"] = "complete" if code == 0 else "failed"
    except (Exception, KeyboardInterrupt) as error:
        record.update(status="failed", error=str(error))
        code = 2
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            # Recover the persistent receipt even when artifact collection itself failed.
            inspected = ci.read(prep_dir / "inventory-status.json") if (prep_dir / "inventory-status.json").is_file() else {}
            if receipt is None:
                receipt = inspected.get("allocation")
                if receipt is None and (prep_dir / "allocation.json").is_file():
                    receipt = ci.read(prep_dir / "allocation.json")
                recovery = ci.read(prep_dir / "recovery.json") if (prep_dir / "recovery.json").is_file() else {}
                owned = receipt is not None and recovery.get("action") != "reuse"
            if receipt and owned:
                record["allocation_cleanup"] = (inspected["allocation_cleanup"]
                    if inspected.get("allocation_cleanup", {}).get("status") == "released"
                    else ci.stop_allocation(receipt, "h3-cross-hardware"))
            elif receipt:
                record["allocation_cleanup"] = {"status": "retained", "reason": "Borrowed allocation remains with its owner"}
        except Exception as error:
            record.update(status="failed", cleanup_error=str(error))
            code = 2
        record.update(finished_at=ci.now(), exit_code=code)
        try:
            ci.write(control / "amd-serving.json", record)
            run_dir.mkdir(parents=True, exist_ok=True)
            ci.write(run_dir / "amd-serving.json", record)
            if prep_dir.is_dir():
                ci.collect(prep_dir, run_dir / "preparation")
            if (control / "prepared-site").is_dir():
                shutil.copytree(control / "prepared-site", run_dir / "prepared-site", dirs_exist_ok=True)
            if (run_dir / "ci.json").is_file():
                state = ci.read(run_dir / "ci.json")
                state.update(allocation_cleanup=record.get("allocation_cleanup"), exit_code=code)
                if code != 0:
                    state.update(phase="failed", ci_accepted=False)
                ci.write(run_dir / "ci.json", state)
                manifest = ci.read(run_dir / "manifest.json")
                manifest.update(exit_code=code, allocation_cleanup=record.get("allocation_cleanup"))
                manifest["evidence"].update({name: ci.digest(run_dir / name) for name in ("ci.json", "amd-serving.json")})
                ci.write(run_dir / "manifest.json", manifest)
            ci.collect(run_dir, output)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(run(args.source_run_id, args.output))
