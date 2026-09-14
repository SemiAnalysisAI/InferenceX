"""Bounded fake-scheduler checks; these do not establish AMD runtime readiness."""
from datetime import datetime, timedelta, timezone

import pytest

import ci
import run_amd_serving_ci as campaign


def test_allocation_deadline_includes_probe_and_preserves_cleanup_budget():
    now = datetime.now(timezone.utc)
    minutes, ready = campaign.allocation_budget({"StartTime": (now - timedelta(minutes=12)).isoformat(),
                                                "EndTime": (now + timedelta(minutes=108)).isoformat()})
    assert 105 <= minutes <= 107
    assert 175 < (ready - now).total_seconds() <= 180
    with pytest.raises(ValueError, match="readiness deadline"):
        campaign.allocation_budget({"StartTime": (now - timedelta(minutes=16)).isoformat(),
                                    "EndTime": (now + timedelta(minutes=104)).isoformat()})
    with pytest.raises(ValueError, match="75 serving minutes"):
        campaign.allocation_budget({"StartTime": now.isoformat(),
                                    "EndTime": (now + timedelta(minutes=70)).isoformat()})


@pytest.mark.parametrize("seconds", [None, 901, 899])
def test_readiness_requires_actual_server_evidence_before_deadline(tmp_path, seconds):
    start = datetime.now(timezone.utc)
    path = tmp_path / "gpu/c1/gpu-job.json"
    path.parent.mkdir(parents=True)
    role = {"status": "starting", "startup_timing_window": {"start_utc": start.isoformat()}}
    if seconds is not None:
        role["startup_seconds"] = seconds
    ci.write(path, {"roles": {"baseline": role}})
    if seconds == 899:
        assert campaign.ready_before(tmp_path, start + timedelta(minutes=15))["ready_at"]
    else:
        with pytest.raises(ValueError, match="readiness"):
            campaign.ready_before(tmp_path, start + timedelta(minutes=15))


@pytest.mark.parametrize("failure", ["inspection", "stage", None])
@pytest.mark.parametrize("reused", [False, True])
def test_one_lease_released_after_failure_or_measurement_and_resealed(tmp_path, monkeypatch, failure, reused):
    workspace = tmp_path / "work"
    monkeypatch.setattr(campaign.stage_amd_site, "WORKSPACE", workspace)
    monkeypatch.setenv("H3_RUN_ID", "456")
    monkeypatch.setenv("H3_RUN_ATTEMPT", "1")
    monkeypatch.setattr(campaign, "source_spec", lambda *args: ({"plan": ci.read(campaign.stage_amd_site.INPUTS / "formal-8s-plan.json")}, {"source_run_id": "123"}))
    monkeypatch.setattr(campaign, "prepare_source", lambda *args: workspace)
    monkeypatch.setattr(campaign.stage_amd_site, "timing_source", lambda *args: (workspace, {}))
    monkeypatch.setattr(campaign.stage_amd_site, "runtime_probe", lambda *args: {})
    monkeypatch.setattr(campaign.signal, "setitimer", lambda *args: None)
    receipt = {"identity": {"JobId": "789"}}
    run_dir = workspace / "results/h3-cross-hardware/github-456-1"
    def inspect(path, output, **kwargs):
        assert kwargs == {"prepare_runtime": True, "serving_continuation": True}
        prep = run_dir.with_name(run_dir.name + "-runtime")
        prep.mkdir(parents=True)
        ci.write(prep / "allocation.json", receipt)
        ci.write(prep / "recovery.json", {"action": "reuse" if reused else "allocate"})
        status = {"allocation": receipt, "allocation_reused": reused, "status": "failed" if failure == "inspection" else "complete"}
        ci.write(prep / "inventory-status.json", status)
        ci.collect(prep, output)
        ci.write(workspace / "campaigns/h3-cross-hardware/runtime-inspected.json", {})
        return 2 if failure == "inspection" else 0
    monkeypatch.setattr(campaign.inspect_amd_node, "inspect", inspect)
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(ci, "job_record", lambda job: {"StartTime": now.isoformat(), "EndTime": (now + timedelta(minutes=108)).isoformat()})
    monkeypatch.setattr(ci, "verify_identity", lambda *args: None)
    def stage(spec, output, **kwargs):
        if failure == "stage":
            raise ValueError("runtime differs after inspection")
        ci.write(output / "site.json", {"resources": {"minutes": kwargs["allocation_minutes"]}})
        return {"site_config": str(output / "site.json")}
    monkeypatch.setattr(campaign.stage_amd_site, "stage", stage)
    launched = []
    def launch(config, output, *, required_allocation):
        launched.append(required_allocation)
        run_dir.mkdir()
        ci.write(run_dir / "ci.json", {"allocation_cleanup": {"status": "retained"}})
        ci.write(run_dir / "manifest.json", {"evidence": {}})
        role_path = run_dir / "gpu/c1/gpu-job.json"
        role_path.parent.mkdir(parents=True)
        ci.write(role_path, {"roles": {"baseline": {"startup_seconds": 1, "startup_timing_window": {"start_utc": now.isoformat()}}}})
        ci.collect(run_dir, output)
        return 0
    monkeypatch.setattr(ci, "launch", launch)
    released = []
    monkeypatch.setattr(ci, "stop_allocation", lambda owned, task: released.append(owned) or {"status": "released"})
    output = tmp_path / "artifact"
    assert campaign.run("123", output) == (2 if failure else 0)
    assert released == ([] if reused else [receipt])
    assert launched == ([] if failure else ["789"])
    assert ci.read(output / "amd-serving.json")["allocation_cleanup"]["status"] == ("retained" if reused else "released")
    assert (output / "preparation/allocation.json").is_file()
    if not failure:
        manifest = ci.read(output / "manifest.json")
        assert manifest["allocation_cleanup"]["status"] == ("retained" if reused else "released")
        assert manifest["evidence"]["ci.json"] == ci.digest(output / "ci.json")
    sums = dict(line.split("  ", 1)[::-1] for line in (output / "SHA256SUMS").read_text().splitlines())
    assert sums["amd-serving.json"] == ci.digest(output / "amd-serving.json")
