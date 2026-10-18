"""Fake-scheduler inventory tests; no GPU or model execution."""
import copy
import sys
from pathlib import Path

import pytest

import ci
import inventory_ci as inv
from test_ci import allocation, config
from test_mvp_gpu_job import spec


def prepared(tmp_path, spec):
    cfg = config(tmp_path)
    Path(cfg["runtime"]["rootfs"]).mkdir()
    Path(cfg["runtime"]["ready_marker"]).write_text("prepared test runtime")
    Path(cfg["runtime"]["entry"]).write_text("test-only entry")
    cfg["runtime"]["entry_sha256"] = ci.digest(cfg["runtime"]["entry"])
    python = ci.host_path(cfg, cfg["runtime"]["python"])
    python.parent.mkdir(parents=True)
    python.touch()
    ci.write(cfg["spec"]["path"], spec)
    cfg["spec"]["sha256"] = ci.digest(cfg["spec"]["path"])
    return cfg


def test_inventory_keeps_pins_without_loading_or_requiring_model_files(tmp_path, spec, monkeypatch):
    cfg = prepared(tmp_path, spec)
    original = copy.deepcopy(cfg)
    for item in spec["model"]["files"]:
        (Path(spec["model"]["path"]) / item["path"]).unlink()
    monkeypatch.setattr(ci, "prepared_spec", lambda cfg: pytest.fail("Model preparation must not run"))
    bounded, approval = inv.prepare(cfg)
    assert bounded["resources"] == {"gpus": 4, "cpus": 4, "memory_gb": 8, "minutes": 10}
    assert cfg == original and approval["compute_approved"] is True
    Path(cfg["runtime"]["entry"]).write_text("changed")
    with pytest.raises(ValueError, match="entry script changed"):
        inv.prepare(cfg)


def test_inventory_rejects_missing_approval_before_scheduler(tmp_path, spec, monkeypatch):
    spec["authorization"]["compute_approved"] = False
    cfg = prepared(tmp_path, spec)
    monkeypatch.setattr(ci, "command", lambda *args: pytest.fail("Must reject before scheduler"))
    with pytest.raises(ValueError, match="lacks approval"):
        inv.collect_inventory(cfg, tmp_path / "out")


@pytest.mark.parametrize("reused", [False, True])
@pytest.mark.parametrize("step_fails", [False, True])
def test_inventory_lifecycle_retains_logs_and_only_releases_owned_holder(tmp_path, spec, monkeypatch, reused, step_fails):
    cfg = prepared(tmp_path, spec)
    receipt, record = allocation(tmp_path)
    monkeypatch.setenv("GITHUB_RUN_ID", "456")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setenv("H3_SOURCE_SHA", "a" * 40)
    monkeypatch.setattr(ci, "command", lambda argv, **kw: "a" * 40 if "rev-parse" in argv else "")
    monkeypatch.setattr(ci, "stage_package", lambda *args: {})
    monkeypatch.setattr(inv, "source_target", lambda *args: {"node": "h200-node", "gpu_uuids": [], "sources": []})
    monkeypatch.setattr(ci, "recover", lambda *args, **kwargs: {"action": "reuse", "receipt": receipt} if reused else {"action": "allocate"})
    monkeypatch.setattr(ci, "allocate", lambda *args, **kwargs: receipt)
    monkeypatch.setattr(ci, "job_record", lambda job: record)
    drained, released = [], []
    monkeypatch.setattr(ci, "drain_step", lambda *args: drained.append("step") or {"status": "ended"})
    monkeypatch.setattr(ci, "stop_allocation", lambda *args: released.append("holder") or {"status": "released"})
    def run_step(argv, log, seconds):
        assert "--gpus-per-task=4" in argv and "--cpus-per-task=4" in argv
        assert "--mem=8G" in argv and "--time=5" in argv
        assert Path(argv[-3]).name == "inventory_ci.py" and argv[-2] == "--enter"
        assert seconds == 360
        log.write_text("retained inventory query output")
        if step_fails:
            raise RuntimeError("test inventory failure")
        ci.write(log.parent / "step-result.json", {"exit_code": 0, "inventory_completed": True})
        ci.write(log.parent / "hardware-profile.json", {"observation_kind": "test-only"})
        return 0
    monkeypatch.setattr(ci, "run_step", run_step)
    output = tmp_path / "out"
    assert inv.collect_inventory(cfg, output) == (2 if step_fails else 0)
    state = ci.read(output / "ci.json")
    assert state["phase"] == ("failed" if step_fails else "complete")
    assert drained == ["step"] and released == ([] if reused else ["holder"])
    assert (output / "srun.log").read_text() == "retained inventory query output"
    assert ci.read(output / "manifest.json")["evidence"]["ci.json"] == ci.digest(output / "ci.json")
    assert "srun.log" in (output / "SHA256SUMS").read_text()


def test_entry_retains_step_before_rejecting_changed_runtime(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    Path(cfg["runtime"]["entry"]).write_text("drift")
    receipt, record = allocation(tmp_path)
    ci.write(tmp_path / "context.json", {"config": cfg, "allocation": receipt, "node": record["NodeList"]})
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_STEP_ID", "7")
    monkeypatch.setenv("SLURMD_NODENAME", record["NodeList"])
    monkeypatch.setattr(inv.os, "sched_getaffinity", lambda pid: {1, 2, 3, 4}, raising=False)
    monkeypatch.setattr(inv.os, "execv", lambda *args: pytest.fail("Drifted runtime must not start"))
    with pytest.raises(ValueError, match="Entry changed"):
        inv.enter(tmp_path)
    assert ci.read(tmp_path / "binding.json")["step_id"] == "7"


def inside_context(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg["runtime"]["python"] = sys.executable
    cfg["resources"] = dict(inv.RESOURCES)
    receipt, record = allocation(tmp_path)
    ci.write(tmp_path / "context.json", {"config": cfg, "allocation": receipt, "node": record["NodeList"],
             "package_files": {}, "source_sha": "a" * 40, "run_id": "456", "run_attempt": "1",
             "target": {"node": record["NodeList"], "gpu_uuids": ["GPU-a", "GPU-b", "GPU-c", "GPU-d"], "sources": []},
             "ci": {"run_id": "456", "run_attempt": "1", "repository": "SemiAnalysisAI/InferenceX", "run_url": "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/456"}})
    ci.write(tmp_path / "binding.json", {"job_id": "123", "step_id": "7", "node": record["NodeList"], "cpu_affinity": [1, 2, 3, 4]})
    for key, value in {"SLURM_JOB_ID": "123", "SLURM_STEP_ID": "7", "SLURMD_NODENAME": record["NodeList"], "SLURM_PROCID": "0", "SLURM_NTASKS": "1", "H3_ASSIGNED_GPU_UUIDS": "GPU-a,GPU-b,GPU-c,GPU-d"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(ci, "inventory", lambda *args: {})
    monkeypatch.setattr(inv.os, "sched_getaffinity", lambda pid: {1, 2, 3, 4}, raising=False)
    monkeypatch.setattr(inv, "cuda_devices", lambda: ["GPU-a", "GPU-b", "GPU-c", "GPU-d"])


def test_inventory_rejects_uuid_mismatch_before_any_query(tmp_path, monkeypatch):
    inside_context(tmp_path, monkeypatch)
    monkeypatch.setenv("H3_ASSIGNED_GPU_UUIDS", "GPU-foreign,GPU-b,GPU-c,GPU-d")
    monkeypatch.setattr(inv, "GpuProbe", lambda *args, **kwargs: pytest.fail("Must reject before probe"))
    assert inv.inside(tmp_path) == 2
    assert "UUIDs differ" in ci.read(tmp_path / "step-result.json")["error"]


def test_inventory_collects_current_limits_without_inventing_variant_or_history(tmp_path, monkeypatch):
    inside_context(tmp_path, monkeypatch)
    calls = []
    class Probe:
        def __init__(self, devices, timeout):
            self.devices = devices
        def snapshot(self):
            calls.append("snapshot")
            return {"compute_apps": [], "gpus": [{"uuid": uuid, "name": "NVIDIA H200"} for uuid in self.devices]}
        def power_configuration(self):
            calls.append("power_configuration")
            return {"status": "recorded", "gpus": [{"uuid": uuid, "configured_limit_w": 700} for uuid in self.devices]}
    monkeypatch.setattr(inv, "GpuProbe", Probe)
    def capture(argv, path):
        calls.append(argv)
        path.write_text("test-only raw inventory")
    monkeypatch.setattr(inv, "capture", capture)
    assert inv.inside(tmp_path) == 0
    profile = ci.read(tmp_path / "hardware-profile.json")
    assert calls.count("snapshot") == 2 and "power_configuration" in calls
    assert ["nvidia-smi", "-q", "-x"] in calls and ["nvidia-smi", "topo", "-m"] in calls
    assert profile["hardware_variant"] is None
    assert profile["historical_benchmark_power_limits"] == "not_observed"
    assert profile["power_configuration"]["gpus"][0]["configured_limit_w"] == 700
    assert profile["run_id"] == "456" and profile["slurm"]["step_id"] == "7"


def test_capture_preserves_partial_query_output_on_timeout(tmp_path, monkeypatch):
    import subprocess
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(["nvidia-smi"], 20, output=b"partial XML", stderr=b"query stalled")
    monkeypatch.setattr(inv.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        inv.capture(["nvidia-smi", "-q", "-x"], tmp_path / "inventory.xml")
    assert (tmp_path / "inventory.xml").read_text() == "partial XML"
    assert (tmp_path / "inventory.xml.stderr.log").read_text() == "query stalled"


def test_inventory_preserves_foreign_compute_evidence_and_stops_queries(tmp_path, monkeypatch):
    inside_context(tmp_path, monkeypatch)
    class BusyProbe:
        def __init__(self, *args, **kwargs):
            pass
        def snapshot(self):
            return {"compute_apps": [{"gpu_uuid": "GPU-a", "pid": 999}], "gpus": []}
    monkeypatch.setattr(inv, "GpuProbe", BusyProbe)
    monkeypatch.setattr(inv, "capture", lambda *args: pytest.fail("Busy GPU must stop inventory"))
    assert inv.inside(tmp_path) == 2
    assert ci.read(tmp_path / "gpu-before.json")["compute_apps"][0]["pid"] == 999
    assert "active compute" in ci.read(tmp_path / "step-result.json")["error"]


def xml_gpu(uuid="GPU-a", device="0x233510DE", subsystem="0x18BE10DE", name="NVIDIA H200"):
    return f"<gpu><uuid>{uuid}</uuid><product_name>{name}</product_name><pci><pci_device_id>{device}</pci_device_id><pci_sub_system_id>{subsystem}</pci_sub_system_id></pci></gpu>"


def test_tdp_uses_vendor_pci_identity_and_distinguishes_nvl():
    sxm = inv.classify_tdp("<nvidia_smi_log>" + xml_gpu() + "</nvidia_smi_log>", ["GPU-a"])
    assert sxm["status"] == "verified" and sxm["watts_per_gpu"] == 700
    assert sxm["hardware_variant"] == "H200 SXM"
    nvl = inv.classify_tdp("<nvidia_smi_log>" + xml_gpu(device="0x233B10DE", subsystem="0x199610DE", name="NVIDIA H200 NVL") + "</nvidia_smi_log>", ["GPU-a"])
    assert nvl["status"] == "verified" and nvl["watts_per_gpu"] == 600
    assert nvl["hardware_variant"] == "H200 NVL"


@pytest.mark.parametrize("rows,devices", [
    (xml_gpu(device="0x233B10DE", subsystem="0x199610DE"), ["GPU-a"]),
    (xml_gpu(subsystem="0xFFFFFFFF"), ["GPU-a"]),
    (xml_gpu(), ["GPU-foreign"]),
    (xml_gpu() + xml_gpu(), ["GPU-a"]),
    (xml_gpu() + xml_gpu(uuid="GPU-b", device="0x233B10DE", subsystem="0x199610DE", name="NVIDIA H200 NVL"), ["GPU-a", "GPU-b"]),
])
def test_tdp_never_guesses_from_generic_name_or_incomplete_identity(rows, devices):
    result = inv.classify_tdp("<nvidia_smi_log>" + rows + "</nvidia_smi_log>", devices)
    assert result["status"] == "unknown" and result["watts_per_gpu"] is None


def saved_source(tmp_path, run_id, *, node="h200-node", devices=None):
    import export_ci
    cfg = config(tmp_path)
    directory = tmp_path / "results" / cfg["task_id"] / f"github-{run_id}-1"
    (directory / "gpu").mkdir(parents=True)
    receipt, record = allocation(tmp_path)
    record["NodeList"] = node
    devices = devices or [f"GPU-00000000-0000-0000-0000-{index:012d}" for index in range(4)]
    url = f"https://github.com/{export_ci.REPOSITORY}/actions/runs/{run_id}"
    metadata = {"repository": export_ci.REPOSITORY, "run_url": url}
    state = {"task_id": cfg["task_id"], "run_id": run_id, "run_attempt": "1", "source_sha": "b" * 40,
             "ci": metadata, "exit_code": 0, "phase": "complete", "smoke_completed": True,
             "allocation": receipt, "slurm_job": record, "step_cleanup": {"status": "ended", "step_id": "123.0"}}
    context = {"config": cfg, "run_id": directory.name, "source_sha": "b" * 40, "allocation": receipt, "node": node}
    binding = {"node": node, "job_id": "123", "step_id": "0", "gpu_uuids": devices}
    for name, value in {"ci.json": state, "context.json": context, "binding.json": binding, "gpu/spec.json": {"gpu_uuids": devices}}.items():
        ci.write(directory / name, value)
    manifest = {"task_id": cfg["task_id"], "run_id": run_id, "run_attempt": "1", "git_commit": "b" * 40,
                "ci": metadata, "exit_code": 0, "slurm_allocation": receipt, "evidence": ci.inventory(directory)}
    ci.write(directory / "manifest.json", manifest)
    (directory / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in ci.inventory(directory).items()))
    source_ci = {"databaseId": int(run_id), "runAttempt": 1, "headSha": "b" * 40, "status": "completed", "conclusion": "success", "url": url}
    return directory, source_ci


def test_source_pin_joins_both_successful_runs_to_same_hardware(tmp_path, monkeypatch):
    import export_ci
    _, first = saved_source(tmp_path, "101")
    _, second = saved_source(tmp_path, "102")
    monkeypatch.setattr(export_ci, "verified_execution", lambda run: ({"101": first, "102": second}[run], {"name": "source-artifact"}))
    target = inv.source_target(config(tmp_path), ["101", "102"])
    assert target["node"] == "h200-node" and len(target["gpu_uuids"]) == 4
    assert [source["run_id"] for source in target["sources"]] == ["101", "102"]
    assert target["sources"][0]["receipt_hashes"]["binding.json"]


def test_source_pin_rejects_tampered_receipt_before_allocation(tmp_path, monkeypatch):
    import export_ci
    directory, source_ci = saved_source(tmp_path, "101")
    (directory / "binding.json").write_text("{}")
    monkeypatch.setattr(export_ci, "verified_execution", lambda run: (source_ci, {}))
    with pytest.raises(ValueError, match="differs from its seal"):
        inv.source_target(config(tmp_path), ["101"])


def test_source_pin_rejects_different_historical_nodes(tmp_path, monkeypatch):
    import export_ci
    _, first = saved_source(tmp_path, "101")
    _, second = saved_source(tmp_path, "102", node="other-h200")
    monkeypatch.setattr(export_ci, "verified_execution", lambda run: ({"101": first, "102": second}[run], {}))
    with pytest.raises(ValueError, match="different physical hardware"):
        inv.source_target(config(tmp_path), ["101", "102"])


def test_inventory_rejects_valid_slurm_assignment_of_different_historical_gpu(tmp_path, monkeypatch):
    inside_context(tmp_path, monkeypatch)
    context = ci.read(tmp_path / "context.json")
    context["target"]["gpu_uuids"][0] = "GPU-other-historical"
    ci.write(tmp_path / "context.json", context)
    monkeypatch.setattr(inv, "GpuProbe", lambda *args, **kwargs: pytest.fail("Historical mismatch must stop before probe"))
    assert inv.inside(tmp_path) == 2
    assert "historical source hardware" in ci.read(tmp_path / "step-result.json")["error"]


def test_source_pin_rejects_local_receipts_for_another_trusted_commit(tmp_path, monkeypatch):
    import export_ci
    _, source_ci = saved_source(tmp_path, "101")
    source_ci["headSha"] = "c" * 40
    monkeypatch.setattr(export_ci, "verified_execution", lambda run: (source_ci, {}))
    with pytest.raises(ValueError, match="CI/Git/task identities"):
        inv.source_target(config(tmp_path), ["101"])


def test_prepare_accepts_interpreter_symlink_resolved_only_inside_container(tmp_path, spec):
    cfg = prepared(tmp_path, spec)
    cfg["runtime"]["python"] = "/usr/bin/python3"
    rootfs = Path(cfg["runtime"]["rootfs"])
    interpreter = rootfs / "usr/bin/python3"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to("/etc/alternatives/h3-inventory-test-python")
    target = rootfs / "etc/alternatives/h3-inventory-test-python"
    target.parent.mkdir(parents=True)
    target.write_text("container-only interpreter target")
    assert interpreter.is_symlink() and not interpreter.is_file()
    assert inv.prepare(cfg)[0]["runtime"]["python"] == "/usr/bin/python3"
    interpreter.unlink()
    with pytest.raises(ValueError, match="Prepared interpreter missing"):
        inv.prepare(cfg)


@pytest.mark.parametrize("fault", ["source_ids", "config"])
def test_cli_retains_preflight_errors_without_allocation(tmp_path, monkeypatch, capsys, fault):
    import sys
    config_path = tmp_path / "invalid.json"
    ci.write(config_path, {})
    output = tmp_path / "diagnostics"
    monkeypatch.setattr(sys, "argv", ["inventory_ci.py", "--config", str(config_path), "--output", str(output),
                                     "--source-run-ids", "invalid" if fault == "source_ids" else "101"])
    monkeypatch.setattr(ci, "allocate", lambda *args, **kwargs: pytest.fail("Preflight must not allocate"))
    assert inv.main() == 2
    error = ci.read(output / "preflight-error.json")
    assert error["exit_code"] == 2 and error["error"]
    assert error["error"] in capsys.readouterr().err


def test_inside_rejects_different_interpreter_before_gpu_inventory(tmp_path, monkeypatch):
    inside_context(tmp_path, monkeypatch)
    other = tmp_path / "different-python"
    other.write_text("test-only executable")
    other.chmod(0o755)
    context = ci.read(tmp_path / "context.json")
    context["config"]["runtime"]["python"] = str(other)
    ci.write(tmp_path / "context.json", context)
    monkeypatch.setattr(inv, "cuda_devices", lambda: pytest.fail("Reject wrong interpreter before any GPU query"))
    assert inv.inside(tmp_path) == 2
    assert "configured container interpreter" in ci.read(tmp_path / "step-result.json")["error"]


def test_tdp_accepts_unprefixed_hex_from_live_nvidia_smi_xml():
    xml = "<nvidia_smi_log>" + xml_gpu(device="233510DE", subsystem="18BE10DE") + "</nvidia_smi_log>"
    result = inv.classify_tdp(xml, ["GPU-a"])
    assert result["status"] == "verified"
    assert result["hardware_variant"] == "H200 SXM" and result["watts_per_gpu"] == 700
    assert result["evidence"]["devices"][0]["pci_device_id"] == "233510DE"


def test_tdp_rejects_invalid_hex_even_with_matching_product_name():
    xml = "<nvidia_smi_log>" + xml_gpu(device="233510DG", subsystem="18BE10DE") + "</nvidia_smi_log>"
    result = inv.classify_tdp(xml, ["GPU-a"])
    assert result["status"] == "unknown" and result["watts_per_gpu"] is None
