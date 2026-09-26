"""CPU checks for observed AMD SMI shapes; these are not GPU measurements."""
import json
from types import SimpleNamespace

import pytest

from evaluator import mvp_amd_gpu as amd


GPU = "11111111-0000-1000-8000-000000000001"
OTHER = "11111111-0000-1000-8000-000000000002"


def observations():
    # Reduced, anonymized shape from CI 34411615416 / AMD SMI 26.2.1.
    return {
        "list": [{"gpu": 0, "bdf": "0000:05:00.0", "uuid": GPU, "partition_id": 0}],
        "static": {"gpu_data": [{"gpu": 0, "bus": {"bdf": "0000:05:00.0"},
                    "asic": {"market_name": "AMD Instinct MI355X"}, "driver": {"version": "6.16.6"},
                    "limit": {"ppt0": {"socket_power_limit": {"value": 1400, "unit": "W"},
                                       "max_power_limit": {"value": 1400, "unit": "W"}},
                              "ppt1": {"socket_power_limit": "N/A", "max_power_limit": "N/A"}}}]},
        "metric": {"gpu_data": [{"gpu": 0,
                   "mem_usage": {"total_vram": {"value": 294896, "unit": "MB"}, "used_vram": {"value": 283, "unit": "MB"}},
                   "usage": {"gfx_activity": {"value": 0, "unit": "%"}},
                   "power": {"socket_power": {"value": 239, "unit": "W"}},
                   "temperature": {"hotspot": {"value": 36, "unit": "C"}}}]},
        "process": [{"gpu": 0, "process_list": [{"process_info": {"pid": 42, "memory_usage": {"vram_mem": {"value": 0, "unit": "B"}}}}]}],
    }


@pytest.mark.parametrize("legacy_static", [False, True])
def test_snapshot_preserves_zero_memory_process_and_measured_power(monkeypatch, legacy_static):
    data = observations()
    if legacy_static:
        data["static"] = data["static"]["gpu_data"]
        data["static"][0]["limit"] = {
            "socket_power": {"value": 1400, "unit": "W"},
            "max_power": {"value": 1400, "unit": "W"},
        }
    monkeypatch.setattr(amd, "_command", lambda command, **kwargs: json.dumps(data[command[1]]))
    probe = amd.AmdGpuProbe([GPU], 2)
    result = probe.snapshot()
    assert result["compute_apps"] == [{"gpu_uuid": GPU, "pid": 42, "memory_used_mib": 0.0}]
    assert result["gpus"][0]["power_watts"] == 239
    assert result["gpus"][0]["memory_used_mib"] == 283
    limits = probe.power_configuration()["gpus"][0]
    assert limits["configured_limit_w"] == 1400
    assert limits["maximum_limit_w"] == 1400
    assert limits["enforced_limit_w"] is None
    data["metric"]["gpu_data"][0]["power"]["socket_power"] = "N/A"
    assert probe.snapshot()["gpus"][0]["power_watts"] is None


@pytest.mark.parametrize("payload", [
    {"error": "query failed"}, {"gpu_data": None}, {"gpu_data": []},
    {"gpu_data": ["not a GPU record"]}, {"gpu_data": [{"gpu": True}]},
    {"gpu_data": [{"gpu": 0}, {"gpu": 0}]},
])
def test_smi_rejects_missing_malformed_or_duplicate_records(monkeypatch, payload):
    monkeypatch.setattr(amd, "_command", lambda *args, **kwargs: json.dumps(payload))
    with pytest.raises(RuntimeError):
        amd.smi("static", 2)


def test_missing_optional_values_are_not_zero_and_required_units_fail(monkeypatch):
    data = observations()
    data["static"]["gpu_data"][0]["limit"]["ppt0"] = {"socket_power_limit": "N/A"}
    data["process"][0]["process_list"][0]["process_info"]["memory_usage"]["vram_mem"] = "N/A"
    monkeypatch.setattr(amd, "_command", lambda command, **kwargs: json.dumps(data[command[1]]))
    probe = amd.AmdGpuProbe([GPU], 2)
    limits = probe.power_configuration()["gpus"][0]
    assert limits["configured_limit_w"] is None and limits["maximum_limit_w"] is None
    assert probe.snapshot()["compute_apps"][0]["memory_used_mib"] is None
    data["metric"]["gpu_data"][0]["mem_usage"]["used_vram"]["unit"] = "GB"
    with pytest.raises(RuntimeError, match="unit unavailable: MB"):
        probe.snapshot()


@pytest.mark.parametrize("field,value", [("bdf", "0000:15:00.0"), ("market_name", "Other GPU")])
def test_static_identity_must_match_assigned_gpu(monkeypatch, field, value):
    data = observations()
    row = data["static"]["gpu_data"][0]
    row["bus" if field == "bdf" else "asic"][field] = value
    monkeypatch.setattr(amd, "_command", lambda command, **kwargs: json.dumps(data[command[1]]))
    with pytest.raises(RuntimeError, match="identity changed or unsupported GPU"):
        amd.AmdGpuProbe([GPU], 2)


def test_hip_order_joins_physical_identity_instead_of_smi_ordinal(monkeypatch):
    rows = observations()["list"] + [{"gpu": 1, "bdf": "0000:15:00.0", "uuid": OTHER, "partition_id": 0}]
    monkeypatch.setattr(amd, "smi", lambda *args: rows)
    def count(pointer):
        pointer._obj.value = 2
        return 0
    def bdf(buffer, size, ordinal):
        buffer.value = [b"0000:15:00.0", b"0000:05:00.0"][ordinal]
        return 0
    monkeypatch.setattr(amd.ctypes, "CDLL", lambda _: SimpleNamespace(hipGetDeviceCount=count, hipDeviceGetPCIBusId=bdf))
    assert amd.hip_devices() == [OTHER, GPU]


@pytest.mark.parametrize("mutation", [
    lambda rows: rows.append(dict(rows[0])),
    lambda rows: rows[0].update(partition_id=1),
    lambda rows: rows[0].update(bdf="../../other"),
])
def test_ambiguous_or_partitioned_device_inventory_is_rejected(mutation):
    rows = observations()["list"]
    mutation(rows)
    with pytest.raises(RuntimeError):
        amd.inventory(rows)


@pytest.mark.parametrize("value", [{"value": 239, "unit": "mW"}, {"value": True, "unit": "W"}, {"value": float("nan"), "unit": "W"}])
def test_invalid_power_unit_or_value_is_not_a_measurement(value):
    assert amd.number(value, "W", optional=True) is None
    with pytest.raises(RuntimeError):
        amd.number(value, "W")


@pytest.mark.parametrize("change", ["pid", "start_ticks", "executable", "uid", "memory", "unverified"])
def test_monitor_exception_rejects_changed_or_active_context(monkeypatch, change):
    identity = {"pid": 42, "start_ticks": 50, "executable": "/opt/gpuagent/gpuagent", "uid": 0}
    receipt = {"status": "verified", "process": identity}
    app = {"pid": 42, "memory_used_mib": 0}
    observed = dict(identity)
    if change == "memory":
        app["memory_used_mib"] = 1
    elif change == "unverified":
        receipt["status"] = "unverified"
    elif change == "executable":
        observed[change] = "/tmp/other"
    else:
        observed[change] += 1
    monkeypatch.setattr(amd, "monitor_process", lambda pid: observed)
    assert not amd.is_system_monitor(app, receipt)


def test_verified_monitor_is_retained_separately_from_workload_contexts(monkeypatch):
    data = observations()
    monkeypatch.setattr(amd, "_command", lambda command, **kwargs: json.dumps(data[command[1]]))
    probe = amd.AmdGpuProbe([GPU], 2)
    identity = {"pid": 42, "start_ticks": 50, "executable": "/opt/gpuagent/gpuagent", "uid": 0}
    probe.monitor = {"status": "verified", "service": "gpuagent.service", "process": identity}
    monkeypatch.setattr(amd, "monitor_process", lambda pid: dict(identity))
    result = probe.snapshot()
    assert result["compute_apps"] == []
    excluded = result["excluded_system_monitor_contexts"]
    assert len(excluded) == 1 and excluded[0]["pid"] == 42 and excluded[0]["gpu_uuid"] == GPU
    assert excluded[0]["identity"]["service"] == "gpuagent.service"


def test_service_pid_mismatch_never_approves_monitor(monkeypatch):
    monkeypatch.setattr(amd, "_command", lambda *a, **k: "MainPID=42\nExecMainPID=43\nExecStart={ path=/opt/gpuagent/gpuagent ; }\nActiveState=active\nSubState=running\nControlGroup=/system.slice/gpuagent.service\nType=simple\n")
    monkeypatch.setattr(amd, "monitor_process", lambda pid: pytest.fail("mismatched service PID must be rejected"))
    assert amd.observe_system_monitor(1)["status"] == "unverified"


def test_bytes_service_description_is_decoded_before_parsing(monkeypatch):
    # _command returns bytes by contract; parsing must decode instead of raising TypeError.
    description = (b"MainPID=42\nExecMainPID=43\nExecStart={ path=/opt/gpuagent/gpuagent ; }\n"
                   b"ActiveState=active\nSubState=running\nControlGroup=/system.slice/gpuagent.service\nType=simple\n")
    monkeypatch.setattr(amd, "_command", lambda *a, **k: description)
    monkeypatch.setattr(amd, "monitor_process", lambda pid: pytest.fail("mismatched service PID must be rejected"))
    record = amd.observe_system_monitor(1)
    assert record["status"] == "unverified" and record["error_type"] == "ValueError"


def test_unreadable_monitor_identity_remains_a_foreign_process(monkeypatch):
    def unreadable(pid):
        raise PermissionError("process identity unavailable")
    monkeypatch.setattr(amd, "monitor_process", unreadable)
    assert not amd.is_system_monitor({"pid": 42, "memory_used_mib": 0}, {"status": "verified", "process": {"pid": 42}})
