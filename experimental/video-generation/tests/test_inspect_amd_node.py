"""Fake scheduler checks for ownership and cleanup, not AMD hardware evidence."""
from types import SimpleNamespace

import pytest

import ci
import inspect_amd_node as amd
from test_ci import allocation


def test_wrong_node_stops_before_device_queries(tmp_path, monkeypatch):
    ci.write(tmp_path / "context.json", {"allocation": {"identity": {"JobId": "123"}}, "node": "amd-node"})
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURMD_NODENAME", "another-node")
    monkeypatch.setattr(amd, "observation", lambda argv: pytest.fail("foreign node must not be queried"))
    with pytest.raises(ValueError, match="Wrong AMD inventory allocation"):
        amd.inspect_node(tmp_path)
    assert not (tmp_path / "binding.json").exists()


@pytest.mark.parametrize("value", ["0-7", "0,1,2,3,4,5,6,7"])
def test_full_amd_step_assignment(value):
    assert amd.step_gpu_indices(value) == set(range(8))


@pytest.mark.parametrize("value", ["", "0-99", "7-0", "0;id"])
def test_invalid_amd_step_assignment(value):
    with pytest.raises(ValueError, match="AMD step GPU assignment"):
        amd.step_gpu_indices(value)


@pytest.mark.parametrize("reused", [False, True])
def test_failed_inventory_drains_only_owned_step_and_preserves_borrowed_allocation(tmp_path, monkeypatch, reused):
    monkeypatch.setenv("H3_RUN_ID", "456")
    monkeypatch.setenv("H3_RUN_ATTEMPT", "1")
    account = "cameronamd@semianalysis.com"
    monkeypatch.setattr(amd.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_name=account))
    receipt, record = allocation(tmp_path)
    site = {"cluster": "mi355x-amds", "partition": "compute", "account": account, "gpu_model": "MI355X"}
    receipt.update(task_id="h3-cross-hardware", site=site)
    receipt["identity"].update(Account=account, Partition="compute")
    record.update(receipt["identity"])
    monkeypatch.setattr(ci, "recover", lambda *args: {"action": "reuse" if reused else "allocate", "receipt": receipt, "active_steps": ""})
    def allocate(config, root):
        assert not reused
        ci.write(root / "allocation.json", receipt)
        return receipt
    monkeypatch.setattr(ci, "allocate", allocate)
    monkeypatch.setattr(ci, "job_record", lambda job: record)
    monkeypatch.setattr(ci, "run_step", lambda *args: 1)
    cleanup = []
    monkeypatch.setattr(ci, "drain_step", lambda owned, task, root: cleanup.append(("step", owned["identity"]["JobId"])) or {"status": "ended"})
    monkeypatch.setattr(ci, "stop_allocation", lambda owned, task: cleanup.append(("allocation", owned["identity"]["JobId"])) or {"status": "released"})
    output = tmp_path / "output"
    assert amd.inspect(tmp_path / "work", output) == 2
    assert cleanup == [("step", "123")] + ([] if reused else [("allocation", "123")])
    assert ci.read(output / "inventory-status.json")["generation_executed"] is False
    assert (output / "SHA256SUMS").is_file()
