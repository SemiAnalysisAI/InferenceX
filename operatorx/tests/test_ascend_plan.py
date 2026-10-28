import copy
import importlib
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "ascend"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("prepare"), importlib.import_module("fp8_plan")


def entry(m=240):
    return {
        "type": "gemm",
        "args": {
            "m": m,
            "n": 2304,
            "k": 7168,
            "out": "bf16",
            "a": {
                "dtype": "e4m3",
                "scale": {"dtype": "fp32", "group": [1, 128], "static": False},
            },
            "b": {
                "dtype": "e4m3",
                "scale": {"dtype": "fp32", "group": [128, 128], "static": True},
            },
        },
    }


def observation(case, sha, base_m):
    a = case["args"]
    required = {240: 61440, 272: 69632}[base_m]
    return {
        "case_id": case["case_id"],
        "opType": "gemm",
        "args": a,
        "status": "inspected",
        "hardware_identity": "a" * 64,
        "launch_exception": "launch rejected: 100000",
        "provenance": {"worker_sha256": sha, "task_queue_enable": "0"},
        "guard": {
            "observed": True,
            "mode": 1,
            "forward_allowed": False,
            "decision": "within_capacity" if base_m == 240 else "over_capacity",
            "dispatch_key": 5124,
            "kernel_name": "QuantBatchMatmulV3_ND_ND_fp8e4m3_fp8e4m3_fp32_bf16_high_performance_5124",
            "m": a["m"],
            "n": 2304,
            "k": 7168,
            "expected_m": a["m"],
            "expected_n": 2304,
            "expected_k": 7168,
            "base_m": base_m,
            "base_n": 128,
            "base_k": 128,
            "l0a_bytes": required,
            "l0a_capacity_bytes": 65536,
        },
    }


def test_prepare_deduplicates_complete_arguments_and_preserves_sources(modules):
    prepare, _ = modules
    first = {
        "type": "gemm",
        "args": {
            "m": 8,
            "n": 16,
            "k": 32,
            "a": {"dtype": "bf16"},
            "b": {"dtype": "bf16"},
            "out": "bf16",
        },
        "sources": ["org/model/first"],
    }
    second = copy.deepcopy(first)
    second["sources"] = ["org/model/second"]
    other = copy.deepcopy(first)
    other["args"]["out"] = "fp32"
    plan = prepare.make_plan([first, second, other], "unquantized", 4 * 1024**3)
    assert plan["counts"] == {"new": 2}
    bf16 = next(c for c in plan["cases"] if c["args"]["out"] == "bf16")
    assert bf16["sources"] == ["org/model/first", "org/model/second"]
    assert bf16["memory"]["output_bytes"] == 256


def test_prepare_keeps_unsupported_and_over_budget_cases(modules):
    prepare, _ = modules
    tail = entry()
    tail["args"]["k"] = 129
    plan = prepare.make_plan([entry(), tail], "fp8", 1024)
    assert plan["counts"] == {"implementation_required": 1, "memory_review": 1}
    assert plan["smoke_case_ids"] == []


def test_guard_skips_oversized_tile_and_rejects_plan_tampering(modules):
    prepare, fp8 = modules
    plan = prepare.make_plan([entry(), entry(264)], "fp8", 8 * 1024**3)
    rows = [
        observation(c, plan["worker_sha256"], 240 if c["args"]["m"] == 240 else 272)
        for c in plan["cases"]
    ]
    approved = fp8.apply_guard(plan, rows)
    assert approved["counts"] == {"new": 1, "skipped_l0a_capacity": 1}
    assert len(approved["smoke_case_ids"]) == 1
    fp8.verify_plan(approved)
    bad = copy.deepcopy(approved)
    next(c for c in bad["cases"] if c["args"]["m"] == 264)["phase"] = "new"
    with pytest.raises(ValueError, match="differs"):
        fp8.verify_plan(bad)
    with pytest.raises(ValueError, match="Incomplete"):
        fp8.apply_guard(plan, rows[:1])


def test_guard_rejects_fault_or_different_hardware(modules):
    prepare, fp8 = modules
    plan = prepare.make_plan([entry(), entry(264)], "fp8", 8 * 1024**3)
    rows = [
        observation(c, plan["worker_sha256"], 240 if c["args"]["m"] == 240 else 272)
        for c in plan["cases"]
    ]
    rows[0]["launch_exception"] += " 507015"
    with pytest.raises(ValueError, match="prelaunch"):
        fp8.apply_guard(plan, rows)
    rows[0]["launch_exception"] = "launch rejected: 100000"
    rows[1]["hardware_identity"] = "b" * 64
    with pytest.raises(ValueError, match="mixed hardware"):
        fp8.apply_guard(plan, rows)


def test_controller_requires_fp8_approval_before_loading_device(modules, tmp_path):
    import json
    from types import SimpleNamespace

    prepare, _ = modules
    batch = importlib.import_module("operatorx_batch_campaign")
    plan = prepare.make_plan([entry()], "fp8", 8 * 1024**3)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    args = SimpleNamespace(
        plan=path,
        phase="smoke",
        run_dir=tmp_path / "run",
        worker_script=SCRIPTS / "operatorx_fp8_block_worker.py",
    )
    with pytest.raises(ValueError, match="approved prelaunch"):
        batch.run_locked(args)


def test_controller_rejects_changed_timing_dependency(modules, tmp_path):
    import json
    from types import SimpleNamespace

    prepare, _ = modules
    batch = importlib.import_module("operatorx_batch_campaign")
    plan = prepare.make_plan([entry()], "fp8", 8 * 1024**3)
    plan["dependency_hashes"]["operatorx_queued_timing.py"] = "0" * 64
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    args = SimpleNamespace(
        plan=path,
        phase="smoke",
        run_dir=tmp_path / "run",
        worker_script=SCRIPTS / "operatorx_fp8_block_worker.py",
    )
    with pytest.raises(ValueError, match="dependency changed"):
        batch.run_locked(args)


def test_preflight_rejects_health_report_for_another_device(modules):
    preflight = importlib.import_module("operatorx_fp8_l0_preflight")
    with pytest.raises(ValueError, match="different logical device"):
        preflight.validate_health_report({"identity": {"device": 0}}, 1)
