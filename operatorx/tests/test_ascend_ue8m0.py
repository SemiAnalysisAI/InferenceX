import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "scripts" / "ascend")
    )
    return importlib.import_module("operatorx_mxfp8_128_triton_queued_worker")


def case():
    return {
        "opType": "gemm",
        "args": {
            "m": 3,
            "n": 7,
            "k": 128,
            "out": "bf16",
            "a": {
                "dtype": "e4m3",
                "scale": {"dtype": "ue8m0", "group": [1, 128], "static": False},
            },
            "b": {
                "dtype": "e4m3",
                "scale": {"dtype": "ue8m0", "group": [128, 128], "static": True},
            },
        },
    }


def test_plan_preserves_exact_semantics_and_two_live_outputs(worker):
    original = case()
    result = worker.plan_case(original, cache_write_bytes=0, workspace_reserve_bytes=0)
    assert result["args"] == original["args"]
    assert result["status"] == "unvalidated"
    assert result["metrics"] is None
    memory = result["memory_estimate"]
    assert memory["a_bytes"] == 768
    assert memory["weight_bytes"] == 896
    assert memory["raw_activation_scale_bytes"] == 6
    assert memory["activation_scale_bytes"] == 12
    assert memory["output_bytes"] == 42
    assert memory["output_live_buffer_count"] == 2


@pytest.mark.parametrize(
    "change,reason",
    [
        ("tail", "k_alignment_not_implemented"),
        ("fp32", "scale_semantics_not_implemented"),
        ("static", "scale_semantics_not_implemented"),
        ("output", "output_not_implemented"),
    ],
)
def test_rejects_other_contracts_before_allocation(worker, change, reason):
    request = copy.deepcopy(case())
    if change == "tail":
        request["args"]["k"] = 129
    elif change == "fp32":
        request["args"]["a"]["scale"]["dtype"] = "fp32"
    elif change == "static":
        request["args"]["a"]["scale"]["static"] = True
    else:
        request["args"]["out"] = "fp32"
    result = worker.plan_case(request)
    assert result["status"] == "unsupported"
    assert result["support"]["reason_code"] == reason
    assert result["metrics"] is None


def test_prepare_ue8m0_retains_unsupported_cases(worker):
    prepare = importlib.import_module("prepare")
    tail = case()
    tail["args"]["k"] = 129
    plan = prepare.make_plan([case(), tail], "ue8m0", 96 * 1024**3)
    assert plan["counts"] == {"new": 1, "implementation_required": 1}
    assert len(plan["smoke_case_ids"]) == 1
    assert next(c for c in plan["cases"] if c["phase"] == "new")["args"]["k"] == 128


def test_scale_expansion_ignores_storage_padding(worker):
    torch = pytest.importorskip("torch")
    raw = torch.tensor([[[120, 255]], [[127, 254]]], dtype=torch.uint8)
    expanded = worker.expand_activation_scales(raw, 2, 128)
    assert expanded.tolist() == [[[120, 120], [120, 120]], [[127, 127], [127, 127]]]
    assert worker.decode_scale_bytes(
        torch, torch.tensor([0, 120, 127, 128], dtype=torch.uint8)
    ).tolist() == [2**-127, 2**-7, 1.0, 2.0]
    assert torch.isnan(
        worker.decode_scale_bytes(torch, torch.tensor([255], dtype=torch.uint8))
    ).item()


def test_static_scale_validation_detects_corrupted_replica(worker):
    torch = pytest.importorskip("torch")
    coarse = torch.tensor([[120]], dtype=torch.uint8)
    expanded = torch.tensor([[[120, 120], [120, 120]]], dtype=torch.uint8)
    assert worker.validate_static_scales(torch, coarse, expanded, 1, 128)["passed"]
    expanded[0, 1, 0] = 121
    result = worker.validate_static_scales(torch, coarse, expanded, 1, 128)
    assert not result["passed"]
    assert result["mismatched_expanded_bytes"] == 1
