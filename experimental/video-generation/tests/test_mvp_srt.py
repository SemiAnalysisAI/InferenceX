"""CPU tests for the srt-slurm H3 adapters."""

from pathlib import Path

import pytest

from evaluator import mvp_srt


def test_bind_visible_gpus_fills_uuids_from_cuda_inventory(monkeypatch):
    monkeypatch.setattr(mvp_srt, "cuda_devices", lambda: ["GPU-a", "GPU-b"])
    monkeypatch.setattr(
        mvp_srt,
        "validate_gpu_job",
        lambda spec: {**spec, "validated": True},
    )
    bound = mvp_srt.bind_visible_gpus({"gpu_uuids": ["placeholder"], "allocation": {"mode": "x"}})
    assert bound["gpu_uuids"] == ["GPU-a", "GPU-b"]
    assert bound["allocation"]["mode"] == "x"
    assert bound["validated"] is True


def test_bind_visible_gpus_rejects_empty_inventory(monkeypatch):
    monkeypatch.setattr(mvp_srt, "cuda_devices", lambda: [])
    with pytest.raises(RuntimeError, match="no visible GPUs"):
        mvp_srt.bind_visible_gpus({"gpu_uuids": ["placeholder"]})


def test_run_srt_gpu_job_delegates_to_supervisor(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(mvp_srt, "bind_visible_gpus", lambda spec: {**spec, "bound": True})
    monkeypatch.setattr(
        mvp_srt,
        "run_gpu_job",
        lambda spec, output: calls.append((spec, output)) or {"status": "complete"},
    )
    result = mvp_srt.run_srt_gpu_job({"plan": {}}, tmp_path / "out")
    assert result["status"] == "complete"
    assert calls[0][0]["bound"] is True
    assert calls[0][1] == tmp_path / "out"
