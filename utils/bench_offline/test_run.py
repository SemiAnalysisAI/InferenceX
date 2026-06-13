import json
import sys
from types import ModuleType

from run import classify_failure, git_revision
from trt_mpi_entry import worker_main


def test_classify_graph_failure_before_generic_runtime():
    result = {"error": "CUDA graph capture failed"}
    assert classify_failure(result, "") == "cuda_graph"


def test_classify_capacity_failure():
    result = {
        "phase": "engine_init",
        "error": "Total tokens exceeds max_num_tokens",
    }
    assert classify_failure(result, "") == "capacity"


def test_classify_timeout():
    assert classify_failure({}, "", timed_out=True) == "timeout"


def test_git_revision_prefers_explicit_benchmark_revision(monkeypatch):
    revision = "a" * 40
    monkeypatch.setenv("TRT_BENCH_GIT_REVISION", revision)
    assert git_revision() == revision


def test_mpi_entry_sets_router_before_real_worker(
    tmp_path,
    monkeypatch,
):
    calls = []
    worker_module = ModuleType("tensorrt_llm.executor.worker")

    def fake_worker(*args, **kwargs):
        calls.append((args, kwargs))
        return "done"

    worker_module.worker_main = fake_worker
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm.executor.worker",
        worker_module,
    )
    marker = tmp_path / "marker.jsonl"
    monkeypatch.setenv("TRTLLM_ENABLE_PERFECT_ROUTER", "1")
    monkeypatch.setenv("TRTLLM_PERFECT_ROUTER_MARKER", str(marker))
    monkeypatch.delenv("ENABLE_PERFECT_ROUTER", raising=False)

    assert worker_main(1, value=2) == "done"
    assert calls == [((1,), {"value": 2})]
    row = json.loads(marker.read_text(encoding="utf-8"))
    assert row["perfect_router"] == "1"
    assert row["source"] == "trt_mpi_entry"
