import json
import os
import queue
import subprocess
import sys
import time
from types import ModuleType

from run import (
    ALLOWED_GLOBAL_BATCH_SIZES,
    classify_failure,
    git_revision,
    latest_worker_progress,
    wait_for_worker_process,
)
from trt_mpi_entry import _install_fixed_batch_request_barrier, worker_main
from trt_worker import read_perfect_router_marker


def test_controller_accepts_only_huawei_global_batches():
    assert ALLOWED_GLOBAL_BATCH_SIZES == (16, 64, 128)


def test_classify_fixed_batch_validation_before_runtime():
    result = {
        "error": "TRT began decode before the fixed local batch was active"
    }
    assert classify_failure(result, "") == "fixed_batch_validation"


def test_classify_fixed_batch_barrier_failure():
    result = {"error": "Fixed-batch barrier timed out with 63/64 requests"}
    assert classify_failure(result, "") == "fixed_batch_validation"


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


def test_latest_worker_progress_ignores_native_trt_lines(tmp_path):
    worker_log = tmp_path / "worker.log"
    worker_log.write_text(
        "native TRT output\n"
        "[offline-trt-worker 2026-06-13T12:00:00+00:00] warmup start\n"
        "more native output\n"
        "[offline-trt-worker 2026-06-13T12:01:00+00:00] "
        "measured: generation start global_batch=64\n",
        encoding="utf-8",
    )
    assert latest_worker_progress(worker_log) == (
        "measured: generation start global_batch=64"
    )


def test_wait_for_worker_process_reports_heartbeat(tmp_path, capsys):
    class FakeProcess:
        args = ["fake-worker"]

        def __init__(self):
            self.wait_calls = 0

        def wait(self, timeout):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired(self.args, timeout)
            return 0

    worker_log = tmp_path / "worker.log"
    worker_log.write_text(
        "[offline-trt-worker 2026-06-13T12:00:00+00:00] "
        "engine initialization start\n",
        encoding="utf-8",
    )
    process = FakeProcess()
    assert (
        wait_for_worker_process(
            process,
            label="gbs16",
            worker_log=worker_log,
            started=time.perf_counter(),
            timeout_seconds=10,
            heartbeat_seconds=0.01,
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "gbs16: still running" in output
    assert "last_worker_progress=engine initialization start" in output


def test_mpi_entry_sets_fixed_environment_before_real_worker(
    tmp_path,
    monkeypatch,
):
    calls = []
    trt_module = ModuleType("tensorrt_llm")
    trt_module.__path__ = []
    torch_module = ModuleType("tensorrt_llm._torch")
    torch_module.__path__ = []
    pyexecutor_module = ModuleType("tensorrt_llm._torch.pyexecutor")
    pyexecutor_module.__path__ = []
    request_queue_module = ModuleType(
        "tensorrt_llm._torch.pyexecutor.executor_request_queue"
    )
    executor_module = ModuleType("tensorrt_llm.executor")
    executor_module.__path__ = []
    worker_module = ModuleType("tensorrt_llm.executor.worker")

    class FakeExecutorRequestQueue:
        def get_from_request_queue(self, timeout):
            return []

    def fake_worker(*args, **kwargs):
        calls.append((args, kwargs))
        return "done"

    request_queue_module.ExecutorRequestQueue = FakeExecutorRequestQueue
    worker_module.worker_main = fake_worker
    trt_module._torch = torch_module
    trt_module.executor = executor_module
    torch_module.pyexecutor = pyexecutor_module
    pyexecutor_module.executor_request_queue = request_queue_module
    executor_module.worker = worker_module
    for name, module in (
        ("tensorrt_llm", trt_module),
        ("tensorrt_llm._torch", torch_module),
        ("tensorrt_llm._torch.pyexecutor", pyexecutor_module),
        (
            "tensorrt_llm._torch.pyexecutor.executor_request_queue",
            request_queue_module,
        ),
        ("tensorrt_llm.executor", executor_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm.executor.worker",
        worker_module,
    )
    marker = tmp_path / "marker.jsonl"
    cache_dir = tmp_path / "cute-cache"
    expected = {
        "ENABLE_CONFIGURABLE_MOE": "1",
        "TRTLLM_BENCH_ENABLE_CONFIGURABLE_MOE": "1",
        "TRTLLM_BENCH_FIXED_BATCH_TIMEOUT_SECONDS": "120",
        "TRTLLM_BENCH_GLOBAL_BATCH_SIZE": "64",
        "TRTLLM_GEN_MOE_AUTOTUNE_DUMMY_DISTRIBUTION": "random",
    }
    monkeypatch.setenv("TRTLLM_ENABLE_PERFECT_ROUTER", "1")
    monkeypatch.setenv("TRTLLM_PERFECT_ROUTER_MARKER", str(marker))
    monkeypatch.setenv(
        "TRTLLM_BENCH_CUTE_DSL_CACHE_DIR",
        str(cache_dir),
    )
    monkeypatch.setenv(
        "TRTLLM_BENCH_EXPECTED_RANK_ENV",
        json.dumps(expected),
    )
    monkeypatch.setenv("TRTLLM_BENCH_ENABLE_CONFIGURABLE_MOE", "1")
    monkeypatch.setenv(
        "TRTLLM_GEN_MOE_AUTOTUNE_DUMMY_DISTRIBUTION",
        "random",
    )
    monkeypatch.setenv(
        "TRTLLM_BENCH_FIXED_BATCH_TIMEOUT_SECONDS",
        "120",
    )
    monkeypatch.setenv("TRTLLM_BENCH_GLOBAL_BATCH_SIZE", "64")
    monkeypatch.delenv("ENABLE_CONFIGURABLE_MOE", raising=False)
    monkeypatch.delenv("ENABLE_PERFECT_ROUTER", raising=False)
    monkeypatch.delenv("CUTE_DSL_CACHE_DIR", raising=False)

    assert worker_main(1, value=2) == "done"
    assert calls == [((1,), {"value": 2})]
    row = json.loads(marker.read_text(encoding="utf-8"))
    assert row["perfect_router"] == "1"
    assert row["cute_dsl_cache_dir"] == str(cache_dir)
    assert row["benchmark_environment"] == expected
    assert row["fixed_batch_global_size"] == "64"
    assert os.environ["ENABLE_CONFIGURABLE_MOE"] == "1"
    assert row["source"] == "trt_mpi_entry"


def test_fixed_batch_barrier_waits_for_one_complete_global_batch(
    monkeypatch,
):
    class Item:
        is_normal_request = True

    class FakeExecutorRequestQueue:
        def __init__(self):
            self.request_queue = queue.Queue()

        def get_from_request_queue(self, timeout):
            return [self.request_queue.get(timeout=1)]

    request_queue_module = ModuleType(
        "tensorrt_llm._torch.pyexecutor.executor_request_queue"
    )
    request_queue_module.ExecutorRequestQueue = FakeExecutorRequestQueue
    pyexecutor_module = ModuleType("tensorrt_llm._torch.pyexecutor")
    pyexecutor_module.__path__ = []
    pyexecutor_module.executor_request_queue = request_queue_module
    torch_module = ModuleType("tensorrt_llm._torch")
    torch_module.__path__ = []
    torch_module.pyexecutor = pyexecutor_module
    trt_module = ModuleType("tensorrt_llm")
    trt_module.__path__ = []
    trt_module._torch = torch_module
    for name, module in (
        ("tensorrt_llm", trt_module),
        ("tensorrt_llm._torch", torch_module),
        ("tensorrt_llm._torch.pyexecutor", pyexecutor_module),
        (
            "tensorrt_llm._torch.pyexecutor.executor_request_queue",
            request_queue_module,
        ),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("TRTLLM_BENCH_GLOBAL_BATCH_SIZE", "3")
    monkeypatch.setenv(
        "TRTLLM_BENCH_FIXED_BATCH_TIMEOUT_SECONDS",
        "1",
    )

    installed = _install_fixed_batch_request_barrier()
    request_queue = FakeExecutorRequestQueue()
    for _ in range(3):
        request_queue.request_queue.put(Item())

    assert installed["global_batch_size"] == 3
    assert len(request_queue.get_from_request_queue(None)) == 3


def test_marker_reports_exact_rank_and_cache_coverage(tmp_path):
    marker = tmp_path / "marker.jsonl"
    rows = [
        {
            "pid": 100 + rank,
            "rank": str(rank),
            "perfect_router": "1",
            "cute_dsl_cache_dir": "/cache",
            "source": "trt_mpi_entry",
        }
        for rank in range(8)
    ]
    marker.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    parsed = read_perfect_router_marker(marker)
    assert parsed["mpi_entry_processes"] == 8
    assert parsed["mpi_entry_ranks"] == list(range(8))
    assert parsed["mpi_entry_cute_cache_processes"] == 8
    assert parsed["mpi_entry_cute_cache_paths"] == ["/cache"]
