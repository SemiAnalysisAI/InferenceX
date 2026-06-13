import json
import subprocess
import sys
import time
from types import ModuleType

from run import (
    ALLOWED_CONCURRENCIES,
    classify_failure,
    git_revision,
    latest_worker_progress,
    wait_for_worker_process,
)
from trt_mpi_entry import worker_main
from trt_worker import measured_pass_count, read_perfect_router_marker


def test_controller_accepts_workflow_concurrencies():
    assert ALLOWED_CONCURRENCIES == (
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
    )


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


def test_worker_honors_requested_tuning_pass_count():
    assert measured_pass_count(3) == 3


def test_latest_worker_progress_ignores_native_trt_lines(tmp_path):
    worker_log = tmp_path / "worker.log"
    worker_log.write_text(
        "native TRT output\n"
        "[offline-trt-worker 2026-06-13T12:00:00+00:00] warmup start\n"
        "more native output\n"
        "[offline-trt-worker 2026-06-13T12:01:00+00:00] "
        "measured pass 1/3: generation start requests=8\n",
        encoding="utf-8",
    )

    assert latest_worker_progress(worker_log) == (
        "measured pass 1/3: generation start requests=8"
    )


def test_wait_for_worker_process_reports_heartbeat(
    tmp_path,
    capsys,
):
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
            label="tune_01_wait30",
            worker_log=worker_log,
            started=time.perf_counter(),
            timeout_seconds=10,
            heartbeat_seconds=0.01,
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "tune_01_wait30: still running" in output
    assert "last_worker_progress=engine initialization start" in output


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
    cache_dir = tmp_path / "cute-cache"
    monkeypatch.setenv("TRTLLM_ENABLE_PERFECT_ROUTER", "1")
    monkeypatch.setenv("TRTLLM_PERFECT_ROUTER_MARKER", str(marker))
    monkeypatch.setenv(
        "TRTLLM_BENCH_CUTE_DSL_CACHE_DIR",
        str(cache_dir),
    )
    monkeypatch.delenv("ENABLE_PERFECT_ROUTER", raising=False)
    monkeypatch.delenv("CUTE_DSL_CACHE_DIR", raising=False)

    assert worker_main(1, value=2) == "done"
    assert calls == [((1,), {"value": 2})]
    row = json.loads(marker.read_text(encoding="utf-8"))
    assert row["perfect_router"] == "1"
    assert row["cute_dsl_cache_dir"] == str(cache_dir)
    assert row["source"] == "trt_mpi_entry"


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
        for rank in range(4)
    ]
    marker.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    parsed = read_perfect_router_marker(marker)
    assert parsed["mpi_entry_processes"] == 4
    assert parsed["mpi_entry_ranks"] == [0, 1, 2, 3]
    assert parsed["mpi_entry_cute_cache_processes"] == 4
    assert parsed["mpi_entry_cute_cache_paths"] == ["/cache"]
