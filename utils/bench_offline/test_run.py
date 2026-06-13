import json
import os
import subprocess
import sys
import time
from types import ModuleType

import pytest

from run import (
    ALLOWED_CONCURRENCIES,
    classify_failure,
    collect_profile_artifacts,
    git_revision,
    latest_worker_progress,
    wait_for_worker_process,
)
from trt_config import CandidateConfig
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


def test_classify_kernel_dtype_before_perfect_router():
    result = {
        "phase": "engine_init",
        "traceback": (
            "FP8 Paged MQA Logits dtype errors: "
            "q must be float8_e4m3fn, got torch.int8"
        ),
    }
    assert classify_failure(result, "perfect_router enabled") == "kernel_dtype"


def test_classify_timeout():
    assert classify_failure({}, "", timed_out=True) == "timeout"


def test_git_revision_prefers_explicit_benchmark_revision(monkeypatch):
    revision = "a" * 40
    monkeypatch.setenv("TRT_BENCH_GIT_REVISION", revision)
    assert git_revision() == revision


def test_worker_enforces_one_measured_pass():
    assert measured_pass_count(1) == 1
    with pytest.raises(ValueError, match="exactly 1"):
        measured_pass_count(3)


def test_latest_worker_progress_ignores_native_trt_lines(tmp_path):
    worker_log = tmp_path / "worker.log"
    worker_log.write_text(
        "native TRT output\n"
        "[offline-trt-worker 2026-06-13T12:00:00+00:00] warmup start\n"
        "more native output\n"
        "[offline-trt-worker 2026-06-13T12:01:00+00:00] "
        "measured pass 1/1: generation start requests=8\n",
        encoding="utf-8",
    )

    assert latest_worker_progress(worker_log) == (
        "measured pass 1/1: generation start requests=8"
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
    monkeypatch.setenv(
        "TRTLLM_BENCH_EXPECTED_RANK_ENV",
        json.dumps(
            {
                "ENABLE_CONFIGURABLE_MOE": "0",
                "TRTLLM_BENCH_ENABLE_CONFIGURABLE_MOE": "0",
                "TRTLLM_ENABLE_PDL": "0",
            }
        ),
    )
    monkeypatch.setenv("TRTLLM_BENCH_ENABLE_CONFIGURABLE_MOE", "0")
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "0")
    monkeypatch.delenv("ENABLE_CONFIGURABLE_MOE", raising=False)
    monkeypatch.delenv("ENABLE_PERFECT_ROUTER", raising=False)
    monkeypatch.delenv("CUTE_DSL_CACHE_DIR", raising=False)

    assert worker_main(1, value=2) == "done"
    assert calls == [((1,), {"value": 2})]
    row = json.loads(marker.read_text(encoding="utf-8"))
    assert row["perfect_router"] == "1"
    assert row["cute_dsl_cache_dir"] == str(cache_dir)
    assert row["benchmark_environment"] == {
        "ENABLE_CONFIGURABLE_MOE": "0",
        "TRTLLM_BENCH_ENABLE_CONFIGURABLE_MOE": "0",
        "TRTLLM_ENABLE_PDL": "0",
    }
    assert os.environ["ENABLE_CONFIGURABLE_MOE"] == "0"
    assert row["source"] == "trt_mpi_entry"


def test_mpi_entry_records_dsv4_backport_source(
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
    import trt_backports

    monkeypatch.setattr(
        trt_backports,
        "inspect_dsv4_source",
        lambda: {
            "path": "/site-packages/modeling_deepseekv4.py",
            "sha256": "patched-sha",
            "skip_premoe_allreduce_backport": True,
        },
    )
    marker = tmp_path / "marker.jsonl"
    expected = {
        "TRTLLM_BENCH_DSV4_PATCHED_SHA256": "patched-sha",
        "TRTLLM_DSV4_SKIP_PREMOE_ALLREDUCE": "1",
    }
    monkeypatch.setenv("TRTLLM_PERFECT_ROUTER_MARKER", str(marker))
    monkeypatch.setenv(
        "TRTLLM_BENCH_EXPECTED_RANK_ENV",
        json.dumps(expected),
    )
    for name, value in expected.items():
        monkeypatch.setenv(name, value)

    assert worker_main() == "done"
    assert calls == [((), {})]
    row = json.loads(marker.read_text(encoding="utf-8"))
    assert row["benchmark_environment"] == expected
    assert row["dsv4_source"]["sha256"] == "patched-sha"
    assert row["dsv4_source"]["skip_premoe_allreduce_backport"] is True


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


def test_profile_artifacts_require_every_active_rank(tmp_path):
    candidate = CandidateConfig(
        name="profile",
        batching_wait_iters=0,
        parallelism="dep4",
        profile_iterations="50-51",
    )
    label = "experiment_01_profile"
    for rank in range(4):
        (tmp_path / f"{label}_torch_profile-rank-{rank}.json").write_text(
            '{"traceEvents":[]}\n',
            encoding="utf-8",
        )

    manifest = collect_profile_artifacts(
        output_dir=tmp_path,
        label=label,
        candidate=candidate,
    )

    assert manifest is not None
    assert manifest["complete"] is True
    assert manifest["observed_ranks"] == [0, 1, 2, 3]
    assert (tmp_path / manifest["manifest_path"]).is_file()
