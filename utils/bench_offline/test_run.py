import json
import os
import queue
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace

from run import (
    ALLOWED_GLOBAL_BATCH_SIZES,
    classify_failure,
    git_revision,
    latest_worker_progress,
    wait_for_worker_process,
)
from trt_mpi_entry import (
    _install_engine_warmup_token_cap,
    _install_fixed_batch_request_barrier,
    _install_large_prefill_fp8_quant_guard,
    worker_main,
)
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
    model_engine_module = ModuleType(
        "tensorrt_llm._torch.pyexecutor.model_engine"
    )
    request_queue_module = ModuleType(
        "tensorrt_llm._torch.pyexecutor.executor_request_queue"
    )
    custom_ops_package = ModuleType("tensorrt_llm._torch.custom_ops")
    custom_ops_package.__path__ = []
    torch_custom_ops_module = ModuleType(
        "tensorrt_llm._torch.custom_ops.torch_custom_ops"
    )
    executor_module = ModuleType("tensorrt_llm.executor")
    executor_module.__path__ = []
    worker_module = ModuleType("tensorrt_llm.executor.worker")

    class FakeExecutorRequestQueue:
        def get_from_request_queue(self, timeout):
            return []

    class FakeModelEngine:
        def warmup(self, resource_manager):
            return None

    class FakeFp8QuantKernelRunner:
        TACTIC_TRITON = 1

        def get_valid_tactics(self, inputs, profile, **kwargs):
            return [0, self.TACTIC_TRITON]

    def fake_fp8_quantize(input_tensor, tactic):
        return input_tensor, tactic

    def fake_worker(*args, **kwargs):
        calls.append((args, kwargs))
        return "done"

    request_queue_module.ExecutorRequestQueue = FakeExecutorRequestQueue
    model_engine_module.ModelEngine = FakeModelEngine
    torch_custom_ops_module.Fp8QuantKernelRunner = (
        FakeFp8QuantKernelRunner
    )
    torch_custom_ops_module._fp8_quantize_1x128_ue8m0 = (
        fake_fp8_quantize
    )
    custom_ops_package.torch_custom_ops = torch_custom_ops_module
    worker_module.worker_main = fake_worker
    trt_module._torch = torch_module
    trt_module.executor = executor_module
    torch_module.pyexecutor = pyexecutor_module
    torch_module.custom_ops = custom_ops_package
    pyexecutor_module.executor_request_queue = request_queue_module
    pyexecutor_module.model_engine = model_engine_module
    executor_module.worker = worker_module
    for name, module in (
        ("tensorrt_llm", trt_module),
        ("tensorrt_llm._torch", torch_module),
        ("tensorrt_llm._torch.pyexecutor", pyexecutor_module),
        (
            "tensorrt_llm._torch.pyexecutor.model_engine",
            model_engine_module,
        ),
        ("tensorrt_llm._torch.custom_ops", custom_ops_package),
        (
            "tensorrt_llm._torch.custom_ops.torch_custom_ops",
            torch_custom_ops_module,
        ),
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
        "TRTLLM_BENCH_ENGINE_WARMUP_MAX_TOKENS": "65536",
        "TRTLLM_BENCH_FP8_FUSED_QUANT_MAX_ROWS": "32768",
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
    monkeypatch.setenv(
        "TRTLLM_BENCH_ENGINE_WARMUP_MAX_TOKENS",
        "65536",
    )
    monkeypatch.setenv(
        "TRTLLM_BENCH_FP8_FUSED_QUANT_MAX_ROWS",
        "32768",
    )
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
    assert row["engine_warmup_max_tokens"] == "65536"
    assert row["fp8_fused_quant_max_rows"] == "32768"
    assert os.environ["ENABLE_CONFIGURABLE_MOE"] == "1"
    assert row["source"] == "trt_mpi_entry"


def test_engine_warmup_token_cap_restores_runtime_capacity(monkeypatch):
    seen = []
    model_engine_module = ModuleType(
        "tensorrt_llm._torch.pyexecutor.model_engine"
    )

    class FakeModelEngine:
        def __init__(self, max_num_tokens):
            self.max_num_tokens = max_num_tokens

        def warmup(self, resource_manager):
            seen.append(self.max_num_tokens)
            return resource_manager

    model_engine_module.ModelEngine = FakeModelEngine
    pyexecutor_module = ModuleType("tensorrt_llm._torch.pyexecutor")
    pyexecutor_module.__path__ = []
    pyexecutor_module.model_engine = model_engine_module
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.pyexecutor",
        pyexecutor_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.pyexecutor.model_engine",
        model_engine_module,
    )
    monkeypatch.setenv(
        "TRTLLM_BENCH_ENGINE_WARMUP_MAX_TOKENS",
        "65536",
    )

    installed = _install_engine_warmup_token_cap()
    large = FakeModelEngine(131072)
    small = FakeModelEngine(16384)

    assert installed == {
        "max_warmup_tokens": 65536,
        "already_installed": False,
    }
    assert large.warmup("large") == "large"
    assert large.max_num_tokens == 131072
    assert small.warmup("small") == "small"
    assert small.max_num_tokens == 16384
    assert seen == [65536, 16384]


def test_large_prefill_fp8_guard_keeps_decode_and_routes_prefill(
    monkeypatch,
):
    custom_ops_package = ModuleType("tensorrt_llm._torch.custom_ops")
    custom_ops_package.__path__ = []
    torch_custom_ops_module = ModuleType(
        "tensorrt_llm._torch.custom_ops.torch_custom_ops"
    )

    class FakeFp8QuantKernelRunner:
        TACTIC_TRITON = 1

        def get_valid_tactics(self, inputs, profile, **kwargs):
            return [0, self.TACTIC_TRITON]

    quantize_calls = []

    def fake_fp8_quantize(input_tensor, tactic):
        quantize_calls.append((input_tensor.shape, tactic))
        return input_tensor, tactic

    torch_custom_ops_module.Fp8QuantKernelRunner = (
        FakeFp8QuantKernelRunner
    )
    torch_custom_ops_module._fp8_quantize_1x128_ue8m0 = (
        fake_fp8_quantize
    )
    custom_ops_package.torch_custom_ops = torch_custom_ops_module
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.custom_ops",
        custom_ops_package,
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.custom_ops.torch_custom_ops",
        torch_custom_ops_module,
    )
    monkeypatch.setenv(
        "TRTLLM_BENCH_FP8_FUSED_QUANT_MAX_ROWS",
        "32768",
    )

    installed = _install_large_prefill_fp8_quant_guard()
    runner = FakeFp8QuantKernelRunner()
    small = [SimpleNamespace(shape=(16, 7168))]
    large = [SimpleNamespace(shape=(65536, 7168))]

    assert installed == {
        "max_fused_rows": 32768,
        "already_installed": False,
    }
    assert runner.get_valid_tactics(small, None) == [0, 1]
    assert runner.get_valid_tactics(large, None) == [1]
    assert torch_custom_ops_module._fp8_quantize_1x128_ue8m0(
        small[0], 0
    ) == (small[0], 0)
    assert torch_custom_ops_module._fp8_quantize_1x128_ue8m0(
        large[0], -1
    ) == (large[0], 1)
    assert quantize_calls == [
        ((16, 7168), 0),
        ((65536, 7168), 1),
    ]


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
