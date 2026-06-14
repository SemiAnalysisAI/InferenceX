import json
import os
import queue
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

from run import (
    ALLOWED_GLOBAL_BATCH_SIZES,
    classify_failure,
    git_revision,
    latest_rank_progress,
    latest_worker_fatal,
    latest_worker_progress,
    WorkerFatalLogError,
    wait_for_worker_process,
)
from trt_mpi_entry import (
    _install_attention_workspace_preallocation,
    _install_engine_warmup_shape_cap,
    _install_fixed_batch_request_barrier,
    _install_large_prefill_fp8_quant_guard,
    worker_main,
)
from trt_worker import (
    arm_fixed_batch_request_barrier,
    read_perfect_router_marker,
    validate_rank_propagation,
)


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


def test_classify_cuda_illegal_address():
    assert (
        classify_failure({}, "CUDA error: an illegal memory access")
        == "cuda_illegal_address"
    )


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


def test_latest_worker_progress_surfaces_mpi_warmup_marker(tmp_path):
    worker_log = tmp_path / "worker.log"
    worker_log.write_text(
        "[offline-trt-worker 2026-06-13T12:00:00+00:00] "
        "engine initialization start\n"
        "[offline-trt-mpi] rank=0 event=engine_warmup_shape_capped "
        "requested_tokens=84087 tuned_tokens=65536\n"
        "native TRT output\n",
        encoding="utf-8",
    )
    assert latest_worker_progress(worker_log) == (
        "[offline-trt-mpi] rank=0 event=engine_warmup_shape_capped "
        "requested_tokens=84087 tuned_tokens=65536"
    )


def test_latest_rank_progress_groups_each_ranks_latest_event(tmp_path):
    marker = tmp_path / "perfect_router.jsonl"
    rows = [
        {"source": "trt_mpi_entry", "rank": "0", "event": "entry_ready"},
        {
            "source": "trt_mpi_entry",
            "rank": "1",
            "event": "engine_warmup_start",
        },
        {
            "source": "trt_mpi_entry",
            "rank": "0",
            "event": "engine_warmup_complete",
        },
        {"source": "other", "rank": "7", "event": "ignored"},
    ]
    marker.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    assert latest_rank_progress(marker) == (
        "engine_warmup_complete:[0],engine_warmup_start:[1]"
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


def test_wait_for_worker_process_stops_on_native_fatal_log(tmp_path):
    class FakeProcess:
        args = ["fake-worker"]

        def wait(self, timeout):
            raise subprocess.TimeoutExpired(self.args, timeout)

    worker_log = tmp_path / "worker.log"
    fatal_line = (
        "[TRT-LLM] Fatal error detected, initiating shutdown: "
        "CUDA error: an illegal memory access was encountered"
    )
    worker_log.write_text(fatal_line + "\n", encoding="utf-8")

    assert latest_worker_fatal(worker_log) == fatal_line
    with pytest.raises(WorkerFatalLogError, match="initiating shutdown"):
        wait_for_worker_process(
            FakeProcess(),
            label="gbs128",
            worker_log=worker_log,
            started=time.perf_counter(),
            timeout_seconds=10,
            heartbeat_seconds=0.01,
        )


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
    py_executor_impl_module = ModuleType(
        "tensorrt_llm._torch.pyexecutor.py_executor"
    )
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
        def _create_warmup_request(
            self,
            resource_manager,
            num_tokens,
            num_gen_requests,
        ):
            return num_tokens, num_gen_requests

        def warmup(self, resource_manager):
            return None

    class FakePyExecutor:
        def _set_global_steady_clock_offset(self):
            return None

        def start_worker(self):
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
    model_engine_module.PyTorchModelEngine = FakeModelEngine
    py_executor_impl_module.PyExecutor = FakePyExecutor
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
    pyexecutor_module.py_executor = py_executor_impl_module
    executor_module.worker = worker_module
    for name, module in (
        ("tensorrt_llm", trt_module),
        ("tensorrt_llm._torch", torch_module),
        ("tensorrt_llm._torch.pyexecutor", pyexecutor_module),
        (
            "tensorrt_llm._torch.pyexecutor.model_engine",
            model_engine_module,
        ),
        (
            "tensorrt_llm._torch.pyexecutor.py_executor",
            py_executor_impl_module,
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
    fixed_batch_arm_file = tmp_path / "fixed-batch.armed.json"
    cache_dir = tmp_path / "cute-cache"
    expected = {
        "ENABLE_CONFIGURABLE_MOE": "1",
        "TRTLLM_BENCH_ENABLE_CONFIGURABLE_MOE": "1",
        "TRTLLM_BENCH_ATTENTION_WORKSPACE_BYTES": "0",
        "TRTLLM_BENCH_ENGINE_WARMUP_MAX_TOKENS": "65536",
        "TRTLLM_BENCH_FP8_FUSED_QUANT_MAX_ROWS": "32768",
        "TRTLLM_BENCH_FIXED_BATCH_ARM_FILE": str(fixed_batch_arm_file),
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
    monkeypatch.setenv(
        "TRTLLM_BENCH_FIXED_BATCH_ARM_FILE",
        str(fixed_batch_arm_file),
    )
    monkeypatch.setenv("TRTLLM_BENCH_GLOBAL_BATCH_SIZE", "64")
    monkeypatch.setenv(
        "TRTLLM_BENCH_ATTENTION_WORKSPACE_BYTES",
        "0",
    )
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
    assert row["fixed_batch_arm_file"] == str(fixed_batch_arm_file)
    assert row["fixed_batch_barrier_armed"] is False
    assert row["engine_warmup_max_tokens"] == "65536"
    assert row["attention_workspace_target_bytes"] == "0"
    assert row["fp8_fused_quant_max_rows"] == "32768"
    assert os.environ["ENABLE_CONFIGURABLE_MOE"] == "1"
    assert row["source"] == "trt_mpi_entry"


def test_engine_warmup_shape_cap_preserves_runtime_capacity(monkeypatch):
    seen_runtime_capacity = []
    seen_requests = []
    model_engine_module = ModuleType(
        "tensorrt_llm._torch.pyexecutor.model_engine"
    )

    class FakeModelEngine:
        def warmup(self, resource_manager):
            raise AssertionError("abstract base warmup must not be patched")

    class FakePyTorchModelEngine(FakeModelEngine):
        def __init__(self, max_num_tokens):
            self.max_num_tokens = max_num_tokens

        def _create_warmup_request(
            self,
            resource_manager,
            num_tokens,
            num_gen_requests,
        ):
            seen_requests.append((num_tokens, num_gen_requests))
            return resource_manager

        def warmup(self, resource_manager):
            seen_runtime_capacity.append(self.max_num_tokens)
            self._create_warmup_request(
                resource_manager,
                self.max_num_tokens,
                num_gen_requests=0,
            )
            self._create_warmup_request(resource_manager, 16, 16)
            return self.max_num_tokens

    model_engine_module.ModelEngine = FakeModelEngine
    model_engine_module.PyTorchModelEngine = FakePyTorchModelEngine
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

    installed = _install_engine_warmup_shape_cap()
    large = FakePyTorchModelEngine(131072)
    small = FakePyTorchModelEngine(16384)

    assert installed == {
        "max_warmup_tokens": 65536,
        "target": "FakePyTorchModelEngine._create_warmup_request",
        "trace_target": "FakePyTorchModelEngine.warmup",
        "already_installed": False,
    }
    assert not getattr(
        FakeModelEngine.warmup,
        "_offline_engine_warmup_trace",
        False,
    )
    assert large.warmup("large") == 131072
    assert large.max_num_tokens == 131072
    assert small.warmup("small") == 16384
    assert small.max_num_tokens == 16384
    assert seen_runtime_capacity == [131072, 16384]
    assert seen_requests == [
        (65536, 0),
        (16, 16),
        (16384, 0),
        (16, 16),
    ]


def test_attention_workspace_preallocation_keeps_graph_workspace(
    tmp_path,
    monkeypatch,
):
    allocations = []

    class FakeTensor:
        def __init__(self, elements, element_bytes=1):
            self.elements = elements
            self.element_bytes = element_bytes

        def numel(self):
            return self.elements

        def element_size(self):
            return self.element_bytes

        def new_empty(self, shape):
            tensor = FakeTensor(shape[0], self.element_bytes)
            allocations.append(tensor)
            return tensor

    eager_workspace = FakeTensor(4)
    graph_workspace = FakeTensor(7)

    class FakePyTorchModelEngine:
        def __init__(self):
            self.max_num_tokens = 131072
            self.attn_metadata = None

        def _set_up_attn_metadata(self, kv_cache_manager):
            if self.attn_metadata is None:
                self.attn_metadata = SimpleNamespace(
                    workspace=eager_workspace,
                    cuda_graph_workspace=graph_workspace,
                )
            return self.attn_metadata

    model_engine_module = ModuleType(
        "tensorrt_llm._torch.pyexecutor.model_engine"
    )
    model_engine_module.PyTorchModelEngine = FakePyTorchModelEngine
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
    marker = tmp_path / "marker.jsonl"
    monkeypatch.setenv(
        "TRTLLM_BENCH_ATTENTION_WORKSPACE_BYTES",
        "100",
    )
    monkeypatch.setenv("TRTLLM_PERFECT_ROUTER_MARKER", str(marker))
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "3")

    installed = _install_attention_workspace_preallocation()
    engine = FakePyTorchModelEngine()
    first = engine._set_up_attn_metadata(object())
    first_reserved = first.workspace
    second = engine._set_up_attn_metadata(object())

    assert installed == {
        "enabled": True,
        "target_bytes": 100,
        "target": "FakePyTorchModelEngine._set_up_attn_metadata",
        "already_installed": False,
    }
    assert first is second
    assert first_reserved.numel() == 100
    assert second.workspace is first_reserved
    assert second.cuda_graph_workspace is graph_workspace
    assert allocations == [first_reserved]
    event = json.loads(marker.read_text(encoding="utf-8"))
    assert event["event"] == "attention_workspace_preallocated"
    assert event["rank"] == "3"
    assert event["runtime_max_tokens"] == 131072
    assert event["previous_bytes"] == 4
    assert event["target_bytes"] == 100
    assert event["allocated_bytes"] == 100
    assert event["cuda_graph_workspace_bytes"] == 7


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


def test_fixed_batch_barrier_bypasses_init_then_waits_for_global_batch(
    tmp_path,
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
    arm_file = tmp_path / "fixed-batch.armed.json"
    monkeypatch.setenv(
        "TRTLLM_BENCH_FIXED_BATCH_ARM_FILE",
        str(arm_file),
    )

    installed = _install_fixed_batch_request_barrier()
    request_queue = FakeExecutorRequestQueue()
    request_queue.request_queue.put(Item())
    assert len(request_queue.get_from_request_queue(None)) == 1

    arm_file.write_text("{}\n", encoding="utf-8")
    for _ in range(3):
        request_queue.request_queue.put(Item())

    assert installed["global_batch_size"] == 3
    assert installed["arm_file"] == str(arm_file)
    assert installed["armed_at_install"] is False
    assert len(request_queue.get_from_request_queue(None)) == 3


def test_arm_fixed_batch_request_barrier_is_single_use(tmp_path):
    arm_file = tmp_path / "fixed-batch.armed.json"
    marker = arm_fixed_batch_request_barrier(arm_file, 128)

    assert marker["global_batch_size"] == 128
    assert json.loads(arm_file.read_text(encoding="utf-8")) == marker
    with pytest.raises(RuntimeError, match="already exists"):
        arm_fixed_batch_request_barrier(arm_file, 128)


def test_marker_reports_exact_rank_and_cache_coverage(tmp_path):
    marker = tmp_path / "marker.jsonl"
    rows = [
        {
            "pid": 100 + rank,
            "rank": str(rank),
            "perfect_router": "1",
            "cute_dsl_cache_dir": "/cache",
            "event": "entry_ready",
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
    assert parsed["event_ranks"]["entry_ready"] == list(range(8))


def test_rank_validation_requires_exact_attention_workspace_events(
    tmp_path,
    monkeypatch,
):
    marker = tmp_path / "marker.jsonl"
    rank_environment = {
        "TRTLLM_BENCH_ATTENTION_WORKSPACE_BYTES": "100",
    }
    rows = [
        {
            "pid": 100 + rank,
            "rank": str(rank),
            "perfect_router": "1",
            "cute_dsl_cache_dir": "/cache",
            "benchmark_environment": rank_environment,
            "event": "attention_workspace_preallocated",
            "target_bytes": 100,
            "allocated_bytes": 100,
            "cuda_graph_workspace_bytes": 0,
            "source": "trt_mpi_entry",
        }
        for rank in range(8)
    ]
    marker.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    monkeypatch.setenv("CUTE_DSL_CACHE_DIR", "/cache")

    parsed = validate_rank_propagation(marker, rank_environment)
    assert parsed["event_ranks"][
        "attention_workspace_preallocated"
    ] == list(range(8))

    rows[-1]["cuda_graph_workspace_bytes"] = 1
    marker.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="reservation mismatch"):
        validate_rank_propagation(marker, rank_environment)
