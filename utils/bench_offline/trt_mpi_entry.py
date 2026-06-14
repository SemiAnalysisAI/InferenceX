"""MPI worker entry shim for the pinned TRT perfect-router environment."""

from __future__ import annotations

import json
import os
import queue
import time
from pathlib import Path
from typing import Any


def _emit_rank_event(event: str, **details: Any) -> None:
    rank = os.getenv("OMPI_COMM_WORLD_RANK", "unknown")
    _write_marker(event=event, **details)
    detail_text = " ".join(
        f"{name}={value}" for name, value in details.items()
    )
    suffix = f" {detail_text}" if detail_text else ""
    print(
        f"[offline-trt-mpi] rank={rank} event={event}{suffix}",
        flush=True,
    )


def _install_engine_warmup_shape_cap() -> dict[str, Any]:
    max_warmup_tokens = int(
        os.environ["TRTLLM_BENCH_ENGINE_WARMUP_MAX_TOKENS"]
    )
    if max_warmup_tokens <= 0:
        raise RuntimeError(
            "TRT engine warmup shape cap needs a positive limit"
        )

    from tensorrt_llm._torch.pyexecutor import model_engine

    model_engine_class = model_engine.PyTorchModelEngine
    create_warmup_request = model_engine_class._create_warmup_request
    cap_already_installed = getattr(
        create_warmup_request,
        "_offline_engine_warmup_shape_cap",
        False,
    )
    if not cap_already_installed:

        def bounded_warmup_request(
            engine: Any,
            resource_manager: Any,
            num_tokens: int,
            num_gen_requests: int,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            requested_tokens = int(num_tokens)
            bounded_tokens = (
                min(requested_tokens, max_warmup_tokens)
                if int(num_gen_requests) == 0
                else requested_tokens
            )
            if bounded_tokens != requested_tokens:
                _emit_rank_event(
                    "engine_warmup_shape_capped",
                    requested_tokens=requested_tokens,
                    tuned_tokens=bounded_tokens,
                )
            return create_warmup_request(
                engine,
                resource_manager,
                bounded_tokens,
                num_gen_requests,
                *args,
                **kwargs,
            )

        bounded_warmup_request._offline_engine_warmup_shape_cap = True
        model_engine_class._create_warmup_request = bounded_warmup_request

    warmup = model_engine_class.warmup
    trace_already_installed = getattr(
        warmup,
        "_offline_engine_warmup_trace",
        False,
    )
    if not trace_already_installed:

        def traced_warmup(
            engine: Any,
            resource_manager: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            runtime_max_num_tokens = int(engine.max_num_tokens)
            started_at = time.monotonic()
            _emit_rank_event(
                "engine_warmup_start",
                runtime_max_tokens=runtime_max_num_tokens,
                max_context_warmup_tokens=max_warmup_tokens,
            )
            warmup_error: BaseException | None = None
            try:
                return warmup(engine, resource_manager, *args, **kwargs)
            except BaseException as error:
                warmup_error = error
                raise
            finally:
                _emit_rank_event(
                    (
                        "engine_warmup_complete"
                        if warmup_error is None
                        else "engine_warmup_error"
                    ),
                    elapsed_seconds=f"{time.monotonic() - started_at:.3f}",
                    runtime_max_tokens=runtime_max_num_tokens,
                    **(
                        {}
                        if warmup_error is None
                        else {"error_type": type(warmup_error).__name__}
                    ),
                )

        traced_warmup._offline_engine_warmup_trace = True
        model_engine_class.warmup = traced_warmup

    return {
        "max_warmup_tokens": max_warmup_tokens,
        "target": (
            f"{model_engine_class.__name__}._create_warmup_request"
        ),
        "trace_target": f"{model_engine_class.__name__}.warmup",
        "already_installed": (
            cap_already_installed and trace_already_installed
        ),
    }


def _install_large_prefill_fp8_quant_guard() -> dict[str, Any]:
    max_fused_rows = int(
        os.environ["TRTLLM_BENCH_FP8_FUSED_QUANT_MAX_ROWS"]
    )
    if max_fused_rows <= 0:
        raise RuntimeError(
            "Large-prefill FP8 quant guard needs a positive row limit"
        )

    from tensorrt_llm._torch.custom_ops import torch_custom_ops

    runner_class = torch_custom_ops.Fp8QuantKernelRunner
    tactic_selector = runner_class.get_valid_tactics
    tactic_filter_installed = getattr(
        tactic_selector,
        "_offline_large_prefill_guard",
        False,
    )
    if not tactic_filter_installed:

        def guarded_get_valid_tactics(
            runner: Any,
            inputs: list[Any],
            profile: Any,
            **kwargs: Any,
        ) -> list[Any]:
            tactics = tactic_selector(runner, inputs, profile, **kwargs)
            rows = int(inputs[0].shape[0])
            if rows <= max_fused_rows:
                return tactics
            triton_tactic = runner.TACTIC_TRITON
            if triton_tactic not in tactics:
                raise RuntimeError(
                    "TRT large-prefill FP8 quantization has no Triton tactic"
                )
            return [triton_tactic]

        guarded_get_valid_tactics._offline_large_prefill_guard = True
        runner_class.get_valid_tactics = guarded_get_valid_tactics

    quantize = torch_custom_ops._fp8_quantize_1x128_ue8m0
    dispatch_guard_installed = getattr(
        quantize,
        "_offline_large_prefill_guard",
        False,
    )
    if not dispatch_guard_installed:

        def guarded_quantize(input_tensor: Any, tactic: int) -> Any:
            if int(input_tensor.shape[0]) > max_fused_rows:
                tactic = runner_class.TACTIC_TRITON
            return quantize(input_tensor, tactic)

        guarded_quantize._offline_large_prefill_guard = True
        torch_custom_ops._fp8_quantize_1x128_ue8m0 = guarded_quantize

    return {
        "max_fused_rows": max_fused_rows,
        "already_installed": (
            tactic_filter_installed and dispatch_guard_installed
        ),
    }


def _install_executor_lifecycle_trace() -> dict[str, Any]:
    from tensorrt_llm._torch.pyexecutor import py_executor

    executor_class = py_executor.PyExecutor
    installed: list[str] = []
    already_installed: list[str] = []
    for method_name, event_prefix in (
        ("_set_global_steady_clock_offset", "clock_sync"),
        ("start_worker", "executor_worker_start"),
    ):
        original = getattr(executor_class, method_name)
        if getattr(original, "_offline_lifecycle_trace", False):
            already_installed.append(method_name)
            continue

        def traced(
            executor: Any,
            *args: Any,
            _original: Any = original,
            _event_prefix: str = event_prefix,
            **kwargs: Any,
        ) -> Any:
            _emit_rank_event(f"{_event_prefix}_enter")
            try:
                result = _original(executor, *args, **kwargs)
            except BaseException as error:
                _emit_rank_event(
                    f"{_event_prefix}_error",
                    error_type=type(error).__name__,
                )
                raise
            _emit_rank_event(f"{_event_prefix}_exit")
            return result

        traced._offline_lifecycle_trace = True
        setattr(executor_class, method_name, traced)
        installed.append(method_name)
    return {
        "target": executor_class.__name__,
        "installed": installed,
        "already_installed": already_installed,
    }


def _install_fixed_batch_request_barrier() -> dict[str, Any]:
    expected = int(os.environ["TRTLLM_BENCH_GLOBAL_BATCH_SIZE"])
    arm_file_raw = os.getenv("TRTLLM_BENCH_FIXED_BATCH_ARM_FILE")
    if not arm_file_raw:
        raise RuntimeError(
            "TRTLLM_BENCH_FIXED_BATCH_ARM_FILE is required"
        )
    arm_file = Path(arm_file_raw)
    if not arm_file.is_absolute():
        raise RuntimeError(
            "Fixed-batch barrier arm file must use an absolute path"
        )
    timeout_seconds = float(
        os.getenv("TRTLLM_BENCH_FIXED_BATCH_TIMEOUT_SECONDS", "120")
    )
    if expected <= 0 or timeout_seconds <= 0:
        raise RuntimeError(
            "Fixed-batch request barrier needs positive size and timeout"
        )

    from tensorrt_llm._torch.pyexecutor import executor_request_queue

    request_queue_class = executor_request_queue.ExecutorRequestQueue
    original = request_queue_class.get_from_request_queue
    if getattr(original, "_offline_fixed_batch_barrier", False):
        return {
            "global_batch_size": expected,
            "timeout_seconds": timeout_seconds,
            "arm_file": str(arm_file),
            "armed_at_install": arm_file.is_file(),
            "already_installed": True,
        }

    armed = arm_file.is_file()

    def fixed_batch_get_from_request_queue(
        request_queue: Any,
        timeout: Any,
    ) -> list[Any]:
        nonlocal armed
        items = original(request_queue, timeout)
        # The MPI executor passes timeout=None only when there are no active or
        # waiting requests. That is the boundary between benchmark passes.
        if timeout is not None or not items:
            return items
        # TRT submits internal dummy requests while sizing the KV cache. The
        # parent creates this file only after LLM initialization has completed,
        # immediately before the benchmark's first real generate() call.
        if not armed:
            if not arm_file.is_file():
                return items
            armed = True
            _emit_rank_event(
                "fixed_batch_barrier_armed",
                arm_file=str(arm_file),
                global_batch_size=expected,
            )
        if any(not item.is_normal_request for item in items):
            return items
        if len(items) > expected:
            raise RuntimeError(
                "Fixed-batch barrier received too many requests: "
                f"{len(items)} > {expected}"
            )

        deadline = time.monotonic() + timeout_seconds
        while len(items) < expected:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "Fixed-batch barrier timed out after "
                    f"{timeout_seconds}s with {len(items)}/{expected} "
                    "requests"
                )
            try:
                item = request_queue.request_queue.get(timeout=remaining)
            except queue.Empty as error:
                raise RuntimeError(
                    "Fixed-batch barrier timed out after "
                    f"{timeout_seconds}s with {len(items)}/{expected} "
                    "requests"
                ) from error
            if not item.is_normal_request:
                raise RuntimeError(
                    "Fixed-batch barrier received a control, cancellation, "
                    "or shutdown item before the batch was complete"
                )
            items.append(item)
        return items

    fixed_batch_get_from_request_queue._offline_fixed_batch_barrier = True
    request_queue_class.get_from_request_queue = (
        fixed_batch_get_from_request_queue
    )
    return {
        "global_batch_size": expected,
        "timeout_seconds": timeout_seconds,
        "arm_file": str(arm_file),
        "armed_at_install": armed,
        "already_installed": False,
    }


def _write_marker(event: str = "entry_ready", **details: Any) -> None:
    marker = os.getenv("TRTLLM_PERFECT_ROUTER_MARKER")
    if not marker:
        return
    expected_environment_raw = os.getenv(
        "TRTLLM_BENCH_EXPECTED_RANK_ENV",
        "{}",
    )
    try:
        expected_environment = json.loads(expected_environment_raw)
    except json.JSONDecodeError:
        expected_environment = {}
    if not isinstance(expected_environment, dict):
        expected_environment = {}
    payload = {
        "pid": os.getpid(),
        "rank": os.getenv("OMPI_COMM_WORLD_RANK"),
        "perfect_router": os.getenv("ENABLE_PERFECT_ROUTER"),
        "cute_dsl_cache_dir": os.getenv("CUTE_DSL_CACHE_DIR"),
        "benchmark_environment": {
            str(name): os.getenv(str(name))
            for name in expected_environment
        },
        "fixed_batch_global_size": os.getenv(
            "TRTLLM_BENCH_GLOBAL_BATCH_SIZE"
        ),
        "fixed_batch_arm_file": os.getenv(
            "TRTLLM_BENCH_FIXED_BATCH_ARM_FILE"
        ),
        "fixed_batch_barrier_armed": (
            Path(os.environ["TRTLLM_BENCH_FIXED_BATCH_ARM_FILE"]).is_file()
            if os.getenv("TRTLLM_BENCH_FIXED_BATCH_ARM_FILE")
            else False
        ),
        "engine_warmup_max_tokens": os.getenv(
            "TRTLLM_BENCH_ENGINE_WARMUP_MAX_TOKENS"
        ),
        "fp8_fused_quant_max_rows": os.getenv(
            "TRTLLM_BENCH_FP8_FUSED_QUANT_MAX_ROWS"
        ),
        "event": event,
        "source": "trt_mpi_entry",
    }
    payload.update(details)
    path = Path(marker)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def worker_main(*args: Any, **kwargs: Any) -> Any:
    """Restore benchmark aliases before importing TRT's real worker entry."""
    if os.getenv("TRTLLM_ENABLE_PERFECT_ROUTER") == "1":
        os.environ["ENABLE_PERFECT_ROUTER"] = "1"
    configurable_moe = os.getenv(
        "TRTLLM_BENCH_ENABLE_CONFIGURABLE_MOE"
    )
    if configurable_moe is not None:
        os.environ["ENABLE_CONFIGURABLE_MOE"] = configurable_moe
    cute_cache_dir = os.getenv("TRTLLM_BENCH_CUTE_DSL_CACHE_DIR")
    if cute_cache_dir:
        os.environ["CUTE_DSL_CACHE_DIR"] = cute_cache_dir
    warmup_cap = _install_engine_warmup_shape_cap()
    fp8_guard = _install_large_prefill_fp8_quant_guard()
    lifecycle_trace = _install_executor_lifecycle_trace()
    fixed_batch_barrier = _install_fixed_batch_request_barrier()
    _write_marker()
    print(
        "[offline-trt-mpi] fixed-batch request barrier "
        f"global_batch={fixed_batch_barrier['global_batch_size']} "
        f"arm_file={fixed_batch_barrier['arm_file']} "
        f"armed_at_install={fixed_batch_barrier['armed_at_install']} "
        f"already_installed={fixed_batch_barrier['already_installed']}",
        flush=True,
    )
    print(
        "[offline-trt-mpi] synthetic engine warmup shape cap "
        f"max_tokens={warmup_cap['max_warmup_tokens']} "
        f"target={warmup_cap['target']} "
        f"trace_target={warmup_cap['trace_target']} "
        f"already_installed={warmup_cap['already_installed']}",
        flush=True,
    )
    print(
        "[offline-trt-mpi] large-prefill FP8 quant guard "
        f"max_fused_rows={fp8_guard['max_fused_rows']} "
        f"already_installed={fp8_guard['already_installed']}",
        flush=True,
    )
    print(
        "[offline-trt-mpi] executor lifecycle trace "
        f"target={lifecycle_trace['target']} "
        f"installed={lifecycle_trace['installed']} "
        f"already_installed={lifecycle_trace['already_installed']}",
        flush=True,
    )

    from tensorrt_llm.executor.worker import worker_main as trt_worker_main

    return trt_worker_main(*args, **kwargs)
