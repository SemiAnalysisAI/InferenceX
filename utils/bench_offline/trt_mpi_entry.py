"""MPI worker entry shim for the pinned TRT perfect-router environment."""

from __future__ import annotations

import json
import os
import queue
import time
from pathlib import Path
from typing import Any


def _install_engine_warmup_token_cap() -> dict[str, Any]:
    max_warmup_tokens = int(
        os.environ["TRTLLM_BENCH_ENGINE_WARMUP_MAX_TOKENS"]
    )
    if max_warmup_tokens <= 0:
        raise RuntimeError(
            "TRT engine warmup token cap needs a positive limit"
        )

    from tensorrt_llm._torch.pyexecutor import model_engine

    model_engine_class = model_engine.ModelEngine
    original = model_engine_class.warmup
    if getattr(original, "_offline_engine_warmup_token_cap", False):
        return {
            "max_warmup_tokens": max_warmup_tokens,
            "already_installed": True,
        }

    def bounded_warmup(
        engine: Any,
        resource_manager: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        runtime_max_num_tokens = int(engine.max_num_tokens)
        tuned_max_num_tokens = min(
            runtime_max_num_tokens,
            max_warmup_tokens,
        )
        engine.max_num_tokens = tuned_max_num_tokens
        started_at = time.monotonic()
        print(
            "[offline-trt-mpi] synthetic engine warmup start "
            f"runtime_max_tokens={runtime_max_num_tokens} "
            f"tuned_max_tokens={tuned_max_num_tokens}",
            flush=True,
        )
        try:
            return original(engine, resource_manager, *args, **kwargs)
        finally:
            engine.max_num_tokens = runtime_max_num_tokens
            print(
                "[offline-trt-mpi] synthetic engine warmup complete "
                f"elapsed_seconds={time.monotonic() - started_at:.3f} "
                f"restored_max_tokens={runtime_max_num_tokens}",
                flush=True,
            )

    bounded_warmup._offline_engine_warmup_token_cap = True
    model_engine_class.warmup = bounded_warmup
    return {
        "max_warmup_tokens": max_warmup_tokens,
        "already_installed": False,
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


def _install_fixed_batch_request_barrier() -> dict[str, Any]:
    expected = int(os.environ["TRTLLM_BENCH_GLOBAL_BATCH_SIZE"])
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
            "already_installed": True,
        }

    def fixed_batch_get_from_request_queue(
        request_queue: Any,
        timeout: Any,
    ) -> list[Any]:
        items = original(request_queue, timeout)
        # The MPI executor passes timeout=None only when there are no active or
        # waiting requests. That is the boundary between benchmark passes.
        if timeout is not None or not items:
            return items
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
        "already_installed": False,
    }


def _write_marker() -> None:
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
        "engine_warmup_max_tokens": os.getenv(
            "TRTLLM_BENCH_ENGINE_WARMUP_MAX_TOKENS"
        ),
        "fp8_fused_quant_max_rows": os.getenv(
            "TRTLLM_BENCH_FP8_FUSED_QUANT_MAX_ROWS"
        ),
        "source": "trt_mpi_entry",
    }
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
    warmup_cap = _install_engine_warmup_token_cap()
    fp8_guard = _install_large_prefill_fp8_quant_guard()
    _install_fixed_batch_request_barrier()
    _write_marker()
    print(
        "[offline-trt-mpi] synthetic engine warmup token cap "
        f"max_tokens={warmup_cap['max_warmup_tokens']} "
        f"already_installed={warmup_cap['already_installed']}",
        flush=True,
    )
    print(
        "[offline-trt-mpi] large-prefill FP8 quant guard "
        f"max_fused_rows={fp8_guard['max_fused_rows']} "
        f"already_installed={fp8_guard['already_installed']}",
        flush=True,
    )

    from tensorrt_llm.executor.worker import worker_main as trt_worker_main

    return trt_worker_main(*args, **kwargs)
