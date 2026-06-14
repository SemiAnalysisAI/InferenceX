"""MPI worker entry shim for the pinned TRT perfect-router environment."""

from __future__ import annotations

import json
import os
import queue
import time
from pathlib import Path
from typing import Any


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
    _install_fixed_batch_request_barrier()
    _write_marker()

    from tensorrt_llm.executor.worker import worker_main as trt_worker_main

    return trt_worker_main(*args, **kwargs)
