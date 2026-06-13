"""MPI worker entry shim for the pinned TRT perfect-router environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _write_marker() -> None:
    marker = os.getenv("TRTLLM_PERFECT_ROUTER_MARKER")
    if not marker:
        return
    payload = {
        "pid": os.getpid(),
        "rank": os.getenv("OMPI_COMM_WORLD_RANK"),
        "perfect_router": os.getenv("ENABLE_PERFECT_ROUTER"),
        "cute_dsl_cache_dir": os.getenv("CUTE_DSL_CACHE_DIR"),
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
    cute_cache_dir = os.getenv("TRTLLM_BENCH_CUTE_DSL_CACHE_DIR")
    if cute_cache_dir:
        os.environ["CUTE_DSL_CACHE_DIR"] = cute_cache_dir
    _write_marker()

    from tensorrt_llm.executor.worker import worker_main as trt_worker_main

    return trt_worker_main(*args, **kwargs)
