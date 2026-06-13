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
        "source": "trt_mpi_entry",
    }
    path = Path(marker)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def worker_main(*args: Any, **kwargs: Any) -> Any:
    """Set the alias before importing TRT's real model worker entry."""
    if os.getenv("TRTLLM_ENABLE_PERFECT_ROUTER") == "1":
        os.environ["ENABLE_PERFECT_ROUTER"] = "1"
    _write_marker()

    from tensorrt_llm.executor.worker import worker_main as trt_worker_main

    return trt_worker_main(*args, **kwargs)
