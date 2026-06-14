"""Environment aliases that must exist in TensorRT-LLM MPI workers.

TensorRT-LLM's ``MpiPoolSession`` explicitly forwards ``TRTLLM_*`` and
``TLLM_*`` variables. The perfect-router implementation reads the unprefixed
``ENABLE_PERFECT_ROUTER`` variable, so recreate it in every spawned Python
process before TensorRT-LLM imports its model code.
"""

from __future__ import annotations

import os
from pathlib import Path

from io_utils import append_json_line


def apply_trtllm_env_aliases() -> None:
    if os.getenv("TRTLLM_ENABLE_PERFECT_ROUTER") == "1":
        os.environ["ENABLE_PERFECT_ROUTER"] = "1"

    marker = os.getenv("TRTLLM_PERFECT_ROUTER_MARKER")
    if not marker:
        return

    payload = {
        "pid": os.getpid(),
        "perfect_router": os.getenv("ENABLE_PERFECT_ROUTER"),
        "source": "sitecustomize",
    }
    try:
        append_json_line(Path(marker), payload)
    except OSError:
        # This module runs during interpreter startup. The benchmark validates
        # the marker later and reports a useful error there.
        pass


apply_trtllm_env_aliases()
