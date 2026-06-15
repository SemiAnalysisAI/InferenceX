"""Environment aliases that must be normalized in TensorRT-LLM MPI workers.

TensorRT-LLM's ``MpiPoolSession`` explicitly forwards ``TRTLLM_*`` and
``TLLM_*`` variables. Huawei mode recreates the unprefixed perfect-router
alias before TensorRT-LLM imports its model code; PR-max removes any stale
unprefixed value so learned routing remains active.
"""

from __future__ import annotations

import os
from pathlib import Path

from io_utils import append_json_line


def apply_trtllm_env_aliases() -> None:
    if os.getenv("TRTLLM_ENABLE_PERFECT_ROUTER") == "1":
        os.environ["ENABLE_PERFECT_ROUTER"] = "1"
    else:
        os.environ.pop("ENABLE_PERFECT_ROUTER", None)

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
