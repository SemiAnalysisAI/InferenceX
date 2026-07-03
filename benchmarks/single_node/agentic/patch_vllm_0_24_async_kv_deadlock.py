#!/usr/bin/env python3
"""Backport vLLM PR #45406 to the v0.24.0 scheduler."""

from importlib.metadata import distribution
from pathlib import Path


OLD_BLOCK = """\
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break
"""

NEW_BLOCK = """\
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    if self.running:
                        # Running requests will free blocks when they
                        # complete; stop here to preserve queue-order
                        # admission.
                        break
                    # Nothing is running, so no future event frees blocks and
                    # stopping at this request would freeze this state
                    # permanently. Requests behind this one may hold blocks
                    # while parked (async KV loads in WAITING_FOR_REMOTE_KVS)
                    # and are only promoted when this traversal reaches them.
                    # Keep scanning so they can be promoted, scheduled, and
                    # eventually free the blocks this request needs.
                    # See https://github.com/vllm-project/vllm/issues/45388
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue
"""


def main() -> None:
    vllm_distribution = distribution("vllm")
    installed_version = vllm_distribution.version
    if installed_version != "0.24.0":
        raise RuntimeError(
            f"Expected vLLM 0.24.0, found {installed_version}; "
            "review whether PR #45406 is still needed"
        )

    scheduler_path = Path(
        vllm_distribution.locate_file("vllm/v1/core/sched/scheduler.py")
    )
    source = scheduler_path.read_text()
    if NEW_BLOCK in source:
        print("vLLM async-KV scheduler fix already present")
        return
    if source.count(OLD_BLOCK) != 1:
        raise RuntimeError(
            f"Could not uniquely locate the vLLM 0.24.0 scheduler block in "
            f"{scheduler_path}"
        )

    scheduler_path.write_text(source.replace(OLD_BLOCK, NEW_BLOCK))
    print(f"Applied vLLM PR #45406 scheduler fix to {scheduler_path}")


if __name__ == "__main__":
    main()
