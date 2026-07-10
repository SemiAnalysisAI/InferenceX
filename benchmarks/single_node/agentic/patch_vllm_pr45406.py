#!/usr/bin/env python3
"""Backport the scheduler fix from vllm-project/vllm PR #45406."""

from __future__ import annotations

import importlib.util
from pathlib import Path


PATCH_MARKER = "See https://github.com/vllm-project/vllm/issues/45388"


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Could not locate the installed vllm package")

    package_root = Path(next(iter(spec.submodule_search_locations)))
    scheduler = package_root / "v1" / "core" / "sched" / "scheduler.py"
    text = scheduler.read_text()
    if PATCH_MARKER in text:
        print(f"vLLM PR #45406 backport already present in {scheduler}")
        return

    old = """                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                # KVTransfer: the connector uses this info to determine
"""
    new = """                    if request.has_encoder_inputs:
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

                # KVTransfer: the connector uses this info to determine
"""
    if text.count(old) != 1:
        raise RuntimeError(
            f"Expected exactly one vLLM PR #45406 patch target in {scheduler}"
        )

    scheduler.write_text(text.replace(old, new, 1))
    compile(scheduler.read_text(), str(scheduler), "exec")
    print(f"Applied vLLM PR #45406 backport to {scheduler}")


if __name__ == "__main__":
    main()
