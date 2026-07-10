#!/usr/bin/env python3
"""Backport vLLM MooncakeStore and scheduler correctness fixes to v0.24.0.

v0.24.0 keeps a connector-local set of preempted request IDs to decide
whether cached-request block IDs replace or extend the tracked block table.
That set can drift from the scheduler, causing a normal decode delta to
replace the full table after preemption.  The Mooncake send thread then walks
past the shortened block table and silently drops the request's store work.

Upstream now uses ``CachedRequestData.resumed_req_ids``, which is the
scheduler-owned source of truth for replacement semantics.  It also logs the
request ID and traceback when a transfer thread raises.

The v0.24.0 receive path also assumes every worker resolves at least one local
Mooncake key.  With hybrid KV layouts, a worker can legitimately have no local
chunks for a request.  Rotating an empty key list raises before the request is
marked complete, so the all-worker completion aggregator waits forever.  Treat
that case as a successful no-op load.

The scheduler can also strand completed async loads behind an unschedulable
queue head.  If nothing is running, stopping at that head means no future work
can free blocks and the engine remains permanently idle.  Backport the focused
recovery from vLLM PR #45406: in that terminal state only, skip the head and
continue traversal so a completed parked load can be promoted and make progress.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path


EXPECTED_VERSION = "0.24.0"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new and new in text:
        return
    if text.count(old) != 1:
        raise RuntimeError(
            f"Expected exactly one vLLM v{EXPECTED_VERSION} patch target in {path}"
        )
    path.write_text(text.replace(old, new, 1))


def main() -> None:
    version = importlib.metadata.version("vllm")
    if version != EXPECTED_VERSION:
        raise RuntimeError(
            f"Mooncake resumed-request backport targets vLLM {EXPECTED_VERSION}; "
            f"found {version}"
        )

    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Could not locate the installed vllm package")
    package_root = Path(next(iter(spec.submodule_search_locations)))
    store_root = (
        package_root
        / "distributed"
        / "kv_transfer"
        / "kv_connector"
        / "v1"
        / "mooncake"
        / "store"
    )

    scheduler = store_root / "scheduler.py"
    worker = store_root / "worker.py"
    core_scheduler = package_root / "v1" / "core" / "sched" / "scheduler.py"

    resumed_requests_patched = (
        "if req_id in cached_reqs.resumed_req_ids:" in scheduler.read_text()
        and 'logger.exception("Error in %s (req=%s)"' in worker.read_text()
    )
    empty_load_patched = "Mooncake load has no local keys" in worker.read_text()
    parked_load_recovery_patched = (
        "Nothing is running, so no future event frees blocks" in core_scheduler.read_text()
    )
    if (
        resumed_requests_patched
        and empty_load_patched
        and parked_load_recovery_patched
    ):
        print(
            "vLLM MooncakeStore correctness backports already present in "
            f"{package_root}"
        )
        return

    if not resumed_requests_patched:
        replace_once(
            scheduler,
            '        self._preempted_req_ids: set[str] = set()  # preempted requests\n',
            "",
        )
        replace_once(
            scheduler,
            '            self._preempted_req_ids.discard(finished_req_id)\n',
            "",
        )
        replace_once(
            scheduler,
            '        self._preempted_req_ids.update(preempted_ids)\n',
            "",
        )
        replace_once(
            scheduler,
            '                if req_id in self._preempted_req_ids:\n',
            '                if req_id in cached_reqs.resumed_req_ids:\n',
        )
        replace_once(
            scheduler,
            '                    self._preempted_req_ids.discard(req_id)\n',
            "",
        )

        replace_once(
            worker,
            """        while True:
            try:
                request_data = self.request_queue.get()
""",
            """        while True:
            request_data = None
            try:
                request_data = self.request_queue.get()
""",
        )
        replace_once(
            worker,
            """            except Exception as e:
                logger.error("Error in %s: %s", self.name, e)
""",
            """            except Exception:
                req_id = getattr(request_data, "req_id", "<unknown>")
                logger.exception("Error in %s (req=%s)", self.name, req_id)
""",
        )

    if not empty_load_patched:
        replace_once(
            worker,
            """        # Rotate aligned lists by tp_rank for load balancing.
        rotation = self.tp_rank % len(key_list)
""",
            """        # A worker can legitimately own no local chunks for a request
        # (for example with a hybrid KV layout). It must still acknowledge the
        # async load so the cross-worker completion aggregator can release the
        # request instead of waiting forever for this rank.
        if not key_list:
            logger.debug(
                "Mooncake load has no local keys for request %s on tp_rank %d; "
                "marking the no-op load complete",
                req_id,
                self.tp_rank,
            )
            self.set_finished_request(req_id)
            self.request_queue.task_done()
            return

        # Rotate aligned lists by tp_rank for load balancing.
        rotation = self.tp_rank % len(key_list)
""",
        )

    if not parked_load_recovery_patched:
        replace_once(
            core_scheduler,
            """                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                # KVTransfer: the connector uses this info to determine
""",
            """                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    if self.running:
                        # Running requests will eventually free blocks; stop
                        # here to preserve queue-order admission.
                        break
                    # Nothing is running, so no future event frees blocks and
                    # stopping at this request would freeze this state. A
                    # completed async load behind it may hold blocks and is
                    # only promoted when traversal reaches it. Skip the head
                    # so that parked load can run and release capacity.
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # KVTransfer: the connector uses this info to determine
""",
        )

    compile(scheduler.read_text(), str(scheduler), "exec")
    compile(worker.read_text(), str(worker), "exec")
    compile(core_scheduler.read_text(), str(core_scheduler), "exec")
    print(f"Applied vLLM MooncakeStore correctness backports to {package_root}")


if __name__ == "__main__":
    main()
