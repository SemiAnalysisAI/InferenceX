#!/usr/bin/env python3
"""Backport vLLM MooncakeStore correctness fixes to v0.24.0.

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
    diagnostics_patched = (
        "MOONCAKE_DIAG load-enqueue" in worker.read_text()
        and "MOONCAKE_DIAG scheduler-waiting" in core_scheduler.read_text()
        and "MOONCAKE_DIAG scheduler-finished" in core_scheduler.read_text()
    )
    if resumed_requests_patched and empty_load_patched and diagnostics_patched:
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

    if not diagnostics_patched:
        replace_once(
            worker,
            """            load_spec.token_len = load_spec.kvpool_cached_tokens

            assert self.kv_recv_thread is not None
            self.kv_recv_thread.add_request(request)
""",
            """            load_spec.token_len = load_spec.kvpool_cached_tokens

            logger.warning(
                "MOONCAKE_DIAG load-enqueue req=%s tp_rank=%d token_len=%d",
                request.req_id,
                self.tp_rank,
                load_spec.token_len,
            )
            assert self.kv_recv_thread is not None
            self.kv_recv_thread.add_request(request)
""",
        )
        replace_once(
            worker,
            """        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        req_id = req_meta.req_id
""",
            """        token_len = req_meta.load_spec.token_len  # type: ignore[union-attr]
        req_id = req_meta.req_id
        logger.warning(
            "MOONCAKE_DIAG load-start req=%s tp_rank=%d token_len=%d",
            req_id,
            self.tp_rank,
            token_len,
        )
""",
        )
        replace_once(
            worker,
            """        self.set_finished_request(req_id)
        self.request_queue.task_done()


# ============================================================
""",
            """        logger.warning(
            "MOONCAKE_DIAG load-complete req=%s tp_rank=%d",
            req_id,
            self.tp_rank,
        )
        self.set_finished_request(req_id)
        self.request_queue.task_done()


# ============================================================
""",
        )
        replace_once(
            worker,
            """        logger.debug(
            "Completed send: %d, recv: %d, tp_rank: %d",
""",
            """        if done_recving:
            logger.warning(
                "MOONCAKE_DIAG load-report tp_rank=%d reqs=%s",
                self.tp_rank,
                sorted(done_recving),
            )
        logger.debug(
            "Completed send: %d, recv: %d, tp_rank: %d",
""",
        )
        replace_once(
            core_scheduler,
            """                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    step_skipped_waiting.prepend_request(request)
""",
            """                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    logger.warning(
                        "MOONCAKE_DIAG scheduler-waiting req=%s external_tokens=%d",
                        request_id,
                        num_external_computed_tokens,
                    )
                    step_skipped_waiting.prepend_request(request)
""",
        )
        replace_once(
            core_scheduler,
            """        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
""",
            """        for req_id in kv_connector_output.finished_recving or ():
            logger.warning("MOONCAKE_DIAG scheduler-finished req=%s", req_id)
            logger.debug("Finished recving KV transfer for request %s", req_id)
""",
        )

    compile(scheduler.read_text(), str(scheduler), "exec")
    compile(worker.read_text(), str(worker), "exec")
    compile(core_scheduler.read_text(), str(core_scheduler), "exec")
    print(f"Applied vLLM MooncakeStore correctness backports to {package_root}")


if __name__ == "__main__":
    main()
