#!/usr/bin/env python3
"""Backport vLLM MooncakeStore fixes to v0.24.0.

v0.24.0 keeps a connector-local set of preempted request IDs to decide
whether cached-request block IDs replace or extend the tracked block table.
That set can drift from the scheduler, causing a normal decode delta to
replace the full table after preemption.  The Mooncake send thread then walks
past the shortened block table and silently drops the request's store work.

Upstream now uses ``CachedRequestData.resumed_req_ids``, which is the
scheduler-owned source of truth for replacement semantics.  It also logs the
request ID and traceback when a transfer thread raises.

vLLM #45971 replaces the single serial Mooncake receive thread with a
configurable pool.  A single long ``batch_get_into_multi_buffers`` otherwise
head-of-line blocks every later async load on that rank, leaving scheduler
requests indefinitely deferred under sustained cache pressure.
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

    resumed_requests_patched = (
        "if req_id in cached_reqs.resumed_req_ids:" in scheduler.read_text()
        and 'logger.exception("Error in %s (req=%s)"' in worker.read_text()
    )
    receive_pool_patched = (
        "self.kv_recv_threads: list[KVCacheStoreRecvingThread] = []"
        in worker.read_text()
        and "VLLM_MOONCAKE_LOAD_RECV_THREADS" in worker.read_text()
    )
    if resumed_requests_patched and receive_pool_patched:
        print(f"vLLM MooncakeStore backports already present in {package_root}")
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

    if not receive_pool_patched:
        replace_once(
            worker,
            """        name: str,
        record_operation: Callable[..., None] | None = None,
    ):
""",
            """        name: str,
        record_operation: Callable[..., None] | None = None,
        request_queue: queue.Queue[Any] | None = None,
    ):
""",
        )
        replace_once(
            worker,
            """        self.request_queue: queue.Queue[Any] = queue.Queue()
""",
            """        self.request_queue: queue.Queue[Any] = request_queue or queue.Queue()
""",
        )
        replace_once(
            worker,
            """        disk_offload_buffer_budget_bytes: int | None = None,
        record_operation: Callable[..., None] | None = None,
    ):
""",
            """        disk_offload_buffer_budget_bytes: int | None = None,
        record_operation: Callable[..., None] | None = None,
        request_queue: queue.Queue[Any] | None = None,
    ):
""",
        )
        replace_once(
            worker,
            """            name="KVCacheStoreRecvingThread",
            record_operation=record_operation,
        )
""",
            """            name="KVCacheStoreRecvingThread",
            record_operation=record_operation,
            request_queue=request_queue,
        )
""",
        )
        replace_once(
            worker,
            """        self.kv_send_thread: KVCacheStoreSendingThread | None = None
        self.kv_recv_thread: KVCacheStoreRecvingThread | None = None
""",
            """        self.kv_send_thread: KVCacheStoreSendingThread | None = None
        self.kv_recv_threads: list[KVCacheStoreRecvingThread] = []
        self.num_recv_threads = max(
            1, int(os.getenv("VLLM_MOONCAKE_LOAD_RECV_THREADS", "1"))
        )
        self.recv_request_queue: queue.Queue[Any] = queue.Queue()
""",
        )
        replace_once(
            worker,
            """        ready_event_recving = threading.Event()
        self.kv_recv_thread = KVCacheStoreRecvingThread(
            self.store,
            self.coord,
            self.token_dbs,
            self.block_size,
            self.tp_rank,
            ready_event_recving,
            disk_offload_buffer_budget_bytes=self.disk_offload_buffer_budget_bytes,
            record_operation=self._record_kv_connector_operation,
        )
        self.kv_recv_thread.start()
        ready_event_recving.wait()
""",
            """        ready_events_recving: list[threading.Event] = []
        for i in range(self.num_recv_threads):
            ready_event_recving = threading.Event()
            recv_thread = KVCacheStoreRecvingThread(
                self.store,
                self.coord,
                self.token_dbs,
                self.block_size,
                self.tp_rank,
                ready_event_recving,
                disk_offload_buffer_budget_bytes=self.disk_offload_buffer_budget_bytes,
                record_operation=self._record_kv_connector_operation,
                request_queue=self.recv_request_queue,
            )
            recv_thread.name = f"KVCacheStoreRecvingThread-{i}"
            recv_thread.start()
            self.kv_recv_threads.append(recv_thread)
            ready_events_recving.append(ready_event_recving)
        for ready_event_recving in ready_events_recving:
            ready_event_recving.wait()
        logger.info(
            "Started %d Mooncake KV-load receive thread(s)", self.num_recv_threads
        )
""",
        )
        replace_once(
            worker,
            """            assert self.kv_recv_thread is not None
            self.kv_recv_thread.add_request(request)
""",
            """            self.recv_request_queue.put(request)
""",
        )
        replace_once(
            worker,
            """        done_recving = (
            self.kv_recv_thread.get_and_clear_finished_requests()
            if self.load_async and self.kv_recv_thread is not None
            else set()
        )
""",
            """        done_recving: set[str] = set()
        if self.load_async:
            for recv_thread in self.kv_recv_threads:
                done_recving |= recv_thread.get_and_clear_finished_requests()
""",
        )
        replace_once(
            worker,
            """    def get_block_ids_with_load_errors(self) -> set[int]:
        if self.kv_recv_thread is None:
            return set()
        return self.kv_recv_thread.get_and_clear_block_ids_with_load_errors()
""",
            """    def get_block_ids_with_load_errors(self) -> set[int]:
        block_ids: set[int] = set()
        for recv_thread in self.kv_recv_threads:
            block_ids |= recv_thread.get_and_clear_block_ids_with_load_errors()
        return block_ids
""",
        )

    compile(scheduler.read_text(), str(scheduler), "exec")
    compile(worker.read_text(), str(worker), "exec")
    print(f"Applied vLLM #46595 and #45971 MooncakeStore backports to {package_root}")


if __name__ == "__main__":
    main()
