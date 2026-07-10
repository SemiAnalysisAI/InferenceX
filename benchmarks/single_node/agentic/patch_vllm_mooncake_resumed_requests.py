#!/usr/bin/env python3
"""Backport vLLM #46595 to the v0.24.0 MooncakeStore connector.

v0.24.0 keeps a connector-local set of preempted request IDs to decide
whether cached-request block IDs replace or extend the tracked block table.
That set can drift from the scheduler, causing a normal decode delta to
replace the full table after preemption.  The Mooncake send thread then walks
past the shortened block table and silently drops the request's store work.

Upstream now uses ``CachedRequestData.resumed_req_ids``, which is the
scheduler-owned source of truth for replacement semantics.  It also logs the
request ID and traceback when a transfer thread raises.
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

    if (
        "if req_id in cached_reqs.resumed_req_ids:" in scheduler.read_text()
        and 'logger.exception("Error in %s (req=%s)"' in worker.read_text()
    ):
        print(f"vLLM #46595 MooncakeStore backport already present in {package_root}")
        return

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

    compile(scheduler.read_text(), str(scheduler), "exec")
    compile(worker.read_text(), str(worker), "exec")
    print(f"Applied vLLM #46595 MooncakeStore backport to {package_root}")


if __name__ == "__main__":
    main()
