#!/usr/bin/env python3
"""Apply vLLM issue #46240 workaround to Scheduler._update_from_kv_xfer_finished.

https://github.com/vllm-project/vllm/issues/46240

When async KV connectors (MultiConnector + MooncakeStore) report finished_recving
and finished_sending for the same request in one step, or report completion after
the scheduler already freed the request, the hard assert kills EngineCore.

Replace ``assert req_id in self.requests`` with a skip for untracked ids.
"""
from __future__ import annotations

import sys
from pathlib import Path

PATCH_MARKER = "INFERENCEX_PATCH_46240"

RECV_OLD = """        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests"""

RECV_NEW = f"""        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            if req_id not in self.requests:
                continue  # {PATCH_MARKER}: stale async KV recv completion"""

SEND_OLD = """        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests"""

SEND_NEW = f"""        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            if req_id not in self.requests:
                continue  # {PATCH_MARKER}: stale async KV send completion"""


def find_scheduler_py() -> Path:
    import vllm

    return Path(vllm.__file__).resolve().parent / "v1" / "core" / "sched" / "scheduler.py"


def apply_patch(path: Path) -> None:
    text = path.read_text()
    if PATCH_MARKER in text:
        print(f"[patch-46240] Already applied: {path}")
        return

    if RECV_OLD not in text:
        raise SystemExit(f"[patch-46240] recv pattern not found in {path}")
    if SEND_OLD not in text:
        raise SystemExit(f"[patch-46240] send pattern not found in {path}")

    text = text.replace(RECV_OLD, RECV_NEW, 1).replace(SEND_OLD, SEND_NEW, 1)
    if PATCH_MARKER not in text:
        raise SystemExit(f"[patch-46240] patch application failed for {path}")

    path.write_text(text)
    print(f"[patch-46240] Patched {path}")


def main() -> None:
    path = find_scheduler_py()
    if not path.is_file():
        raise SystemExit(f"[patch-46240] scheduler.py not found: {path}")
    apply_patch(path)


if __name__ == "__main__":
    main()
