#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Never block the scheduler on the auxiliary KV-metrics PUSH socket.

WHAT: with KV events enabled, `KvEventsPublisher` also PUSHes a `KvMetrics`
record every scheduler step to an endpoint that nothing binds in an Infera
deployment (the router reads KV events and /metrics, not this socket). Once the
ZMQ send queue fills, `sock_send` blocks and the scheduler stops: a GLM-5.2
prefill wedged this way (yihou agentx44_20261009, R001, py-spy 2026-10-09).

FIX: send with zmq.NOBLOCK and drop the record on zmq.Again. KV events and the
Prometheus metrics are unaffected. The v0.5.21 deployments carried this as a
bind-mounted file; v0.5.22 still sends blocking.

Self-locating and idempotent; exits non-zero when the anchor drifted.
Marker: `_infera_kv_metrics_noblock`.
"""

import importlib.util
import py_compile
import sys
from pathlib import Path

TAG = "[kv-metrics-noblock]"
MARKER = "_infera_kv_metrics_noblock"
OLD = (
    "        if not self.send_metrics_from_scheduler.closed:\n"
    "            sock_send(self.send_metrics_from_scheduler, kv_metrics)\n"
)
NEW = (
    "        if not self.send_metrics_from_scheduler.closed:\n"
    "            # Infera: nothing may bind this PUSH peer; a blocking send would\n"
    "            # stall the scheduler once the queue fills, so drop instead.\n"
    "            try:\n"
    "                sock_send(self.send_metrics_from_scheduler, kv_metrics, flags=zmq.NOBLOCK)\n"
    f"            except zmq.Again:  # {MARKER}\n"
    "                pass\n"
)


def main() -> int:
    spec = importlib.util.find_spec("sglang")
    if not spec or not spec.origin:
        print(f"{TAG} cannot locate sglang")
        return 1
    path = (
        Path(spec.origin).parent
        / "srt"
        / "managers"
        / "scheduler_components"
        / "kv_events_publisher.py"
    )
    source = path.read_text()
    if MARKER in source:
        print(f"{TAG} already present — skipping")
        return 0
    if source.count(OLD) != 1 or "\nimport zmq\n" not in source:
        print(f"{TAG} anchor not found exactly once in {path} — re-cut the patch")
        return 1
    path.write_text(source.replace(OLD, NEW, 1))
    for cached in (path.parent / "__pycache__").glob(path.stem + ".*.pyc"):
        cached.unlink()
    py_compile.compile(str(path), doraise=True)
    print(f"{TAG} patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
