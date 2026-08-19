"""Instrument SimpleCPUOffloadConnector batch memcpys and keep descriptors alive.

vllm/v1/simple_kv_offload/cuda_mem_ops.py:copy_blocks turns (base, block_id, bpb)
triples into raw host/device addresses and hands them to hipMemcpyBatchAsync. On
Kimi-K3 TP8PP2 agentic that call segfaults at conc>=16 while conc<=12 is clean, and
the faulting thread sits inside the memcpy itself, so an address in the batch is bad
rather than the batch being too large (observed batches are tiny: n=35, 5 layers).

This replacement logs the block-id range of every call so the last batch before the
crash is recoverable, and retains the descriptor arrays so a lifetime bug in the async
call cannot masquerade as a bad address.
"""
from __future__ import annotations

import os
import sys
import threading

_TARGET = "vllm.v1.simple_kv_offload.copy_backend"
_OPS = "vllm.v1.simple_kv_offload.cuda_mem_ops"


def _make_instrumented(mod_ops):
    import collections
    import ctypes

    import numpy as np

    keepalive = collections.deque(maxlen=4096)
    try:
        log = open("/tmp/kvoff_%d.log" % os.getpid(), "w", buffering=1)
    except Exception:
        log = None
    logged_bases = False

    def copy_blocks(src_block_ids, dst_block_ids, params):
        nonlocal logged_bases
        n = len(src_block_ids)
        if n == 0:
            return

        src_ids = np.array(src_block_ids, dtype=np.uint64)
        dst_ids = np.array(dst_block_ids, dtype=np.uint64)

        if log is not None:
            if not logged_bases:
                log.write(
                    "bases src=%r dst=%r bpb=%r num_layers=%d\n"
                    % (
                        list(params.src_bases),
                        list(params.dst_bases),
                        list(params.bpb),
                        params.num_layers,
                    )
                )
                logged_bases = True
            log.write(
                "n=%d src=[%d,%d] dst=[%d,%d]\n"
                % (
                    n,
                    int(src_ids.min()),
                    int(src_ids.max()),
                    int(dst_ids.min()),
                    int(dst_ids.max()),
                )
            )

        src_all = (
            params.src_bases[:, None] + src_ids[None, :] * params.bpb[:, None]
        ).ravel()
        dst_all = (
            params.dst_bases[:, None] + dst_ids[None, :] * params.bpb[:, None]
        ).ravel()
        sz_all = np.repeat(params.bpb, n)
        total = n * params.num_layers
        keepalive.append((src_all, dst_all, sz_all))

        num_attrs = 0 if mod_ops.current_platform.is_rocm() else 1
        err = mod_ops._batch_memcpy_fn(
            dst_all.ctypes.data,
            src_all.ctypes.data,
            sz_all.ctypes.data,
            total,
            ctypes.addressof(params.attrs),
            ctypes.byref(params.attrs_idx),
            num_attrs,
            ctypes.byref(params.fail_idx),
            params.stream_handle,
        )
        if err != 0:
            raise RuntimeError(
                "batch memcpy failed: err=%s failIdx=%s"
                % (err, params.fail_idx.value)
            )

    return copy_blocks


def _patch_when_imported() -> None:
    import time

    while True:
        mod = sys.modules.get(_TARGET)
        mod_ops = sys.modules.get(_OPS)
        if (
            mod is not None
            and mod_ops is not None
            and not getattr(mod, "_kv_chunk_patched", False)
        ):
            mod.copy_blocks = _make_instrumented(mod_ops)
            mod._kv_chunk_patched = True
            return
        time.sleep(0.2)


def install() -> None:
    threading.Thread(
        target=_patch_when_imported,
        name="kv-offload-instrument-patch",
        daemon=True,
    ).start()
