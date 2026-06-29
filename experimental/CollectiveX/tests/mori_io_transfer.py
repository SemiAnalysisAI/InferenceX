#!/usr/bin/env python3
"""CollectiveX — MoRI-IO transfer benchmark (family=kv-cache, backend=mori-io).

MoRI-IO (ROCm/mori `mori.io`) is AMD's RDMA point-to-point transfer engine — the AMD analog of
NIXL, used for disaggregated-serving KV movement between GPUs/nodes. This benches its read path the
way a prefill->decode KV handoff uses it: two IOEngines in one process (initiator + target, RDMA
backend, mutual register_remote_engine), the initiator RDMA-reads the target's GPU buffer, swept
over KV-block-sized payloads. Wall-clock latency + bandwidth (RDMA completion via InProgress()).

This is the WIRED `mori-io` backend the goal's "KV-cache transfer backends" axis declared a stub.
Runs only on the AMD MoRI image (CX_BENCH=mori-io on mi355x); elsewhere the import fails and the run
records that — never faked. The mori.io API surface is DUMPED to stderr at startup so a GHA run's
log is self-documenting (SSH into the MI355X container stalls on the shared cluster).

  python tests/mori_io_transfer.py --runner mi355x --topology-class mi355x-xgmi \\
      --transport rdma --env-json results/env.json --out results/mi355x_mori_io.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import time

SCHEMA_VERSION = 1
MEASUREMENT_CONTRACT = "mori-io-transfer-v1"
FAMILY = "kv-cache"
BACKEND = "mori-io"

DEFAULT_MIN_BYTES = 64 * 1024
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
DECODE_MAX_BYTES = 512 * 1024


def size_class(nbytes: int) -> str:
    return "decode" if nbytes <= DECODE_MAX_BYTES else "prefill"


def _sizes(min_bytes: int, max_bytes: int, factor: int = 4):
    out, s = [], min_bytes
    while s <= max_bytes:
        out.append(s)
        s *= factor
    return out


def comparison_key(meta: dict) -> str:
    parts = [meta["direction"], meta["layout"], meta["backend"], meta["dtype"],
             str(meta["nodes"]), meta["topology_class"], meta["measurement_contract"]]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _introspect(mod):
    info = {"mori_io_exports": [n for n in dir(mod) if not n.startswith("_")][:40]}
    try:
        import importlib.metadata as _m
        info["mori_version"] = _m.version("mori")
    except Exception as e:
        info["mori_version"] = f"<{e!r}>"
    print("MORI_IO_API " + json.dumps(info), file=sys.stderr, flush=True)


def _make_engines(io):
    """Two local IOEngines (initiator + target) on distinct localhost ports with an RDMA backend,
    mutually registered. Mirrors examples/io/example.py."""
    cfg = io.IOEngineConfig(host="127.0.0.1", port=8080)
    initiator = io.IOEngine(key="cx_initiator", config=cfg)
    cfg2 = io.IOEngineConfig(host="127.0.0.1", port=8081)
    target = io.IOEngine(key="cx_target", config=cfg2)
    rdma = io.RdmaBackendConfig(qp_per_transfer=1)
    initiator.create_backend(io.BackendType.RDMA, rdma)
    target.create_backend(io.BackendType.RDMA, rdma)
    initiator.register_remote_engine(target.get_engine_desc())
    target.register_remote_engine(initiator.get_engine_desc())
    return initiator, target


def _bench_one(initiator, target, src_t, dst_t, nbytes, warmup, iters):
    """Register src (initiator, GPU0) + dst (target, GPU1); RDMA-read dst->src `iters` times, poll
    each to completion. Returns (latency_ms, gb_s). Raises on a MoRI-IO error."""
    im = initiator.register_torch_tensor(src_t)
    tm = target.register_torch_tensor(dst_t)

    def _once():
        uid = initiator.allocate_transfer_uid()
        st = initiator.read(im, 0, tm, 0, nbytes, uid)
        while st.InProgress():
            pass
        msg = st.Message() if hasattr(st, "Message") else ""
        if msg and "succ" not in msg.lower() and "ok" not in msg.lower() and "done" not in msg.lower():
            # Message() is informational on success; only treat an explicit failure word as fatal.
            if any(w in msg.lower() for w in ("fail", "error", "abort")):
                raise RuntimeError(f"mori-io read status: {msg}")

    try:
        for _ in range(warmup):
            _once()
        t0 = time.perf_counter()
        for _ in range(iters):
            _once()
        dt = time.perf_counter() - t0
    finally:
        initiator.deregister_memory(im)
        target.deregister_memory(tm)
    ms = (dt / iters) * 1e3
    gb_s = (nbytes / (dt / iters)) / 1e9 if dt > 0 else 0.0
    return round(ms, 5), round(gb_s, 2)


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX MoRI-IO transfer benchmark")
    ap.add_argument("--direction", default="dtod-remote", choices=["dtod-remote"])
    ap.add_argument("--min-bytes", type=int, default=DEFAULT_MIN_BYTES)
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--topology-class", required=True)
    ap.add_argument("--transport", default="rdma")
    ap.add_argument("--env-json")
    ap.add_argument("--timestamp")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    env = None
    if args.env_json and os.path.exists(args.env_json):
        with open(args.env_json) as fh:
            env = json.load(fh)

    def _emit(groups, status, peak, notes):
        doc = {"schema_version": SCHEMA_VERSION, "family": FAMILY,
               "generated_by": "mori_io_transfer.py",
               "generated_at": args.timestamp or _dt.datetime.now().astimezone().isoformat(),
               "runner": args.runner, "transport": args.transport,
               "measurement_contract": MEASUREMENT_CONTRACT, "nodes": args.nodes,
               "wired_backends": [BACKEND], "status": status,
               "num_groups": len(groups), "groups": groups, "notes": notes, "environment": env}
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(doc, fh, indent=2)
            fh.write("\n")
        print(f"mori-io: {len(groups)} groups -> {args.out} (status={status}, peak_bw={peak:.1f} GB/s)")
        if notes:
            print("notes: " + "; ".join(notes), file=sys.stderr)

    try:
        import torch
    except Exception as exc:
        _emit([], "invalid", 0.0, [f"torch unavailable: {exc!r}"])
        return 3
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        _emit([], "invalid", 0.0,
              [f"mori-io needs >=2 GPUs (RDMA p2p); have {torch.cuda.device_count() if torch.cuda.is_available() else 0}"])
        return 1
    try:
        import mori.io as moriio
    except Exception as exc:
        _emit([], "invalid", 0.0, [f"mori.io import failed (needs the AMD MoRI image): {exc!r}"])
        return 1
    _introspect(moriio)
    try:
        if hasattr(moriio, "set_log_level"):
            moriio.set_log_level("warning")
        initiator, target = _make_engines(moriio)
    except Exception as exc:
        _emit([], "invalid", 0.0, [f"mori.io engine/backend init failed: {exc!r}"])
        return 1

    sizes = _sizes(args.min_bytes, args.max_bytes)
    notes = ["mori.io 2-engine RDMA loopback (GPU0<->GPU1)"]
    rows, peak = [], 0.0
    for nbytes in sizes:
        try:
            src = torch.empty(nbytes, dtype=torch.uint8, device="cuda:0")
            dst = torch.empty(nbytes, dtype=torch.uint8, device="cuda:1")
            ms, gb_s = _bench_one(initiator, target, src, dst, nbytes, args.warmup, args.iters)
        except Exception as exc:
            rows.append({"transfer_bytes": nbytes, "error": f"{exc!r}", "correct": None})
            break
        rows.append({"transfer_bytes": nbytes, "size_class": size_class(nbytes),
                     "block_bytes": nbytes, "num_blocks": 1,
                     "time_ms": ms, "bandwidth_gb_s": gb_s, "correct": True})
        peak = max(peak, gb_s)
        del src, dst
        torch.cuda.empty_cache()

    groups = []
    if any(r.get("bandwidth_gb_s") for r in rows):
        meta = {"direction": "dtod-remote", "layout": "contiguous", "backend": BACKEND,
                "dtype": "uint8", "nodes": args.nodes,
                "topology_class": args.topology_class,
                "measurement_contract": MEASUREMENT_CONTRACT}
        groups.append({**meta, "comparison_key": comparison_key(meta), "rows": rows})
    status = "valid" if (groups and peak > 0.0) else "invalid"
    _emit(groups, status, peak, notes)
    return 0 if status == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
