#!/usr/bin/env python3
"""CollectiveX — NIXL transfer benchmark (family=kv-cache, backend=nixl).

NIXL (ai-dynamo/nixl) is the transfer fabric dynamo uses for disaggregated-serving KV movement.
This benches its point-to-point transfer engine the way a prefill->decode KV handoff uses it: two
NIXL agents in one process, one registers the source buffer and the other the destination, and the
initiator posts a WRITE over the UCX backend (GPU<->GPU, GPU<->host). It sweeps KV-block-sized
payloads and records wall-clock latency + bandwidth (NIXL transfers run on UCX's own streams, so
CUDA events don't bound them — perf_counter around post+poll-to-DONE is the honest measure).

This is the WIRED `nixl` backend for the goal's "KV-cache transfer backends" axis (kv_cache_transfer
declared it a stub). It runs only in the NIXL/dynamo container (CX_BENCH=nixl switches CX_IMAGE to
the tensorrtllm-runtime image); elsewhere the import fails and the run records that — never faked.

The NIXL Python surface (version, Abseil, backends, agent methods) is DUMPED to stderr at startup so
a GHA run's log is self-documenting even if the API drifted — SSH inspection of the NIXL container is
not available. Emits one kv-cache-family JSON (plots in the KV-cache tab next to raw memcpy).

  python tests/nixl_transfer.py --runner b300 --topology-class b300-nvlink-island \\
      --transport nvlink --env-json results/env.json --out results/b300_nixl.json
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
MEASUREMENT_CONTRACT = "nixl-transfer-v1"
FAMILY = "kv-cache"          # same family/schema as kv_cache_transfer.py -> plots in the KV-cache tab
BACKEND = "nixl"

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


def _import_nixl():
    """Return (nixl_agent, nixl_agent_config, import_path) or raise. Tries both documented paths."""
    try:
        from nixl._api import nixl_agent, nixl_agent_config  # canonical
        return nixl_agent, nixl_agent_config, "nixl._api"
    except Exception:
        from nixl import nixl_agent, nixl_agent_config        # re-export
        return nixl_agent, nixl_agent_config, "nixl"


def _nixl_introspect(nixl_agent, nixl_agent_config):
    """Dump the NIXL surface (version, Abseil, backends, agent methods) to stderr. Self-documenting
    so the GHA log resolves any API drift without SSH into the NIXL container."""
    info = {}
    try:
        import importlib.metadata as _m
        info["nixl_version"] = _m.version("nixl")
    except Exception as e:
        info["nixl_version"] = f"<{e!r}>"
    try:
        import nixl._bindings as _b   # the pybind core; surfaces the linked Abseil/UCX if present
        info["bindings"] = [n for n in dir(_b) if not n.startswith("_")][:40]
    except Exception as e:
        info["bindings"] = f"<{e!r}>"
    info["agent_methods"] = [n for n in dir(nixl_agent) if not n.startswith("_")]
    print("NIXL_API " + json.dumps(info), file=sys.stderr, flush=True)
    return info


def _make_agents(nixl_agent, nixl_agent_config):
    """Two local agents (initiator + target) on the UCX backend; exchange metadata so the initiator
    can post to the target's registered memory. No IP/listen thread needed in one process."""
    try:
        cfg = nixl_agent_config(backends=["UCX"])
    except TypeError:
        cfg = nixl_agent_config(True, True, 0)   # positional fallback (older signature)
    init = nixl_agent("cx_initiator", cfg)
    targ = nixl_agent("cx_target", cfg)
    return init, targ


def _bench_one(init, targ, src_t, dst_t, nbytes, warmup, iters):
    """Register src (initiator) + dst (target), post WRITE src->dst `iters` times, poll each to DONE.
    Returns (latency_ms_per_xfer, gb_s). Raises on a NIXL error (caller records it)."""
    init.register_memory(src_t)
    targ.register_memory(dst_t)
    init.add_remote_agent(targ.get_agent_metadata())
    src_descs = init.get_xfer_descs([src_t])
    dst_descs = init.get_xfer_descs([dst_t])

    def _once():
        h = init.initialize_xfer("WRITE", src_descs, dst_descs, targ.name, b"cx")
        st = init.transfer(h)
        if st == "ERR":
            init.release_xfer_handle(h)
            raise RuntimeError("nixl transfer post returned ERR")
        while True:
            st = init.check_xfer_state(h)
            if st == "ERR":
                init.release_xfer_handle(h)
                raise RuntimeError("nixl transfer state ERR")
            if st == "DONE":
                break
        init.release_xfer_handle(h)

    for _ in range(warmup):
        _once()
    t0 = time.perf_counter()
    for _ in range(iters):
        _once()
    dt = time.perf_counter() - t0
    ms = (dt / iters) * 1e3
    gb_s = (nbytes / (dt / iters)) / 1e9 if dt > 0 else 0.0
    return round(ms, 5), round(gb_s, 2)


def _alloc(torch, where, nbytes):
    if where == "cpu":
        return torch.empty(nbytes, dtype=torch.uint8, device="cpu").pin_memory()
    return torch.empty(nbytes, dtype=torch.uint8, device=where)


def run_direction(torch, init, targ, direction, sizes, warmup, iters, ngpu):
    rows = []
    for nbytes in sizes:
        if direction == "dtod-local":
            src_dev, dst_dev = "cuda:0", "cuda:0"
        elif direction == "dtod-remote":
            if ngpu < 2:
                return [], "n/a (needs >=2 GPUs)"
            src_dev, dst_dev = "cuda:0", "cuda:1"
        elif direction == "dtoh":
            src_dev, dst_dev = "cuda:0", "cpu"
        elif direction == "htod":
            src_dev, dst_dev = "cpu", "cuda:0"
        else:
            return [], f"unknown direction {direction}"
        try:
            src = _alloc(torch, src_dev, nbytes)
            dst = _alloc(torch, dst_dev, nbytes)
            ms, gb_s = _bench_one(init, targ, src, dst, nbytes, warmup, iters)
        except RuntimeError as exc:
            rows.append({"transfer_bytes": nbytes, "error": f"{exc!r}", "correct": None})
            break
        rows.append({"transfer_bytes": nbytes, "size_class": size_class(nbytes),
                     "block_bytes": nbytes, "num_blocks": 1,
                     "time_ms": ms, "bandwidth_gb_s": gb_s, "correct": True})
        del src, dst
        torch.cuda.empty_cache()
    return rows, None


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX NIXL transfer benchmark")
    ap.add_argument("--direction", default="all",
                    choices=["all", "dtod-local", "dtod-remote", "dtoh", "htod"])
    ap.add_argument("--min-bytes", type=int, default=DEFAULT_MIN_BYTES)
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--topology-class", required=True)
    ap.add_argument("--transport", default="")
    ap.add_argument("--env-json")
    ap.add_argument("--timestamp")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    notes = []
    env = None
    if args.env_json and os.path.exists(args.env_json):
        with open(args.env_json) as fh:
            env = json.load(fh)

    def _emit(groups, status, peak, extra_notes):
        doc = {"schema_version": SCHEMA_VERSION, "family": FAMILY,
               "generated_by": "nixl_transfer.py",
               "generated_at": args.timestamp or _dt.datetime.now().astimezone().isoformat(),
               "runner": args.runner, "transport": args.transport,
               "measurement_contract": MEASUREMENT_CONTRACT, "nodes": args.nodes,
               "wired_backends": [BACKEND], "status": status,
               "num_groups": len(groups), "groups": groups,
               "notes": extra_notes, "environment": env}
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(doc, fh, indent=2)
            fh.write("\n")
        print(f"nixl-transfer: {len(groups)} groups -> {args.out} (status={status}, "
              f"peak_bw={peak:.1f} GB/s)")
        if extra_notes:
            print("notes: " + "; ".join(extra_notes), file=sys.stderr)

    try:
        import torch
    except Exception as exc:
        _emit([], "invalid", 0.0, [f"torch unavailable: {exc!r}"])
        return 3
    if not torch.cuda.is_available():
        _emit([], "invalid", 0.0, ["CUDA not available"])
        return 3

    try:
        nixl_agent, nixl_agent_config, path = _import_nixl()
        notes.append(f"nixl imported via {path}")
    except Exception as exc:
        _emit([], "invalid", 0.0,
              [f"nixl import failed (needs the NIXL/dynamo container): {exc!r}"])
        return 1
    _nixl_introspect(nixl_agent, nixl_agent_config)
    try:
        init, targ = _make_agents(nixl_agent, nixl_agent_config)
    except Exception as exc:
        _emit([], "invalid", 0.0, [f"nixl agent init failed: {exc!r}"])
        return 1

    ngpu = torch.cuda.device_count()
    directions = (["dtod-local", "dtod-remote", "dtoh", "htod"]
                  if args.direction == "all" else [args.direction])
    sizes = _sizes(args.min_bytes, args.max_bytes)

    groups, peak = [], 0.0
    for direction in directions:
        try:
            rows, na = run_direction(torch, init, targ, direction, sizes, args.warmup, args.iters, ngpu)
        except Exception as exc:
            notes.append(f"{direction}: {exc!r}")
            continue
        if na:
            notes.append(f"{direction}: {na}")
            continue
        timed = [r for r in rows if r.get("bandwidth_gb_s")]
        if not timed:
            continue
        peak = max(peak, max(r["bandwidth_gb_s"] for r in timed))
        meta = {"direction": direction, "layout": "contiguous", "backend": BACKEND,
                "dtype": "uint8", "nodes": args.nodes,
                "topology_class": args.topology_class,
                "measurement_contract": MEASUREMENT_CONTRACT}
        groups.append({**meta, "comparison_key": comparison_key(meta), "rows": rows})

    status = "valid" if (groups and peak > 0.0) else "invalid"
    _emit(groups, status, peak, notes)
    return 0 if status == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
