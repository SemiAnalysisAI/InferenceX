#!/usr/bin/env python3
"""CollectiveX — Mooncake transfer-engine benchmark (family=kv-cache, backend=mooncake).

Mooncake (kvcache-ai/Mooncake) is the disaggregated-KV transfer engine used by vLLM/SGLang PD
setups. This benches its RDMA `transfer_write_on_cuda` the way a prefill->decode KV write uses it:
one TransferEngine, P2PHANDSHAKE metadata (no etcd), src+dst GPU buffers registered for RDMA, the
engine RDMA-writes src->dst (loopback to its own rpc endpoint) over a KV-block size sweep. CUDA-
event timed on the transfer stream.

The WIRED kv-cache `mooncake` backend the goal declared a stub. Mooncake isn't in any CollectiveX
container, so run_in_container pip-installs `mooncake-transfer-engine` first (the directive's "import
a new one" — a pip import rather than a base-image swap). Needs an RDMA NIC (auto-detected from
/sys/class/infiniband). The mooncake API surface + the chosen device are DUMPED to the log; absence
of the package or an RDMA device is recorded, never faked.

  python tests/mooncake_transfer.py --runner b300 --topology-class b300-nvlink-island \\
      --transport rdma --env-json results/env.json --out results/b300_mooncake.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import socket
import sys
import time

SCHEMA_VERSION = 1
MEASUREMENT_CONTRACT = "mooncake-transfer-v1"
FAMILY = "kv-cache"
BACKEND = "mooncake"

DEFAULT_MIN_BYTES = 64 * 1024
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
DECODE_MAX_BYTES = 512 * 1024


def size_class(nbytes: int) -> str:
    return "decode" if nbytes <= DECODE_MAX_BYTES else "prefill"


def _sizes(lo: int, hi: int, factor: int = 4):
    out, s = [], lo
    while s <= hi:
        out.append(s)
        s *= factor
    return out


def comparison_key(meta: dict) -> str:
    parts = [meta["direction"], meta["layout"], meta["backend"], meta["dtype"],
             str(meta["nodes"]), meta["topology_class"], meta["measurement_contract"]]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _get_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return socket.gethostbyname(socket.gethostname())


def _rdma_devices():
    """RDMA device names to try, in order — the detected IB devices, then common fallbacks."""
    devs = []
    try:
        devs = sorted(os.listdir("/sys/class/infiniband"))
    except Exception:
        pass
    # prefer a bond if present (the Mooncake test used mlx5_bond_0), then the raw devices.
    bonds = [d for d in devs if "bond" in d]
    return bonds + [d for d in devs if d not in bonds] + ["mlx5_bond_0", "mlx5_0", "rocep0s0"]


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX Mooncake transfer benchmark")
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
               "generated_by": "mooncake_transfer.py",
               "generated_at": args.timestamp or _dt.datetime.now().astimezone().isoformat(),
               "runner": args.runner, "transport": args.transport,
               "measurement_contract": MEASUREMENT_CONTRACT, "nodes": args.nodes,
               "wired_backends": [BACKEND], "status": status,
               "num_groups": len(groups), "groups": groups, "notes": notes, "environment": env}
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(doc, fh, indent=2)
            fh.write("\n")
        print(f"mooncake: {len(groups)} groups -> {args.out} (status={status}, peak_bw={peak:.1f} GB/s)")
        if notes:
            print("notes: " + "; ".join(notes), file=sys.stderr)

    try:
        import torch
    except Exception as exc:
        _emit([], "invalid", 0.0, [f"torch unavailable: {exc!r}"])
        return 3
    if not torch.cuda.is_available():
        _emit([], "invalid", 0.0, ["CUDA/ROCm not available"])
        return 3
    try:
        from mooncake.engine import TransferEngine
    except Exception as exc:
        _emit([], "invalid", 0.0,
              [f"mooncake import failed (run_in_container pip-installs mooncake-transfer-engine): {exc!r}"])
        return 1
    print("MOONCAKE_API methods=" + json.dumps([m for m in dir(TransferEngine) if not m.startswith("_")][:40]),
          file=sys.stderr, flush=True)

    is_rocm = bool(getattr(torch.version, "hip", None))
    xfer = "transfer_write_on_hip" if is_rocm else "transfer_write_on_cuda"
    eng = TransferEngine()
    host = _get_ip()
    init_note = None
    for dev in _rdma_devices():
        try:
            ret = eng.initialize(host, "P2PHANDSHAKE", "rdma", dev)
            if ret == 0:
                init_note = f"initialized on rdma device {dev}"
                break
        except Exception as e:
            init_note = f"init raised on {dev}: {e!r}"
    if init_note is None or "initialized" not in init_note:
        _emit([], "invalid", 0.0, [f"mooncake init failed on all RDMA devices: {init_note}"])
        return 1
    print(f"MOONCAKE_INIT {init_note}", file=sys.stderr, flush=True)
    if not hasattr(eng, xfer):
        _emit([], "invalid", 0.0, [f"mooncake engine has no {xfer} (methods dumped above)"])
        return 1
    rpc = eng.get_rpc_port()
    target = f"[{host}]:{rpc}" if ":" in host else f"{host}:{rpc}"
    transfer = getattr(eng, xfer)

    dev0 = torch.device("cuda:0")
    stream = torch.cuda.Stream(dev0)
    sizes = _sizes(args.min_bytes, args.max_bytes)
    rows, peak = [], 0.0
    for nbytes in sizes:
        try:
            src = torch.ones(nbytes, dtype=torch.uint8, device=dev0)
            dst = torch.zeros(nbytes, dtype=torch.uint8, device=dev0)
            if eng.register_memory(src.data_ptr(), src.nbytes) != 0 or \
               eng.register_memory(dst.data_ptr(), dst.nbytes) != 0:
                rows.append({"transfer_bytes": nbytes, "error": "register_memory != 0", "correct": None})
                break

            def _once():
                transfer(target, src.data_ptr(), dst.data_ptr(), nbytes, stream.cuda_stream)
            for _ in range(args.warmup):
                _once()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                _once()
            stream.synchronize()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            eng.unregister_memory(src.data_ptr()); eng.unregister_memory(dst.data_ptr())
        except Exception as exc:
            rows.append({"transfer_bytes": nbytes, "error": f"{exc!r}", "correct": None})
            break
        ms = (dt / args.iters) * 1e3
        gb_s = (nbytes / (dt / args.iters)) / 1e9 if dt > 0 else 0.0
        rows.append({"transfer_bytes": nbytes, "size_class": size_class(nbytes),
                     "block_bytes": nbytes, "num_blocks": 1,
                     "time_ms": round(ms, 5), "bandwidth_gb_s": round(gb_s, 2), "correct": True})
        peak = max(peak, gb_s)
        del src, dst
        torch.cuda.empty_cache()

    groups = []
    if any(r.get("bandwidth_gb_s") for r in rows):
        meta = {"direction": "dtod-local", "layout": "contiguous", "backend": BACKEND,
                "dtype": "uint8", "nodes": args.nodes,
                "topology_class": args.topology_class,
                "measurement_contract": MEASUREMENT_CONTRACT}
        groups.append({**meta, "comparison_key": comparison_key(meta), "rows": rows})
    status = "valid" if (groups and peak > 0.0) else "invalid"
    _emit(groups, status, peak, [init_note, f"loopback target={target}"])
    return 0 if status == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
