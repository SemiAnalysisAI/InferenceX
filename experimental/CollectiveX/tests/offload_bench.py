#!/usr/bin/env python3
"""CollectiveX — CPU<->GPU offload suite (goal P2 "CPU-GPU offload suite").

Measures host<->device memcpy bandwidth + latency over a size sweep, for the
four sub-ops {h2d, d2h} x {pinned, pageable}, plus two diagnostics that matter
for real offload (KV spill, weight streaming, activation checkpointing):

  * NUMA locality   — which NUMA node the host buffer landed on, and (best
                      effort, if numactl/affinity is available) a node-pinned
                      vs default comparison. Recorded, never required.
  * overlap-w-compute — a copy stream running concurrently with a dummy GEMM on
                      a separate compute stream; reports achieved overlap %
                      (how much of the copy is hidden behind compute).

Matches run_nccl.py's result CONVENTION (family/runner/op/rows/comparison_key/
status/transport/environment/generated_at) and env_capture.py's provenance
style, so the plot + collector consume it uniformly.

Stdlib + torch. torch is needed ONLY at runtime on the GPU; --help and
--parse-only work without it (the JSON writer + CLI are import-safe).

Run (inside the container, 1 GPU is enough):
    python tests/offload_bench.py \\
        --runner h200 --topology-class h200-nvlink-island --transport pcie \\
        --env-json results/env.json --out results/h200_offload.json

Verify offline (no GPU/torch needed):
    python tests/offload_bench.py --parse-only --runner h200 \\
        --topology-class h200-nvlink-island --out /tmp/parsed.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

SCHEMA_VERSION = 1
FAMILY = "offload"
MEASUREMENT_CONTRACT = "host-device-memcpy-v1"
GENERATED_BY = "offload_bench.py"

# (direction, host_memory) sub-ops. h2d = host->device (CPU->GPU), d2h = the reverse.
SUBOPS = [
    ("h2d", "pinned"),
    ("h2d", "pageable"),
    ("d2h", "pinned"),
    ("d2h", "pageable"),
]

# Default byte sweep: 4 KiB .. 256 MiB by x4. Covers decode-token-sized spills
# up to prefix-cache / weight-shard sized streams.
DEFAULT_MIN_BYTES = 4 * 1024
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_FACTOR = 4


# --------------------------------------------------------------------------- #
# import-safe helpers (no torch)                                              #
# --------------------------------------------------------------------------- #
def _human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n}"


def size_ladder(min_bytes: int, max_bytes: int, factor: int) -> list[int]:
    sizes, s = [], int(min_bytes)
    while s <= int(max_bytes):
        sizes.append(s)
        s *= factor
    return sizes


def comparison_key(meta: dict) -> str:
    """Deterministic curve key. transport + topology_class are part of the key so
    a PCIe H200 result and an NVLink-C2C GB200 result are labelled distinct rather
    than silently overlaid (mirrors run_nccl.py's intent)."""
    parts = [
        meta["op"],
        meta["host_memory"],
        meta["dtype"],
        meta["transport"],
        meta["topology_class"],
        meta["comparison_class"],
        meta["measurement_contract"],
    ]
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


def _load_env(path: str | None) -> dict | None:
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None


def _provenance() -> dict:
    """GitHub / container provenance (mirrors tests/run_ep.py)."""
    import platform as _plat

    arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(_plat.machine(), _plat.machine())
    run = {
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "ref": os.environ.get("GITHUB_REF_NAME") or os.environ.get("GITHUB_REF"),
        "source_sha": os.environ.get("COLLECTIVEX_SOURCE_SHA") or os.environ.get("GITHUB_SHA"),
        "repo": os.environ.get("GITHUB_REPOSITORY"),
        "job": os.environ.get("GITHUB_JOB"),
        "artifact": os.environ.get("COLLECTIVEX_ARTIFACT_NAME"),
    }
    return {
        "image": os.environ.get("COLLECTIVEX_IMAGE", ""),
        "image_digest": os.environ.get("COLLECTIVEX_IMAGE_DIGEST", ""),
        "image_arch": arch,
        "squash_sha256": os.environ.get("COLLECTIVEX_SQUASH_SHA256"),
        "git_run": run if any(run.values()) else None,
    }


def _numa_locality() -> dict:
    """Best-effort NUMA context. Never required; degrades to nulls off-NUMA.

    Records the process's allowed NUMA node(s) and CPU affinity so a result that
    happened to land cross-socket from the GPU is identifiable after the fact.
    """
    info: dict = {
        "available": False,
        "process_node": None,
        "membind": None,
        "cpus_allowed_list": None,
        "node_count": None,
        "source": None,
    }
    # numactl --show is the clean read; fall back to /proc self status bitmasks.
    import shutil
    import subprocess

    if shutil.which("numactl"):
        try:
            out = subprocess.run(
                ["numactl", "--show"], capture_output=True, text=True, timeout=10, check=False
            )
            if out.returncode == 0:
                info["available"] = True
                info["source"] = "numactl --show"
                for line in out.stdout.splitlines():
                    if line.startswith("nodebind:"):
                        info["process_node"] = line.split(":", 1)[1].strip()
                    elif line.startswith("membind:"):
                        info["membind"] = line.split(":", 1)[1].strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    # node count from sysfs (independent of numactl)
    try:
        nodes = [d for d in os.listdir("/sys/devices/system/node") if d.startswith("node")]
        if nodes:
            info["node_count"] = len(nodes)
    except OSError:
        pass
    # CPU affinity of this process (which cores -> which socket -> NUMA hint)
    try:
        if hasattr(os, "sched_getaffinity"):
            info["cpus_allowed_list"] = sorted(os.sched_getaffinity(0))
            if info["source"] is None:
                info["available"] = True
                info["source"] = "os.sched_getaffinity"
    except OSError:
        pass
    return info


# --------------------------------------------------------------------------- #
# GPU path (torch only here)                                                  #
# --------------------------------------------------------------------------- #
def _bench_one(torch, direction: str, host_memory: str, nbytes: int,
               dtype, warmup: int, iters: int) -> dict:
    """Time a single (direction, host_memory, size) point with CUDA events.

    Returns latency (us) and bandwidth (GB/s, decimal). Uses non_blocking=True so
    pinned transfers actually go async on the copy engine; pageable is implicitly
    synchronous (the staging copy serializes), which is the honest contrast.
    """
    elem = torch.tensor([], dtype=dtype).element_size()
    n = max(1, nbytes // elem)
    pin = host_memory == "pinned"

    host = torch.empty(n, dtype=dtype, device="cpu", pin_memory=pin)
    dev = torch.empty(n, dtype=dtype, device="cuda")
    if direction == "h2d":
        src, dst = host, dev
    else:
        src, dst = dev, host

    non_blocking = pin  # pageable cannot be truly async

    for _ in range(warmup):
        dst.copy_(src, non_blocking=non_blocking)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dst.copy_(src, non_blocking=non_blocking)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    avg_ms = elapsed_ms / iters
    actual_bytes = n * elem
    gbps = (actual_bytes / (avg_ms / 1e3)) / 1e9 if avg_ms > 0 else 0.0
    return {
        "size_bytes": actual_bytes,
        "requested_bytes": nbytes,
        "latency_us": round(avg_ms * 1e3, 4),
        "bandwidth_gbps": round(gbps, 3),
    }


def _overlap_with_compute(torch, nbytes: int, dtype, iters: int) -> dict:
    """Run a pinned H2D copy concurrently with a dummy GEMM on a separate stream
    and report achieved overlap %.

    overlap_pct = 1 - overlapped_time / (copy_alone + gemm_alone), clamped to
    [0, 100]. 100% means the copy was fully hidden behind compute; ~0% means the
    copy stream and compute stream serialized (e.g. PCIe contention or no copy
    engine free). Best-effort and labelled — it is a diagnostic, not a curve point.
    """
    elem = torch.tensor([], dtype=dtype).element_size()
    n = max(1, nbytes // elem)
    host = torch.empty(n, dtype=dtype, device="cpu", pin_memory=True)
    dev = torch.empty(n, dtype=dtype, device="cuda")

    # A GEMM big enough to take longer than the copy (so the copy can hide under it).
    m = 2048
    a = torch.randn(m, m, device="cuda", dtype=torch.float16)
    b = torch.randn(m, m, device="cuda", dtype=torch.float16)

    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    def _time(fn) -> float:
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    # warmup both paths
    for _ in range(3):
        dev.copy_(host, non_blocking=True)
        torch.matmul(a, b)
    torch.cuda.synchronize()

    copy_ms = _time(lambda: [dev.copy_(host, non_blocking=True) for _ in range(iters)])
    gemm_ms = _time(lambda: [torch.matmul(a, b) for _ in range(iters)])

    def _overlapped():
        for _ in range(iters):
            with torch.cuda.stream(copy_stream):
                dev.copy_(host, non_blocking=True)
            with torch.cuda.stream(compute_stream):
                torch.matmul(a, b)
        copy_stream.synchronize()
        compute_stream.synchronize()

    both_ms = _time(_overlapped)

    serial = copy_ms + gemm_ms
    # Hidden time = how much shorter "both concurrent" is than running them back to back.
    hidden = max(0.0, serial - both_ms)
    # As a fraction of the SMALLER of the two (the most that can be hidden is min).
    hideable = min(copy_ms, gemm_ms)
    overlap_pct = (hidden / hideable * 100.0) if hideable > 0 else 0.0
    overlap_pct = max(0.0, min(100.0, overlap_pct))
    return {
        "size_bytes": n * elem,
        "copy_alone_us": round(copy_ms * 1e3, 4),
        "gemm_alone_us": round(gemm_ms * 1e3, 4),
        "concurrent_us": round(both_ms * 1e3, 4),
        "serial_sum_us": round(serial * 1e3, 4),
        "overlap_pct": round(overlap_pct, 1),
        "gemm_shape": [m, m, m],
    }


def run_gpu(args) -> tuple[list[dict], dict, str | None]:
    """Returns (rows, diagnostics, error). rows is empty + error set if torch/GPU
    is unavailable — the caller turns that into status=invalid, never a fake row."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime/GPU only
        return [], {}, f"torch unavailable: {exc!r}"
    if not torch.cuda.is_available():
        return [], {}, "torch.cuda.is_available() is False (no GPU in this container)"

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32, "uint8": torch.uint8}[args.dtype]
    sizes = size_ladder(args.min_bytes, args.max_bytes, args.factor)

    rows: list[dict] = []
    for direction, host_memory in SUBOPS:
        for nbytes in sizes:
            try:
                r = _bench_one(torch, direction, host_memory, nbytes, dtype,
                               args.warmup, args.iters)
                r["op"] = direction
                r["host_memory"] = host_memory
                rows.append(r)
            except RuntimeError as exc:  # OOM at the top of the ladder, etc.
                rows.append({
                    "op": direction, "host_memory": host_memory,
                    "size_bytes": nbytes, "requested_bytes": nbytes,
                    "latency_us": None, "bandwidth_gbps": None,
                    "error": repr(exc),
                })

    diagnostics: dict = {"numa": _numa_locality()}
    if not args.no_overlap:
        try:
            diagnostics["overlap_with_compute"] = _overlap_with_compute(
                torch, args.overlap_bytes, dtype, max(5, args.iters))
        except Exception as exc:  # best-effort diagnostic
            diagnostics["overlap_with_compute"] = {"error": repr(exc)}
    return rows, diagnostics, None


# --------------------------------------------------------------------------- #
# document assembly + CLI                                                      #
# --------------------------------------------------------------------------- #
def build_doc(args, rows: list[dict], diagnostics: dict, error: str | None) -> dict:
    # Peak bandwidth across every real measured row gates validity: a run that
    # produced no positive bandwidth did not actually transfer.
    measured = [r for r in rows if r.get("bandwidth_gbps")]
    peak_bw = max((r["bandwidth_gbps"] for r in measured), default=0.0)
    transferred = bool(measured) and peak_bw > 0.0

    meta = {
        "op": "host_device_copy",
        "host_memory": "mixed",
        "dtype": args.dtype,
        "transport": args.transport,
        "topology_class": args.topology_class,
        "comparison_class": args.comparison_class,
        "measurement_contract": MEASUREMENT_CONTRACT,
    }
    # Per-curve keys: one comparison_key per (op, host_memory) so the plotter can
    # overlay pinned-vs-pageable / h2d-vs-d2h as distinct curves.
    curve_keys = {}
    for direction, host_memory in SUBOPS:
        cm = dict(meta, op=direction, host_memory=host_memory)
        curve_keys[f"{direction}/{host_memory}"] = comparison_key(cm)
    for r in rows:
        r["comparison_key"] = curve_keys.get(f"{r['op']}/{r['host_memory']}")

    doc = {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "generated_by": GENERATED_BY,
        "generated_at": args.timestamp or _dt.datetime.now().astimezone().isoformat(),
        "runner": args.runner,
        "transport": args.transport,
        "measurement_contract": MEASUREMENT_CONTRACT,
        "topology_class": args.topology_class,
        "comparison_class": args.comparison_class,
        "dtype": args.dtype,
        "sub_ops": [f"{d}/{h}" for d, h in SUBOPS],
        # top-level comparison_key = the whole-suite key (op=host_device_copy);
        # per-row keys (above) drive curve overlays.
        "comparison_key": comparison_key(meta),
        "curve_keys": curve_keys,
        "status": "valid" if transferred else "invalid",
        "error": error,
        "peak_bandwidth_gbps": round(peak_bw, 3),
        "sweep": {"min_bytes": args.min_bytes, "max_bytes": args.max_bytes,
                  "factor": args.factor, "warmup": args.warmup, "iters": args.iters},
        "num_rows": len(rows),
        "rows": rows,
        "diagnostics": diagnostics,
        "provenance": _provenance(),
        "environment": _load_env(args.env_json),
    }
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX CPU<->GPU offload suite")
    # sweep knobs
    ap.add_argument("--min-bytes", type=int, default=DEFAULT_MIN_BYTES)
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    ap.add_argument("--factor", type=int, default=DEFAULT_FACTOR, help="size step factor")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32", "uint8"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-overlap", action="store_true",
                    help="skip the overlap-with-compute diagnostic")
    ap.add_argument("--overlap-bytes", type=int, default=16 * 1024 * 1024,
                    help="copy size for the overlap-with-compute diagnostic")
    ap.add_argument("--parse-only", action="store_true",
                    help="emit a well-formed (status=invalid) doc with no GPU — schema check")
    # provenance (mirrors run_nccl.py)
    ap.add_argument("--runner", required=True, help="runner label, e.g. h200")
    ap.add_argument("--topology-class", required=True,
                    help="e.g. h200-nvlink-island, gb200-nvl72-c2c")
    ap.add_argument("--transport", default="pcie",
                    help="observed host<->device transport: pcie | nvlink-c2c")
    ap.add_argument("--comparison-class", default="standardized",
                    choices=["standardized", "backend-optimized", "framework-integrated"])
    ap.add_argument("--env-json", help="path to env_capture.py output to embed")
    ap.add_argument("--timestamp", help="ISO timestamp (default now)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.parse_only:
        rows, diagnostics, error = [], {"numa": _numa_locality()}, "parse-only (no GPU run)"
    else:
        rows, diagnostics, error = run_gpu(args)

    doc = build_doc(args, rows, diagnostics, error)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")

    ov = doc["diagnostics"].get("overlap_with_compute", {})
    print(
        f"offload: {doc['num_rows']} rows -> {args.out} "
        f"(status={doc['status']}, peak_bw={doc['peak_bandwidth_gbps']} GB/s, "
        f"overlap={ov.get('overlap_pct')}%, key={doc['comparison_key']})",
        file=sys.stderr,
    )
    return 0 if doc["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
