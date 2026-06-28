#!/usr/bin/env python3
"""CollectiveX — Copy-engine / SDMA collectives (goal P2).

Compares the NVIDIA COPY-ENGINE (DMA) path against an SM-based copy:

  * copy-engine path  — cudaMemcpyAsync (torch .copy_/Tensor copy that lowers to
                        cudaMemcpyDeviceToDevice) issued on a DEDICATED copy
                        stream. Hardware routes device-to-device memcpy through a
                        copy engine (DMA), not the SMs.
  * SM path           — an elementwise kernel (torch mul-add) that necessarily
                        occupies SMs to move the same bytes.

For each it reports latency + bandwidth across a size sweep (DtoD, and HtoD as a
second op). It then VALIDATES that the copy-engine path uses ~0 SMs:

  Primary  : if pynvml is importable, sample SM utilization (nvmlDeviceGetUtilization
             / process-SM) during a sustained copy-engine loop vs a sustained SM-copy
             loop. copy-engine should read near-zero, SM-copy should read high.
  Fallback : a concurrent-kernel NON-INTERFERENCE probe. Run a long SM-bound
             "victim" kernel alone (t_victim). Then run it concurrently with a
             copy-engine copy on a separate stream (t_with_ce) and with an
             SM-copy on a separate stream (t_with_sm). If the copy engine truly
             uses no SMs, t_with_ce ~ t_victim (the copy is hidden), whereas
             t_with_sm > t_victim (the SM-copy steals SM cycles from the victim).
             The ratio is reported as evidence; the proxy is documented in the doc.

family="copy-engine". NVIDIA only (AMD SDMA is out of scope) — refuses on ROCm.

Stdlib + torch; --help / --parse-only work without torch (import-safe writer+CLI).

Run (inside the container, 1 GPU is enough):
    python tests/copy_engine_bench.py \\
        --runner h200 --topology-class h200-nvlink-island --transport nvlink \\
        --env-json results/env.json --out results/h200_copy_engine.json

Verify offline (no GPU/torch needed):
    python tests/copy_engine_bench.py --parse-only --runner h200 \\
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
FAMILY = "copy-engine"
MEASUREMENT_CONTRACT = "copy-engine-vs-sm-v1"
GENERATED_BY = "copy_engine_bench.py"

# (op, engine) sub-ops. engine = copy-engine (DMA) vs sm (kernel).
SUBOPS = [
    ("dtod", "copy-engine"),
    ("dtod", "sm"),
    ("htod", "copy-engine"),
    ("htod", "sm"),
]

DEFAULT_MIN_BYTES = 64 * 1024
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_FACTOR = 4


# --------------------------------------------------------------------------- #
# import-safe helpers (no torch)                                              #
# --------------------------------------------------------------------------- #
def size_ladder(min_bytes: int, max_bytes: int, factor: int) -> list[int]:
    sizes, s = [], int(min_bytes)
    while s <= int(max_bytes):
        sizes.append(s)
        s *= factor
    return sizes


def comparison_key(meta: dict) -> str:
    parts = [
        meta["op"],
        meta["engine"],
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


# --------------------------------------------------------------------------- #
# GPU path (torch only here)                                                  #
# --------------------------------------------------------------------------- #
def _copy_engine_copy(torch, dst, src, stream):
    """DtoD/HtoD memcpy that lowers to cudaMemcpyAsync on `stream` (copy engine)."""
    with torch.cuda.stream(stream):
        dst.copy_(src, non_blocking=True)


def _sm_copy(torch, dst, src, stream):
    """Bytes moved by an elementwise KERNEL (occupies SMs): dst = src * 1 + 0.

    mul/add lowers to a CUDA elementwise kernel scheduled on the SMs — the
    deliberate SM-based contrast to the copy engine. Same byte volume as .copy_."""
    with torch.cuda.stream(stream):
        torch.add(src, 0, out=dst) if dst.dtype == src.dtype else dst.copy_(src)


def _time_loop(torch, fn, iters: int) -> float:
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms/iter


def _bench_one(torch, op: str, engine: str, nbytes: int, dtype,
               warmup: int, iters: int, copy_stream) -> dict:
    elem = torch.tensor([], dtype=dtype).element_size()
    n = max(1, nbytes // elem)

    dev_dst = torch.empty(n, dtype=dtype, device="cuda")
    if op == "dtod":
        src = torch.randn(n, dtype=dtype, device="cuda") if dtype.is_floating_point \
            else torch.zeros(n, dtype=dtype, device="cuda")
    else:  # htod
        src = torch.empty(n, dtype=dtype, device="cpu", pin_memory=True)

    if engine == "copy-engine":
        fn = lambda: _copy_engine_copy(torch, dev_dst, src, copy_stream)
    else:
        # SM kernel copy. For HtoD an add kernel can't read host memory directly,
        # so stage to device first then SM-copy device->device (still SM-bound).
        if op == "htod":
            staged = torch.empty(n, dtype=dtype, device="cuda")
            staged.copy_(src)
            torch.cuda.synchronize()
            src = staged
        fn = lambda: _sm_copy(torch, dev_dst, src, copy_stream)

    for _ in range(warmup):
        fn()
    copy_stream.synchronize()
    torch.cuda.synchronize()

    avg_ms = _time_loop(torch, fn, iters)
    actual_bytes = n * elem
    gbps = (actual_bytes / (avg_ms / 1e3)) / 1e9 if avg_ms > 0 else 0.0
    return {
        "op": op,
        "engine": engine,
        "size_bytes": actual_bytes,
        "requested_bytes": nbytes,
        "latency_us": round(avg_ms * 1e3, 4),
        "bandwidth_gbps": round(gbps, 3),
    }


# ---- SM-utilization validation (primary: nvml; fallback: non-interference) -- #
def _victim_kernel_factory(torch, device):
    """A long SM-bound kernel used as the 'victim' in the non-interference probe.

    Repeated matmuls saturate the SMs for a measurable, stable duration; if a
    concurrent copy steals SM cycles, the victim slows down."""
    m = 2048
    a = torch.randn(m, m, device=device, dtype=torch.float16)
    b = torch.randn(m, m, device=device, dtype=torch.float16)
    inner = 8

    def victim():
        c = a
        for _ in range(inner):
            c = torch.matmul(c, b)
        return c

    return victim, [m, m, m, inner]


def _attention_victim_factory(torch, device):
    """An SM-bound ATTENTION victim (scaled_dot_product_attention = the flash-attention kernel) for
    the copy-vs-attention interference probe (goal "Interference with attention kernels"). Decode-ish
    attention shape [batch, heads, seq, head_dim]; repeated to saturate the SMs for a stable duration."""
    import torch.nn.functional as _F
    b_, h_, s_, d_ = 8, 32, 2048, 128
    q = torch.randn(b_, h_, s_, d_, device=device, dtype=torch.float16)
    k = torch.randn(b_, h_, s_, d_, device=device, dtype=torch.float16)
    v = torch.randn(b_, h_, s_, d_, device=device, dtype=torch.float16)
    inner = 6

    def victim():
        o = q
        for _ in range(inner):
            o = _F.scaled_dot_product_attention(o, k, v)
        return o

    return victim, [b_, h_, s_, d_, inner]


def _probe_victim(torch, victim, copy_engine_copy, sm_copy, dst, src, copy_stream, iters):
    """Time a victim alone vs concurrent with a copy-engine copy vs concurrent with an SM-copy.
    Returns (t_victim_us, t_with_ce_us, t_with_sm_us, ce_slowdown, sm_slowdown, near_zero)."""
    for _ in range(3):
        victim(); copy_engine_copy(); sm_copy()
    torch.cuda.synchronize()
    t_victim = _time_loop(torch, lambda: victim(), iters)
    t_with_ce = _time_loop(torch, lambda: (copy_engine_copy(), victim()), iters)
    t_with_sm = _time_loop(torch, lambda: (sm_copy(), victim()), iters)
    copy_stream.synchronize()
    ce_slow = (t_with_ce / t_victim) if t_victim > 0 else None
    sm_slow = (t_with_sm / t_victim) if t_victim > 0 else None
    near_zero = (ce_slow is not None and sm_slow is not None
                 and ce_slow < 1.15 and (sm_slow - ce_slow) > 0.05)
    return (round(t_victim * 1e3, 4), round(t_with_ce * 1e3, 4), round(t_with_sm * 1e3, 4),
            round(ce_slow, 4) if ce_slow else None, round(sm_slow, 4) if sm_slow else None, bool(near_zero))


def _sm_validation(torch, device, nbytes: int, iters: int) -> dict:
    """Return evidence the copy-engine path uses ~0 SMs.

    Tries pynvml SM utilization sampling first; always also runs the
    concurrent-kernel non-interference probe and records BOTH. The doc documents
    which signal is authoritative."""
    elem = 2  # float16
    n = max(1, nbytes // elem)
    src = torch.randn(n, dtype=torch.float16, device=device)
    dst = torch.empty(n, dtype=torch.float16, device=device)
    copy_stream = torch.cuda.Stream()
    victim, gemm_shape = _victim_kernel_factory(torch, device)

    result: dict = {
        "method": None,
        "nvml": None,
        "non_interference": None,
        "copy_engine_uses_near_zero_sms": None,
        "proxy_doc": (
            "Non-interference proxy: a long SM-bound victim kernel timed alone "
            "(t_victim) vs concurrent with a copy-engine copy on a separate "
            "stream (t_with_ce) vs concurrent with an SM-copy (t_with_sm). "
            "ce_slowdown=t_with_ce/t_victim ~1.0 => the copy engine stole no SM "
            "cycles; sm_slowdown=t_with_sm/t_victim >1.0 => the SM-copy did. "
            "copy_engine_uses_near_zero_sms is asserted when ce_slowdown is "
            "materially smaller than sm_slowdown (and < ce_slowdown_threshold)."
        ),
    }

    # ---- primary: pynvml SM utilization while copying on the copy engine ----
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        idx = torch.cuda.current_device()
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)

        def _sample_during(fn, n_samples=40) -> float:
            # launch a long stream of the op, sample SM util repeatedly, take max
            import time
            for _ in range(3):
                fn()
            samples = []
            # keep the queue full while sampling
            for _ in range(n_samples):
                for _ in range(8):
                    fn()
                u = pynvml.nvmlDeviceGetUtilizationRates(handle)
                samples.append(u.gpu)
                time.sleep(0.001)
            torch.cuda.synchronize()
            return max(samples) if samples else 0.0

        ce_util = _sample_during(
            lambda: _copy_engine_copy(torch, dst, src, copy_stream))
        sm_util = _sample_during(
            lambda: _sm_copy(torch, dst, src, copy_stream))
        result["nvml"] = {
            "source": "pynvml nvmlDeviceGetUtilizationRates (whole-GPU SM util %)",
            "copy_engine_max_sm_util_pct": ce_util,
            "sm_copy_max_sm_util_pct": sm_util,
            "note": "whole-GPU util is a coarse proxy; copy-engine should read low, SM-copy high",
        }
        pynvml.nvmlShutdown()
    except Exception as exc:
        result["nvml"] = {"available": False, "error": repr(exc)}

    # ---- always: concurrent-kernel non-interference probe ----
    try:
        # warmup
        for _ in range(3):
            victim()
            _copy_engine_copy(torch, dst, src, copy_stream)
            _sm_copy(torch, dst, src, copy_stream)
        torch.cuda.synchronize()

        t_victim = _time_loop(torch, lambda: victim(), iters)

        def _victim_with_ce():
            _copy_engine_copy(torch, dst, src, copy_stream)
            victim()

        def _victim_with_sm():
            _sm_copy(torch, dst, src, copy_stream)
            victim()

        t_with_ce = _time_loop(torch, _victim_with_ce, iters)
        t_with_sm = _time_loop(torch, _victim_with_sm, iters)
        copy_stream.synchronize()

        ce_slow = (t_with_ce / t_victim) if t_victim > 0 else None
        sm_slow = (t_with_sm / t_victim) if t_victim > 0 else None
        threshold = 1.15
        near_zero = (
            ce_slow is not None and sm_slow is not None
            and ce_slow < threshold and (sm_slow - ce_slow) > 0.05
        )
        result["non_interference"] = {
            "victim_kernel": "matmul x8 (fp16 2048^3)",
            "gemm_shape": gemm_shape,
            "t_victim_us": round(t_victim * 1e3, 4),
            "t_victim_with_copy_engine_us": round(t_with_ce * 1e3, 4),
            "t_victim_with_sm_copy_us": round(t_with_sm * 1e3, 4),
            "ce_slowdown": round(ce_slow, 4) if ce_slow else None,
            "sm_slowdown": round(sm_slow, 4) if sm_slow else None,
            "ce_slowdown_threshold": threshold,
        }
        result["copy_engine_uses_near_zero_sms"] = bool(near_zero)
        result["method"] = ("nvml+non-interference"
                            if result.get("nvml", {}).get("source") else "non-interference")
    except Exception as exc:
        result["non_interference"] = {"error": repr(exc)}
        result["method"] = result["method"] or "failed"

    # ---- copy-vs-ATTENTION interference (goal "Interference with attention kernels") ----
    # Same probe with a flash-attention (scaled_dot_product_attention) victim instead of GEMM, so
    # the copy engine's non-interference is shown against BOTH expert-GEMM and attention kernels.
    try:
        avictim, ashape = _attention_victim_factory(torch, device)
        tv, tce, tsm, ce_s, sm_s, az = _probe_victim(
            torch, avictim,
            lambda: _copy_engine_copy(torch, dst, src, copy_stream),
            lambda: _sm_copy(torch, dst, src, copy_stream),
            dst, src, copy_stream, iters)
        result["non_interference_attention"] = {
            "victim_kernel": "scaled_dot_product_attention x6 (fp16 [8,32,2048,128])",
            "attn_shape": ashape, "t_victim_us": tv,
            "t_victim_with_copy_engine_us": tce, "t_victim_with_sm_copy_us": tsm,
            "ce_slowdown": ce_s, "sm_slowdown": sm_s, "ce_slowdown_threshold": 1.15}
        result["copy_engine_uses_near_zero_sms_attention"] = az
    except Exception as exc:
        result["non_interference_attention"] = {"error": repr(exc)}

    return result


def run_gpu(args) -> tuple[list[dict], dict, str | None]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        return [], {}, f"torch unavailable: {exc!r}"
    if not torch.cuda.is_available():
        return [], {}, "torch.cuda.is_available() is False (no GPU in this container)"
    # Accelerator-aware: on NVIDIA the off-SM DMA path is the copy engine; on AMD/ROCm the same
    # async stream-copy lowers to the SDMA (System DMA) engines (the "AMD SDMA path"). The bench
    # body is identical (torch.cuda maps to HIP); we label the DMA engine honestly per accelerator
    # and let the non-interference probe characterize SDMA-vs-CU interference (pynvml is absent on
    # ROCm, so _sm_validation falls back to the pure-torch non-interference path automatically).
    is_rocm = bool(getattr(torch.version, "hip", None))
    accel = "rocm" if is_rocm else "cuda"
    copy_engine_kind = "sdma" if is_rocm else "copy-engine"

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    sizes = size_ladder(args.min_bytes, args.max_bytes, args.factor)
    copy_stream = torch.cuda.Stream()

    rows: list[dict] = []
    for op, engine in SUBOPS:
        for nbytes in sizes:
            try:
                rows.append(_bench_one(torch, op, engine, nbytes, dtype,
                                       args.warmup, args.iters, copy_stream))
            except RuntimeError as exc:
                rows.append({"op": op, "engine": engine, "size_bytes": nbytes,
                             "requested_bytes": nbytes, "latency_us": None,
                             "bandwidth_gbps": None, "error": repr(exc)})

    diagnostics = {
        "sm_validation": _sm_validation(torch, torch.device("cuda"),
                                        args.validation_bytes, max(10, args.iters)),
        "device_name": torch.cuda.get_device_name(0),
        "multiprocessor_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "accelerator": accel,
        "copy_engine_kind": copy_engine_kind,   # "sdma" on AMD/ROCm, "copy-engine" on NVIDIA
        "hip_version": getattr(torch.version, "hip", None),
    }
    return rows, diagnostics, None


# --------------------------------------------------------------------------- #
# document assembly + CLI                                                      #
# --------------------------------------------------------------------------- #
def build_doc(args, rows: list[dict], diagnostics: dict, error: str | None) -> dict:
    measured = [r for r in rows if r.get("bandwidth_gbps")]
    peak_bw = max((r["bandwidth_gbps"] for r in measured), default=0.0)
    # gate: must have transferred on BOTH the copy-engine and SM paths with bw>0
    ce_ok = any(r["engine"] == "copy-engine" and r.get("bandwidth_gbps") for r in rows)
    sm_ok = any(r["engine"] == "sm" and r.get("bandwidth_gbps") for r in rows)
    transferred = bool(measured) and peak_bw > 0.0 and ce_ok and sm_ok

    meta = {
        "op": "memcpy", "engine": "mixed", "dtype": args.dtype,
        "transport": args.transport, "topology_class": args.topology_class,
        "comparison_class": args.comparison_class,
        "measurement_contract": MEASUREMENT_CONTRACT,
    }
    curve_keys = {}
    for op, engine in SUBOPS:
        curve_keys[f"{op}/{engine}"] = comparison_key(dict(meta, op=op, engine=engine))
    for r in rows:
        r["comparison_key"] = curve_keys.get(f"{r['op']}/{r['engine']}")

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
        "sub_ops": [f"{o}/{e}" for o, e in SUBOPS],
        "comparison_key": comparison_key(meta),
        "curve_keys": curve_keys,
        "status": "valid" if transferred else "invalid",
        "error": error,
        # "copy-engine" on NVIDIA, "sdma" on AMD/ROCm (same off-SM DMA-engine role) — labeled so the
        # AMD SDMA result is not conflated with the NVIDIA copy-engine result in the plot.
        "accelerator": diagnostics.get("accelerator"),
        "copy_engine_kind": diagnostics.get("copy_engine_kind"),
        "peak_bandwidth_gbps": round(peak_bw, 3),
        "copy_engine_uses_near_zero_sms": diagnostics.get("sm_validation", {}).get(
            "copy_engine_uses_near_zero_sms"),
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
    ap = argparse.ArgumentParser(description="CollectiveX copy-engine vs SM copy bench (NVIDIA)")
    ap.add_argument("--min-bytes", type=int, default=DEFAULT_MIN_BYTES)
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    ap.add_argument("--factor", type=int, default=DEFAULT_FACTOR)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--validation-bytes", type=int, default=16 * 1024 * 1024,
                    help="copy size used by the SM-utilization validation probe")
    ap.add_argument("--parse-only", action="store_true",
                    help="emit a well-formed (status=invalid) doc with no GPU — schema check")
    ap.add_argument("--runner", required=True)
    ap.add_argument("--topology-class", required=True)
    ap.add_argument("--transport", default="nvlink",
                    help="DtoD transport: nvlink (intra-node) | pcie")
    ap.add_argument("--comparison-class", default="standardized",
                    choices=["standardized", "backend-optimized", "framework-integrated"])
    ap.add_argument("--env-json")
    ap.add_argument("--timestamp")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.parse_only:
        rows, diagnostics, error = [], {}, "parse-only (no GPU run)"
    else:
        rows, diagnostics, error = run_gpu(args)

    doc = build_doc(args, rows, diagnostics, error)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")

    sv = doc["diagnostics"].get("sm_validation", {})
    print(
        f"copy-engine: {doc['num_rows']} rows -> {args.out} "
        f"(status={doc['status']}, peak_bw={doc['peak_bandwidth_gbps']} GB/s, "
        f"ce_near_zero_sms={doc['copy_engine_uses_near_zero_sms']}, "
        f"method={sv.get('method')}, key={doc['comparison_key']})",
        file=sys.stderr,
    )
    return 0 if doc["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
