#!/usr/bin/env python3
"""CollectiveX — framework custom all-reduce benchmark (family=allreduce-fw).

Goal P2 "Low-latency all-reduce suite", framework-integrated tier. The standardized
NCCL all-reduce is already covered by run_nccl.py (nccl-tests); this benchmark times the
CUSTOM all-reduce kernels the serving frameworks ship — the ones that beat NCCL in the
small-to-medium, latency-bound regime (TP all-reduce of activations: a few KiB .. tens of
MiB) by doing a single one-shot or two-shot NVLink reduction instead of a ring.

It runs under torchrun (multi-process, one rank per GPU) and, for EACH importable
framework, times an all-reduce-sum of a bf16/fp32 tensor across the whole world over a
latency-focused size ladder, CUDA-event timed, validating the result against a known
reference. NCCL (torch.distributed.all_reduce) is the always-present baseline.

Implementations measured (each IMPORT-GUARDED — a framework that isn't importable in the
container is recorded as skipped, never faked):
  * nccl                 — torch.distributed.all_reduce (baseline)
  * flashinfer-oneshot   } flashinfer custom all-reduce (trtllm fusion / vLLM-style
  * flashinfer-twoshot   } custom-allreduce), one-shot and two-shot recorded separately
  * sglang               — sgl_kernel / sglang custom all-reduce
  * vllm                 — vllm custom all-reduce (vllm may or may not be in the image)

Each measured impl is one group:
  {impl, dtype, world_size, rows:[{size_bytes, latency_us, algbw_gbps, busbw_gbps, correct}]}
busbw uses the all-reduce factor 2*(n-1)/n (same as nccl-tests) so framework and NCCL bus
bandwidth are directly comparable. status=valid iff nccl + >=1 framework impl produced rows
with bw>0. A top-level frameworks_available dict records which frameworks were importable.

Stdlib + torch; torch (and every framework) is imported lazily so `--help` works on a login
node with no GPU. One provenance-tagged JSON like rl_mesh_bench.py / run_nccl.py.

  torchrun --nproc_per_node=8 tests/allreduce_fw_bench.py --runner h200-dgxc \\
      --topology-class h200-nvlink-island --transport nvlink \\
      --env-json results/env.json --out results/h200_allreduce_fw.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

SCHEMA_VERSION = 1
MEASUREMENT_CONTRACT = "allreduce-fw-v1"
FAMILY = "allreduce-fw"

# Latency-focused ladder: 1 KiB .. 64 MiB. This is the regime where a custom one-shot /
# two-shot NVLink all-reduce beats the NCCL ring (small messages are latency-bound; the
# ring's 2*(n-1) hops dominate). Above ~tens of MiB NCCL's bandwidth-optimal ring wins, so
# we deliberately stop at 64 MiB — past the crossover the framework kernels stop being the
# point. Geometric x4 keeps the sweep short (9 points) so per-impl warmup cost stays bounded.
DEFAULT_MIN_BYTES = 1 << 10   # 1 KiB
DEFAULT_MAX_BYTES = 64 << 20  # 64 MiB

# Custom all-reduce kernels are written for fp16/bf16 activations (TP all-reduce); a few also
# take fp32. bf16 is the headline serving dtype. Map to torch dtype lazily (torch imported in main).
_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}


def _sizes(lo: int, hi: int, factor: int = 4):
    out, s = [], lo
    while s <= hi:
        out.append(s)
        s *= factor
    return out


def comparison_key(meta: dict) -> str:
    """Rows may share a curve only within the same (impl, dtype, world, topology, contract).
    impl + topology-class are part of the key so e.g. flashinfer-oneshot on H200(NVLink) is
    never silently overlaid on sglang or on a different topology."""
    parts = [meta["impl"], meta["dtype"], str(meta["world_size"]),
             meta["topology_class"], meta["measurement_contract"]]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _bench(fn, torch, warmup: int, iters: int) -> float:
    """CUDA-event timed mean ms/iter (identical pattern to rl_mesh_bench._bench)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms/iter


def _bandwidths(nbytes: int, ms: float, world: int):
    """algbw + busbw (GB/s) for an all-reduce, matching nccl-tests so framework numbers are
    directly comparable to run_nccl.py. algbw = size/time; busbw = algbw * 2*(n-1)/n."""
    if ms <= 0:
        return 0.0, 0.0
    sec = ms / 1e3
    algbw = (nbytes / sec) / 1e9
    factor = (2.0 * (world - 1) / world) if world > 1 else 1.0
    return algbw, algbw * factor


# --------------------------------------------------------------------------------------
# Implementation registry. Each entry is a builder: given (torch, dist, dev, world, rank,
# dtype_str) it returns either None (framework/kernel not available -> skipped) or a dict
#   {"runner": fn(tensor)->None in-place all-reduce-sum, "free": optional teardown}.
# Every builder is fully import-guarded and never raises out — an unavailable framework is a
# recorded skip with a note, never a fake row. Several framework entrypoints are GUESSED
# defensively across plausible API surfaces (flashinfer/sglang/vllm reorganize these often);
# each guess is tried under try/except and simply yields "skipped" if absent, so a wrong guess
# degrades to a skip rather than a crash.
# --------------------------------------------------------------------------------------

def _build_nccl(torch, dist, dev, world, rank, dtype):
    """Baseline: torch.distributed.all_reduce (NCCL). Always available when dist is up."""
    def run(t):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return {"runner": run, "note": "torch.distributed.all_reduce (NCCL ring)"}


def _build_flashinfer(torch, dist, dev, world, rank, dtype, variant):
    """FlashInfer custom all-reduce, one-shot vs two-shot as distinct impls.

    FlashInfer's custom AR lives under flashinfer.comm and has moved across releases. We try,
    in order, the surfaces that have existed (all guarded; first that yields a working closure
    wins). The `variant` ("oneshot"/"twoshot") selects the strategy where the API exposes one.
    GUESSED entrypoints (no GPU here to confirm against 0.6.8): trtllm_allreduce_fusion,
    trtllm_custom_all_reduce, the CustomAllReduce/AllReduce workspace classes, and a one_shot/
    two_shot_all_reduce free function. If none import or none accept this world/dtype, return
    None -> recorded as skipped."""
    try:
        import flashinfer  # noqa: F401
    except Exception:
        return None
    try:
        import flashinfer.comm as ficomm
    except Exception:
        ficomm = None
    if ficomm is None:
        return {"runner": None, "skip": "flashinfer present but flashinfer.comm absent"}

    want_oneshot = (variant == "oneshot")
    inp_holder = {}

    # (a) trtllm fusion all-reduce — flashinfer's TRT-LLM-derived one/two-shot fused AR. The
    #     signature varies by release; we probe for an enum/kwarg that selects the strategy and
    #     wrap it so .runner(t) does an in-place all-reduce-sum. Heavily guarded + GUESSED.
    fusion = getattr(ficomm, "trtllm_allreduce_fusion", None)
    if fusion is not None:
        try:
            # Strategy/pattern enums live in flashinfer.comm in recent releases; absence is fine.
            strat_enum = getattr(ficomm, "AllReduceStrategyType", None) \
                or getattr(ficomm, "AllReduceStrategy", None)
            one = two = None
            if strat_enum is not None:
                one = getattr(strat_enum, "ONESHOT", None) or getattr(strat_enum, "ONE_SHOT", None)
                two = getattr(strat_enum, "TWOSHOT", None) or getattr(strat_enum, "TWO_SHOT", None)
            chosen = one if want_oneshot else two
            if chosen is None:
                # API present but can't express this variant -> let the explicit one/two-shot
                # free functions (branch c) or the class (branch b) try instead.
                raise RuntimeError("strategy enum lacks requested variant")

            def run(t, _f=fusion, _s=chosen):
                # Defensive call: try the (allreduce_in, strategy=) shape; if the real signature
                # differs the first warmup call raises and the impl is dropped (caught upstream).
                _f(t, strategy=_s)
            return {"runner": run, "note": f"flashinfer.comm.trtllm_allreduce_fusion strategy={variant}"}
        except Exception:
            pass  # fall through to other surfaces

    # (b) a CustomAllReduce / AllReduce workspace object (vLLM-style: construct once with a
    #     buffer, call per tensor). GUESSED class names + ctor; if it constructs and exposes a
    #     callable that does an in-place AR we use it. one-shot vs two-shot usually a ctor flag.
    cls = getattr(ficomm, "CustomAllReduce", None) or getattr(ficomm, "AllReduce", None)
    if cls is not None:
        try:
            obj = None
            for kwargs in ({"group": dist.group.WORLD, "device": dev},
                           {"world_size": world, "rank": rank, "device": dev},
                           {"max_size": DEFAULT_MAX_BYTES}, {}):
                try:
                    obj = cls(**kwargs)
                    break
                except Exception:
                    continue
            if obj is not None:
                method = None
                for name in ("all_reduce", "custom_all_reduce", "one_shot_all_reduce" if want_oneshot
                             else "two_shot_all_reduce", "__call__"):
                    if hasattr(obj, name):
                        method = getattr(obj, name)
                        break
                if method is not None:
                    def run(t, _m=method):
                        out = _m(t)
                        if out is not None and out.data_ptr() != t.data_ptr():
                            t.copy_(out)
                    free = getattr(obj, "close", None) or getattr(obj, "destroy", None)
                    return {"runner": run, "free": free,
                            "note": f"flashinfer.comm.{cls.__name__} ({variant})"}
        except Exception:
            pass

    # (c) explicit one_shot_all_reduce / two_shot_all_reduce free functions. GUESSED names.
    fn_name = "one_shot_all_reduce" if want_oneshot else "two_shot_all_reduce"
    fn = getattr(ficomm, fn_name, None) or getattr(ficomm, fn_name.replace("_all_reduce", "_custom_all_reduce"), None)
    if fn is not None:
        try:
            def run(t, _f=fn):
                out = _f(t)
                if out is not None and out.data_ptr() != t.data_ptr():
                    t.copy_(out)
            return {"runner": run, "note": f"flashinfer.comm.{fn_name}"}
        except Exception:
            pass
    _ = inp_holder  # (kept for symmetry; explicit workspaces would stash here)
    return {"runner": None,
            "skip": f"flashinfer.comm present but no usable {variant} all-reduce entrypoint "
                    f"(probed trtllm_allreduce_fusion / CustomAllReduce / {fn_name})"}


def _build_sglang(torch, dist, dev, world, rank, dtype):
    """SGLang 'quick all-reduce' / custom all-reduce (sgl_kernel). SGLang wraps its custom AR
    in sglang.srt.distributed.device_communicators.custom_all_reduce.CustomAllreduce; the raw
    kernels are in sgl_kernel.allreduce. We try the high-level wrapper first (it owns the IPC
    workspace setup), then the raw kernel. Both GUESSED + fully guarded -> skip on absence."""
    # (a) the SGLang distributed wrapper (preferred — manages the shared IPC buffer).
    try:
        from sglang.srt.distributed.device_communicators import custom_all_reduce as sgl_car
    except Exception:
        sgl_car = None
    if sgl_car is not None:
        cls = getattr(sgl_car, "CustomAllreduce", None) or getattr(sgl_car, "CustomAllReduce", None)
        if cls is not None:
            try:
                obj = None
                for kwargs in ({"group": dist.group.WORLD, "device": dev},
                               {"group": dist.group.WORLD, "device": local_device_index(dev)},
                               {"device": dev}, {}):
                    try:
                        obj = cls(**kwargs)
                        break
                    except Exception:
                        continue
                if obj is not None:
                    method = None
                    for name in ("custom_all_reduce", "all_reduce", "quick_all_reduce", "__call__"):
                        if hasattr(obj, name):
                            method = getattr(obj, name)
                            break
                    if method is not None:
                        def run(t, _m=method):
                            out = _m(t)
                            if out is not None and out.data_ptr() != t.data_ptr():
                                t.copy_(out)
                        free = getattr(obj, "close", None)
                        return {"runner": run, "free": free,
                                "note": f"sglang.srt...custom_all_reduce.{cls.__name__}"}
            except Exception:
                pass
    # (b) raw sgl_kernel custom/quick all-reduce. The raw API needs explicit IPC handle setup we
    #     can't reliably reconstruct here; probe for a self-contained entrypoint, else skip.
    try:
        import sgl_kernel  # noqa: F401
        allreduce_mod = getattr(__import__("sgl_kernel.allreduce", fromlist=["allreduce"]),
                                "allreduce", None) if _module_exists("sgl_kernel.allreduce") else None
    except Exception:
        allreduce_mod = None
    if allreduce_mod is not None:
        for fname in ("all_reduce", "custom_all_reduce", "quick_all_reduce"):
            fn = getattr(allreduce_mod, fname, None)
            if callable(fn):
                # Raw kernels generally require a registered IPC buffer / meta handle as extra
                # args; without the wrapper we cannot supply those safely. Record as present-
                # but-not-self-wireable rather than guess a buffer layout and risk corruption.
                return {"runner": None,
                        "skip": f"sgl_kernel.allreduce.{fname} present but needs IPC-buffer setup "
                                f"only the sglang wrapper provides (wrapper import failed)"}
    return {"runner": None,
            "skip": "sglang present but no usable custom/quick all-reduce wrapper "
                    "(probed sglang.srt...custom_all_reduce.CustomAllreduce + sgl_kernel.allreduce)"}


def _build_vllm(torch, dist, dev, world, rank, dtype):
    """vLLM in-tree custom all-reduce. vllm.distributed.device_communicators.custom_all_reduce.
    CustomAllreduce owns the IPC workspace; we construct it against the world group and call its
    custom_all_reduce/all_reduce. vLLM may not be installed -> skip. GUESSED ctor shapes."""
    mod = None
    for path in ("vllm.distributed.device_communicators.custom_all_reduce",
                 "vllm.distributed.custom_all_reduce"):
        if _module_exists(path):
            try:
                mod = __import__(path, fromlist=["x"])
                break
            except Exception:
                mod = None
    if mod is None:
        return None
    cls = getattr(mod, "CustomAllreduce", None) or getattr(mod, "CustomAllReduce", None)
    if cls is None:
        return {"runner": None, "skip": "vllm custom_all_reduce module present but no CustomAllreduce class"}
    try:
        obj = None
        for kwargs in ({"group": dist.group.WORLD, "device": dev},
                       {"group": dist.group.WORLD, "device": local_device_index(dev)},
                       {"device": dev}, {}):
            try:
                obj = cls(**kwargs)
                break
            except Exception:
                continue
        if obj is None:
            return {"runner": None, "skip": "vllm CustomAllreduce present but no ctor signature accepted"}
        method = None
        for name in ("custom_all_reduce", "all_reduce", "__call__"):
            if hasattr(obj, name):
                method = getattr(obj, name)
                break
        if method is None:
            return {"runner": None, "skip": "vllm CustomAllreduce has no all_reduce method"}

        def run(t, _m=method):
            out = _m(t)
            if out is not None and out.data_ptr() != t.data_ptr():
                t.copy_(out)
        free = getattr(obj, "close", None)
        return {"runner": run, "free": free, "note": f"vllm...custom_all_reduce.{cls.__name__}"}
    except Exception as exc:
        return {"runner": None, "skip": f"vllm custom all-reduce setup raised: {exc!r}"}


def _module_exists(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def local_device_index(dev) -> int:
    return dev.index if getattr(dev, "index", None) is not None else 0


# (impl-name, builder, top-level framework key). flashinfer one/two-shot share the "flashinfer"
# framework key; nccl's framework is "torch". The framework key drives frameworks_available.
def _impl_registry():
    return [
        ("nccl", lambda *a: _build_nccl(*a), "torch"),
        ("flashinfer-oneshot", lambda *a: _build_flashinfer(*a, variant="oneshot"), "flashinfer"),
        ("flashinfer-twoshot", lambda *a: _build_flashinfer(*a, variant="twoshot"), "flashinfer"),
        ("sglang", lambda *a: _build_sglang(*a), "sglang"),
        ("vllm", lambda *a: _build_vllm(*a), "vllm"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="CollectiveX framework custom all-reduce benchmark")
    ap.add_argument("--min-bytes", type=int, default=DEFAULT_MIN_BYTES)
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    ap.add_argument("--dtype", default="bf16", choices=sorted(_DTYPE_BYTES))
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--impls", default="",
                    help="comma/space-separated subset of impls to run (default: all). "
                         "e.g. 'nccl,flashinfer-oneshot' — nccl is always included as baseline.")
    ap.add_argument("--runner", required=True)
    ap.add_argument("--topology-class", required=True)
    ap.add_argument("--transport", default="nvlink")
    ap.add_argument("--env-json")
    ap.add_argument("--timestamp")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:
        import torch
        import torch.distributed as dist
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: torch unavailable: {exc!r}", file=sys.stderr)
        return 3

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world < 2:
        if rank == 0:
            print(f"ERROR: allreduce-fw needs world_size >= 2 (got {world}); "
                  f"launch under torchrun --nproc_per_node=N", file=sys.stderr)
        return 5
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12359")
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    elem_bytes = _DTYPE_BYTES[args.dtype]
    sizes = _sizes(args.min_bytes, args.max_bytes)

    # Which impls to attempt. nccl baseline is always included.
    want = {s for s in args.impls.replace(",", " ").split() if s}
    registry = _impl_registry()
    if want:
        registry = [e for e in registry if e[0] in want or e[0] == "nccl"]

    # frameworks_available: framework key -> {available: bool, note/skip-reason}. Probed once.
    frameworks_available: dict = {}

    def _note_framework(fwkey: str, available: bool, detail: str):
        prev = frameworks_available.get(fwkey)
        # importable wins over a per-variant skip (flashinfer may import yet a variant be absent).
        if prev is None or (available and not prev.get("available")):
            frameworks_available[fwkey] = {"available": available, "detail": detail}

    groups = []
    peak_bw = 0.0
    nccl_ok = False
    framework_ok = False

    for impl_name, builder, fwkey in registry:
        # Build the impl on every rank (custom AR needs collective IPC setup on all ranks).
        try:
            built = builder(torch, dist, dev, world, rank, args.dtype)
        except Exception as exc:
            built = {"runner": None, "skip": f"builder raised: {exc!r}"}

        if built is None:
            _note_framework(fwkey, False, "framework not importable")
            if rank == 0:
                print(f"  {impl_name}: skipped (framework '{fwkey}' not importable)", file=sys.stderr)
            continue
        if built.get("runner") is None:
            reason = built.get("skip", "no usable entrypoint")
            # framework imported (we got past `is None`) but this impl/variant isn't wireable.
            _note_framework(fwkey, fwkey == "torch", reason if fwkey != "torch" else "baseline")
            if rank == 0:
                print(f"  {impl_name}: skipped ({reason})", file=sys.stderr)
            continue

        _note_framework(fwkey, True, built.get("note", "available"))
        run = built["runner"]
        rows = []
        impl_failed = False
        for nbytes in sizes:
            numel = max(1, nbytes // elem_bytes)
            actual_bytes = numel * elem_bytes
            # Known inputs so the reduced result has a closed form: every rank fills with its
            # (rank+1); all-reduce-sum -> world*(world+1)/2 in every element. Lets us validate
            # custom kernels against a reference without trusting the kernel to define "correct".
            base = float(rank + 1)
            expected = float(world * (world + 1) // 2)
            try:
                t = torch.full((numel,), base, dtype=torch_dtype, device=dev)

                def step(_t=t):
                    run(_t)
                ms = _bench(step, torch, args.warmup, args.iters)
            except Exception as exc:
                rows.append({"size_bytes": actual_bytes, "latency_us": None,
                             "algbw_gbps": 0.0, "busbw_gbps": 0.0, "correct": None,
                             "error": repr(exc)})
                impl_failed = True
                break

            # Correctness: re-run once on a fresh known buffer and compare to the reference.
            correct = None
            try:
                chk = torch.full((numel,), base, dtype=torch_dtype, device=dev)
                run(chk)
                ref = torch.full((numel,), expected, dtype=torch_dtype, device=dev)
                # bf16/fp16 accumulate with rounding; tolerance scales with the magnitude.
                atol = 0.0 if args.dtype == "fp32" else max(1.0, expected * 0.02)
                correct = bool(torch.allclose(chk, ref, atol=atol, rtol=0.0))
            except Exception:
                correct = None

            # Reduce timing across ranks (max = slowest rank) for a stable cross-rank number,
            # exactly like rl_mesh_bench. Done with the always-present NCCL collective on a tiny
            # tensor (not the impl under test).
            tt = torch.tensor([ms], device=dev)
            dist.all_reduce(tt, op=dist.ReduceOp.MAX)
            ms_max = float(tt.item())
            algbw, busbw = _bandwidths(actual_bytes, ms_max, world)
            peak_bw = max(peak_bw, busbw)
            rows.append({"size_bytes": actual_bytes,
                         "latency_us": round(ms_max * 1e3, 3),
                         "algbw_gbps": round(algbw, 3),
                         "busbw_gbps": round(busbw, 3),
                         "correct": correct})

        if built.get("free"):
            try:
                built["free"]()
            except Exception:
                pass

        had_bw = any((r.get("busbw_gbps") or 0.0) > 0.0 for r in rows)
        if had_bw:
            if impl_name == "nccl":
                nccl_ok = True
            else:
                framework_ok = True
        meta = {"impl": impl_name, "framework": fwkey, "dtype": args.dtype,
                "world_size": world, "topology_class": args.topology_class,
                "measurement_contract": MEASUREMENT_CONTRACT}
        groups.append({**meta, "comparison_key": comparison_key(meta),
                       "note": built.get("note"), "rows": rows,
                       "incomplete": impl_failed})
        if rank == 0:
            mn = min((r["latency_us"] for r in rows if r.get("latency_us")), default=None)
            print(f"  {impl_name}: {len(rows)} sizes, min latency "
                  f"{mn if mn is not None else float('nan')} us, peak busbw "
                  f"{max((r.get('busbw_gbps') or 0.0) for r in rows):.1f} GB/s", file=sys.stderr)

    if rank != 0:
        dist.barrier()
        dist.destroy_process_group()
        return 0

    env = None
    if args.env_json and os.path.exists(args.env_json):
        with open(args.env_json) as fh:
            env = json.load(fh)

    # valid iff the NCCL baseline AND at least one framework custom kernel produced real (bw>0)
    # rows. A run where every framework was skipped (only nccl ran) is NOT valid for this family —
    # the whole point is the framework comparison; that case should be read as "no framework AR
    # available on this image", not as a green result.
    status = "valid" if (nccl_ok and framework_ok) else "invalid"

    doc = {
        "schema_version": SCHEMA_VERSION, "family": FAMILY,
        "generated_by": "allreduce_fw_bench.py",
        "generated_at": args.timestamp or _dt.datetime.now().astimezone().isoformat(),
        "runner": args.runner, "transport": args.transport,
        "measurement_contract": MEASUREMENT_CONTRACT,
        "world_size": world, "dtype": args.dtype,
        "size_min_bytes": args.min_bytes, "size_max_bytes": args.max_bytes,
        "status": status,
        "peak_busbw_gbps": round(peak_bw, 2),
        "frameworks_available": frameworks_available,
        "num_groups": len(groups), "groups": groups, "environment": env,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")

    avail = sorted(k for k, v in frameworks_available.items() if v.get("available"))
    print(f"allreduce-fw: {len(groups)} impl group(s) -> {args.out} "
          f"(status={status}, world={world}, dtype={args.dtype}, "
          f"frameworks_available={avail}, peak_busbw={peak_bw:.1f} GB/s)")
    dist.barrier()
    dist.destroy_process_group()
    return 0 if status == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
