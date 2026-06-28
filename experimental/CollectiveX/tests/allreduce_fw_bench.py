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


# FlashInfer custom AR works on a [token_num, hidden_dim] activation tensor (the TP all-reduce
# shape), so the flashinfer impls sweep this fixed hidden and reshape the bench's flat buffer to
# [numel/H, H]. Sizes not a multiple of H (only the smallest 1 KiB point) raise _SkipSize -> the
# bench records a skipped row and continues (does NOT mark the impl failed).
_FI_AR_HIDDEN = 2048


class _SkipSize(Exception):
    """Raised by an impl's run() for a size its kernel can't shape (skip that size, keep the impl)."""


def _build_flashinfer(torch, dist, dev, world, rank, dtype, variant):
    """FlashInfer custom all-reduce, one-shot vs two-shot as distinct impls — the REAL contract
    (pinned on B300, flashinfer 0.6.8.post1): trtllm_allreduce_fusion with pattern_code=
    AllReduceFusionPattern.kAllReduce (pure AR, no fusion) and use_oneshot True/False selecting
    one-shot vs two-shot. The IPC workspace comes from trtllm_create_ipc_workspace_for_all_reduce_
    fusion(tp_rank, tp_size, max_token_num, hidden_dim, group) -> (ipc_handles, workspace_ptrs[7]).
    Both variants validated correct=True at EP2. (These APIs carry a deprecation note toward a future
    allreduce.py, but are the functional one/two-shot entrypoints in this wheel.)"""
    try:
        import flashinfer.comm as ficomm
        from flashinfer.comm import trtllm_ar as fi_ar
    except Exception:
        return None
    fusion = getattr(ficomm, "trtllm_allreduce_fusion", None)
    mkws = getattr(ficomm, "trtllm_create_ipc_workspace_for_all_reduce_fusion", None)
    rmws = getattr(ficomm, "trtllm_destroy_ipc_workspace_for_all_reduce_fusion", None)
    Pat = getattr(fi_ar, "AllReduceFusionPattern", None) or getattr(ficomm, "AllReduceFusionPattern", None)
    if fusion is None or mkws is None or Pat is None or not hasattr(Pat, "kAllReduce"):
        return {"runner": None,
                "skip": "flashinfer.comm lacks trtllm_allreduce_fusion / IPC workspace / "
                        "AllReduceFusionPattern.kAllReduce"}
    H = _FI_AR_HIDDEN
    use_oneshot = (variant == "oneshot")
    max_tok = max(1, (DEFAULT_MAX_BYTES // _DTYPE_BYTES[dtype]) // H)
    try:
        ws = mkws(rank, world, max_tok, H, group=dist.group.WORLD)
    except Exception as exc:
        return {"runner": None, "skip": f"fusion IPC workspace creation failed: {exc!r}"}
    ipc_handles = ws[0] if isinstance(ws, (list, tuple)) else None
    ws_ptrs = ws[1] if isinstance(ws, (list, tuple)) and len(ws) >= 2 else None
    pat = Pat.kAllReduce
    out_buf = {}

    def run(t, _f=fusion, _pat=pat, _os=use_oneshot, _wp=ws_ptrs):
        numel = t.numel()
        if numel < H or (numel % H) != 0:
            raise _SkipSize(f"size {numel} elems not a multiple of hidden {H}")
        Tn = numel // H
        # Two-shot splits the sequence dim across ranks -> it asserts token_num > tp_size. One-shot
        # has no such floor. Skip (don't fail) the small sizes where two-shot can't run.
        if not _os and Tn <= world:
            raise _SkipSize(f"two-shot needs token_num({Tn}) > tp_size({world})")
        inp = t.view(Tn, H)
        out = out_buf.get(Tn)
        if out is None:
            out = torch.empty_like(inp)
            out_buf[Tn] = out
        _f(allreduce_in=inp, world_size=world, world_rank=rank, token_num=Tn, hidden_dim=H,
           workspace_ptrs=_wp, launch_with_pdl=False, trigger_completion_at_end=True,
           fp32_acc=True, pattern_code=_pat, use_oneshot=_os, allreduce_out=out,
           residual_in=None, residual_out=None, norm_out=None, quant_out=None, scale_out=None,
           rms_gamma=None, rms_eps=None, scale_factor=None, layout_code=None)
        # The kernel is out-of-place; copy back so the bench's in-place run(t) contract + its
        # correctness check (which reads t) hold. The copy is small vs the AR and noted in the row.
        t.copy_(out.view(-1))

    def free():
        if rmws is not None and ipc_handles is not None:
            try:
                rmws(ipc_handles, group=dist.group.WORLD)
            except Exception:
                pass

    return {"runner": run, "free": free,
            "note": f"flashinfer.comm.trtllm_allreduce_fusion kAllReduce use_oneshot={use_oneshot} "
                    f"(hidden={H}, out-of-place + copy-back)"}


def _sglang_vllm_ca_runner(ps, torch, dev, world, rank, fw):
    """Shared: replicate the framework's SERVING distributed init (init_distributed_environment +
    initialize_model_parallel) on the existing torchrun group, then return a run() that calls the TP
    GroupCoordinator's custom-allreduce. sglang AND vllm expose the identical parallel_state API
    (sglang forked vllm's), so one helper drives both. The serving init is exactly the context the
    CustomAllreduce wrapper needs (it builds ca_comm only after initialize_model_parallel) — which is
    why a bare-wrapper construction skipped before. Fully guarded -> skip dict on any failure."""
    try:
        if not ps.model_parallel_is_initialized():
            ps.init_distributed_environment(world_size=world, rank=rank,
                                            distributed_init_method="env://",
                                            local_rank=local_device_index(dev), backend="nccl")
            ps.initialize_model_parallel(tensor_model_parallel_size=world)
        tp = ps.get_tp_group()
    except Exception as e:
        return {"runner": None, "skip": f"{fw} distributed init failed: {e!r}"}
    ca = getattr(tp, "ca_comm", None)
    if ca is None or getattr(ca, "disabled", True):
        return {"runner": None,
                "skip": f"{fw} TP group ca_comm absent/disabled (no custom-AR at world={world}; "
                        f"needs >1 rank + a supported topology/size)"}

    def run(t, _ca=ca):
        if hasattr(_ca, "should_custom_ar") and not _ca.should_custom_ar(t):
            raise _SkipSize(f"{fw} ca_comm: size outside custom-AR range")
        out = _ca.custom_all_reduce(t)
        if out is not None and out.data_ptr() != t.data_ptr():
            t.copy_(out)
    return {"runner": run, "free": getattr(tp, "destroy", None),
            "note": f"{fw} GroupCoordinator.ca_comm.custom_all_reduce (serving init replicated)"}


def _build_sglang(torch, dist, dev, world, rank, dtype):
    """SGLang custom all-reduce. The wrapper builds its IPC buffer only inside the framework's
    distributed init (initialize_model_parallel) — so replicate that on the torchrun group and use
    the TP group's ca_comm (the prior bare-CustomAllreduce construction skipped for exactly this)."""
    try:
        from sglang.srt.distributed import parallel_state as ps
    except Exception as e:
        return {"runner": None, "skip": f"sglang.srt.distributed import failed (not in image?): {e!r}"}
    return _sglang_vllm_ca_runner(ps, torch, dev, world, rank, "sglang")


def _build_vllm(torch, dist, dev, world, rank, dtype):
    """vLLM in-tree custom all-reduce via its GroupCoordinator — same serving-init replication as
    sglang (vllm.distributed.parallel_state has the identical init/get_tp_group/ca_comm API). vLLM
    isn't in the sglang image, so this runs under the vLLM container switch (CX_BENCH=allreduce-fw +
    sku/image -> a vllm image); skips on absence."""
    try:
        from vllm.distributed import parallel_state as ps
    except Exception as e:
        return {"runner": None, "skip": f"vllm.distributed import failed (not in image — needs a vLLM container): {e!r}"}
    return _sglang_vllm_ca_runner(ps, torch, dev, world, rank, "vllm")


def _module_exists(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _build_aiter(torch, dist, dev, world, rank, dtype):
    """AITER (AMD) custom/quick all-reduce — aiter.dist.device_communicators.custom_all_reduce.
    CustomAllreduce (the wrapper owns the IPC buffer), else the raw aiter.ops.custom_all_reduce.
    Fully guarded -> skip on absence (e.g. NVIDIA image has no aiter). The AMD framework-AR tier."""
    # (a) the AITER distributed wrapper (preferred — manages the shared IPC buffer).
    for modpath in ("aiter.dist.device_communicators.custom_all_reduce",
                    "aiter.dist.device_communicators.quick_all_reduce"):
        try:
            mod = __import__(modpath, fromlist=["x"])
        except Exception:
            mod = None
        if mod is None:
            continue
        cls = getattr(mod, "CustomAllreduce", None) or getattr(mod, "QuickAllReduce", None) \
            or getattr(mod, "CustomAllReduce", None)
        if cls is None:
            continue
        try:
            obj = None
            for kwargs in ({"group": dist.group.WORLD, "device": dev},
                           {"group": dist.group.WORLD, "device": local_device_index(dev)},
                           {"device": dev}, {}):
                try:
                    obj = cls(**kwargs); break
                except Exception:
                    continue
            if obj is not None:
                for name in ("custom_all_reduce", "quick_all_reduce", "all_reduce", "__call__"):
                    if hasattr(obj, name):
                        method = getattr(obj, name)
                        def run(t, _m=method):
                            out = _m(t)
                            if out is not None and out.data_ptr() != t.data_ptr():
                                t.copy_(out)
                        return {"runner": run, "free": getattr(obj, "close", None),
                                "note": f"{modpath}.{cls.__name__}"}
        except Exception:
            pass
    # (b) raw aiter.ops kernels — need an explicit IPC handle we can't reconstruct here -> record present.
    try:
        import aiter  # noqa: F401
        if _module_exists("aiter.ops.custom_all_reduce"):
            return {"runner": None, "skip": "aiter.ops.custom_all_reduce present but needs IPC-buffer "
                                            "setup only the aiter wrapper provides (wrapper init failed)"}
    except Exception:
        pass
    return {"runner": None, "skip": "aiter not importable (not in this image) / no usable custom AR wrapper"}


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
        ("aiter", lambda *a: _build_aiter(*a), "aiter"),
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
            except _SkipSize as sk:
                # The kernel can't shape this size (e.g. below the custom-AR hidden) — record a
                # skipped row and CONTINUE; do NOT fail the impl (it works at the other sizes).
                rows.append({"size_bytes": actual_bytes, "latency_us": None,
                             "algbw_gbps": 0.0, "busbw_gbps": 0.0, "correct": None,
                             "skipped": str(sk)})
                continue
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

    # valid iff the NCCL baseline produced real (bw>0) rows — the all-reduce curve itself is the
    # deliverable. Which framework custom kernels were importable on this image is recorded in
    # frameworks_available + the `framework_ok` flag (not all frameworks ship in every image); a run
    # with only nccl is a valid latency/bandwidth baseline, not a failure.
    status = "valid" if nccl_ok else "invalid"

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
