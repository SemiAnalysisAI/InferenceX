"""Run every testlist × every backend on the detected platform.

Invoke as ``python -m operatorx`` (or ``python -m operatorx.main``).

Reads testlists from ``<repo>/testlists/*.json`` (one file per testlist; filename
without extension = testlist name). Default = all testlists; use ``--testlists a,b,c``
to filter.

Backend op-type support is derived from each backend module's ``IMPLS`` list. If a
backend doesn't claim an op_type, the (op, backend) combination is silently skipped
— no Result row is emitted. Args-level rejection (e.g. dtype mismatch) is captured
as ``status="unsupported"``. Other exceptions become ``status="error"``.

Platform is detected from ``$OPERATORX_CLUSTER`` → ``CLUSTERS[id].platform``, or
overridden with ``--platform``. Backend filter via ``--backends`` (CSV) or env
``OPERATORX_BACKENDS``. World size from ``WORLD_SIZE`` env (set by torchrun, or
manually for TPU); auto-detected for TPU when unset.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pkgutil
from pathlib import Path

import operatorx.ops  # noqa: F401  populates op registry
from operatorx import Op, Result, UnsupportedOpError, write_run_result
from operatorx.clusters import CLUSTER_PLATFORMS
from operatorx.runtime import runtime_snapshot, utc_now_iso


# Repo root (directory containing testlists/ and results/), one level above the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent


TESTLIST_DIR = _REPO_ROOT / "testlists"
RESULTS_DIR = _REPO_ROOT / "results"


def _discover_backends(platform: str) -> list[str]:
    """All backends for a platform = the python modules under
    ``operatorx/runners/<platform>/backends/``
    """
    try:
        pkg = importlib.import_module(f"operatorx.runners.{platform}.backends")
    except ImportError:
        return []
    return sorted(
        info.name for info in pkgutil.iter_modules(pkg.__path__)
        if not info.name.startswith("_")
    )


def _csv(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _load_testlists(names: list[str] | None) -> dict[str, list[dict]]:
    """Returns {testlist_name: [shape_dict, ...]}. Default = all available."""
    available = {p.stem: p for p in sorted(TESTLIST_DIR.glob("*.json"))}
    if names:
        wanted = {n: available[n] for n in names if n in available}
        missing = [n for n in names if n not in available]
        if missing:
            raise SystemExit(f"unknown testlist(s): {missing}; available: {sorted(available)}")
    else:
        wanted = available
    return {name: json.loads(path.read_text()) for name, path in wanted.items()}


def _backend_supported_ops(platform: str, backends: list[str]) -> dict[str, set[str]]:
    """Discover per-backend supported op_types from each module's IMPLS list."""
    out: dict[str, set[str]] = {}
    for b in backends:
        try:
            mod = importlib.import_module(f"operatorx.runners.{platform}.backends.{b}")
        except Exception as e:
            print(f"[run_smoke] backend {platform}/{b} not importable: {e}", file=sys.stderr)
            out[b] = set()
            continue
        out[b] = {impl.op_type for impl in getattr(mod, "IMPLS", [])}
    return out


def _collect_backend_versions(platform: str, backends: list[str]) -> dict[str, str]:
    """Ask each backend module to report its own library versions.

    Each backend exports ``versions() -> dict[str, str]``. Returned keys are
    merged into ``RunInfo.software``. Missing or failing probes are skipped.
    """
    out: dict[str, str] = {}
    for b in backends:
        try:
            mod = importlib.import_module(f"operatorx.runners.{platform}.backends.{b}")
        except Exception:
            continue
        fn = getattr(mod, "versions", None)
        if fn is None:
            continue
        try:
            out.update(fn())
        except Exception:
            pass
    return out


def _resolve_platform(args_platform: str | None, run_cluster: str | None) -> str:
    if args_platform:
        return args_platform
    if run_cluster:
        platform = CLUSTER_PLATFORMS.get(run_cluster)
        if platform:
            return platform
    raise SystemExit(
        "can't determine platform: pass --platform or set $OPERATORX_CLUSTER"
    )


def _resolve_world_size(platform: str) -> int:
    ws = os.environ.get("WORLD_SIZE")
    if ws:
        return int(ws)
    if platform == "tpu":
        try:
            import jax
            return jax.device_count()
        except Exception:
            pass
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--platform", default=None,
                    help="platform override (otherwise inferred from $OPERATORX_CLUSTER)")
    ap.add_argument("--testlists", default=None,
                    help="comma-separated testlist names; default = all in operatorx/testlists/")
    ap.add_argument("--backends", default=None,
                    help="comma-separated backend names; default = all for the platform")
    args = ap.parse_args()

    run = runtime_snapshot()
    platform = _resolve_platform(args.platform, run.cluster)

    # Backends
    requested = _csv(args.backends) or _csv(os.environ.get("OPERATORX_BACKENDS"))
    backends = requested if requested else _discover_backends(platform)
    if not backends:
        raise SystemExit(f"no backends configured for platform={platform!r}")

    backend_ops = _backend_supported_ops(platform, backends)
    # Merge per-backend library versions into the run's software dict.
    run.software.update(_collect_backend_versions(platform, backends))
    ws = _resolve_world_size(platform)
    rank = int(os.environ.get("RANK", "0"))

    # Testlists
    testlists = _load_testlists(_csv(args.testlists) or _csv(os.environ.get("OPERATORX_TESTLISTS")) or None)

    # submit_run.py fans out one job per (ep, routed_tp, shared_tp) MoE combo
    # plus one job per ws for non-MoE ops, since sglang's group state is
    # process-global. OPERATORX_MOE_PARALLELISM scopes a job to one combo;
    # absence scopes it to non-MoE ops only.
    moe_par_env = os.environ.get("OPERATORX_MOE_PARALLELISM")
    moe_filter: tuple[int, int, int] | None = None
    if moe_par_env:
        parts = moe_par_env.split(":")
        if len(parts) != 3:
            raise SystemExit(f"OPERATORX_MOE_PARALLELISM must be ep:routed_tp:shared_tp; got {moe_par_env!r}")
        moe_filter = (int(parts[0]), int(parts[1]), int(parts[2]))

    # Load runner
    runner_mod = importlib.import_module(f"operatorx.runners.{platform}.runner")

    # Build entries: (op, testlist_name).
    entries: list[tuple[Op, str]] = []
    for tl_name, shapes in testlists.items():
        for shape in shapes:
            # In a multi-rank job (ws>1) only run ops explicitly tagged with the
            # matching world_size — running single-rank ops on every rank is
            # both wasteful and meaningless (each rank would redo the same work
            # in parallel). Per-rank ops belong in the ws=1 job.
            shape_ws = shape["args"].get("world_size")
            if shape_ws is None:
                if ws != 1:
                    continue
            elif int(shape_ws) != ws:
                continue
            # Combo job (moe_filter set): only moe_forward shapes matching the
            # parallelism triple. Non-combo job: everything except moe_forward.
            if moe_filter is not None:
                if shape["type"] != "moe_forward":
                    continue
                a = shape["args"]
                triple = (
                    a.get("expert_parallel_size", 1),
                    a.get("routed_tensor_parallel_size", 1),
                    a.get("shared_tensor_parallel_size", 1),
                )
                if triple != moe_filter:
                    continue
            elif shape["type"] == "moe_forward":
                continue
            for backend in backends:
                if shape["type"] not in backend_ops.get(backend, set()):
                    continue  # silent skip — backend doesn't claim this op type
                entries.append((
                    Op(type=shape["type"], args=shape["args"], backend=backend,
                       name=shape.get("name")),
                    tl_name,
                ))

    if rank == 0:
        print(f"[run_smoke] platform={platform} cluster={run.cluster!r} ws={ws}")
        print(f"[run_smoke] backends={backends}")
        print(f"[run_smoke] testlists={list(testlists)}  -> {len(entries)} (op,backend) entries")

    import time as _time
    results: list[Result] = []
    counts = {"ok": 0, "unsupported": 0, "error": 0}
    for op, tl in entries:
        _t0 = _time.perf_counter()
        try:
            r = runner_mod.run(op)
            results.append(Result(op=op, metrics=r.metrics, status="ok",
                                  testlist=tl))
            counts["ok"] += 1
            status = "ok"
            latency_str = f"{r.metrics['latency_us']:>10.2f} us"
        except UnsupportedOpError as e:
            results.append(Result(op=op, metrics={}, status="unsupported",
                                  message=str(e), testlist=tl))
            counts["unsupported"] += 1
            status = "unsupported"
            latency_str = "         unsupported"
        except Exception as e:
            import traceback as _tb
            tb_tail = "".join(_tb.format_exception(type(e), e, e.__traceback__)).splitlines()[-6:]
            results.append(Result(op=op, metrics={}, status="error",
                                  message=f"{type(e).__name__}: {e} | tb: " + " || ".join(tb_tail),
                                  testlist=tl))
            counts["error"] += 1
            status = "error"
            latency_str = f"        ERROR ({type(e).__name__})"
        wall_s = _time.perf_counter() - _t0
        if rank == 0:
            shape_str = " ".join(
                f"{k}={v}" for k, v in op.args.items() if not k.startswith("dtype")
            )
            print(f"[ws={ws}] {tl:12} {status:11} {op.type:18} {op.backend:10}  "
                  f"{latency_str}  wall={wall_s:6.1f}s   {shape_str}", flush=True)
        # Release per-op tensors back to the CUDA driver so the next op's
        # allocator sees the full GPU. Without this, PyTorch's caching
        # allocator hangs on to large output buffers and the next big shape
        # OOMs even though the prior tensors are no longer referenced.
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass

    run.finished_at = utc_now_iso()

    if rank == 0 and results:
        cluster_dir = run.cluster or "unknown"
        out_path = RESULTS_DIR / platform / cluster_dir / f"{run.id}.json"
        write_run_result(out_path, run, results)
        print(f"\n[run_smoke] {counts} -> {out_path}")

    # Multi-rank cleanup (torch.distributed used by NVIDIA + Trainium)
    if ws > 1:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
