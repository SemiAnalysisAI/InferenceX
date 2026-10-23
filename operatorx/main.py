"""``python -m operatorx``: run every testlist x backend on this platform.

Op types a backend's IMPLS doesn't claim are skipped (no row, unless --strict);
UnsupportedOpError -> status="unsupported", other exceptions -> "error".
Env: OPERATORX_CLUSTER, OPERATORX_BACKENDS, OPERATORX_PARALLEL (JSON), WORLD_SIZE/RANK.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pkgutil
import re
import sys
from pathlib import Path

import operatorx.ops  # noqa: F401  populates op registry
from operatorx.core import op_registry, parallel
from operatorx import Op, Result, UnsupportedOpError, write_run_result
from operatorx.clusters import CLUSTER_PLATFORMS
from operatorx.runtime import runtime_snapshot, utc_now_iso

# a testlist source: "<org>/<model>/<role>"
_SOURCE = re.compile(r"[^/\s]+/[^/\s]+/[^/\s]+")


_REPO_ROOT = Path(__file__).resolve().parent


TESTLIST_DIR = _REPO_ROOT / "testlists"
RESULTS_DIR = _REPO_ROOT / "results"


def _discover_backends(platform: str) -> list[str]:
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


def _load_testlists(names: list[str] | None, directory: Path = TESTLIST_DIR) -> dict[str, list[dict]]:
    """{testlist_name: [shape_dict, ...]}; all available by default."""
    available = {p.stem: p for p in sorted(directory.glob("*.json"))}
    if names:
        wanted = {n: available[n] for n in names if n in available}
        missing = [n for n in names if n not in available]
        if missing:
            raise SystemExit(f"unknown testlist(s): {missing}; available: {sorted(available)}")
    else:
        wanted = available
    lists = {name: json.loads(path.read_text()) for name, path in wanted.items()}
    for name, entries in lists.items():
        for i, entry in enumerate(entries):
            v = entry.get("sources")
            if not isinstance(v, list) or not all(isinstance(s, str) and _SOURCE.fullmatch(s) for s in v):
                raise SystemExit(f"{name}[{i}]: every testlist entry needs a 'sources' list of "
                                 f"'<org>/<model>/<role>' strings (empty for a shape from no model)")
            if "name" in entry:
                raise SystemExit(f"{name}[{i}]: 'name' is gone; the role is the last part of each source")
    return lists


def _backend_supported_ops(platform: str, backends: list[str], strict: bool = False) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for b in backends:
        try:
            mod = importlib.import_module(f"operatorx.runners.{platform}.backends.{b}")
        except Exception as e:
            if strict:
                raise RuntimeError(f"requested backend {platform}/{b} cannot be loaded") from e
            print(f"[run_smoke] backend {platform}/{b} not importable: {e}", file=sys.stderr)
            out[b] = set()
            continue
        out[b] = {impl.op_type for impl in getattr(mod, "IMPLS", [])}
    return out


def _collect_backend_versions(platform: str, backends: list[str]) -> dict[str, str]:
    """Each backend's versions(); failing probes are skipped."""
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


def _resolve_split() -> dict:
    """This process's split (OPERATORX_PARALLEL, JSON; default one device); must match WORLD_SIZE."""
    split = parallel.normalize(json.loads(os.environ.get("OPERATORX_PARALLEL") or "null"))
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if parallel.world_size(split) != ws:
        raise SystemExit(f"OPERATORX_PARALLEL={split} needs {parallel.world_size(split)} ranks, launched {ws}")
    return split


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--platform", default=None,
                    help="platform override (otherwise inferred from $OPERATORX_CLUSTER)")
    ap.add_argument("--testlists", default=None,
                    help="comma-separated testlist names; default = all in operatorx/testlists/")
    ap.add_argument("--backends", default=None,
                    help="comma-separated backend names; default = all for the platform")
    ap.add_argument("--testlist-dir", type=Path, default=TESTLIST_DIR)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--strict", action="store_true", help="fail on errors or zero successful rows")
    args = ap.parse_args()

    run = runtime_snapshot()
    platform = _resolve_platform(args.platform, run.cluster)

    requested = _csv(args.backends) or _csv(os.environ.get("OPERATORX_BACKENDS"))
    backends = requested if requested else _discover_backends(platform)
    if not backends:
        raise SystemExit(f"no backends configured for platform={platform!r}")

    backend_ops = _backend_supported_ops(platform, backends, strict=args.strict)
    run.software.update(_collect_backend_versions(platform, backends))
    split = _resolve_split()
    ws = parallel.world_size(split)
    rank = int(os.environ.get("RANK", "0"))

    testlists = _load_testlists(
        _csv(args.testlists) or _csv(os.environ.get("OPERATORX_TESTLISTS")) or None,
        args.testlist_dir,
    )

    runner_mod = importlib.import_module(f"operatorx.runners.{platform}.runner")

    entries: list[tuple[Op, str]] = []
    for tl_name, shapes in testlists.items():
        for shape in shapes:
            # one parallel state per process; other splits run in their own process
            if parallel.normalize(shape["args"].get("parallel")) != split:
                continue
            for backend in backends:
                if not args.strict and shape["type"] not in backend_ops.get(backend, set()):
                    continue  # strict CI keeps unsupported backend/op pairs
                entries.append((
                    Op(type=shape["type"], args=shape["args"], backend=backend,
                       sources=shape["sources"]),
                    tl_name,
                ))

    if rank == 0:
        print(f"[run_smoke] platform={platform} cluster={run.cluster!r} parallel={split}")
        print(f"[run_smoke] backends={backends}")
        print(f"[run_smoke] testlists={list(testlists)}  -> {len(entries)} (op,backend) entries")

    import time as _time
    results: list[Result] = []
    counts = {"ok": 0, "unsupported": 0, "error": 0}
    out_path = args.results_dir / platform / (run.cluster or "unknown") / f"{run.id}.json"
    for op, tl in entries:
        _t0 = _time.perf_counter()
        try:
            if op.type not in backend_ops.get(op.backend, set()):
                raise UnsupportedOpError(f"{platform}/{op.backend} has no implementation for {op.type}")
            # a case the op's schema rejects is a bad testlist entry: an error, never measured
            op_registry.validate(op)
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
            # checkpoint after each op, outside timing, so cancellation keeps completed rows
            if args.strict:
                run.finished_at = utc_now_iso()
                write_run_result(out_path, run, results)
            shape_str = " ".join(
                f"{k}={json.dumps(v, separators=(',', ':')) if isinstance(v, (dict, list)) else v}"
                for k, v in op.args.items() if not k.startswith("dtype")
            )
            print(f"[ws={ws}] {tl:12} {status:11} {op.type:18} {op.backend:10}  "
                  f"{latency_str}  wall={wall_s:6.1f}s   {shape_str}", flush=True)
        # release cached blocks, else the next big shape can OOM on the previous op's buffers
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass

    run.finished_at = utc_now_iso()

    if rank == 0 and results:
        write_run_result(out_path, run, results)
        print(f"\n[run_smoke] {counts} -> {out_path}")

    if ws > 1:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
        except Exception:
            pass

    return int(args.strict and (counts["error"] > 0 or counts["ok"] == 0))


if __name__ == "__main__":
    raise SystemExit(main())
