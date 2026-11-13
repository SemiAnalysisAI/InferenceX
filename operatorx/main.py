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
from operatorx.core.ladder import layer_key
from operatorx import Op, Result, UnsupportedOpError, write_run_result
from operatorx.core.errors import FreshProcessError
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
    available = {}
    for p in sorted(directory.rglob("*.json")):  # official/ and smoke/, or one flat staged dir
        if p.stem in available:
            raise SystemExit(f"testlist {p.stem!r} is in both {available[p.stem]} and {p}")
        available[p.stem] = p
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
                                 f"'<org>/<model>/<role>' strings (operatorx/Synthetic-Shapes/m<M>_n<N>_k<K> for a shape from no model)")
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


# Recovery (CI parts; OPERATORX_PROGRESS names this rank's progress file). A case that raised
# an error (OOM, device fault), or that failed with no layer or engine left cached while the
# process still holds device memory (a failed startup vLLM never freed), ends the process after
# its checkpoint (RESTART); ci.py's rank wrapper starts a fresh one, which runs that case once
# more and goes on. A process that dies mid-case is resumed by the wrapper too (one device):
# the case is recorded as an error. Every rank of a group decides together.
RESTART = 75
TIMED_OUT = 76  # the case watchdog ended a case that did not finish
# a case (build, timing, profile) that runs longer is stopped: a collective that hangs otherwise
# waits for NCCL's 10-minute watchdog; a process's first case also compiles and builds
_CASE_TIMEOUT_S = float(os.environ.get("OPERATORX_CASE_TIMEOUT_S", "300"))
# the first case of a layer builds it and compiles its kernels (DeepSeek-V4 attention on B200:
# over 15 minutes cold, eight ranks at once)
_FIRST_CASE_TIMEOUT_S = float(os.environ.get("OPERATORX_FIRST_CASE_TIMEOUT_S", "1800"))
_USED_FLOOR = float(os.environ.get("OPERATORX_RECOVER_USED_GIB", "8")) * 2**30
_USED_FRACTION = float(os.environ.get("OPERATORX_RECOVER_USED_FRACTION", "0.12"))


def _read_progress(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _write_progress(path: Path, state: dict) -> None:
    """The group's progress; its rank 0 writes it (every rank reaches the same points)."""
    if os.environ.get("RANK", "0") != "0":
        return
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


class _Watchdog:
    """Stops this process (TIMED_OUT, stacks on stderr) if a case runs past its limit."""

    def __init__(self, progress: Path):
        import threading
        self.progress, self.lock, self.timer = progress, threading.Lock(), None

    def arm(self, index: int, seconds: float, state: dict) -> None:
        import faulthandler
        import threading
        self.cancel()
        self.timer = threading.Timer(seconds, self._fire, (index, seconds, state))
        self.timer.daemon = True
        self.timer.start()
        # a thread in native code holding the GIL keeps the timer from running: faulthandler's
        # own thread needs no GIL; it dumps the stacks and exits (a crash for the supervisor)
        faulthandler.dump_traceback_later(seconds + 60, exit=True)

    def cancel(self) -> None:
        import faulthandler
        faulthandler.cancel_dump_traceback_later()
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

    def _fire(self, index: int, seconds: float, state: dict) -> None:
        import faulthandler
        faulthandler.cancel_dump_traceback_later()
        _write_progress(self.progress, {**state, "next": index, "running": index, "timeout_s": int(seconds)})
        print(f"[run_smoke] case {index} did not finish within {seconds:.0f} s; stopping this process",
              file=sys.stderr, flush=True)
        faulthandler.dump_traceback(all_threads=True)
        os._exit(TIMED_OUT)


def _exit_restart() -> None:
    """End this process for a restart once rank 0 has written the group's progress and results
    (every rank waits here, so no rank's exit cuts rank 0's writes short)."""
    _agree_any(False)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(RESTART)  # skip vLLM's exit-time teardown of a broken state


def _agree_any(flag: bool) -> bool:
    from operatorx.runners.common import ranks
    try:
        return ranks.any_(flag)
    except Exception:  # noqa: BLE001 - a broken group: decide alone (the process restarts anyway)
        return flag


def _is_oom(e: BaseException) -> bool:
    """A device allocation failure (torch's, or one a kernel library reports in its message)."""
    try:
        import torch
        if isinstance(e, torch.OutOfMemoryError):
            return True
    except Exception:  # noqa: BLE001 - torch is not importable here
        pass
    text = str(e).lower()
    return "out of memory" in text or "hipErrorOutOfMemory".lower() in text


def _restart_reason(status: str, residents: list, oom: bool = False) -> str | None:
    """Why this process should not run another case, agreed by every rank; None to go on."""
    import gc

    from operatorx.runners.common import ranks
    reason = "the case raised an error" if status == "error" else "out of memory" if oom else None
    try:
        import torch
        # with a layer or engine cached, the failure was this case's own (e.g. a batch the engine
        # rejected); without one, whatever device memory remains is a failed build's leftovers
        if reason is None and torch.cuda.is_available() and not any(r() for r in residents):
            gc.collect()
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            used = total - free
            if used > max(_USED_FLOOR, _USED_FRACTION * total):
                reason = f"{used / 2**30:.1f} of {total / 2**30:.0f} GiB in use with nothing cached"
    except Exception as e:  # noqa: BLE001 - a device that cannot even report is not reusable
        reason = f"recovery check failed: {type(e).__name__}: {e}"[:300]
    try:
        if ranks.any_(reason is not None):
            return reason or "another rank must restart"
    except Exception as e:  # noqa: BLE001 - a broken group cannot run more cases
        return f"ranks could not agree: {type(e).__name__}: {e}"[:300]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--platform", default=None,
                    help="platform override (otherwise inferred from $OPERATORX_CLUSTER)")
    ap.add_argument("--testlists", default=None,
                    help="comma-separated testlist names; default = all under operatorx/testlists/")
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
    progress = Path(os.environ["OPERATORX_PROGRESS"]) if os.environ.get("OPERATORX_PROGRESS") else None
    if progress is not None:  # stacks on SIGUSR1: the supervisor dumps a rank it must stop
        import faulthandler
        import signal
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    start = 0
    residents = []
    retried = None
    failed_layers: dict[str, list] = {}
    repeats: dict[str, int] = {}  # cases of a layer that failed again in a fresh process
    if progress is not None:
        state = _read_progress(progress)
        start = int(state.get("next", 0))
        retried = state.get("retry")
        failed_layers = state.get("failed_layers", {})
        repeats = state.get("repeats", {})
        residents = [r for b in backends for r in [getattr(importlib.import_module(
            f"operatorx.runners.{platform}.backends.{b}"), "resident", None)] if r]
        crashed = state.get("running")
        restarted = os.environ.get("OPERATORX_CRASH_RC") == str(RESTART)  # not a crash
        if not restarted and crashed is not None and int(crashed) == start and start < len(entries) \
                and retried != start:
            # the first time a case kills or hangs the process, run it once more in this fresh
            # one (a cold compile or a stale state can be the cause), unless a case of its
            # layer already did
            if layer_key(entries[start][0].type, entries[start][0].args) not in repeats:
                retried = start
                _write_progress(progress, {**{k: v for k, v in state.items() if k != "running"},
                                           "retry": start})
                crashed = None
        if not restarted and crashed is not None and int(crashed) == start and start < len(entries):
            op, tl = entries[start]  # the process died (or hung) during this case
            rc = os.environ.get("OPERATORX_CRASH_RC", "?")
            what = (f"did not finish within {state.get('timeout_s', '?')} s (the harness stopped it)"
                    if rc == str(TIMED_OUT) else f"the process exited with code {rc} during this case")
            results.append(Result(op=op, metrics={}, status="error", testlist=tl, message=what))
            counts["error"] += 1
            # a layer two of whose cases kill or hang the process: the rest is recorded the same
            layer = layer_key(op.type, op.args)
            repeats[layer] = repeats.get(layer, 0) + 1
            if repeats[layer] >= 2:
                failed_layers[layer] = ["error", what]
            start += 1
        if start and rank == 0:
            print(f"[run_smoke] resuming at case {start} of {len(entries)}", flush=True)
        watchdog = _Watchdog(progress)
    first_case = True  # a process's first case run also compiles kernels and builds
    last_layer = None
    for index, (op, tl) in enumerate(entries):
        if index < start:
            continue
        layer = layer_key(op.type, op.args)
        if progress is not None:
            running = {"next": index, "running": index, "failed_layers": failed_layers, "repeats": repeats}
            _write_progress(progress, running)
            new_layer = first_case or layer != last_layer  # it builds and compiles
            watchdog.arm(index, _FIRST_CASE_TIMEOUT_S if new_layer else _CASE_TIMEOUT_S, running)
            first_case, last_layer = False, layer
        _t0 = _time.perf_counter()
        stage = ""
        oom = False
        if layer in failed_layers:  # its build failed in a fresh process too: not run again
            status, message = failed_layers[layer]
            results.append(Result(op=op, metrics={}, status=status, testlist=tl,
                                  message=f"{message} | not run: this layer's build failed in a fresh process"))
            counts[status] += 1
            if rank == 0:
                print(f"[ws={ws}] {tl:12} {status:11} {op.type:18} {op.backend:10}  (layer's build failed)", flush=True)
            if progress is not None:
                _write_progress(progress, {"next": index + 1, "failed_layers": failed_layers, "repeats": repeats})
            continue
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
        except FreshProcessError as e:  # before any result: a CI part restarts and runs it first
            if rank == 0:
                run.finished_at = utc_now_iso()
                write_run_result(out_path, run, results)
                print(f"[run_smoke] {counts} -> {out_path}", flush=True)
                print(f"[run_smoke] fresh process for case {index}: {e}", flush=True)
            _write_progress(progress, {"next": index, "failed_layers": failed_layers, "repeats": repeats})
            _exit_restart()
        except UnsupportedOpError as e:
            stage = getattr(e, "operatorx_stage", "")
            results.append(Result(op=op, metrics={}, status="unsupported",
                                  message=str(e), testlist=tl))
            counts["unsupported"] += 1
            status = "unsupported"
            latency_str = "         unsupported"
        except Exception as e:
            stage = getattr(e, "operatorx_stage", "")
            import traceback as _tb
            tb_tail = "".join(_tb.format_exception(type(e), e, e.__traceback__)).splitlines()[-6:]
            # the case does not fit this device: unsupported (a CI part first retries it in a
            # fresh process, so memory an earlier case left behind is not the cause)
            status = "unsupported" if _is_oom(e) else "error"
            oom = status == "unsupported"
            results.append(Result(op=op, metrics={}, status=status,
                                  message=("out of memory: " if oom else "")
                                  + f"{type(e).__name__}: {e} | tb: " + " || ".join(tb_tail),
                                  testlist=tl))
            counts[status] += 1
            latency_str = "  unsupported (OOM)" if oom else f"        ERROR ({type(e).__name__})"
        if progress is not None:
            watchdog.cancel()
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
            lane = f" lane={os.environ['OPERATORX_LANE']}" if os.environ.get("OPERATORX_LANES", "1") != "1" else ""
            print(f"[ws={ws}{lane}] {tl:12} {status:11} {op.type:18} {op.backend:10}  "
                  f"{latency_str}  wall={wall_s:6.1f}s   {shape_str}", flush=True)
        # release cached blocks, else the next big shape can OOM on the previous op's buffers
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass
        if progress is not None:
            _write_progress(progress, {"next": index + 1, "failed_layers": failed_layers, "repeats": repeats})
            reason = _restart_reason(status, residents, oom) if status != "ok" else None
            if reason is not None and retried != index:  # run this case once more, in a fresh process
                results.pop()
                counts[status] -= 1
                _write_progress(progress, {"next": index, "retry": index, "failed_layers": failed_layers,
                                           "repeats": repeats})
            elif reason is not None:
                # failed again in a fresh process: after a build failure, or a second such case,
                # the rest of the layer is recorded the same way without running
                repeats[layer] = repeats.get(layer, 0) + 1
                if _agree_any(stage.startswith("build")) or repeats[layer] >= 2:
                    failed_layers[layer] = [status, results[-1].message or ""]
                _write_progress(progress, {"next": index + 1, "failed_layers": failed_layers,
                                           "repeats": repeats})
            if reason is not None and (retried != index or index + 1 < len(entries)):
                if rank == 0:
                    run.finished_at = utc_now_iso()
                    write_run_result(out_path, run, results)
                    print(f"[run_smoke] {counts} -> {out_path}", flush=True)
                    print(f"[run_smoke] restarting after case {index}: {reason}", flush=True)
                _exit_restart()

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

    # a CI part's processes exit 0, so a lane or part with no supported cases neither stops its
    # siblings (srun --kill-on-bad-exit) nor fails the shard; the shard summary fails errors or
    # a shard without ok rows
    if os.environ.get("OPERATORX_PART") is not None:
        return 0
    return int(args.strict and (counts["error"] > 0 or counts["ok"] == 0))


if __name__ == "__main__":
    raise SystemExit(main())
