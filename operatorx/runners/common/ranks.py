"""Agreement and reduction across ranks; no-ops on one rank.

Every rank must take the same branch wherever a collective launches, or the others hang.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch

from operatorx.core import UnsupportedOpError


def _group():
    import torch.distributed as dist
    return dist if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1 else None


def world() -> int:
    d = _group()
    return d.get_world_size() if d else 1


def agree(ok: bool) -> bool:
    """True only when every rank passes True."""
    d = _group()
    if d is None:
        return ok
    t = torch.tensor([1 if ok else 0], dtype=torch.int32, device="cuda")
    d.all_reduce(t, op=d.ReduceOp.MIN)
    return bool(t.item())


def any_(flag: bool) -> bool:
    return not agree(not flag)


_TOKEN: dict[int, torch.Tensor] = {}


def align(cycles: int) -> None:
    """Device-side rank barrier then a spin, without a host sync (CollectiveX _graph_align)."""
    d = _group()
    if d is None:
        return
    dev = torch.cuda.current_device()
    if dev not in _TOKEN:
        _TOKEN[dev] = torch.zeros(1, device="cuda")
    d.all_reduce(_TOKEN[dev])
    torch.cuda._sleep(cycles)


@contextmanager
def together(what: str):
    """Every rank reports once whether it got through; all stop if any did not."""
    try:
        yield
    except BaseException:
        agree(False)
        raise
    if not agree(True):
        raise UnsupportedOpError(f"another rank could not {what}")


LAST: dict = {}  # the last timing's reduction, for summary()


def iterations(times: list[float]) -> dict:
    """Per-iteration latencies reduced across ranks (CollectiveX ep_harness._reduce_vec)."""
    d = _group()
    if d is None:
        out = {"max": list(times), "min": list(times), "spread": [0.0] * len(times)}
    else:
        from operatorx.runners.common import collectivex
        h = collectivex.harness()
        dev = torch.device("cuda", torch.cuda.current_device())
        hi = h._reduce_vec(torch, d, dev, times, d.ReduceOp.MAX)
        lo = h._reduce_vec(torch, d, dev, times, d.ReduceOp.MIN)
        out = {"max": hi, "min": lo, "spread": [a - b for a, b in zip(hi, lo)]}
    LAST.clear()
    LAST.update(out, own=list(times))
    return out


def summary(metrics: dict) -> dict | None:
    """Every rank's latency, timeline and telemetry, gathered on rank 0; None on one device."""
    d = _group()
    if d is None:
        return None

    def med(xs):
        return sorted(xs)[len(xs) // 2] if xs else None

    mine = {"rank": d.get_rank(), "latency_us": med(LAST.get("own", [])),
            "profile": metrics.get("profile"), "telemetry": metrics.get("telemetry")}
    gathered = [None] * d.get_world_size() if d.get_rank() == 0 else None
    d.gather_object(mine, gathered, dst=0)
    capped = [p["rank"] for p in gathered or () if (p.get("telemetry") or {}).get("capped")]
    return {"world": d.get_world_size(), "latency_us_min": med(LAST.get("min", [])),
            "skew_us": med(LAST.get("spread", [])), "capped_ranks": capped, "per_rank": gathered}
