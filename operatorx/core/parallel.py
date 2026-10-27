"""How an op is split over the devices of one node.

    "parallel": {"tp": T, "dp": D, "ep": E, "dcp": C}   (each optional, default 1)

  tp: weights (attention: heads) split T ways; the reduction is part of the op.
  dp: D groups each bring their own batch (MoE experts see D x tokens in total).
  ep: whole experts partitioned E ways; E is 1 or T x D.
  dcp: each request's KV cache split C ways within a tp group, outputs merged; C divides T.

Runs on T x D devices. Args keep the full unsplit shape and each dp group's batch; how
the split is carried out is the backend's choice. No "parallel": one device.
"""
from __future__ import annotations

from typing import Any

AXES = ("tp", "dp", "ep", "dcp")
GEMM_AXES = ("tp",)
MOE_AXES = ("tp", "dp", "ep")
ATTENTION_AXES = ("tp", "dp", "dcp")


def check(p: Any, allowed: tuple[str, ...] = AXES) -> None:
    if p is None:
        return
    if not isinstance(p, dict) or set(p) - set(allowed):
        raise ValueError(f"parallel must be a dict with keys from {list(allowed)}, got {p!r}")
    for k, v in p.items():
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise ValueError(f"parallel.{k} must be a positive int, got {v!r}")
    if p.get("ep", 1) not in (1, world_size(p)):
        raise ValueError("parallel.ep must be 1 or tp x dp")
    if p.get("tp", 1) % p.get("dcp", 1):
        raise ValueError("parallel.dcp must divide parallel.tp")


def world_size(p: dict | None) -> int:
    p = p or {}
    return p.get("tp", 1) * p.get("dp", 1)


def normalize(p: dict | None) -> dict:
    """Every axis spelled out: the key a process's parallel state is built for."""
    return {k: (p or {}).get(k, 1) for k in AXES}
