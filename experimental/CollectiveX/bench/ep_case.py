"""Case identity, command-line inputs, and token ladders; safe without torch."""
from __future__ import annotations

import argparse
import re

_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
_NON_SLUG = re.compile(r"[^a-z0-9]+")


def is_case_id(value) -> bool:
    return bool(isinstance(value, str) and _CASE_ID.fullmatch(value))


def case_id(sku: str, case: dict) -> str:
    parts = (
        sku,
        case["backend"],
        case["workload"],
        case["mode"],
        case["phase"],
        f"ep{int(case['ep'])}",
        case["routing"],
        case["precision"],
    )
    values = [_NON_SLUG.sub("-", str(part).lower()).strip("-") for part in parts]
    if not all(values):
        raise ValueError("case ID contains an empty factor")
    return "-".join(values)


def format_collective_version(raw) -> str:
    """Normalize PyTorch's tuple or packed NCCL/RCCL version representation."""
    if isinstance(raw, int):
        if raw < 10_000:
            return f"{raw // 1000}.{raw // 100 % 10}.{raw % 100}"
        return f"{raw // 10_000}.{raw // 100 % 100}.{raw % 100}"
    if isinstance(raw, (tuple, list)):
        return ".".join(map(str, raw))
    return str(raw) if raw not in (None, "") else "unknown"


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """Add the varying v1 inputs; fixed profile values are not CLI axes."""
    ap.add_argument("--mode", required=True, choices=["normal", "low-latency"])
    ap.add_argument("--precision", required=True, choices=["bf16", "fp8"],
                    help="dispatch payload precision; combine is always BF16")
    ap.add_argument("--phase", required=True, choices=["decode", "prefill"],
                    help="token-size regime label: decode (small T) / prefill (large T)")
    ap.add_argument("--tokens-ladder", required=True,
                    help="space/comma-separated source-tokens-per-rank sweep; the matrix "
                         "supplies the workload's phase ladder from configs/sweep.json")
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--topk", type=int, required=True)
    ap.add_argument("--experts", type=int, required=True,
                    help="TOTAL experts (fixed across EP degrees)")
    ap.add_argument("--routing", required=True, choices=["uniform"])
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--workload-name", required=True)
    ap.add_argument("--seed", type=int, required=True,
                    help="routing-trace seed; part of the workload identity in configs/sweep.json")
    ap.add_argument(
        "--version",
        type=int,
        required=True,
        help="iterable benchmark version copied verbatim into the emitted result",
    )
    # The single cross-SKU profile lives in configs/sweep.json
    # `timing:`; the matrix bakes it into every scheduled case.
    ap.add_argument("--warmup", type=int, required=True,
                    help="untimed full roundtrips before each trial/point")
    ap.add_argument("--iters", type=int, required=True,
                    help="timed iterations per trial")
    ap.add_argument("--trials", type=int, required=True,
                    help="timed trials")
    # Chain sampling on its own knobs: one call already yields chain_iters free-running pairs, so
    # it converges in far fewer trials than the fresh-entry components. The matrix bakes these from
    # configs/sweep.json `timing:`; the defaults match it, for cases scheduled before the fields.
    ap.add_argument("--chain-iters", type=int, default=128,
                    help="free-running dispatch->combine pairs per chain trial")
    ap.add_argument("--chain-trials", type=int, default=4,
                    help="chain trials per ladder point")
    ap.add_argument("--chain-drop", type=int, default=16,
                    help="head pairs discarded per chain trial (pipeline fill, not period)")
    # provenance / output
    ap.add_argument("--runner", required=True)
    ap.add_argument("--topology-class", required=True)
    ap.add_argument("--transport", required=True)
    ap.add_argument("--scope", required=True, choices=["scale-up", "scale-out"])
    ap.add_argument("--scale-up-transport", required=True)
    ap.add_argument("--scale-out-transport", required=True)
    ap.add_argument("--gpus-per-node", type=int, required=True)
    ap.add_argument("--scale-up-domain", type=int, required=True)
    ap.add_argument("--out", required=True)


def token_ladder(spec: str, cap: int | None) -> tuple[list[int], list[int]]:
    """Return (ladder, dropped) from an explicit spec (there is no default — the
    model-specific ladders live in configs/sweep.json); positive ints; clamped to
    `cap` with dropped points reported (never silently truncated)."""
    want = sorted({t for t in (int(t) for t in spec.replace(",", " ").split() if t) if t > 0})
    if cap is not None:
        return [t for t in want if t <= cap], [t for t in want if t > cap]
    return want, []
