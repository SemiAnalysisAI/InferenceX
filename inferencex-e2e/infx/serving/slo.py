"""SLO tuples for sweep scenarios: ``metric:stat:op:threshold`` joined by ``;``.

    python -m infx.serving.slo "time_to_first_token:avg:lt:8000;inter_token_latency:p90:lt:50"

prints one tuple per line (the form AIPerf's ``--search-sla`` takes), or exits 2.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass

STATS = ("avg", "p50", "p90", "p95", "p99")
OPS = ("lt", "le", "gt", "ge")
_METRIC = re.compile(r"[a-z][a-z0-9_]*")


class SLOError(ValueError):
    pass


@dataclass(frozen=True)
class SLO:
    metric: str
    stat: str
    op: str
    threshold: float

    def tuple(self) -> str:
        return f"{self.metric}:{self.stat}:{self.op}:{self.threshold:g}"


def parse_slo(text: str) -> list[SLO]:
    parts = [part.strip() for part in text.split(";") if part.strip()]
    if not parts:
        raise SLOError("slo: empty; give metric:stat:op:threshold[;...]")
    slos = []
    for part in parts:
        fields = part.split(":")
        if len(fields) != 4:
            raise SLOError(f"slo: {part!r} must be metric:stat:op:threshold")
        metric, stat, op, raw = (field.strip() for field in fields)
        if not _METRIC.fullmatch(metric):
            raise SLOError(f"slo: metric {metric!r} must be a lowercase AIPerf metric tag")
        if stat not in STATS:
            raise SLOError(f"slo: stat {stat!r} must be one of {', '.join(STATS)}")
        if op not in OPS:
            raise SLOError(f"slo: op {op!r} must be one of {', '.join(OPS)}")
        try:
            threshold = float(raw)
        except ValueError:
            raise SLOError(f"slo: threshold {raw!r} is not a number") from None
        if not math.isfinite(threshold) or threshold <= 0:
            raise SLOError(f"slo: threshold {raw!r} must be a positive finite number")
        slos.append(SLO(metric, stat, op, threshold))
    return slos


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m infx.serving.slo '<tuples>'", file=sys.stderr)
        return 2
    try:
        slos = parse_slo(argv[0])
    except SLOError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 2
    for slo in slos:
        print(slo.tuple())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
