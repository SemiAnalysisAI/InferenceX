"""Parallelism calculations shared by fixed-sequence and AgentX results.

Callers own environment parsing, allocation counts, and validation order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, kw_only=True)
class Parallelism:
    tp: int
    pp: int = 1
    dcp_size: int = 1
    pcp_size: int = 1
    ep: int = 1

    @property
    def gpus_per_worker(self) -> int:
        """EP and DCP share GPUs; only TP, PP, and PCP multiply allocation."""
        return self.tp * self.pp * self.pcp_size

    def for_decode(self, num_gpus: int) -> Parallelism:
        """An aggregate worker has no separate decode parallelism."""
        return self if num_gpus > 0 else Parallelism(tp=0, ep=0)

    def fields(self, prefix: str = "") -> dict[str, int]:
        """Return fresh result fields in their existing serialization order."""
        return {
            f"{prefix}tp": self.tp,
            f"{prefix}pp": self.pp,
            f"{prefix}dcp_size": self.dcp_size,
            f"{prefix}pcp_size": self.pcp_size,
            f"{prefix}ep": self.ep,
        }


def validate_parallelism(
    *layouts: Parallelism,
    error_type: type[BaseException] = ValueError,
) -> None:
    """Validate PP/DCP/PCP after the caller has parsed all its inputs."""
    if any(
        size <= 0 for layout in layouts for size in (layout.pp, layout.dcp_size, layout.pcp_size)
    ):
        dimensions = (
            "Multinode PP, DCP, and PCP sizes"
            if len(layouts) > 1
            else "PP_SIZE, DCP_SIZE, and PCP_SIZE"
        )
        raise error_type(f"{dimensions} must be positive integers.")


def as_int(x: Any, default: int = 0) -> int:
    """Convert a metadata field to int with a fallback."""
    try:
        return int(x)
    except Exception:  # noqa: BLE001
        return default


def as_bool(x: Any, default: bool = False) -> bool:
    """Parse a metadata boolean stored as bool/string/int."""
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    return str(x).lower() == "true"


def eval_topology(meta: dict[str, Any]) -> dict[str, Any]:
    """Physical topology for explicitly typed evals; leave legacy inference alone."""
    if "disagg" not in meta:
        return {}

    def layout(prefix: str = "") -> Parallelism:
        return Parallelism(
            **{
                name: as_int(meta.get(f"{prefix}{name}", meta.get(name, 1)), 1)
                for name in ("tp", "pp", "dcp_size", "pcp_size", "ep")
            }
        )

    disagg = as_bool(meta["disagg"])
    result = {"disagg": disagg, **layout().fields()}
    if disagg:
        for role in ("prefill", "decode"):
            prefix = f"{role}_"
            parallelism = layout(prefix)
            workers = as_int(meta.get(f"{prefix}num_workers", 1), 1)
            result.update(parallelism.fields(prefix))
            result[f"{prefix}num_workers"] = workers
            result[f"num_{role}_gpu"] = parallelism.gpus_per_worker * workers
        result["num_gpus"] = result["num_prefill_gpu"] + result["num_decode_gpu"]
    else:
        multinode = as_bool(meta.get("is_multinode"))
        parallelism = layout("prefill_") if multinode else layout()
        workers = as_int(meta.get("prefill_num_workers", 1), 1) if multinode else 1
        result.update(parallelism.fields())
        result["num_gpus"] = parallelism.gpus_per_worker * workers
        for role in ("prefill", "decode"):
            result.update(parallelism.fields(f"{role}_"))
            result[f"{role}_num_workers"] = 0
        if multinode:
            result["prefill_num_workers"] = workers
            result["num_prefill_gpu"] = result["num_gpus"]
            result["num_decode_gpu"] = 0
            result.update(parallelism.for_decode(0).fields("decode_"))
    return result
