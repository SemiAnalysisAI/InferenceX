"""Parallelism calculations shared by fixed-sequence and AgentX results.

Callers own environment parsing, allocation counts, and validation order.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    *layouts: Parallelism, error_type: type[BaseException] = ValueError,
) -> None:
    """Validate PP/DCP/PCP after the caller has parsed all its inputs."""
    if any(size <= 0 for layout in layouts for size in (layout.pp, layout.dcp_size, layout.pcp_size)):
        dimensions = (
            "Multinode PP, DCP, and PCP sizes" if len(layouts) > 1
            else "PP_SIZE, DCP_SIZE, and PCP_SIZE"
        )
        raise error_type(f"{dimensions} must be positive integers.")
