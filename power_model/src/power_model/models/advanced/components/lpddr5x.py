# SPDX-License-Identifier: GPL-3.0-only
"""Whole Grace memory-pool power from physical DRAM traffic."""

from typing import Literal, Self

from pydantic import model_validator

from power_model.base import Details, Provenance, Watts, validate_watts
from power_model.models.advanced.components.base import GenericComponent


class LPDDR5XMemory(GenericComponent):
    capacity_gb: Literal[512] = 512
    frequency_mhz: Literal[3200] = 3200
    bandwidth_gbps: Watts
    provenance: Provenance = Provenance(
        profile_id="grace-512gb-lpddr5x",
        version="1",
        kind="estimated",
        source="https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/power-thermals.html",
        input_boundary="Entire 512 GB pool attached to one Grace CPU; excludes regulator losses",
        assumptions=(
            "Published 512 GB / 3200 MHz coefficients; bandwidth is total DRAM GB/s.",
            "Model domain is 0 to 384 GB/s; workload bandwidths are estimates.",
            "Use on GB300 is provisional pending qualification of its memory configuration.",
        ),
    )

    @model_validator(mode="after")
    def validate_bandwidth(self) -> Self:
        if self.bandwidth_gbps > 384:
            raise ValueError("512 GB LPDDR5X profile supports bandwidth from 0 to 384 GB/s")
        return self

    def estimate_w(self) -> float:
        b = self.bandwidth_gbps
        return validate_watts((-0.0000603 * b * b + 56.2 * b + 3396) / 1000)

    def _details(self) -> Details:
        return (
            ("capacity_gb_per_pool", self.capacity_gb),
            ("frequency_mhz", self.frequency_mhz),
            ("bandwidth_gbps", self.bandwidth_gbps),
            ("scope", "one Grace memory pool"),
        )
