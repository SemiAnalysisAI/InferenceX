# SPDX-License-Identifier: GPL-3.0-only
"""Cluster aggregation over actual system quantities and allocated network gear."""

from typing import Annotated

from pydantic import Field

from power_model.base import (
    DEFAULT_OPERATING_STATE,
    FrozenModel,
    ITPowerBreakdown,
    OperatingState,
    PowerComponentBreakdown,
    validate_watts,
)
from power_model.models.advanced.networking import NetworkGroup
from power_model.models.advanced.systems.base import SystemGroup


class Cluster(FrozenModel):
    systems: Annotated[tuple[SystemGroup, ...], Field(min_length=1)]
    networking: tuple[NetworkGroup, ...]

    @property
    def gpu_count(self) -> int:
        return sum(group.gpu_count for group in self.systems)

    def estimate_it_power(
        self,
        gpu_level_power_per_gpu: float,
        *,
        cpu_and_dram_measured_power_per_socket: float | None = None,
        operating_state: OperatingState = DEFAULT_OPERATING_STATE,
    ) -> ITPowerBreakdown:
        per_gpu = validate_watts(gpu_level_power_per_gpu)
        if cpu_and_dram_measured_power_per_socket is not None and not any(
            group.system.has_grace_sockets for group in self.systems
        ):
            raise ValueError(
                "cpu_and_dram_measured_power_per_socket requires a system with Grace sockets"
            )
        return ITPowerBreakdown(
            scope="cluster",
            gpu_count=self.gpu_count,
            components=(
                PowerComponentBreakdown.group(
                    "GPU systems",
                    tuple(
                        group.estimate_it_power(
                            per_gpu,
                            cpu_and_dram_measured_power_per_socket=cpu_and_dram_measured_power_per_socket,
                            operating_state=operating_state,
                        )
                        for group in self.systems
                    ),
                ),
                PowerComponentBreakdown.group(
                    "Scale-out networking",
                    tuple(
                        group.estimate_breakdown(operating_state=operating_state)
                        for group in self.networking
                    ),
                ),
            ),
        )
