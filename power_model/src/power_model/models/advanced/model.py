# SPDX-License-Identifier: GPL-3.0-only
"""The advanced implementation delegates IT accounting to its configured cluster."""

from typing import Self

from power_model.base import ITPowerBreakdown, PowerModel, WorkloadState
from power_model.models.advanced.cluster import Cluster
from power_model.models.advanced.systems.base import GPUSystem, SystemGroup
from power_model.models.advanced.systems.catalog import get_system_class


class OSSAllinPowerModel(PowerModel):
    cluster: Cluster

    @classmethod
    def for_system(
        cls,
        system: str | type[GPUSystem],
        *,
        workload_state: WorkloadState | str = WorkloadState.FIXED_SEQ_LEN,
        using_scale_out: bool = False,
        systems: int = 1,
    ) -> Self:
        system_class = get_system_class(system)
        hardware = system_class()
        system_group = SystemGroup(system=hardware, quantity=systems)
        if hardware.default_networking is None:
            raise ValueError(f"{hardware.system_type} must define its scale-out network model")
        networking = tuple(
            group.scaled(system_group.quantity) for group in hardware.default_networking
        )
        return cls(
            cooling=system_class.default_cooling,
            workload_state=workload_state,
            using_scale_out=using_scale_out,
            cluster=Cluster(
                systems=(system_group,),
                networking=networking,
            ),
        )

    def _estimate_it_power(
        self, gpu_level_power_per_gpu: float, *, cpu_socket_measured_power: float | None
    ) -> ITPowerBreakdown:
        return self.cluster.estimate_it_power(
            gpu_level_power_per_gpu,
            cpu_socket_measured_power=cpu_socket_measured_power,
            operating_state=self.operating_state,
        )
