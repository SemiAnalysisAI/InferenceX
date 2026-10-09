# SPDX-License-Identifier: GPL-3.0-only
"""An example using a one-GPU reference scope and the agreed 1.5 IT multiplier."""

from typing import Self

from pydantic import model_validator

from power_model.base import ITPowerBreakdown, PowerComponentBreakdown, PowerModel, Provenance


class ExamplePowerModel(PowerModel):
    @model_validator(mode="after")
    def reject_cpu_offload(self) -> Self:
        if self.cpu_offload:
            raise ValueError("CPU offloading requires OSSAllinPowerModel with DDR5 inventory")
        return self

    def _estimate_it_power(
        self, gpu_level_power_per_gpu: float, *, cpu_socket_measured_power: float | None
    ) -> ITPowerBreakdown:
        if cpu_socket_measured_power is not None:
            raise ValueError("cpu_socket_measured_power requires the oss model's Grace inventory")
        return ITPowerBreakdown(
            scope="per_gpu_reference",
            gpu_count=1,
            components=(
                PowerComponentBreakdown(
                    name="GPU",
                    power_w=gpu_level_power_per_gpu,
                    provenance=Provenance(
                        profile_id="gpu-input",
                        version="1",
                        source="gpu_level_power_per_gpu argument",
                        kind="input",
                        input_boundary="GPU electrical power, excluding other IT components",
                    ),
                ),
                PowerComponentBreakdown(
                    name="Estimated non-GPU IT power",
                    power_w=gpu_level_power_per_gpu * 0.5,
                    provenance=Provenance(
                        profile_id="example",
                        version="1",
                        source="User-specified 1.5 IT multiplier",
                        kind="assumed",
                        input_boundary="All non-GPU IT power, including networking",
                        assumptions=("Non-GPU IT power is 0.5 times GPU power.",),
                    ),
                ),
            ),
        )
