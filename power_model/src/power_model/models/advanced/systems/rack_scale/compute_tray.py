# SPDX-License-Identifier: GPL-3.0-only
"""Four-GPU compute trays, with hardware selection owned by the system models."""

from typing import ClassVar, Self

from pydantic import Field, SerializeAsAny, model_validator

from power_model.base import (
    DEFAULT_OPERATING_STATE,
    FrozenModel,
    OperatingState,
    PowerComponentBreakdown,
    Provenance,
    sum_watts,
)
from power_model.models.advanced.components.bianca import (
    BiancaBoard,
    GB200BiancaBoard,
    GB300BiancaBoard,
)
from power_model.models.advanced.components.dc_converter import (
    DCConverterAssembly,
    compute_tray_converter,
)
from power_model.models.advanced.components.generic import (
    NVMeDrive,
    SystemSide400GTransceiver,
    SystemSide800GTransceiver,
    SystemSideTransceiver,
)
from power_model.models.advanced.components.rack_fans import RackFanAssembly, solve_cooling
from power_model.models.advanced.components.scaleoutnic import (
    ConnectX7NICCard,
    ConnectX8NICCard,
    ScaleOutNICCard,
)


class ComputeTray(FrozenModel):
    board: SerializeAsAny[BiancaBoard]
    nic_type: ClassVar[type[ScaleOutNICCard]]
    optic_type: ClassVar[type[SystemSideTransceiver]]
    board_count: ClassVar[int] = 2
    nic_count: ClassVar[int] = 4
    nvme_count: ClassVar[int] = 8
    fans: RackFanAssembly = RackFanAssembly(fan_count=8, design_air_heat_w=500)
    converter: DCConverterAssembly = Field(default_factory=compute_tray_converter)
    provenance: Provenance = Provenance(
        profile_id="nvl72-compute-tray",
        version="2",
        kind="estimated",
        source="User-specified tray inventory and shared engineering thermal profile",
        input_boundary="One compute-tray 48 V-class bus input; excludes rack AC/DC and PUE",
        assumptions=(
            "Two Bianca boards, four NICs and optics, eight NVMe drives and eight fan modules.",
            "One 48 V-class converter serves the complete tray; 8 kW sizing is provisional.",
            "CPU/GPU chip heat goes to liquid; memory, NICs, optics, drives and losses go to air.",
            "500 W air-load fan anchor is shared by GB200 and GB300; OEM coverage may differ.",
            "Additional OEM DPUs, BMCs and boot devices outside this selected BoM are not modeled.",
        ),
    )

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        if self.fans.fan_count != 8:
            raise ValueError("A compute tray must contain exactly eight fan modules")
        return self

    def estimate_breakdown(
        self,
        gpu_level_power_per_gpu: float,
        *,
        cpu_and_dram_measured_power_per_socket: float | None = None,
        operating_state: OperatingState = DEFAULT_OPERATING_STATE,
    ) -> PowerComponentBreakdown:
        state = OperatingState.model_validate(operating_state)
        board = self.board.estimate_breakdown(
            gpu_level_power_per_gpu,
            cpu_and_dram_measured_power_per_socket=cpu_and_dram_measured_power_per_socket,
            operating_state=state,
        )
        nics = self.nic_type(state=state.nic_state).estimate_breakdown().scaled(self.nic_count)
        optics = self.optic_type().estimate_breakdown().scaled(self.nic_count)
        drives = NVMeDrive().estimate_breakdown().scaled(self.nvme_count)
        components = (board.scaled(self.board_count), nics, optics, drives)
        load = sum_watts(tuple(component.power_w for component in components))
        air = sum_watts(
            (
                self.board_count * dict(board.details)["air_heat_w"],
                nics.power_w,
                optics.power_w,
                drives.power_w,
            )
        )
        fans = solve_cooling(load, air, self.fans, self.converter.loss_w)
        loss = self.converter.estimate_loss_breakdown(load + fans.power_w)
        return PowerComponentBreakdown.group(
            type(self).__name__,
            (*components, fans, loss),
            provenance=self.provenance,
            details=(("gpu_count_per_tray", 4), ("dc_component_power_w", load)),
        )


class GB200ComputeTray(ComputeTray):
    board: SerializeAsAny[GB200BiancaBoard] = GB200BiancaBoard()
    nic_type: ClassVar[type[ScaleOutNICCard]] = ConnectX7NICCard
    optic_type: ClassVar[type[SystemSideTransceiver]] = SystemSide400GTransceiver


class GB300ComputeTray(ComputeTray):
    board: SerializeAsAny[GB300BiancaBoard] = GB300BiancaBoard()
    nic_type: ClassVar[type[ScaleOutNICCard]] = ConnectX8NICCard
    optic_type: ClassVar[type[SystemSideTransceiver]] = SystemSide800GTransceiver
