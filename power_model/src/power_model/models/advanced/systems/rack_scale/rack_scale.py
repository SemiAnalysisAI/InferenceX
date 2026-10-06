# SPDX-License-Identifier: GPL-3.0-only
"""Compose fixed NVL72 tray counts and account for rack electrical stages once."""

from typing import ClassVar

from pydantic import Field, SerializeAsAny

from power_model.base import (
    DEFAULT_OPERATING_STATE,
    OperatingState,
    PositiveCount,
    PowerComponentBreakdown,
    Provenance,
    sum_watts,
    validate_watts,
)
from power_model.cooling import CoolingProfile
from power_model.models.advanced.components.rack_power_supply import RackPowerSupply
from power_model.models.advanced.systems.base import GPUSystem

from .compute_tray import ComputeTray
from .nvswitch_tray import NVSwitchTray


class RackScaleSystem(GPUSystem):
    default_cooling: ClassVar[CoolingProfile] = CoolingProfile(mode="liquid")
    system_unit: ClassVar[str] = "rack"
    gpu_count: PositiveCount = Field(default=72, ge=72, le=72)
    compute_tray_count: PositiveCount = Field(default=18, ge=18, le=18)
    switch_tray_count: PositiveCount = Field(default=9, ge=9, le=9)
    compute_tray: SerializeAsAny[ComputeTray]
    switch_tray: NVSwitchTray = NVSwitchTray()
    power_supply: RackPowerSupply = RackPowerSupply()
    provenance: Provenance = Provenance(
        profile_id="nvl72-rack",
        version="2",
        kind="estimated",
        source="Selected NVL72 project BoM",
        input_boundary="Rack AC input including tray conversion and power shelves; excludes PUE",
        assumptions=(
            "18 compute trays, 9 switch trays, 72 GPUs, 36 Grace CPUs and 36 memory pools.",
            "This is an estimated selected-inventory profile, not a complete measured OEM rack.",
            "Fan and tray-converter curves are provisional shared assumptions.",
            "Each compute tray has one converter; Bianca boards have no separate conversion stage.",
            "Rack management switches and unspecified OEM auxiliary equipment are outside scope.",
        ),
    )

    def estimate_it_power(
        self,
        gpu_level_power_per_gpu: float,
        *,
        operating_state: OperatingState = DEFAULT_OPERATING_STATE,
    ) -> PowerComponentBreakdown:
        per_gpu = validate_watts(gpu_level_power_per_gpu)
        state = OperatingState.model_validate(operating_state)
        compute = self.compute_tray.estimate_breakdown(per_gpu, operating_state=state).scaled(
            self.compute_tray_count
        )
        switches = self.switch_tray.estimate_breakdown().scaled(self.switch_tray_count)
        bus_w = sum_watts((compute.power_w, switches.power_w))
        shelves = self.power_supply.estimate_overhead(bus_w)
        return PowerComponentBreakdown.group(
            self.system_type,
            (compute, switches, shelves),
            provenance=self.provenance,
            details=(
                ("gpu_count_per_system", self.gpu_count),
                ("system_unit", "rack"),
            ),
            configuration=self._configuration(state),
        )
