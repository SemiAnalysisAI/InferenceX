# SPDX-License-Identifier: GPL-3.0-only
"""Parallel Titanium supplies; fans and shelf controls are added before facility PUE."""

from typing import Annotated, Self

from pydantic import Field, model_validator

from power_model.base import (
    FrozenModel,
    PositiveCount,
    PowerComponentBreakdown,
    Provenance,
    Watts,
    validate_watts,
)
from power_model.models.advanced.components.psu import (
    TITANIUM_EFFICIENCY_CURVE,
    PSUEfficiencyPoint,
)
from power_model.models.advanced.components.rack_fans import RackFanAssembly, solve_cooling

SHELF_BASELINE = Provenance(
    profile_id="nvl72-titanium-shelves",
    version="1",
    kind="estimated",
    source="Shared Titanium assumption; NVIDIA and Vertiv 33 kW shelf specifications",
    input_boundary="Rack AC/DC conversion loss plus shelf fans and controls; excludes PUE",
    assumptions=(
        "Eight shelves, six 5500 W supplies each; four shelves of redundant capacity.",
        "All 48 supplies share load normally; active shelf count can model a feed failure.",
        "The shared project Titanium curve is an assumption, not measured NVL72 efficiency.",
        "One representative fan per PSU and 10 W controls per active shelf are provisional.",
        "Auxiliary wattages are referred to the shelf output, including auxiliary rail conversion.",
        "Unpowered shelves are off; energized standby policies require another profile.",
    ),
)


class RackPSU(FrozenModel):
    capacity_w: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 5500.0
    efficiency_curve: Annotated[tuple[PSUEfficiencyPoint, ...], Field(min_length=2)] = (
        TITANIUM_EFFICIENCY_CURVE
    )
    fans: RackFanAssembly = RackFanAssembly(fan_count=1, design_air_heat_w=250)
    provenance: Provenance = SHELF_BASELINE

    @model_validator(mode="after")
    def validate_curve(self) -> Self:
        if (
            self.efficiency_curve[0].load_fraction <= 0
            or self.efficiency_curve[-1].load_fraction != 1
        ):
            raise ValueError("Rack PSU efficiency curve must cover a positive minimum to full load")
        if any(
            left.load_fraction >= right.load_fraction
            for left, right in zip(self.efficiency_curve, self.efficiency_curve[1:], strict=False)
        ):
            raise ValueError("PSU load points must be strictly increasing")
        return self

    def efficiency(self, output_w: float) -> float:
        load = validate_watts(output_w) / self.capacity_w
        if load < self.efficiency_curve[0].load_fraction:
            minimum = self.efficiency_curve[0].load_fraction
            raise ValueError(
                f"Rack PSU load below {minimum:.0%} requires a calibrated low-load profile"
            )
        if load > 1:
            raise ValueError("Rack PSU output exceeds continuous capacity")
        for left, right in zip(self.efficiency_curve, self.efficiency_curve[1:], strict=False):
            if load <= right.load_fraction:
                weight = (load - left.load_fraction) / (right.load_fraction - left.load_fraction)
                return left.efficiency + weight * (right.efficiency - left.efficiency)
        raise ValueError("PSU curve does not cover output")

    def conversion_loss_w(self, output_w: float) -> float:
        output = validate_watts(output_w)
        return validate_watts(output * (1 / self.efficiency(output) - 1))

    def estimate_overhead(self, delivered_dc_w: float) -> PowerComponentBreakdown:
        fan = solve_cooling(delivered_dc_w, 0, self.fans, self.conversion_loss_w)
        output = validate_watts(delivered_dc_w + fan.power_w)
        efficiency = self.efficiency(output)
        loss = PowerComponentBreakdown(
            name="AC/DC conversion loss",
            power_w=self.conversion_loss_w(output),
            provenance=self.provenance,
        )
        return PowerComponentBreakdown.group(
            "Rack PSU overhead",
            (loss, fan),
            provenance=self.provenance,
            details=(
                ("dc_output_w_per_psu", output),
                ("efficiency", efficiency),
                ("load_fraction", output / self.capacity_w),
                ("capacity_w", self.capacity_w),
            ),
        )


class PowerShelf(FrozenModel):
    psu_count: PositiveCount = 6
    psu: RackPSU = RackPSU()
    controller_power_w: Watts = 10.0

    def estimate_overhead(self, delivered_dc_w: float) -> PowerComponentBreakdown:
        load = validate_watts(validate_watts(delivered_dc_w) + self.controller_power_w)
        return PowerComponentBreakdown.group(
            "Power shelf overhead",
            (
                self.psu.estimate_overhead(load / self.psu_count).scaled(self.psu_count),
                PowerComponentBreakdown(
                    name="Shelf controller",
                    power_w=self.controller_power_w,
                    provenance=SHELF_BASELINE,
                ),
            ),
            provenance=SHELF_BASELINE,
        )


class RackPowerSupply(FrozenModel):
    installed_shelves: PositiveCount = 8
    active_shelves: PositiveCount = 8
    surviving_shelves: PositiveCount = 4
    shelf: PowerShelf = PowerShelf()
    provenance: Provenance = SHELF_BASELINE

    @model_validator(mode="after")
    def validate_bank(self) -> Self:
        if not self.surviving_shelves <= self.active_shelves <= self.installed_shelves:
            raise ValueError("Shelf counts require surviving <= active <= installed")
        return self

    @property
    def redundant_capacity_w(self) -> float:
        return self.surviving_shelves * self.shelf.psu_count * self.shelf.psu.capacity_w

    def estimate_overhead(self, rack_bus_w: float) -> PowerComponentBreakdown:
        load = validate_watts(rack_bus_w)
        if load > self.redundant_capacity_w:
            raise ValueError("Rack bus load exceeds redundant power-shelf capacity")
        # Check the surviving bank with its own fan load, not the normal operating fan watts.
        reserve = self.shelf.estimate_overhead(load / self.surviving_shelves)
        current = self.shelf.estimate_overhead(load / self.active_shelves).scaled(
            self.active_shelves
        )
        return PowerComponentBreakdown(
            name="Power shelves",
            power_w=current.power_w,
            quantity=current.quantity,
            children=current.children,
            provenance=self.provenance,
            details=(
                ("rack_bus_w", load),
                ("rack_ac_w", load + current.power_w),
                ("installed_shelves", self.installed_shelves),
                ("active_shelves", self.active_shelves),
                ("redundant_capacity_w", self.redundant_capacity_w),
                ("feed_loss_ac_w", load + reserve.power_w * self.surviving_shelves),
            ),
        )
