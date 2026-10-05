# SPDX-License-Identifier: GPL-3.0-only
"""HGX PSU load sharing, efficiency, and capacity limits."""

from typing import Annotated, Self

from pydantic import Field, model_validator

from power_model.base import (
    FrozenModel,
    PositiveCount,
    PowerComponentBreakdown,
    Provenance,
    validate_watts,
)
from power_model.models.advanced.profiles import COMPONENT_INPUT

PositivePower = Annotated[float, Field(gt=0, allow_inf_nan=False)]
PositiveFraction = Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]


def _psu_provenance(family: str) -> Provenance:
    return Provenance(
        profile_id=f"{family}-psu",
        version="1",
        source="HGX PSU model baseline",
        kind="estimated",
        input_boundary="Chassis AC-to-DC PSU conversion loss; DC load includes fans, excludes PUE",
        assumptions=(
            "Efficiency and capacity are modeling assumptions, not measured chassis profiles.",
            "All HGX families use the same Titanium-class efficiency curve, with "
            "system-specific PSU counts, load sharing, and capacity limits.",
        ),
    )


class PSUEfficiencyPoint(FrozenModel):
    load_fraction: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    efficiency: PositiveFraction


TITANIUM_EFFICIENCY_CURVE = tuple(
    PSUEfficiencyPoint(load_fraction=load, efficiency=efficiency)
    for load, efficiency in (
        (0.05, 0.8864),
        (0.10, 0.9238),
        (0.20, 0.9448),
        (0.50, 0.9640),
        (1.0, 0.9556),
    )
)


class PSUEfficiencyModel(FrozenModel):
    n_installed_psu: PositiveCount
    n_load_sharing_psu: PositiveCount
    n_redundant_capacity_psu: PositiveCount
    psu_capacity_w: PositivePower
    system_max_w: PositivePower | None = None
    redundancy: str
    efficiency_curve: Annotated[tuple[PSUEfficiencyPoint, ...], Field(min_length=2)] = (
        TITANIUM_EFFICIENCY_CURVE
    )
    provenance: Provenance = COMPONENT_INPUT

    @model_validator(mode="after")
    def validate_bank(self) -> Self:
        if not self.n_redundant_capacity_psu <= self.n_load_sharing_psu <= self.n_installed_psu:
            raise ValueError("PSU counts must satisfy redundant <= load-sharing <= installed")
        if any(
            left.load_fraction >= right.load_fraction
            for left, right in zip(self.efficiency_curve, self.efficiency_curve[1:], strict=False)
        ):
            raise ValueError("PSU efficiency load fractions must be strictly increasing")
        validate_watts(self.installed_capacity_w)
        return self

    @property
    def installed_capacity_w(self) -> float:
        return self.n_installed_psu * self.psu_capacity_w

    @property
    def load_sharing_capacity_w(self) -> float:
        return self.n_load_sharing_psu * self.psu_capacity_w

    @property
    def redundant_capacity_w(self) -> float:
        return self.n_redundant_capacity_psu * self.psu_capacity_w

    @property
    def modeled_capacity_w(self) -> float:
        if self.system_max_w is None:
            return self.redundant_capacity_w
        return min(self.system_max_w, self.redundant_capacity_w)

    def efficiency(self, dc_load_w: float) -> float:
        load = validate_watts(dc_load_w)
        if load > self.modeled_capacity_w:
            raise ValueError(
                f"DC load {load:g} W exceeds modeled PSU capacity "
                f"{self.modeled_capacity_w:g} W ({self.redundancy})"
            )
        fraction = load / self.load_sharing_capacity_w
        first, last = self.efficiency_curve[0], self.efficiency_curve[-1]
        if fraction <= first.load_fraction:
            return first.efficiency
        if fraction >= last.load_fraction:
            return last.efficiency
        for left, right in zip(self.efficiency_curve, self.efficiency_curve[1:], strict=False):
            if fraction <= right.load_fraction:
                weight = (fraction - left.load_fraction) / (
                    right.load_fraction - left.load_fraction
                )
                return left.efficiency + weight * (right.efficiency - left.efficiency)
        raise ValueError("PSU efficiency curve does not cover the supplied load")

    def estimate_loss_breakdown(self, dc_load_w: float) -> PowerComponentBreakdown:
        load = validate_watts(dc_load_w)
        efficiency = self.efficiency(load)
        ac_power = validate_watts(load / efficiency)
        return PowerComponentBreakdown(
            name="Power conversion losses",
            power_w=ac_power - load,
            provenance=self.provenance,
            details=(
                ("model", type(self).__name__),
                ("dc_load_w_per_system", load),
                ("ac_wall_w_per_system", ac_power),
                ("efficiency", efficiency),
                ("load_sharing_load_fraction", load / self.load_sharing_capacity_w),
                ("load_sharing_capacity_w", self.load_sharing_capacity_w),
                ("redundant_capacity_w", self.redundant_capacity_w),
                ("modeled_capacity_w", self.modeled_capacity_w),
                ("installed_capacity_w", self.installed_capacity_w),
                ("n_installed_psu", self.n_installed_psu),
                ("n_load_sharing_psu", self.n_load_sharing_psu),
                ("redundancy", self.redundancy),
            ),
        )


class HopperPSUEfficiency(PSUEfficiencyModel):
    n_installed_psu: PositiveCount = 6
    n_load_sharing_psu: PositiveCount = 6
    n_redundant_capacity_psu: PositiveCount = 4
    psu_capacity_w: PositivePower = 3300.0
    redundancy: str = "4+2"
    provenance: Provenance = _psu_provenance("hopper")


class B200PSUEfficiency(PSUEfficiencyModel):
    n_installed_psu: PositiveCount = 6
    n_load_sharing_psu: PositiveCount = 3
    n_redundant_capacity_psu: PositiveCount = 3
    psu_capacity_w: PositivePower = 5250.0
    redundancy: str = "3+3"
    provenance: Provenance = _psu_provenance("b200")


class B300PSUEfficiency(PSUEfficiencyModel):
    n_installed_psu: PositiveCount = 12
    n_load_sharing_psu: PositiveCount = 12
    n_redundant_capacity_psu: PositiveCount = 6
    psu_capacity_w: PositivePower = 3300.0
    system_max_w: PositivePower | None = 15000.0
    redundancy: str = "N+N / 6+6"
    provenance: Provenance = _psu_provenance("b300")


class MI300PSUEfficiency(PSUEfficiencyModel):
    n_installed_psu: PositiveCount = 6
    n_load_sharing_psu: PositiveCount = 6
    n_redundant_capacity_psu: PositiveCount = 3
    psu_capacity_w: PositivePower = 3000.0
    system_max_w: PositivePower | None = 9000.0
    redundancy: str = "N+N / 3+3"
    provenance: Provenance = _psu_provenance("amd_oam")


class MI325PSUEfficiency(MI300PSUEfficiency):
    psu_capacity_w: PositivePower = 5250.0
    system_max_w: PositivePower | None = 15750.0


class MI355PSUEfficiency(MI300PSUEfficiency):
    n_redundant_capacity_psu: PositiveCount = 4
    psu_capacity_w: PositivePower = 6600.0
    system_max_w: PositivePower | None = 26400.0
    redundancy: str = "4+2"
