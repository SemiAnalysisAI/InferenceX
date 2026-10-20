# SPDX-License-Identifier: GPL-3.0-only
"""Air-load fan control and coupled cooling/conversion calculation."""

from collections.abc import Callable
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

Fraction = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class RackFanAssembly(FrozenModel):
    fan_count: PositiveCount
    rated_power_w_per_fan: Watts = 30.24
    minimum_speed_fraction: Fraction = 0.4
    maximum_speed_fraction: Fraction = 0.8
    design_air_heat_w: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 500.0
    provenance: Provenance = Provenance(
        profile_id="rack-air-cooling",
        version="1",
        kind="estimated",
        source="Engineering fan policy; Sanyo Denki 9CRH0412P5J001 rated input reference",
        input_boundary="Complete fan module DC input, including both rotors where applicable",
        assumptions=(
            "30.24 W module is a representative part, not a verified OEM fan SKU.",
            "Linear air-heat-to-speed policy, 40-80% RPM and cubic electrical-power approximation.",
            "Speed fraction is not PWM duty; thermal anchors and fan counts are assumptions.",
        ),
    )

    @model_validator(mode="after")
    def validate_speed(self) -> Self:
        if self.minimum_speed_fraction > self.maximum_speed_fraction:
            raise ValueError("Minimum fan speed must not exceed maximum fan speed")
        return self

    def estimate_breakdown(self, air_heat_w: float) -> PowerComponentBreakdown:
        heat = validate_watts(air_heat_w)
        speed = self.minimum_speed_fraction + (
            self.maximum_speed_fraction - self.minimum_speed_fraction
        ) * min(heat / self.design_air_heat_w, 1)
        return PowerComponentBreakdown(
            name="Fan module",
            quantity=self.fan_count,
            power_w=validate_watts(self.fan_count * self.rated_power_w_per_fan * speed**3),
            provenance=self.provenance,
            details=(
                ("speed_fraction", speed),
                ("air_heat_w", heat),
                ("rated_power_w_per_fan", self.rated_power_w_per_fan),
                ("design_air_heat_w", self.design_air_heat_w),
            ),
        )


def solve_cooling(
    component_power_w: float,
    air_heat_w: float,
    fans: RackFanAssembly,
    conversion_loss: Callable[[float], float],
) -> PowerComponentBreakdown:
    """Include air-cooled conversion loss without counting liquid-cooled chip heat."""
    load = validate_watts(component_power_w)
    air = validate_watts(air_heat_w)
    fan = fans.estimate_breakdown(air)
    for _ in range(128):
        loss = validate_watts(conversion_loss(validate_watts(load + fan.power_w)))
        updated = fans.estimate_breakdown(validate_watts(air + loss))
        if abs(updated.power_w - fan.power_w) < 1e-8:
            return updated
        fan = updated
    raise ValueError("Fan and conversion-loss iteration did not converge")
