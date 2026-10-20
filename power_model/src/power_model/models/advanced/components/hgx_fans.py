# SPDX-License-Identifier: GPL-3.0-only
"""Shared fan control and TDP-based cooling capacity for normalized HGX comparisons."""

from typing import Annotated, Self

from pydantic import Field, computed_field, model_validator

from power_model.base import (
    Details,
    FrozenModel,
    PositiveCount,
    Provenance,
    Watts,
    sum_watts,
    validate_watts,
)
from power_model.models.advanced.components.fan import FanPower

Fraction = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
PositiveValue = Annotated[float, Field(gt=0, allow_inf_nan=False)]

HGX_FAN_BASELINE = Provenance(
    profile_id="normalized-hgx-fans",
    version="1",
    source="User-specified GPU TDPs and shared cooling-policy assumptions",
    kind="assumed",
    input_boundary="Chassis fan DC watts from pre-fan component DC load; excludes PSU loss and PUE",
    assumptions=(
        "Default GPU TDP per device: Hopper 700 W, B200 1000 W, B300 1200 W, "
        "MI300 750 W, MI325 1000 W, MI355 1400 W.",
        "Design heat is GPU count times GPU TDP plus the highest modeled non-GPU power: "
        "CPU offloading and active scale-out networking, including board NICs and optional extras.",
        "All default HGX systems share 22-80% PWM and load-to-PWM exponent 1.0.",
        "Fan watts at design heat use one shared ratio: 1100 * 0.8**3 / 9500, "
        "approximately 5.93%; the effective nameplate is derived from that ratio.",
        "This is a shared comparison baseline, not a measured chassis fan profile.",
    ),
)


class AffinityFanPower(FanPower):
    """Infer PWM from thermal load, then apply electrical nameplate times PWM cubed."""

    electrical_nameplate_w: Watts
    min_pwm_frac: Fraction
    normal_max_pwm_frac: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
    full_cooling_load_w: PositiveValue
    fan_curve_exponent: PositiveValue
    fan_complement: str = "Effective aggregate fan electrical nameplate"

    @model_validator(mode="after")
    def ordered_pwm_range(self) -> Self:
        if self.min_pwm_frac > self.normal_max_pwm_frac:
            raise ValueError("Minimum fan PWM must not exceed normal maximum PWM")
        return self

    def _pwm(self, load: float) -> float:
        fraction = min(1.0, load / self.full_cooling_load_w)
        pwm = self.min_pwm_frac + (self.normal_max_pwm_frac - self.min_pwm_frac) * (
            fraction**self.fan_curve_exponent
        )
        return min(self.normal_max_pwm_frac, max(self.min_pwm_frac, pwm))

    def estimate_w(self, non_fan_power_w: float) -> float:
        pwm = self._pwm(validate_watts(non_fan_power_w))
        return validate_watts(self.electrical_nameplate_w * pwm**3)

    def _details(self, non_fan_power_w: float) -> Details:
        pwm = self._pwm(non_fan_power_w)
        return (
            ("mode", "thermal_load_curve"),
            ("fan_pwm_fraction", pwm),
            ("electrical_nameplate_w", self.electrical_nameplate_w),
            ("fan_complement", self.fan_complement),
            ("cooling_load_fraction", min(1.0, non_fan_power_w / self.full_cooling_load_w)),
        )


class HGXFanPolicy(FrozenModel):
    """A common controller and cooling-efficiency assumption, independent of GPU family."""

    min_pwm_frac: Fraction = 0.22
    normal_max_pwm_frac: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = 0.80
    fan_curve_exponent: PositiveValue = 1.0
    fan_power_ratio_at_design: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = (
        1100.0 * 0.8**3 / 9500.0
    )

    @model_validator(mode="after")
    def ordered_pwm_range(self) -> Self:
        if self.min_pwm_frac > self.normal_max_pwm_frac:
            raise ValueError("Minimum fan PWM must not exceed normal maximum PWM")
        return self


class NormalizedHGXFanPower(FanPower):
    """Size cooling from GPU TDP and host design power, then apply the shared policy."""

    gpu_tdp_w: PositiveValue
    gpu_count: PositiveCount
    design_non_gpu_power_w: Watts
    policy: HGXFanPolicy = HGXFanPolicy()
    provenance: Provenance = HGX_FAN_BASELINE

    @computed_field
    @property
    def full_cooling_load_w(self) -> float:
        return sum_watts((self.gpu_count * self.gpu_tdp_w, self.design_non_gpu_power_w))

    @computed_field
    @property
    def electrical_nameplate_w(self) -> float:
        return validate_watts(
            self.full_cooling_load_w
            * self.policy.fan_power_ratio_at_design
            / self.policy.normal_max_pwm_frac**3
        )

    @model_validator(mode="after")
    def finite_design_power(self) -> Self:
        validate_watts(self.electrical_nameplate_w)
        return self

    @property
    def affinity_curve(self) -> AffinityFanPower:
        return AffinityFanPower(
            electrical_nameplate_w=self.electrical_nameplate_w,
            full_cooling_load_w=self.full_cooling_load_w,
            min_pwm_frac=self.policy.min_pwm_frac,
            normal_max_pwm_frac=self.policy.normal_max_pwm_frac,
            fan_curve_exponent=self.policy.fan_curve_exponent,
            fan_complement="Normalized capacity sized from GPU TDP and host design watts",
            provenance=self.provenance,
        )

    def estimate_w(self, non_fan_power_w: float) -> float:
        return self.affinity_curve.estimate_w(non_fan_power_w)

    def _details(self, non_fan_power_w: float) -> Details:
        return self.affinity_curve._details(non_fan_power_w) + (
            ("gpu_tdp_w", self.gpu_tdp_w),
            ("gpu_count_per_system", self.gpu_count),
            ("design_non_gpu_power_w", self.design_non_gpu_power_w),
            ("full_cooling_load_w", self.full_cooling_load_w),
            ("fan_power_ratio_at_design", self.policy.fan_power_ratio_at_design),
            ("min_pwm_frac", self.policy.min_pwm_frac),
            ("normal_max_pwm_frac", self.policy.normal_max_pwm_frac),
            ("fan_curve_exponent", self.policy.fan_curve_exponent),
        )
