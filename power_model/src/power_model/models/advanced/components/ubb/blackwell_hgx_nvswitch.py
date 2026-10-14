# SPDX-License-Identifier: GPL-3.0-only
"""The user-specified Blackwell HGX NVSwitch baseline."""

from typing import ClassVar, Self

from pydantic import model_validator

from power_model.base import Details, Provenance, Watts
from power_model.models.advanced.components.base import FixedPowerComponent
from power_model.models.advanced.profiles import BLACKWELL_NVSWITCH_BASELINE


class BlackwellHGXNVSwitch(FixedPowerComponent):
    power_w: Watts = 200.0
    bandwidth_tbps: ClassVar[float] = 28.8
    provenance: Provenance = BLACKWELL_NVSWITCH_BASELINE

    @model_validator(mode="after")
    def enforce_baseline(self) -> Self:
        if self.power_w != 200.0:
            raise ValueError("Blackwell HGX NVSwitch power must be 200 W in model version 0.1")
        return self

    def _details(self) -> Details:
        return (("bandwidth_tbps", self.bandwidth_tbps), ("per_switch_power_w", self.power_w))
