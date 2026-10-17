# SPDX-License-Identifier: GPL-3.0-only
"""Fan power interface evaluated at each system's thermal load."""

from abc import ABC, abstractmethod

from power_model.base import (
    Details,
    FrozenModel,
    PowerComponentBreakdown,
    Provenance,
    validate_watts,
)
from power_model.models.advanced.profiles import FAN_INPUT


class FanPower(FrozenModel, ABC):
    provenance: Provenance = FAN_INPUT

    @abstractmethod
    def estimate_w(self, non_fan_power_w: float) -> float:
        """Return fan watts for one system, excluding the fans from its own input."""

    def estimate_breakdown(self, non_fan_power_w: float) -> PowerComponentBreakdown:
        load = validate_watts(non_fan_power_w)
        return PowerComponentBreakdown(
            name="FanPower",
            power_w=self.estimate_w(load),
            provenance=self.provenance,
            details=(
                ("model", type(self).__name__),
                ("non_fan_power_w_per_system", load),
            )
            + self._details(load),
        )

    def _details(self, non_fan_power_w: float) -> Details:
        return ()
