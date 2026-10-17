# SPDX-License-Identifier: GPL-3.0-only
"""Reusable component interfaces and explicit equipment quantities."""

from abc import ABC, abstractmethod

from pydantic import SerializeAsAny

from power_model.base import (
    Details,
    FrozenModel,
    PositiveCount,
    PowerComponentBreakdown,
    Provenance,
    Watts,
)
from power_model.models.advanced.profiles import COMPONENT_INPUT


class GenericComponent(FrozenModel, ABC):
    provenance: Provenance = COMPONENT_INPUT

    @abstractmethod
    def estimate_w(self) -> float:
        """Return electrical watts for one component in its configured state."""

    def _details(self) -> Details:
        return ()

    def estimate_breakdown(self) -> PowerComponentBreakdown:
        return PowerComponentBreakdown(
            name=type(self).__name__,
            power_w=self.estimate_w(),
            provenance=self.provenance,
            details=self._details(),
        )


class FixedPowerComponent(GenericComponent):
    """An explicitly supplied power input, also usable for additional component types."""

    power_w: Watts

    def estimate_w(self) -> float:
        return self.power_w


class ComponentGroup(FrozenModel):
    component: SerializeAsAny[GenericComponent]
    quantity: PositiveCount = 1

    def estimate_breakdown(self) -> PowerComponentBreakdown:
        return self.component.estimate_breakdown().scaled(self.quantity)
