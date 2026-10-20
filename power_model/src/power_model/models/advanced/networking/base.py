# SPDX-License-Identifier: GPL-3.0-only
"""External networking power allocated to the modeled cluster, counted once."""

from abc import ABC, abstractmethod
from typing import Self

from pydantic import SerializeAsAny, TypeAdapter

from power_model.base import (
    DEFAULT_OPERATING_STATE,
    Details,
    FrozenModel,
    NonemptyString,
    OperatingState,
    PositiveCount,
    PositiveQuantity,
    PowerComponentBreakdown,
    Provenance,
    Watts,
)
from power_model.models.advanced.profiles import COMPONENT_INPUT


class ScaleOutNetworkingGear(FrozenModel, ABC):
    name: NonemptyString
    power_w: Watts
    provenance: Provenance = COMPONENT_INPUT

    @abstractmethod
    def for_operating_state(self, state: OperatingState) -> Self:
        """Resolve the equipment's power state for this scenario."""

    @abstractmethod
    def estimate_breakdown(self) -> PowerComponentBreakdown:
        """Return modeled network components and their total power."""


class NetworkGroup(FrozenModel):
    """A device count or fractional allocation of a shared network device."""

    gear: SerializeAsAny[ScaleOutNetworkingGear]
    quantity: PositiveQuantity = 1
    details: Details = ()

    def scaled(self, count: int) -> "NetworkGroup":
        count = TypeAdapter(PositiveCount).validate_python(count)
        return NetworkGroup(gear=self.gear, quantity=self.quantity * count, details=self.details)

    def estimate_breakdown(
        self, *, operating_state: OperatingState = DEFAULT_OPERATING_STATE
    ) -> PowerComponentBreakdown:
        state = OperatingState.model_validate(operating_state)
        result = self.gear.for_operating_state(state).estimate_breakdown().scaled(self.quantity)
        return PowerComponentBreakdown(
            name=result.name,
            power_w=result.power_w,
            quantity=result.quantity,
            children=result.children,
            provenance=result.provenance,
            details=(*result.details, *self.details),
            configuration=result.configuration,
        )
