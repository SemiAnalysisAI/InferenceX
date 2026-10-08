# SPDX-License-Identifier: GPL-3.0-only
"""Complete equipment power and operating-state propagation before facility PUE."""

from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import JsonValue, SerializeAsAny

from power_model.base import (
    DEFAULT_OPERATING_STATE,
    FrozenModel,
    OperatingState,
    PositiveCount,
    PowerComponentBreakdown,
    Provenance,
)
from power_model.cooling import CoolingProfile
from power_model.models.advanced.networking.base import NetworkGroup
from power_model.models.advanced.profiles import COMPONENT_INPUT


def _profile_parameters(model: FrozenModel) -> dict[str, JsonValue]:
    """Export parameters once; provenance belongs to the component result tree."""

    def parameters(value: JsonValue) -> JsonValue:
        if isinstance(value, dict):
            return {key: parameters(item) for key, item in value.items() if key != "provenance"}
        if isinstance(value, list):
            return [parameters(item) for item in value]
        return value

    return {
        key: parameters(value)
        for key, value in model.model_dump(mode="json", serialize_as_any=True).items()
        if key != "provenance"
    }


class GPUSystem(FrozenModel, ABC):
    default_cooling: ClassVar[CoolingProfile]
    system_unit: ClassVar[str] = "chassis"
    default_networking: ClassVar[tuple[NetworkGroup, ...] | None] = None
    gpu_count: PositiveCount
    provenance: Provenance = COMPONENT_INPUT

    @property
    @abstractmethod
    def system_type(self) -> str:
        """Hardware family identifier supplied by concrete systems."""

    @abstractmethod
    def estimate_it_power(
        self,
        gpu_level_power_per_gpu: float,
        *,
        operating_state: OperatingState = DEFAULT_OPERATING_STATE,
    ) -> PowerComponentBreakdown:
        """Evaluate a single complete system without PUE."""

    def _configuration(self, state: OperatingState) -> dict[str, JsonValue]:
        return (
            _profile_parameters(self)
            | state.model_dump(mode="json")
            | {
                "cpu_offload": state.cpu_offload,
            }
        )


class SystemGroup(FrozenModel):
    system: SerializeAsAny[GPUSystem]
    quantity: PositiveCount = 1

    @property
    def gpu_count(self) -> int:
        return self.quantity * self.system.gpu_count

    def estimate_it_power(
        self,
        gpu_level_power_per_gpu: float,
        *,
        operating_state: OperatingState = DEFAULT_OPERATING_STATE,
    ) -> PowerComponentBreakdown:
        return self.system.estimate_it_power(
            gpu_level_power_per_gpu, operating_state=operating_state
        ).scaled(self.quantity)
