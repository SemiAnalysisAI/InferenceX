# SPDX-License-Identifier: GPL-3.0-only
"""NIC capacity metadata and fixed active/idle electrical power."""

from abc import abstractmethod
from enum import StrEnum
from typing import Self

from pydantic import field_validator

from power_model.base import Details, OperatingState
from power_model.models.advanced.components.base import GenericComponent


class NICState(StrEnum):
    ACTIVE = "active"
    IDLE = "idle"


class ScaleOutNICCard(GenericComponent):
    state: NICState

    @property
    @abstractmethod
    def active_power_w(self) -> float:
        """Per-card power when using scale-out networking."""

    @property
    @abstractmethod
    def idle_power_w(self) -> float:
        """Per-card power when scale-out networking is unused."""

    @property
    @abstractmethod
    def bandwidth_gbps(self) -> int:
        """Configured nominal bandwidth, independent of the card's state."""

    @field_validator("state", mode="before")
    @classmethod
    def parse_state(cls, value: object) -> NICState:
        if not isinstance(value, str):
            raise ValueError("NIC state must be 'active' or 'idle'")
        return NICState(value)

    def estimate_w(self) -> float:
        return self.active_power_w if self.state == NICState.ACTIVE else self.idle_power_w

    def for_operating_state(self, state: OperatingState) -> Self:
        return type(self).model_validate(self.model_dump() | {"state": state.nic_state})

    def _details(self) -> Details:
        return (
            ("state", self.state.value),
            ("bandwidth_gbps", self.bandwidth_gbps),
            ("active_power_w", self.active_power_w),
            ("idle_power_w", self.idle_power_w),
        )
