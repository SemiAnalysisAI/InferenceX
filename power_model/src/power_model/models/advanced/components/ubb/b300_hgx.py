# SPDX-License-Identifier: GPL-3.0-only
"""B300 HGX: eight GPUs, two Blackwell NVSwitches, and eight ConnectX-8 NICs."""

from typing import Literal, Self

from pydantic import Field, model_validator

from power_model.base import OperatingState, PowerComponentBreakdown, Provenance, sum_watts
from power_model.models.advanced.components.base import ComponentGroup
from power_model.models.advanced.components.scaleoutnic import ConnectX8NICCard
from power_model.models.advanced.components.ubb.blackwell_hgx_nvswitch import BlackwellHGXNVSwitch
from power_model.models.advanced.components.ubb.nvidia_8way_hgx import NVIDIA8WayHGXBoard
from power_model.models.advanced.profiles import B300_BOARD_BASELINE


class B300HGXBoard(NVIDIA8WayHGXBoard):
    gpu_tdp_w: Literal[1200] = 1200
    nic: ConnectX8NICCard = Field(default_factory=lambda: ConnectX8NICCard(state="idle"))
    provenance: Provenance = B300_BOARD_BASELINE

    def for_operating_state(self, state: OperatingState) -> Self:
        return type(self).model_validate(
            self.model_dump(exclude={"non_gpu_power_w", "nic"})
            | {"nic": self.nic.for_operating_state(state)}
        )

    @staticmethod
    def _board_groups(nic: ConnectX8NICCard) -> tuple[ComponentGroup, ...]:
        return (
            ComponentGroup(component=BlackwellHGXNVSwitch(), quantity=2),
            ComponentGroup(component=nic, quantity=8),
        )

    @model_validator(mode="before")
    @classmethod
    def derive_board_power(cls, values: object) -> object:
        if isinstance(values, dict) and "non_gpu_power_w" not in values:
            nic = ConnectX8NICCard.model_validate(values.get("nic", {"state": "idle"}))
            power = sum_watts(
                tuple(group.estimate_breakdown().power_w for group in cls._board_groups(nic))
            )
            return values | {"nic": nic, "non_gpu_power_w": power}
        return values

    @model_validator(mode="after")
    def enforce_board_power(self) -> Self:
        expected = sum_watts(tuple(node.power_w for node in self._non_gpu_components()))
        if self.non_gpu_power_w != expected:
            raise ValueError("B300 non-GPU board power is determined by its switches and NIC state")
        return self

    def _non_gpu_components(self) -> tuple[PowerComponentBreakdown, ...]:
        return tuple(group.estimate_breakdown() for group in self._board_groups(self.nic))
