# SPDX-License-Identifier: GPL-3.0-only
"""B200 HGX: eight GPUs, two Blackwell NVSwitches, and eight PCIe 5.0 x16 retimers."""

from typing import ClassVar, Literal, Self

from pydantic import model_validator

from power_model.base import PowerComponentBreakdown, Provenance, Watts, sum_watts
from power_model.models.advanced.components.base import ComponentGroup
from power_model.models.advanced.components.generic import PCIe5x16Retimer
from power_model.models.advanced.components.ubb.blackwell_hgx_nvswitch import BlackwellHGXNVSwitch
from power_model.models.advanced.components.ubb.nvidia_8way_hgx import NVIDIA8WayHGXBoard
from power_model.models.advanced.profiles import B200_BOARD_BASELINE


class B200HGXBoard(NVIDIA8WayHGXBoard):
    gpu_tdp_w: Literal[1000] = 1000
    board_components: ClassVar[tuple[ComponentGroup, ...]] = (
        ComponentGroup(component=BlackwellHGXNVSwitch(), quantity=2),
        ComponentGroup(component=PCIe5x16Retimer(), quantity=8),
    )
    non_gpu_power_w: Watts = sum_watts(
        tuple(group.estimate_breakdown().power_w for group in board_components)
    )
    provenance: Provenance = B200_BOARD_BASELINE

    @model_validator(mode="after")
    def enforce_board_power(self) -> Self:
        expected = sum_watts(tuple(node.power_w for node in self._non_gpu_components()))
        if self.non_gpu_power_w != expected:
            raise ValueError("B200 non-GPU board power is determined by its switches and retimers")
        return self

    def _non_gpu_components(self) -> tuple[PowerComponentBreakdown, ...]:
        return tuple(group.estimate_breakdown() for group in self.board_components)
