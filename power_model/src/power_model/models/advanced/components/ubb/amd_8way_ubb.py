# SPDX-License-Identifier: GPL-3.0-only
"""Shared AMD UBB inventory: eight GPUs and eight PCIe 5.0 x16 retimers."""

from typing import ClassVar, Self

from pydantic import model_validator

from power_model.base import PowerComponentBreakdown, Provenance, Watts, sum_watts
from power_model.models.advanced.components.base import ComponentGroup
from power_model.models.advanced.components.generic import PCIe5x16Retimer
from power_model.models.advanced.components.ubb.ubb import UniversalBaseBoard
from power_model.models.advanced.profiles import AMD_UBB_BASELINE


class AMD8WayUBB(UniversalBaseBoard):
    board_components: ClassVar[tuple[ComponentGroup, ...]] = (
        ComponentGroup(component=PCIe5x16Retimer(), quantity=8),
    )
    non_gpu_power_w: Watts = sum_watts(
        tuple(group.estimate_breakdown().power_w for group in board_components)
    )
    provenance: Provenance = AMD_UBB_BASELINE

    @model_validator(mode="after")
    def enforce_board_power(self) -> Self:
        expected = sum_watts(tuple(node.power_w for node in self._non_gpu_components()))
        if self.non_gpu_power_w != expected:
            raise ValueError("AMD UBB non-GPU board power is determined by its eight retimers")
        return self

    def _non_gpu_components(self) -> tuple[PowerComponentBreakdown, ...]:
        return tuple(group.estimate_breakdown() for group in self.board_components)
