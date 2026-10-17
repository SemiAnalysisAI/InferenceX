# SPDX-License-Identifier: GPL-3.0-only
"""The eight-GPU UBB interface with component-owned board power."""

from abc import ABC, abstractmethod
from typing import Self

from pydantic import Field, model_validator

from power_model.base import (
    FrozenModel,
    OperatingState,
    PositiveCount,
    PowerComponentBreakdown,
    Provenance,
    Watts,
    validate_watts,
)
from power_model.models.advanced.profiles import COMPONENT_INPUT, GPU_INPUT


class UniversalBaseBoard(FrozenModel, ABC):
    gpu_tdp_w: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description="Maximum design power of one GPU module in watts at the GPU boundary",
    )
    gpu_count: PositiveCount = 8
    non_gpu_power_w: Watts
    provenance: Provenance = COMPONENT_INPUT

    def for_operating_state(self, state: OperatingState) -> Self:
        """Boards without state-dependent components retain their fixed inventory."""
        return self

    @model_validator(mode="after")
    def eight_gpus(self) -> Self:
        if self.gpu_count != 8:
            raise ValueError("An HGX UniversalBaseBoard must contain exactly 8 GPUs")
        return self

    @abstractmethod
    def _non_gpu_components(self) -> tuple[PowerComponentBreakdown, ...]:
        """Return the inventory contributing non-GPU board power."""

    def estimate_breakdown(self, gpu_level_power_per_gpu: float) -> PowerComponentBreakdown:
        per_gpu = validate_watts(gpu_level_power_per_gpu)
        return PowerComponentBreakdown.group(
            type(self).__name__,
            (
                PowerComponentBreakdown(
                    name="GPUs",
                    power_w=self.gpu_count * per_gpu,
                    quantity=self.gpu_count,
                    provenance=GPU_INPUT,
                    details=(("gpu_level_power_per_gpu", per_gpu),)
                    + ((("gpu_tdp_w", self.gpu_tdp_w),) if self.gpu_tdp_w is not None else ()),
                ),
                *self._non_gpu_components(),
            ),
            provenance=self.provenance,
        )
