# SPDX-License-Identifier: GPL-3.0-only
"""One Grace CPU, one 512 GB memory pool, and exactly two GPUs."""

from typing import ClassVar

from pydantic import Field

from power_model.base import (
    DEFAULT_OPERATING_STATE,
    FrozenModel,
    OperatingState,
    PositiveCount,
    PowerComponentBreakdown,
    Provenance,
    validate_watts,
)
from power_model.models.advanced.components.grace_cpu import GraceCPU, GraceWorkloadProfile
from power_model.models.advanced.components.lpddr5x import LPDDR5XMemory
from power_model.models.advanced.profiles import GPU_INPUT


class BiancaBoard(FrozenModel):
    gpu_count: PositiveCount = Field(default=2, ge=2, le=2)
    gpu_tdp_w: float = Field(
        gt=0,
        allow_inf_nan=False,
        description="Maximum design power of one GPU module; separate from actual input watts",
    )
    gpu_family: ClassVar[str]
    workload_profile: GraceWorkloadProfile = GraceWorkloadProfile()
    provenance: Provenance = Provenance(
        profile_id="bianca-board",
        version="2",
        kind="estimated",
        source="User-specified two-GPU / one-Grace / 512 GB board inventory",
        input_boundary="Modeled GPU, CPU and memory loads downstream of the compute-tray converter",
        assumptions=(
            "GPU input includes local GPU module regulation but excludes tray DC/DC loss.",
            "One 48 V-class converter belongs to the compute tray; no per-Grace converter.",
            "No separate board-level regulation loss is added to this selected inventory.",
            "512 GB memory is one pool, not power per DIMM or SOCAMM module.",
        ),
    )

    def estimate_breakdown(
        self,
        gpu_level_power_per_gpu: float,
        *,
        operating_state: OperatingState = DEFAULT_OPERATING_STATE,
    ) -> PowerComponentBreakdown:
        per_gpu = validate_watts(gpu_level_power_per_gpu)
        state = OperatingState.model_validate(operating_state)
        point = self.workload_profile.resolve(state)
        cpu = GraceCPU(
            power_w=point.cpu_power_w,
            workload_state=state.workload_state,
            provenance=self.workload_profile.provenance,
        ).estimate_breakdown()
        memory = LPDDR5XMemory(bandwidth_gbps=point.memory_bandwidth_gbps).estimate_breakdown()
        return PowerComponentBreakdown.group(
            type(self).__name__,
            (
                PowerComponentBreakdown(
                    name="GPUs",
                    quantity=2,
                    power_w=2 * per_gpu,
                    provenance=GPU_INPUT,
                    details=(
                        ("gpu_family", self.gpu_family),
                        ("gpu_tdp_w", self.gpu_tdp_w),
                        ("gpu_level_power_per_gpu", per_gpu),
                        ("input_boundary", "GPU module, after tray DC/DC"),
                    ),
                ),
                cpu,
                memory,
            ),
            provenance=self.provenance,
        )

    def air_heat_w(self, board: PowerComponentBreakdown) -> float:
        """Baseline cold-plates the GPU and CPU; memory heat reaches air."""
        return board.children[2].power_w
