# SPDX-License-Identifier: GPL-3.0-only
"""Grace workload assumptions, excluding LPDDR5X and conversion losses."""

from power_model.base import (
    Details,
    FrozenModel,
    OperatingState,
    Provenance,
    Watts,
    WorkloadState,
)
from power_model.models.advanced.components.base import GenericComponent

GRACE_BASELINE = Provenance(
    profile_id="grace-workload-baseline",
    version="1",
    kind="estimated",
    source="Proposed project workload assignments",
    input_boundary="One Grace CPU and SysIO after local regulation; excludes DRAM and losses",
    assumptions=(
        "Fixed sequence / agentic / offloading CPU power is 100 / 140 / 180 W.",
        "Corresponding physical LPDDR traffic is 25 / 50 / 250 GB/s per Grace CPU.",
        "These are scenario assumptions, not measured workload results.",
    ),
)


class GraceWorkloadPoint(FrozenModel):
    cpu_power_w: Watts
    memory_bandwidth_gbps: Watts


class GraceWorkloadProfile(FrozenModel):
    fixed_seq_len: GraceWorkloadPoint = GraceWorkloadPoint(
        cpu_power_w=100, memory_bandwidth_gbps=25
    )
    agentic: GraceWorkloadPoint = GraceWorkloadPoint(cpu_power_w=140, memory_bandwidth_gbps=50)
    agentic_cpu_offloading: GraceWorkloadPoint = GraceWorkloadPoint(
        cpu_power_w=180, memory_bandwidth_gbps=250
    )
    idle: GraceWorkloadPoint | None = None
    provenance: Provenance = GRACE_BASELINE

    def resolve(self, state: OperatingState) -> GraceWorkloadPoint:
        state = OperatingState.model_validate(state)
        point = {
            WorkloadState.FIXED_SEQ_LEN: self.fixed_seq_len,
            WorkloadState.AGENTIC: self.agentic,
            WorkloadState.AGENTIC_CPU_OFFLOADING: self.agentic_cpu_offloading,
            WorkloadState.IDLE: self.idle,
        }[state.workload_state]
        if point is None:
            raise ValueError("Grace idle power requires an explicit calibrated workload profile")
        return point


class GraceCPU(GenericComponent):
    power_w: Watts
    workload_state: WorkloadState
    provenance: Provenance = GRACE_BASELINE

    def estimate_w(self) -> float:
        return self.power_w

    def _details(self) -> Details:
        return (
            ("architecture", "Arm"),
            ("workload_state", self.workload_state.value),
            ("per_cpu_power_w", self.power_w),
        )
