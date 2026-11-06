# SPDX-License-Identifier: GPL-3.0-only
"""Grace workload power, excluding LPDDR5X and tray conversion losses."""

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
    version="2",
    kind="estimated",
    source="Provisional calibration from ACPI Grace Power Socket telemetry on GB200 and GB300",
    input_boundary=(
        "One Grace socket less modeled LPDDR5X: CPU and SysIO rails plus socket regulator "
        "loss; excludes tray conversion losses"
    ),
    assumptions=(
        "Fixed sequence / agentic / offloading CPU power is 95 / 90 / 85 W.",
        "Each value is a measured socket-total midpoint less modeled LPDDR5X at the state "
        "bandwidth: 100 W fixed-seq 8k/1k, 95 W AgentX offload off, 102 W AgentX offload on.",
        "Socket total therefore matches the measured socket input when telemetry is absent.",
        "CPU plus SysIO rails alone read about 50-64 W; the ~40 W remainder is LPDDR5X "
        "and regulator loss.",
        "Corresponding physical LPDDR traffic is 25 / 50 / 250 GB/s per Grace CPU; these "
        "bandwidths are estimates, which is why offloading CPU power is lowest.",
        "Load moves socket power by about 8-12 W; offload on vs off is not isolated.",
        "AgentX offload-off covers GB300 at concurrency 1-4 only.",
    ),
)


class GraceWorkloadPoint(FrozenModel):
    cpu_power_w: Watts
    memory_bandwidth_gbps: Watts


class GraceWorkloadProfile(FrozenModel):
    fixed_seq_len: GraceWorkloadPoint = GraceWorkloadPoint(cpu_power_w=95, memory_bandwidth_gbps=25)
    agentic: GraceWorkloadPoint = GraceWorkloadPoint(cpu_power_w=90, memory_bandwidth_gbps=50)
    agentic_cpu_offloading: GraceWorkloadPoint = GraceWorkloadPoint(
        cpu_power_w=85, memory_bandwidth_gbps=250
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
