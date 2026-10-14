# SPDX-License-Identifier: GPL-3.0-only
"""Generic host components; unspecified hardware wattages remain explicit inputs."""

from typing import ClassVar, Literal, Self

from pydantic import model_validator

from power_model.base import (
    Details,
    PositiveCount,
    Provenance,
    Watts,
    WorkloadState,
    validate_cpu_offload,
)
from power_model.models.advanced.components.base import FixedPowerComponent, GenericComponent
from power_model.models.advanced.profiles import (
    DDR5_BASELINE,
    NVME_BASELINE,
    PCIE5_RETIMER_BASELINE,
    PEX89144_BASELINE,
    X86_CPU_BASELINE,
    system_side_transceiver_baseline,
)

OPTICS_800G_POWER_W = 15.0


class CPU(FixedPowerComponent):
    """CPU power under the caller's stated component boundary."""


class X86CPU(CPU):
    """One x86 CPU with fixed state power, excluding separately modeled DIMMs."""

    state: Literal["Idle", "FixedSeqLen", "Agentic", "AgenticOffloadingOn"]
    architecture: ClassVar[str] = "x86"
    provenance: Provenance = X86_CPU_BASELINE
    _state_power_w: ClassVar[dict[str, float]] = {
        "Idle": 80.0,
        "FixedSeqLen": 120.0,
        "Agentic": 150.0,
        "AgenticOffloadingOn": 200.0,
    }

    @classmethod
    def for_workload(cls, workload_state: WorkloadState) -> Self:
        state = {
            WorkloadState.IDLE: "Idle",
            WorkloadState.FIXED_SEQ_LEN: "FixedSeqLen",
            WorkloadState.AGENTIC: "Agentic",
            WorkloadState.AGENTIC_CPU_OFFLOADING: "AgenticOffloadingOn",
        }[workload_state]
        return cls(state=state)

    @model_validator(mode="before")
    @classmethod
    def derive_state_power(cls, values: object) -> object:
        if isinstance(values, dict) and isinstance(state := values.get("state"), str):
            if state in cls._state_power_w and "power_w" not in values:
                return values | {"power_w": cls._state_power_w[state]}
        return values

    @model_validator(mode="after")
    def enforce_state_power(self) -> Self:
        if self.power_w != self._state_power_w[self.state]:
            raise ValueError(
                f"x86 CPU power for {self.state} must be {self._state_power_w[self.state]:g} W"
            )
        return self

    def _details(self) -> Details:
        return (
            ("architecture", self.architecture),
            ("state", self.state),
            ("per_cpu_power_w", self.power_w),
        )


class DDR5DIMM(GenericComponent):
    """One DDR5 DIMM, idle at 3.8 W or active during CPU offloading at 7 W."""

    capacity_gb: PositiveCount = 64
    state: Literal["idle", "active"] = "idle"
    memory_type: ClassVar[str] = "DDR5"
    idle_power_w: ClassVar[float] = 3.8
    active_power_w: ClassVar[float] = 7.0
    provenance: Provenance = DDR5_BASELINE

    @property
    def power_w(self) -> float:
        return self.active_power_w if self.state == "active" else self.idle_power_w

    def estimate_w(self) -> float:
        return self.power_w

    def for_cpu_offload(self, cpu_offload: bool) -> Self:
        """Return a validated scenario instance, preserving the original configuration."""
        enabled = validate_cpu_offload(cpu_offload)
        return type(self).model_validate(
            self.model_dump() | {"state": "active" if enabled else "idle"}
        )

    def _details(self) -> Details:
        return (
            ("memory_type", self.memory_type),
            ("state", self.state),
            ("capacity_gb_per_dimm", self.capacity_gb),
            ("per_dimm_power_w", self.power_w),
            ("idle_power_w", self.idle_power_w),
            ("active_power_w", self.active_power_w),
        )


class SystemSideTransceiver(FixedPowerComponent):
    """System-side optics, excluded from external scale-out gear."""


class SystemSide400GTransceiver(SystemSideTransceiver):
    """One always-powered 400 Gbit/s optical transceiver at 8 W."""

    power_w: Watts = 8.0
    bandwidth_gbps: ClassVar[int] = 400
    provenance: Provenance = system_side_transceiver_baseline(400, 8.0)

    @model_validator(mode="after")
    def enforce_baseline(self) -> Self:
        if self.power_w != 8.0:
            raise ValueError("400G system-side transceiver power must be 8 W")
        return self

    def _details(self) -> Details:
        return (
            ("bandwidth_gbps", self.bandwidth_gbps),
            ("per_transceiver_power_w", self.power_w),
            ("power_policy", "always_on"),
        )


class SystemSide800GTransceiver(SystemSideTransceiver):
    """One always-powered 800 Gbit/s optical transceiver at 15 W."""

    power_w: Watts = OPTICS_800G_POWER_W
    bandwidth_gbps: ClassVar[int] = 800
    provenance: Provenance = system_side_transceiver_baseline(800, OPTICS_800G_POWER_W)

    @model_validator(mode="after")
    def enforce_baseline(self) -> Self:
        if self.power_w != OPTICS_800G_POWER_W:
            raise ValueError("800G system-side transceiver power must be 15 W")
        return self

    def _details(self) -> Details:
        return (
            ("bandwidth_gbps", self.bandwidth_gbps),
            ("per_transceiver_power_w", self.power_w),
            ("power_policy", "always_on"),
        )


class SwitchSide800GSR8Transceiver(FixedPowerComponent):
    """One always-powered switch-side 800G SR8 optic, using the shared 800G power budget."""

    power_w: Watts = OPTICS_800G_POWER_W
    bandwidth_gbps: ClassVar[int] = 800
    provenance: Provenance = Provenance(
        profile_id="switch-side-800g-sr8-transceiver",
        version="1",
        source="User-specified model baseline; shared 800G optics power assumption",
        kind="assumed",
        input_boundary="One switch-side optical transceiver; excludes system-side optics",
        assumptions=("15 W per 800G SR8 optic, always powered in both switch states.",),
    )

    @model_validator(mode="after")
    def enforce_baseline(self) -> Self:
        if self.power_w != OPTICS_800G_POWER_W:
            raise ValueError("800G SR8 switch-side transceiver power must be 15 W")
        return self

    def _details(self) -> Details:
        return (
            ("bandwidth_gbps", self.bandwidth_gbps),
            ("optical_standard", "SR8"),
            ("per_transceiver_power_w", self.power_w),
            ("power_policy", "always_on"),
        )


class FrontendDPU(FixedPowerComponent):
    """A frontend data processing unit."""


class PCIeSwitch(FixedPowerComponent):
    """A PCIe switch that is not already included in the UBB input."""


class PEX89144PCIeSwitch(PCIeSwitch):
    """An always-active Broadcom PCIe 5.0, 144-lane switch at 45 W."""

    state: Literal["active"] = "active"
    power_w: Watts = 45.0
    vendor: ClassVar[str] = "Broadcom"
    part_number: ClassVar[str] = "PEX89144"
    pcie_generation: ClassVar[int] = 5
    lane_count: ClassVar[int] = 144
    provenance: Provenance = PEX89144_BASELINE

    @model_validator(mode="after")
    def enforce_baseline(self) -> Self:
        if self.power_w != 45.0:
            raise ValueError("PEX89144 active power must be 45 W in model version 0.1")
        return self

    def _details(self) -> Details:
        return (
            ("vendor", self.vendor),
            ("part_number", self.part_number),
            ("pcie_generation", self.pcie_generation),
            ("lane_count", self.lane_count),
            ("state", self.state),
            ("per_switch_power_w", self.power_w),
        )


class PCIe4x16Retimer(FixedPowerComponent):
    """A generic PCIe 4.0 x16 retimer with caller-supplied power."""

    pcie_generation: ClassVar[int] = 4
    lane_count: ClassVar[int] = 16

    def _details(self) -> Details:
        return (
            ("pcie_generation", self.pcie_generation),
            ("lane_count", self.lane_count),
            ("per_retimer_power_w", self.power_w),
        )


class PCIe5x16Retimer(FixedPowerComponent):
    """A generic PCIe 5.0 x16 retimer using the agreed 12 W baseline."""

    power_w: Watts = 12.0
    pcie_generation: ClassVar[int] = 5
    lane_count: ClassVar[int] = 16
    provenance: Provenance = PCIE5_RETIMER_BASELINE

    @model_validator(mode="after")
    def enforce_baseline(self) -> Self:
        if self.power_w != 12.0:
            raise ValueError("PCIe 5.0 x16 retimer power must be 12 W in model version 0.1")
        return self

    def _details(self) -> Details:
        return (
            ("pcie_generation", self.pcie_generation),
            ("lane_count", self.lane_count),
            ("per_retimer_power_w", self.power_w),
        )


class NVMeDrive(GenericComponent):
    state: Literal["idle"] = "idle"
    power_w: Watts = 5.0
    provenance: Provenance = NVME_BASELINE

    @model_validator(mode="after")
    def enforce_baseline(self) -> Self:
        if self.power_w != 5.0:
            raise ValueError("NVMe idle power must be 5 W in model version 0.1")
        return self

    def estimate_w(self) -> float:
        return self.power_w

    def _details(self) -> Details:
        return (("state", self.state), ("per_drive_power_w", self.power_w))
